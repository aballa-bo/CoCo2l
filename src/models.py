"""White-preserving linear and HPPCC models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import hue_angle_from_rgb


def _solve_constrained_least_squares(a: np.ndarray, x: np.ndarray, c: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim != 2 or x.ndim != 2:
        raise ValueError("a and x must be 2-D arrays.")
    if c.ndim != 2 or b.ndim != 2:
        raise ValueError("c and b must be 2-D arrays.")
    lhs = np.block(
        [
            [2.0 * a.T @ a, c.T],
            [c, np.zeros((c.shape[0], c.shape[0]), dtype=np.float64)],
        ]
    )
    rhs = np.vstack([2.0 * a.T @ x, b])
    solution = np.linalg.solve(lhs, rhs)
    return solution[: a.shape[1], :]


@dataclass(frozen=True)
class LinearWhitePreservingModel:
    matrix: np.ndarray
    white_rgb: np.ndarray
    white_xyz: np.ndarray

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        return rgb @ self.matrix


def fit_white_preserving_3x3(rgb: np.ndarray, xyz: np.ndarray, white_index: int) -> LinearWhitePreservingModel:
    rgb = np.asarray(rgb, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    white_rgb = rgb[white_index]
    white_xyz = xyz[white_index]

    constraint = np.zeros((1, rgb.shape[1]), dtype=np.float64)
    constraint[0] = white_rgb
    matrix = _solve_constrained_least_squares(rgb, xyz, constraint, white_xyz[np.newaxis, :])
    return LinearWhitePreservingModel(matrix=matrix, white_rgb=white_rgb, white_xyz=white_xyz)


@dataclass(frozen=True)
class HPPCCModel:
    matrices: np.ndarray
    boundaries: np.ndarray
    white_rgb: np.ndarray
    white_xyz: np.ndarray

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        flat = rgb.reshape(-1, 3)
        angles = hue_angle_from_rgb(flat)
        region_indices = np.searchsorted(self.boundaries, angles, side="right") % len(self.boundaries)
        out = np.empty_like(flat)
        for region in range(len(self.boundaries)):
            mask = region_indices == region
            if np.any(mask):
                out[mask] = flat[mask] @ self.matrices[region]
        return out.reshape(rgb.shape)

    def predict_blending(self, rgb: np.ndarray, blend_width: float = 0.15) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        flat = rgb.reshape(-1, 3)
        angles = hue_angle_from_rgb(flat)
        boundaries = np.asarray(self.boundaries, dtype=np.float64)
        k_regions = len(boundaries)
        if k_regions < 2:
            return self.predict(rgb)

        full_boundaries = np.concatenate([boundaries, [2.0 * np.pi]])
        centers = np.empty(k_regions, dtype=np.float64)
        widths = np.empty(k_regions, dtype=np.float64)
        for region in range(k_regions):
            start = full_boundaries[region]
            end = full_boundaries[region + 1]
            widths[region] = end - start
            centers[region] = start + 0.5 * widths[region]

        if np.isscalar(blend_width):
            blend_fraction = float(blend_width)
            if blend_fraction <= 0.0:
                return self.predict(rgb)
            blend_halfwidths = 0.5 * blend_fraction * widths
        else:
            blend_halfwidths = np.asarray(blend_width, dtype=np.float64)
            if blend_halfwidths.shape != (k_regions,):
                raise ValueError("blend_width must be a scalar or have one value per region.")

        predictions = np.empty((k_regions, flat.shape[0], 3), dtype=np.float64)
        for region in range(k_regions):
            predictions[region] = flat @ self.matrices[region]

        delta = np.abs(angles[np.newaxis, :] - centers[:, np.newaxis])
        delta = np.minimum(delta, 2.0 * np.pi - delta)
        safe_halfwidths = np.maximum(blend_halfwidths[:, np.newaxis], 1e-12)
        weights = np.clip(1.0 - delta / safe_halfwidths, 0.0, 1.0)

        if np.any(blend_halfwidths <= 0.0):
            zero_mask = blend_halfwidths <= 0.0
            weights[zero_mask] = 0.0

        normalizer = np.sum(weights, axis=0, keepdims=True)
        hard_region_indices = np.searchsorted(boundaries, angles, side="right") % k_regions
        hard_mask = normalizer[0] <= 1e-12
        if np.any(hard_mask):
            weights[:, hard_mask] = 0.0
            weights[hard_region_indices[hard_mask], np.flatnonzero(hard_mask)] = 1.0
            normalizer = np.sum(weights, axis=0, keepdims=True)

        blended = np.sum(predictions * (weights / normalizer)[:, :, np.newaxis], axis=0)
        return blended.reshape(rgb.shape)


def _equal_count_boundaries(angles: np.ndarray, k_regions: int) -> np.ndarray:
    ordered = np.sort(np.asarray(angles, dtype=np.float64))
    if ordered.ndim != 1:
        raise ValueError("angles must be 1-D.")
    if k_regions < 2:
        raise ValueError("k_regions must be at least 2.")
    if len(ordered) < k_regions:
        raise ValueError("k_regions cannot exceed the number of chromatic samples.")
    idx = [int(np.floor(i * len(ordered) / k_regions)) for i in range(k_regions)]
    boundaries = ordered[idx]
    boundaries[0] = 0.0
    return boundaries


def _assign_regions(angles: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    return np.searchsorted(boundaries, angles, side="right") % len(boundaries)


def fit_hppcc(
    rgb: np.ndarray,
    xyz: np.ndarray,
    *,
    white_index: int,
    chromatic_indices: np.ndarray | list[int],
    k_regions: int = 4,
) -> HPPCCModel:
    """Fit the 2016 constrained least-squares HPPCC model with fixed equal-count hue regions.

    The model fit uses chromatic training patches to define hue partitions and regional least-squares
    blocks. The designated white patch enforces white-point preservation globally.
    """

    rgb = np.asarray(rgb, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    chromatic_indices = np.asarray(chromatic_indices, dtype=int)
    if rgb.shape != xyz.shape or rgb.shape[1] != 3:
        raise ValueError("rgb and xyz must both have shape (N, 3).")
    if len(chromatic_indices) < k_regions:
        raise ValueError("Not enough chromatic patches for the requested number of regions.")

    q = rgb[chromatic_indices]
    p = xyz[chromatic_indices]
    angles = hue_angle_from_rgb(q)
    boundaries = _equal_count_boundaries(angles, k_regions)
    order = np.argsort(angles)
    q_sorted = q[order]
    p_sorted = p[order]
    angles_sorted = angles[order]
    region_ids = _assign_regions(angles_sorted, boundaries)

    counts = np.bincount(region_ids, minlength=k_regions)
    if np.any(counts == 0):
        raise ValueError("Equal-count partitioning produced an empty hue region.")

    a = np.zeros((len(q_sorted), 3 * k_regions), dtype=np.float64)
    for region in range(k_regions):
        mask = region_ids == region
        a[np.ix_(mask, np.arange(3 * region, 3 * (region + 1)))] = q_sorted[mask]

    boundary_samples = np.zeros((k_regions, 3), dtype=np.float64)
    for region in range(k_regions):
        current_last = q_sorted[region_ids == region][-1]
        next_region = (region + 1) % k_regions
        next_first = q_sorted[region_ids == next_region][0]
        boundary_samples[region] = 0.5 * (current_last + next_first)

    c = np.zeros((2 * k_regions, 3 * k_regions), dtype=np.float64)
    b = np.zeros((2 * k_regions, 3), dtype=np.float64)

    for region in range(k_regions - 1):
        c[region, 3 * region : 3 * (region + 1)] = boundary_samples[region]
        c[region, 3 * (region + 1) : 3 * (region + 2)] = -boundary_samples[region]

    c[k_regions - 1, 3 * (k_regions - 1) : 3 * k_regions] = boundary_samples[-1]
    c[k_regions - 1, 0:3] = -boundary_samples[-1]

    white_rgb = rgb[white_index]
    for region in range(k_regions - 1):
        row = k_regions + region
        c[row, 3 * region : 3 * (region + 1)] = white_rgb
        c[row, 3 * (region + 1) : 3 * (region + 2)] = -white_rgb

    c[-1, 0:3] = white_rgb
    b[-1] = xyz[white_index]
    t = _solve_constrained_least_squares(a, p_sorted, c, b)
    matrices = t.reshape(k_regions, 3, 3)
    return HPPCCModel(
        matrices=matrices,
        boundaries=boundaries,
        white_rgb=rgb[white_index].copy(),
        white_xyz=xyz[white_index].copy(),
    )
