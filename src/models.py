"""White-preserving linear, RPCC, and HPPCC models."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

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


def _fit_hppcc_constrained(
    q_sorted: np.ndarray,
    p_sorted: np.ndarray,
    boundaries: np.ndarray,
    region_ids: np.ndarray,
    white_rgb: np.ndarray,
    white_xyz: np.ndarray,
) -> np.ndarray:
    """Fit K regional 3×3 matrices with boundary-continuity and white-point constraints.

    Returns matrices of shape (K, 3, 3).
    """
    k_regions = len(boundaries)
    n = len(q_sorted)

    a = np.zeros((n, 3 * k_regions), dtype=np.float64)
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

    for region in range(k_regions - 1):
        row = k_regions + region
        c[row, 3 * region : 3 * (region + 1)] = white_rgb
        c[row, 3 * (region + 1) : 3 * (region + 2)] = -white_rgb

    c[-1, 0:3] = white_rgb
    b[-1] = white_xyz

    t = _solve_constrained_least_squares(a, p_sorted, c, b)
    return t.reshape(k_regions, 3, 3)


def _exhaustive_boundary_search(
    q_sorted: np.ndarray,
    p_sorted: np.ndarray,
    angles_sorted: np.ndarray,
    k_regions: int,
    white_rgb: np.ndarray,
    white_xyz: np.ndarray,
    min_per_region: int = 1,
) -> np.ndarray:
    """Find hue boundaries minimising constrained LS residual via exhaustive split search.

    Evaluates all C(n-1, K-1) ways to partition n sorted chromatic patches into K groups.
    Falls back to equal-count boundaries if no valid split is found.
    """
    n = len(angles_sorted)
    best_boundaries = _equal_count_boundaries(angles_sorted, k_regions)
    best_residual = float("inf")

    for split_positions in combinations(range(1, n), k_regions - 1):
        # Reject trivially if split would leave fewer than min_per_region patches in any region
        sizes = [split_positions[0]] + [
            split_positions[i] - split_positions[i - 1] for i in range(1, len(split_positions))
        ] + [n - split_positions[-1]]
        if any(s < min_per_region for s in sizes):
            continue

        bounds = np.zeros(k_regions, dtype=np.float64)
        bounds[0] = 0.0
        for i, s in enumerate(split_positions):
            bounds[i + 1] = 0.5 * (angles_sorted[s - 1] + angles_sorted[s])

        region_ids = _assign_regions(angles_sorted, bounds)
        counts = np.bincount(region_ids, minlength=k_regions)
        if np.any(counts < min_per_region):
            continue

        try:
            matrices = _fit_hppcc_constrained(q_sorted, p_sorted, bounds, region_ids, white_rgb, white_xyz)
        except np.linalg.LinAlgError:
            continue

        residual = sum(
            float(np.sum((q_sorted[region_ids == r] @ matrices[r] - p_sorted[region_ids == r]) ** 2))
            for r in range(k_regions)
        )
        if residual < best_residual:
            best_residual = residual
            best_boundaries = bounds.copy()

    return best_boundaries


def fit_hppcc(
    rgb: np.ndarray,
    xyz: np.ndarray,
    *,
    white_index: int,
    chromatic_indices: np.ndarray | list[int],
    k_regions: int = 4,
    optimize_boundaries: bool = False,
) -> HPPCCModel:
    """Fit the 2016 constrained least-squares HPPCC model.

    When optimize_boundaries=True, performs exhaustive search over all C(n-1, K-1) hue
    partitions to find region boundaries that minimise the constrained LS residual.
    Otherwise uses equal-count partitioning.
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
    order = np.argsort(angles)
    q_sorted = q[order]
    p_sorted = p[order]
    angles_sorted = angles[order]
    white_rgb = rgb[white_index]
    white_xyz = xyz[white_index]

    if optimize_boundaries:
        boundaries = _exhaustive_boundary_search(
            q_sorted, p_sorted, angles_sorted, k_regions, white_rgb, white_xyz,
            min_per_region=3,
        )
    else:
        boundaries = _equal_count_boundaries(angles_sorted, k_regions)

    region_ids = _assign_regions(angles_sorted, boundaries)
    counts = np.bincount(region_ids, minlength=k_regions)
    if np.any(counts == 0):
        raise ValueError("Hue partitioning produced an empty region.")

    matrices = _fit_hppcc_constrained(q_sorted, p_sorted, boundaries, region_ids, white_rgb, white_xyz)
    return HPPCCModel(
        matrices=matrices,
        boundaries=boundaries,
        white_rgb=white_rgb.copy(),
        white_xyz=white_xyz.copy(),
    )


@dataclass(frozen=True)
class HPPCCRPCCModel:
    """Two-stage pipeline: HPPCC hue-preserving primary + RPCC global residual correction.

    Prediction: xyz = hppcc.predict(rgb) + rpcc_residual.predict(rgb)
    The RPCC component is trained on xyz residuals (xyz_ref - xyz_hppcc).
    Because HPPCC preserves the white point exactly, the RPCC residual for the white
    patch is zero and its white-point constraint is trivially satisfied.
    """

    hppcc: HPPCCModel
    rpcc_residual: "RPCCModel"

    @property
    def boundaries(self) -> np.ndarray:
        return self.hppcc.boundaries

    @property
    def matrices(self) -> np.ndarray:
        return self.hppcc.matrices

    @property
    def white_rgb(self) -> np.ndarray:
        return self.hppcc.white_rgb

    @property
    def white_xyz(self) -> np.ndarray:
        return self.hppcc.white_xyz

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        return self.hppcc.predict(rgb) + self.rpcc_residual.predict(rgb)

    def predict_blending(self, rgb: np.ndarray, blend_width: float = 0.15) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        return self.hppcc.predict_blending(rgb, blend_width=blend_width) + self.rpcc_residual.predict(rgb)


def _rpcc_features(rgb: np.ndarray) -> np.ndarray:
    """Root-Polynomial feature expansion: [R, G, B, sqrt(RG), sqrt(RB), sqrt(GB)]."""
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = np.sqrt(np.maximum(r * g, 0.0))
    rb = np.sqrt(np.maximum(r * b, 0.0))
    gb = np.sqrt(np.maximum(g * b, 0.0))
    return np.stack([r, g, b, rg, rb, gb], axis=-1)


@dataclass(frozen=True)
class RPCCModel:
    """Root-Polynomial Color Correction (Finlayson) with white-point preservation.

    Maps camera RGB → XYZ using a white-constrained 6×3 matrix over the
    feature vector [R, G, B, √(RG), √(RB), √(GB)].
    """

    matrix: np.ndarray  # shape (6, 3)
    white_rgb: np.ndarray
    white_xyz: np.ndarray

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        features = _rpcc_features(np.asarray(rgb, dtype=np.float64))
        return features @ self.matrix


def fit_rpcc(rgb: np.ndarray, xyz: np.ndarray, white_index: int) -> RPCCModel:
    """Fit a white-preserving Root-Polynomial Color Correction matrix.

    Solves: min ||F @ M - XYZ||²  s.t.  F_white @ M = XYZ_white
    where F is the 6-feature root-polynomial expansion of camera RGB.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    features = _rpcc_features(rgb)
    white_features = features[white_index]
    white_xyz = xyz[white_index].copy()
    constraint = white_features[np.newaxis, :]
    matrix = _solve_constrained_least_squares(features, xyz, constraint, white_xyz[np.newaxis, :])
    return RPCCModel(matrix=matrix, white_rgb=rgb[white_index].copy(), white_xyz=white_xyz)



def fit_hppcc_rpcc(
    rgb: np.ndarray,
    xyz: np.ndarray,
    *,
    white_index: int,
    chromatic_indices: np.ndarray | list[int],
    k_regions: int = 4,
    optimize_boundaries: bool = False,
) -> HPPCCRPCCModel:
    """Fit the inverted two-stage HPPCC + RPCC model.

    Stage 1 — fit_hppcc: hue-preserving regional mapping rgb → xyz_hppcc.
    Stage 2 — fit_rpcc on residuals: global root-polynomial correction of (xyz_ref - xyz_hppcc).
    Because HPPCC preserves the white point exactly, the RPCC residual constraint is zero.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    hppcc = fit_hppcc(
        rgb,
        xyz,
        white_index=white_index,
        chromatic_indices=chromatic_indices,
        k_regions=k_regions,
        optimize_boundaries=optimize_boundaries,
    )
    xyz_hppcc = hppcc.predict(rgb)
    xyz_residual = xyz - xyz_hppcc
    rpcc_residual = fit_rpcc(rgb, xyz_residual, white_index=white_index)
    return HPPCCRPCCModel(hppcc=hppcc, rpcc_residual=rpcc_residual)
