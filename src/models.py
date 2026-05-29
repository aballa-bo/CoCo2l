"""White-preserving linear, RPCC, HPPCC, HLCC, TPS, and LWCC models."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .config import HPPCC_GRADIENT_SMOOTH_SIGMA, HPPCC_REGION_SMOOTHNESS, HLCC_SECTORS, RPCC_RIDGE_LAMBDA
from .metrics import hue_angle_from_rgb, delta_e_2000, xyz_to_lab


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
    try:
        solution = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        solution, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
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

    def predict(self, rgb: np.ndarray, *, hue_rgb: np.ndarray | None = None) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        flat = rgb.reshape(-1, 3)
        hue_flat = np.asarray(hue_rgb, dtype=np.float64).reshape(-1, 3) if hue_rgb is not None else flat
        angles = hue_angle_from_rgb(hue_flat)
        region_indices = np.searchsorted(self.boundaries, angles, side="right") % len(self.boundaries)
        out = np.empty_like(flat)
        for region in range(len(self.boundaries)):
            mask = region_indices == region
            if np.any(mask):
                out[mask] = flat[mask] @ self.matrices[region]
        return out.reshape(rgb.shape)

    def predict_blending(self, rgb: np.ndarray, blend_width: float = 0.15, *, hue_rgb: np.ndarray | None = None) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        flat = rgb.reshape(-1, 3)
        hue_flat = np.asarray(hue_rgb, dtype=np.float64).reshape(-1, 3) if hue_rgb is not None else flat
        angles = hue_angle_from_rgb(hue_flat)
        boundaries = np.asarray(self.boundaries, dtype=np.float64)
        k_regions = len(boundaries)
        if k_regions < 2:
            return self.predict(rgb)

        # searchsorted(boundaries, angle, 'right') % k assigns:
        #   region 0  → angles in [boundaries[k-1], 2π)
        #   region i  → angles in [boundaries[i-1], boundaries[i])  for i ≥ 1
        # so each Gaussian centre must sit at the midpoint of the ACTUAL range of
        # its region, not the interval [boundaries[i], boundaries[i+1]) which is
        # off by one.
        starts = np.concatenate([[boundaries[-1]], boundaries[:-1]])
        ends = np.concatenate([[2.0 * np.pi], boundaries[1:]])
        centers = starts + 0.5 * (ends - starts)
        widths = ends - starts  # actual angular width of each region

        # Gaussian partition of unity over hue angle. Each region's matrix is
        # weighted by a Gaussian of the angular distance from its centre, then
        # the weights are normalised. A Gaussian is positive everywhere, so
        # adjacent regions always overlap and the blend crosses every region
        # boundary smoothly (C-infinity) — no hard region assignment, no
        # two-tone seam. sigma is scaled by each region's ACTUAL width (not
        # the mean width), so narrow regions stay tight while wide regions
        # blend more broadly — preventing narrow-region matrices from
        # contaminating predictions that belong to adjacent wide regions.
        if np.isscalar(blend_width):
            blend_fraction = float(blend_width)
            if blend_fraction <= 0.0:
                return self.predict(rgb)
            sigmas = blend_fraction * widths
        else:
            sigmas = np.asarray(blend_width, dtype=np.float64)
            if sigmas.shape != (k_regions,):
                raise ValueError("blend_width must be a scalar or have one value per region.")
        sigmas = np.maximum(sigmas, 1e-12)

        predictions = np.empty((k_regions, flat.shape[0], 3), dtype=np.float64)
        for region in range(k_regions):
            predictions[region] = flat @ self.matrices[region]

        delta = np.abs(angles[np.newaxis, :] - centers[:, np.newaxis])
        delta = np.minimum(delta, 2.0 * np.pi - delta)
        weights = np.exp(-0.5 * (delta / sigmas[:, np.newaxis]) ** 2)

        normalizer = np.sum(weights, axis=0, keepdims=True)
        # Degenerate guard: with a very small blend_width every Gaussian can
        # underflow to zero for a pixel far from all centres. Fall back to the
        # hard region assignment there so the output stays well-defined.
        hard_mask = normalizer[0] <= 1e-12
        if np.any(hard_mask):
            hard_region_indices = np.searchsorted(boundaries, angles, side="right") % k_regions
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
    lam: float = HPPCC_REGION_SMOOTHNESS,
) -> np.ndarray:
    """Fit K regional 3×3 matrices with boundary-continuity and white-point constraints.

    `lam` adds a Tikhonov penalty ``lam * sum ||M_r - M_{r+1}||^2`` over
    cyclically adjacent region pairs, keeping the matrices from over-fitting
    their few patches and diverging (which would show as a colour transition
    even under a smooth hue blend). Returns matrices of shape (K, 3, 3).
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

    if lam > 0.0 and k_regions >= 2:
        # Couple adjacent regions with a per-pair penalty weight that is
        # inversely proportional to the smaller region's patch count.
        # A region with few patches is over-fitted and its matrix diverges
        # easily; scaling its coupling weight by mean_count / min_count pulls
        # it much more strongly toward its neighbour than the reverse.
        # `lam` is calibrated for an "average-sized" region: a pair where
        # min_count == mean_count gets exactly root_lam; pairs with a smaller
        # region get proportionally more.
        counts = np.bincount(region_ids, minlength=k_regions).astype(np.float64)
        mean_count = counts.mean()
        penalty = np.zeros((3 * k_regions, 3 * k_regions), dtype=np.float64)
        for region in range(k_regions):
            next_region = (region + 1) % k_regions
            min_count = min(counts[region], counts[next_region])
            effective_root_lam = np.sqrt(float(lam) * mean_count / max(min_count, 1.0))
            rows = slice(3 * region, 3 * (region + 1))
            penalty[rows, 3 * region : 3 * (region + 1)] = effective_root_lam * np.eye(3)
            penalty[rows, 3 * next_region : 3 * (next_region + 1)] = -effective_root_lam * np.eye(3)
        a = np.vstack([a, penalty])
        p_sorted = np.vstack([p_sorted, np.zeros((3 * k_regions, 3), dtype=np.float64)])

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
    lam: float = HPPCC_REGION_SMOOTHNESS,
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
            matrices = _fit_hppcc_constrained(
                q_sorted, p_sorted, bounds, region_ids, white_rgb, white_xyz, lam=lam
            )
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
    region_smoothness: float = HPPCC_REGION_SMOOTHNESS,
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

    min_k = 2
    while k_regions >= min_k:
        if optimize_boundaries:
            boundaries = _exhaustive_boundary_search(
                q_sorted, p_sorted, angles_sorted, k_regions, white_rgb, white_xyz,
                min_per_region=3, lam=region_smoothness,
            )
        else:
            boundaries = _equal_count_boundaries(angles_sorted, k_regions)

        region_ids = _assign_regions(angles_sorted, boundaries)
        counts = np.bincount(region_ids, minlength=k_regions)
        if not np.any(counts == 0):
            break
        k_regions -= 1
    else:
        raise ValueError("Hue partitioning produced an empty region even with k_regions=2.")

    matrices = _fit_hppcc_constrained(
        q_sorted, p_sorted, boundaries, region_ids, white_rgb, white_xyz, lam=region_smoothness
    )
    return HPPCCModel(
        matrices=matrices,
        boundaries=boundaries,
        white_rgb=white_rgb.copy(),
        white_xyz=white_xyz.copy(),
    )


@dataclass(frozen=True)
class HPPCCGradientModel:
    """Hue-gradient HPPCC: M(θ) = Σ_k [A_k·cos(kθ) + B_k·sin(kθ)], smooth for all θ.

    The correction matrix varies continuously with hue angle via a truncated Fourier
    series. White-point preservation is enforced for every basis function, so
    M(θ)·white_rgb = white_xyz holds identically for all θ.
    coeffs has shape (1 + 2·n_harmonics, 3, 3): [A₀, A₁, B₁, A₂, B₂, ...].
    """

    coeffs: np.ndarray
    white_rgb: np.ndarray
    white_xyz: np.ndarray

    def _basis(self, angles: np.ndarray) -> np.ndarray:
        n = (self.coeffs.shape[0] - 1) // 2
        parts = [np.ones(len(angles), dtype=np.float64)]
        for k in range(1, n + 1):
            parts.append(np.cos(k * angles))
            parts.append(np.sin(k * angles))
        return np.stack(parts, axis=1)  # (N, 1+2n)

    def predict(self, rgb: np.ndarray, *, smooth_sigma: float = HPPCC_GRADIENT_SMOOTH_SIGMA, **_kwargs) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        flat = rgb.reshape(-1, 3)
        if rgb.ndim == 3 and smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter
            h, w = rgb.shape[:2]
            denom = flat.sum(axis=-1).clip(1e-12).reshape(h, w)
            r_norm = gaussian_filter(flat[:, 0].reshape(h, w) / denom, sigma=smooth_sigma)
            g_norm = gaussian_filter(flat[:, 1].reshape(h, w) / denom, sigma=smooth_sigma)
            angles = np.mod(np.arctan2(g_norm - 1.0 / 3.0, r_norm - 1.0 / 3.0).ravel(), 2.0 * np.pi)
        else:
            angles = hue_angle_from_rgb(flat)
        basis = self._basis(angles)
        out = np.einsum("ik,il,klj->ij", basis, flat, self.coeffs)
        return out.reshape(rgb.shape)

    def predict_blending(self, rgb: np.ndarray, blend_width: float = 0.15, **_kwargs) -> np.ndarray:
        return self.predict(rgb)

    @property
    def boundaries(self) -> np.ndarray:
        return np.array([0.0])

    @property
    def matrices(self) -> np.ndarray:
        return self.coeffs


@dataclass(frozen=True)
class HPPCCRPCCModel:
    """Two-stage pipeline: HPPCC hue-preserving primary + RPCC global residual correction.

    Prediction: xyz = hppcc.predict(rgb) + rpcc_residual.predict(rgb)
    The RPCC component is trained on xyz residuals (xyz_ref - xyz_hppcc).
    Because HPPCC preserves the white point exactly, the RPCC residual for the white
    patch is zero and its white-point constraint is trivially satisfied.
    hppcc accepts both HPPCCModel and HPPCCGradientModel (duck typing).
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

    def predict(self, rgb: np.ndarray, *, hue_rgb: np.ndarray | None = None) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        return self.hppcc.predict(rgb, hue_rgb=hue_rgb) + self.rpcc_residual.predict(rgb)

    def predict_blending(self, rgb: np.ndarray, blend_width: float = 0.15, *, hue_rgb: np.ndarray | None = None) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        return self.hppcc.predict_blending(rgb, blend_width=blend_width, hue_rgb=hue_rgb) + self.rpcc_residual.predict(rgb)


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
    region_smoothness: float = HPPCC_REGION_SMOOTHNESS,
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
        region_smoothness=region_smoothness,
    )
    xyz_hppcc = hppcc.predict(rgb)
    xyz_residual = xyz - xyz_hppcc
    rpcc_residual = fit_rpcc(rgb, xyz_residual, white_index=white_index)
    return HPPCCRPCCModel(hppcc=hppcc, rpcc_residual=rpcc_residual)


def fit_hppcc_gradient(
    rgb: np.ndarray,
    xyz: np.ndarray,
    *,
    white_index: int,
    chromatic_indices: np.ndarray | list[int],
    n_harmonics: int = 2,
) -> HPPCCGradientModel:
    """Fit a trigonometric-series HPPCC: M(θ) varies smoothly with hue angle.

    White-point preservation is enforced for every basis function (A_k·white = 0 for k≥1,
    A_0·white = white_xyz), so M(θ)·white_rgb = white_xyz holds for all θ.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    chromatic_indices = np.asarray(chromatic_indices, dtype=int)
    q = rgb[chromatic_indices]
    p = xyz[chromatic_indices]
    white_rgb = rgb[white_index]
    white_xyz = xyz[white_index]

    angles = hue_angle_from_rgb(q)
    n_basis = 1 + 2 * n_harmonics

    parts: list[np.ndarray] = [np.ones(len(angles), dtype=np.float64)]
    for k in range(1, n_harmonics + 1):
        parts.append(np.cos(k * angles))
        parts.append(np.sin(k * angles))
    basis = np.stack(parts, axis=1)  # (N, n_basis)

    # Design matrix: X[i, k*3+l] = basis[i,k] * q[i,l]
    x = (basis[:, :, np.newaxis] * q[:, np.newaxis, :]).reshape(len(q), -1)

    # White constraints per basis function: A_k · white_rgb = white_xyz (k=0), 0 (k≥1)
    c = np.zeros((n_basis, n_basis * 3), dtype=np.float64)
    b = np.zeros((n_basis, 3), dtype=np.float64)
    for k in range(n_basis):
        c[k, k * 3:(k + 1) * 3] = white_rgb
    b[0] = white_xyz

    w = _solve_constrained_least_squares(x, p, c, b)  # (n_basis*3, 3)
    return HPPCCGradientModel(
        coeffs=w.reshape(n_basis, 3, 3),
        white_rgb=white_rgb.copy(),
        white_xyz=white_xyz.copy(),
    )


# ---------------------------------------------------------------------------
# Ridge-regularised RPCC
# ---------------------------------------------------------------------------

def fit_rpcc_ridge(
    rgb: np.ndarray,
    xyz: np.ndarray,
    white_index: int,
    lambda_ridge: float = RPCC_RIDGE_LAMBDA,
) -> RPCCModel:
    """Fit a Tikhonov-regularised Root-Polynomial CC matrix (6×3).

    Minimises  ||F M - XYZ||² + λ ||M||²  subject to F_white M = XYZ_white.
    Ridge shrinks the polynomial coefficients toward zero, reducing overfitting
    when the training set is small.  λ=0 is equivalent to fit_rpcc.

    Reference: Cheung et al. (2004), Coloration Technology 120(1):19-25.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    features = _rpcc_features(rgb)
    white_features = features[white_index]
    white_xyz = xyz[white_index].copy()
    # Augment the design matrix: appending sqrt(λ)*I adds the Tikhonov penalty
    # as extra "virtual" observations with target 0.
    a_aug = np.vstack([features, np.sqrt(lambda_ridge) * np.eye(features.shape[1])])
    y_aug = np.vstack([xyz, np.zeros((features.shape[1], 3), dtype=np.float64)])
    matrix = _solve_constrained_least_squares(
        a_aug, y_aug, white_features[np.newaxis, :], white_xyz[np.newaxis, :]
    )
    return RPCCModel(matrix=matrix, white_rgb=rgb[white_index].copy(), white_xyz=white_xyz)


# ---------------------------------------------------------------------------
# HLCC — Hue-Linear Color Correction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HLCCModel:
    """Hue-Linear Color Correction (Kawakami-style hat-function interpolation).

    The correction matrix M(θ) is a piecewise-linear (hat-function) blend of
    K sector matrices placed at equal hue-angle intervals θ_k = k·2π/K.
    For a pixel whose sensor hue falls in the interval [θ_i, θ_{i+1}):

        M(θ) = (1−t)·M_i  +  t·M_{(i+1) mod K}

    where t = (θ − θ_i) / (2π/K) ∈ [0,1).

    Because the hat functions form a partition of unity (they sum to 1 for all
    θ) and each M_k maps white_rgb → white_xyz, M(θ) automatically preserves
    the white point for every hue angle.

    Reference: Kawakami et al. (hue-linear interpolation); see also
    Finlayson & Hordley (HPPCC, 2001).
    """

    matrices: np.ndarray  # (K, 3, 3)
    white_rgb: np.ndarray
    white_xyz: np.ndarray

    @property
    def n_sectors(self) -> int:
        return len(self.matrices)

    def predict(self, rgb: np.ndarray, **_kwargs) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        flat = rgb.reshape(-1, 3)
        angles = hue_angle_from_rgb(flat)
        K = self.n_sectors
        step = 2.0 * np.pi / K
        frac = angles / step                     # fractional sector position
        sector = np.floor(frac).astype(int) % K
        t = (frac - np.floor(frac))[:, None]     # (N,1) interpolation weight
        next_sector = (sector + 1) % K
        out = (
            (1.0 - t) * np.einsum("ij,ijk->ik", flat, self.matrices[sector])
            + t * np.einsum("ij,ijk->ik", flat, self.matrices[next_sector])
        )
        return out.reshape(rgb.shape)

    def predict_blending(self, rgb: np.ndarray, **_kwargs) -> np.ndarray:
        return self.predict(rgb)

    @property
    def boundaries(self) -> np.ndarray:
        K = self.n_sectors
        return np.linspace(0.0, 2.0 * np.pi, K, endpoint=False)

    @property
    def matrices_as_3d(self) -> np.ndarray:
        return np.asarray(self.matrices, dtype=np.float64)


def fit_hlcc(
    rgb: np.ndarray,
    xyz: np.ndarray,
    *,
    white_index: int,
    chromatic_indices: np.ndarray | list[int],
    k_sectors: int = HLCC_SECTORS,
) -> HLCCModel:
    """Fit a Hue-Linear Color Correction model with K equal-width hue sectors.

    Design matrix: for patch j at sensor hue θ_j in sector i with fractional
    weight t, row j of A is zero except:
        A[j, 3i:3(i+1)] = (1−t)·rgb_j
        A[j, 3((i+1)%K):3((i+1)%K+1)] += t·rgb_j

    White constraints: M_k·white_rgb = white_xyz for every sector k.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    chromatic_indices = np.asarray(chromatic_indices, dtype=int)
    q = rgb[chromatic_indices]
    p = xyz[chromatic_indices]
    white_rgb = rgb[white_index]
    white_xyz = xyz[white_index]
    N = len(q)
    K = k_sectors
    step = 2.0 * np.pi / K

    angles = hue_angle_from_rgb(q)
    frac = angles / step
    sector = np.floor(frac).astype(int) % K
    t = frac - np.floor(frac)
    next_s = (sector + 1) % K

    a = np.zeros((N, 3 * K), dtype=np.float64)
    for j in range(N):
        s, ns, tj = int(sector[j]), int(next_s[j]), float(t[j])
        a[j, 3 * s : 3 * (s + 1)] += (1.0 - tj) * q[j]
        a[j, 3 * ns : 3 * (ns + 1)] += tj * q[j]

    # White constraints: one per sector
    c = np.zeros((K, 3 * K), dtype=np.float64)
    b = np.zeros((K, 3), dtype=np.float64)
    for k in range(K):
        c[k, 3 * k : 3 * (k + 1)] = white_rgb
        b[k] = white_xyz

    w = _solve_constrained_least_squares(a, p, c, b)  # (3K, 3)
    return HLCCModel(
        matrices=w.reshape(K, 3, 3),
        white_rgb=white_rgb.copy(),
        white_xyz=white_xyz.copy(),
    )


# ---------------------------------------------------------------------------
# TPS — Thin-Plate Spline Color Correction
# ---------------------------------------------------------------------------

class TPSModel:
    """Thin-Plate Spline color correction.

    Maps camera RGB → XYZ by fitting a TPS through the N training patch pairs.
    The TPS minimises the second-derivative bending energy while interpolating
    (or approximating with smoothing>0) the training points exactly.  It
    extrapolates smoothly but may diverge far from the training hull.

    Training data is stored for re-fitting on deserialisation; the scipy
    RBFInterpolator is built lazily and cached.

    Reference: Bookstein (1989), IEEE TPAMI 11(6):567-585;
               Fang et al. (2019), AIP Advances 9(12):125134.
    """

    def __init__(
        self,
        training_rgb: np.ndarray,
        training_xyz: np.ndarray,
        white_rgb: np.ndarray,
        white_xyz: np.ndarray,
        smoothing: float = 0.0,
    ) -> None:
        self.training_rgb = np.asarray(training_rgb, dtype=np.float64)
        self.training_xyz = np.asarray(training_xyz, dtype=np.float64)
        self.white_rgb = np.asarray(white_rgb, dtype=np.float64)
        self.white_xyz = np.asarray(white_xyz, dtype=np.float64)
        self.smoothing = float(smoothing)
        self._interp = None

    def _get_interp(self):
        if self._interp is None:
            from scipy.interpolate import RBFInterpolator
            self._interp = RBFInterpolator(
                self.training_rgb,
                self.training_xyz,
                kernel="thin_plate_spline",
                smoothing=self.smoothing,
            )
        return self._interp

    def predict(self, rgb: np.ndarray, **_kwargs) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        flat = rgb.reshape(-1, 3)
        out = self._get_interp()(flat)
        return out.reshape(rgb.shape)

    def predict_blending(self, rgb: np.ndarray, **_kwargs) -> np.ndarray:
        return self.predict(rgb)


def fit_tps(
    rgb: np.ndarray,
    xyz: np.ndarray,
    white_index: int,
    smoothing: float = 0.0,
) -> TPSModel:
    """Fit a Thin-Plate Spline from camera RGB to XYZ over the N training patches."""
    rgb = np.asarray(rgb, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    return TPSModel(
        training_rgb=rgb.copy(),
        training_xyz=xyz.copy(),
        white_rgb=rgb[white_index].copy(),
        white_xyz=xyz[white_index].copy(),
        smoothing=smoothing,
    )


# ---------------------------------------------------------------------------
# LWCC — Locally Weighted Color Correction
# ---------------------------------------------------------------------------

class LWCCModel:
    """Locally Weighted Color Correction.

    For each training patch i a local 3×3 matrix M_i is fitted from all N
    training patches with Gaussian distance-weighting centred on patch i.
    For any test pixel x the prediction blends the N local matrices by
    Gaussian weights in RGB space:

        M(x) = Σ_i w_i(x)·M_i / Σ_i w_i(x),   y = M(x)·x

    Processing is chunked to stay within a configurable memory budget.

    References: locally-weighted / nearest-neighbour colour correction methods
    surveyed in Cheung et al. (2004).
    """

    def __init__(
        self,
        local_matrices: np.ndarray,
        training_rgb: np.ndarray,
        bandwidth: float,
        white_rgb: np.ndarray,
        white_xyz: np.ndarray,
        chunk_size: int = 100_000,
    ) -> None:
        self.local_matrices = np.asarray(local_matrices, dtype=np.float64)
        self.training_rgb = np.asarray(training_rgb, dtype=np.float64)
        self.bandwidth = float(bandwidth)
        self.white_rgb = np.asarray(white_rgb, dtype=np.float64)
        self.white_xyz = np.asarray(white_xyz, dtype=np.float64)
        self.chunk_size = int(chunk_size)

    def predict(self, rgb: np.ndarray, **_kwargs) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        flat = rgb.reshape(-1, 3)
        N_train = len(self.training_rgb)
        h2 = self.bandwidth ** 2
        out = np.empty_like(flat)
        cs = self.chunk_size
        for start in range(0, len(flat), cs):
            chunk = flat[start : start + cs]              # (C, 3)
            diff = chunk[:, None, :] - self.training_rgb[None, :, :]  # (C, N, 3)
            sq_dist = np.einsum("cni,cni->cn", diff, diff)            # (C, N)
            w = np.exp(-0.5 * sq_dist / h2)                           # (C, N)
            w /= w.sum(axis=1, keepdims=True).clip(1e-12)
            # Blended matrix per pixel: (C, 3, 3)
            blended = np.einsum("cn,nij->cij", w, self.local_matrices)
            out[start : start + cs] = np.einsum("ci,cij->cj", chunk, blended)
        return out.reshape(rgb.shape)

    def predict_blending(self, rgb: np.ndarray, **_kwargs) -> np.ndarray:
        return self.predict(rgb)


def fit_lwcc(
    rgb: np.ndarray,
    xyz: np.ndarray,
    white_index: int,
    bandwidth: float | None = None,
) -> LWCCModel:
    """Fit a Locally Weighted Color Correction model.

    For each training patch i, a local 3×3 matrix M_i is solved from all N
    patches weighted by  w_ij = exp(−||rgb_i − rgb_j||² / (2·h²))  subject
    to the white-point constraint  M_i·white_rgb = white_xyz.

    Bandwidth h defaults to half the median pairwise Euclidean distance in the
    normalised RGB training set (a data-driven Silverman-like rule).
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    N = len(rgb)
    white_rgb = rgb[white_index]
    white_xyz = xyz[white_index]

    if bandwidth is None:
        from scipy.spatial.distance import pdist
        dists = pdist(rgb)
        bandwidth = float(np.median(dists)) * 0.5 if len(dists) > 0 else 1.0
        bandwidth = max(bandwidth, 1e-6)

    h2 = bandwidth ** 2
    local_matrices = np.empty((N, 3, 3), dtype=np.float64)
    c = white_rgb[np.newaxis, :]        # (1, 3)
    b = white_xyz[np.newaxis, :]        # (1, 3)
    for i in range(N):
        sq = np.sum((rgb - rgb[i]) ** 2, axis=1)
        w_vec = np.exp(-0.5 * sq / h2)
        # sqrt-weight augmentation so _solve_constrained_least_squares sees ||Ax-b||²
        sw = np.sqrt(w_vec)[:, None]           # (N, 1)
        local_matrices[i] = _solve_constrained_least_squares(
            rgb * sw, xyz * sw, c, b
        )

    return LWCCModel(
        local_matrices=local_matrices,
        training_rgb=rgb.copy(),
        bandwidth=bandwidth,
        white_rgb=white_rgb.copy(),
        white_xyz=white_xyz.copy(),
    )


# ---------------------------------------------------------------------------
# CIEDE2000-optimised linear model
# ---------------------------------------------------------------------------

def fit_de00_opt(
    rgb: np.ndarray,
    xyz: np.ndarray,
    white_index: int,
    illuminant_white: np.ndarray,
) -> LinearWhitePreservingModel:
    """Fit a 3×3 linear model by minimising the sum of CIEDE2000 errors.

    Starts from the white-preserving OLS solution; refines with scipy SLSQP
    using the dE00 sum as the loss function.  The white-point equality
    constraint is enforced exactly throughout the optimisation.

    Reference: Luo et al. (2001), Color Research & Application 26(5):340-355.
    """
    from scipy.optimize import minimize

    rgb = np.asarray(rgb, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    illuminant_white = np.asarray(illuminant_white, dtype=np.float64)
    white_rgb = rgb[white_index]
    white_xyz = xyz[white_index].copy()

    lab_ref = xyz_to_lab(xyz, illuminant_white)

    linear0 = fit_white_preserving_3x3(rgb, xyz, white_index)
    m0 = linear0.matrix.ravel()

    def loss(m_flat: np.ndarray) -> float:
        M = m_flat.reshape(3, 3)
        xyz_pred = rgb @ M
        lab_pred = xyz_to_lab(xyz_pred, illuminant_white)
        de = delta_e_2000(lab_pred, lab_ref)
        return float(np.sum(de))

    def white_con(m_flat: np.ndarray) -> np.ndarray:
        return white_rgb @ m_flat.reshape(3, 3) - white_xyz

    result = minimize(
        loss,
        m0,
        method="SLSQP",
        constraints={"type": "eq", "fun": white_con},
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    M_opt = result.x.reshape(3, 3)
    return LinearWhitePreservingModel(
        matrix=M_opt,
        white_rgb=white_rgb.copy(),
        white_xyz=white_xyz,
    )
