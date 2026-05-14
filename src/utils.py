import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageCms

from .models import HPPCCModel, HPPCCRPCCModel, LinearWhitePreservingModel, RPCCModel


BRADFORD_MATRIX = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ],
    dtype=np.float64,
)

XYZ_TO_SRGB_MATRIX = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ],
    dtype=np.float64,
)

XYZ_TO_DISPLAY_P3_MATRIX = np.array(
    [
        [2.493496911941425, -0.931383617919124, -0.402710784450717],
        [-0.829488969561575, 1.762664060318346, 0.023624685841943],
        [0.035845830243784, -0.076172389268041, 0.956884524007687],
    ],
    dtype=np.float64,
)

SYSTEM_COLOR_PROFILE_DIR = Path(r"C:\Windows\System32\spool\drivers\color")
LOCAL_COLOR_PROFILE_DIR = Path(__file__).resolve().parent.parent / "assets"
ICC_PROFILE_CANDIDATES = {
    "sRGB": [
        LOCAL_COLOR_PROFILE_DIR / "ICCProfile_sRGB.icc",
        SYSTEM_COLOR_PROFILE_DIR / "ICCProfile_sRGB.icc",
        SYSTEM_COLOR_PROFILE_DIR / "sRGB Color Space Profile.icm",
    ],
    "Display-P3": [
        LOCAL_COLOR_PROFILE_DIR / "Display P3.icc",
        SYSTEM_COLOR_PROFILE_DIR / "Display P3.icc",
        SYSTEM_COLOR_PROFILE_DIR / "DCI-P3-D65.icc",
    ],
}


def estimate_scene_white_from_neutral_patches(
    normalized_rgb: np.ndarray,
    neutral_indices: np.ndarray,
    daylight_wb: tuple,
    reference_white: np.ndarray,
    min_value: float = 0.01,
) -> np.ndarray:
    """Estimate scene illuminant XYZ from the r/g and b/g ratios of neutral patches.

    For any achromatic surface, the camera r/g and b/g ratios are independent of
    reflectance and are equal to the illuminant r/g and b/g in camera space.
    Averaging over multiple neutral patches reduces measurement noise.
    Uses Bradford adaptation (camera space ≈ cone space) to convert to XYZ.

    Patches with values below min_value (too dark, noisy) are excluded.
    """
    neutral_rgb = normalized_rgb[neutral_indices].astype(np.float64)
    valid = np.all(neutral_rgb > min_value, axis=1) & np.all(neutral_rgb < 0.99, axis=1)
    if not np.any(valid):
        return np.asarray(reference_white, dtype=np.float64).copy()
    valid_rgb = neutral_rgb[valid]
    if valid_rgb.shape[0] >= 4:
        order = np.argsort(valid_rgb[:, 1])
        valid_rgb = valid_rgb[order][1:-1]

    r_g = valid_rgb[:, 0] / valid_rgb[:, 1]
    b_g = valid_rgb[:, 2] / valid_rgb[:, 1]
    return _scene_white_from_neutral_ratios(
        float(np.mean(r_g)),
        float(np.mean(b_g)),
        daylight_wb,
        reference_white,
    )


def _scene_white_from_neutral_ratios(
    r_g: float,
    b_g: float,
    daylight_wb: tuple,
    reference_white: np.ndarray,
) -> np.ndarray:
    scene_cam = np.array([r_g, 1.0, b_g], dtype=np.float64)
    day = np.array([float(daylight_wb[0]), float(daylight_wb[1]), float(daylight_wb[2])], dtype=np.float64)
    if day[1] == 0.0:
        return np.asarray(reference_white, dtype=np.float64).copy()
    day_norm = day / day[1]
    # D65 illuminant in camera space ∝ 1/daylight_wb (lower gain → more illuminant light)
    d65_cam = (1.0 / day_norm)
    d65_cam /= d65_cam[1]

    wb_ratio = scene_cam / d65_cam
    ref_white = np.asarray(reference_white, dtype=np.float64)
    ref_bradford = BRADFORD_MATRIX @ ref_white
    scene_bradford = ref_bradford * wb_ratio
    scene_white = np.linalg.inv(BRADFORD_MATRIX) @ scene_bradford
    return scene_white / scene_white[1]


def estimate_cct_from_xyz(xyz: np.ndarray) -> float | None:
    xyz_sum = float(np.sum(xyz))
    if xyz_sum <= 0:
        return None
    x = float(xyz[0]) / xyz_sum
    y = float(xyz[1]) / xyz_sum
    n = (x - 0.3320) / (0.1858 - y)
    return 449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33


def xyz_to_xy(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    xyz_sum = float(np.sum(xyz))
    if xyz_sum <= 0.0:
        return np.array([np.nan, np.nan], dtype=np.float64)
    return np.array([xyz[0] / xyz_sum, xyz[1] / xyz_sum], dtype=np.float64)


def _middle_neutral_indices(normalized_rgb: np.ndarray, neutral_indices: np.ndarray, min_value: float = 0.01) -> np.ndarray:
    neutral_indices = np.asarray(neutral_indices, dtype=int)
    neutral_rgb = np.asarray(normalized_rgb[neutral_indices], dtype=np.float64)
    valid = np.all(neutral_rgb > min_value, axis=1) & np.all(neutral_rgb < 0.99, axis=1)
    selected = neutral_indices[valid]
    if selected.size == 0:
        return neutral_indices
    if selected.size >= 4:
        order = np.argsort(normalized_rgb[selected, 1])
        selected = selected[order][1:-1]
    return selected


def _weighted_linear_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    design = np.column_stack([np.ones_like(x), x])
    weighted_design = design * weights[:, np.newaxis]
    weighted_targets = y * weights
    intercept, slope = np.linalg.lstsq(weighted_design, weighted_targets, rcond=None)[0]
    return float(intercept), float(slope)


def analyze_neutral_illuminant_gradient(
    normalized_rgb: np.ndarray,
    neutral_indices: np.ndarray,
    absolute_patch_centers: np.ndarray,
    daylight_wb: tuple,
    reference_white: np.ndarray,
) -> dict[str, object]:
    neutral_indices = np.asarray(neutral_indices, dtype=int)
    centers = np.asarray(absolute_patch_centers[neutral_indices], dtype=np.float64)
    neutral_rgb = np.asarray(normalized_rgb[neutral_indices], dtype=np.float64)

    order = np.argsort(centers[:, 0])
    neutral_indices = neutral_indices[order]
    centers = centers[order]
    neutral_rgb = neutral_rgb[order]

    green = neutral_rgb[:, 1]
    safe_green = np.maximum(green, 1e-12)
    r_over_g = neutral_rgb[:, 0] / safe_green
    b_over_g = neutral_rgb[:, 2] / safe_green
    weights = green / np.max(green) if np.max(green) > 0.0 else np.ones_like(green)

    x_positions = centers[:, 0]
    x_span = float(np.max(x_positions) - np.min(x_positions))
    if x_span > 0.0:
        x_norm = (x_positions - np.min(x_positions)) / x_span
    else:
        x_norm = np.zeros_like(x_positions)

    local_white_xyz = np.asarray(
        [
            _scene_white_from_neutral_ratios(float(rg), float(bg), daylight_wb, reference_white)
            for rg, bg in zip(r_over_g, b_over_g, strict=False)
        ],
        dtype=np.float64,
    )
    local_white_xy = np.asarray([xyz_to_xy(xyz) for xyz in local_white_xyz], dtype=np.float64)
    local_white_cct = np.array([estimate_cct_from_xyz(xyz) for xyz in local_white_xyz], dtype=np.float64)

    rg_intercept, rg_slope = _weighted_linear_fit(x_norm, r_over_g, weights)
    bg_intercept, bg_slope = _weighted_linear_fit(x_norm, b_over_g, weights)
    xyx_intercept, xyx_slope = _weighted_linear_fit(x_norm, local_white_xy[:, 0], weights)
    xyy_intercept, xyy_slope = _weighted_linear_fit(x_norm, local_white_xy[:, 1], weights)

    horizontal_xy_span = float(np.linalg.norm(local_white_xy[-1] - local_white_xy[0]))
    horizontal_cct_span = (
        float(np.nanmax(local_white_cct) - np.nanmin(local_white_cct))
        if np.any(np.isfinite(local_white_cct))
        else float("nan")
    )
    if horizontal_xy_span >= 0.008:
        severity = "strong"
    elif horizontal_xy_span >= 0.004:
        severity = "moderate"
    elif horizontal_xy_span >= 0.002:
        severity = "possible"
    else:
        severity = "low"

    patches: list[dict[str, object]] = []
    for patch_index, center, rgb, rg, bg, white_xyz, white_xy, cct, weight in zip(
        neutral_indices,
        centers,
        neutral_rgb,
        r_over_g,
        b_over_g,
        local_white_xyz,
        local_white_xy,
        local_white_cct,
        weights,
        strict=False,
    ):
        patches.append(
            {
                "patch_index": int(patch_index),
                "center_xy": center.tolist(),
                "rgb": rgb.tolist(),
                "r_over_g": float(rg),
                "b_over_g": float(bg),
                "local_white_xyz": white_xyz.tolist(),
                "local_white_xy": white_xy.tolist(),
                "local_white_cct": None if np.isnan(cct) else float(cct),
                "weight": float(weight),
            }
        )

    return {
        "supports_horizontal_test": True,
        "supports_vertical_test": False,
        "severity": severity,
        "x_min": float(np.min(x_positions)),
        "x_max": float(np.max(x_positions)),
        "horizontal_xy_span": horizontal_xy_span,
        "horizontal_cct_span": horizontal_cct_span,
        "r_over_g_span": float(np.max(r_over_g) - np.min(r_over_g)),
        "b_over_g_span": float(np.max(b_over_g) - np.min(b_over_g)),
        "weighted_rg_slope_per_chart_width": rg_slope,
        "weighted_bg_slope_per_chart_width": bg_slope,
        "weighted_x_slope_per_chart_width": xyx_slope,
        "weighted_y_slope_per_chart_width": xyy_slope,
        "left_patch_index": int(neutral_indices[0]),
        "right_patch_index": int(neutral_indices[-1]),
        "left_local_white_xyz": local_white_xyz[0].tolist(),
        "right_local_white_xyz": local_white_xyz[-1].tolist(),
        "left_local_white_xy": local_white_xy[0].tolist(),
        "right_local_white_xy": local_white_xy[-1].tolist(),
        "left_local_white_cct": None if np.isnan(local_white_cct[0]) else float(local_white_cct[0]),
        "right_local_white_cct": None if np.isnan(local_white_cct[-1]) else float(local_white_cct[-1]),
        "patches": patches,
        "weighted_line": {
            "r_over_g": {"intercept": rg_intercept, "slope": rg_slope},
            "b_over_g": {"intercept": bg_intercept, "slope": bg_slope},
            "x": {"intercept": xyx_intercept, "slope": xyx_slope},
            "y": {"intercept": xyy_intercept, "slope": xyy_slope},
        },
    }

def select_scene_white_source(
    src_module,
    measured_rgb: np.ndarray,
    reference_xyz: np.ndarray,
    reference_white: np.ndarray,
    *,
    white_index: int,
    chromatic_indices: np.ndarray,
    neutral_indices: np.ndarray,
    requested_source: str,
    camera_wb_white: np.ndarray | None,
    neutral_patch_white: np.ndarray | None,
) -> tuple[str, np.ndarray, dict[str, dict[str, float]]]:
    candidates: dict[str, np.ndarray] = {"reference": np.asarray(reference_white, dtype=np.float64).copy()}
    if camera_wb_white is not None:
        candidates["camera-wb"] = np.asarray(camera_wb_white, dtype=np.float64).copy()
    if neutral_patch_white is not None:
        candidates["neutral-patches"] = np.asarray(neutral_patch_white, dtype=np.float64).copy()

    if requested_source != "auto":
        selected_white = candidates.get(requested_source)
        if selected_white is None:
            return "reference", candidates["reference"], {}
        return requested_source, selected_white, {}

    scoring_indices = _middle_neutral_indices(measured_rgb, neutral_indices)
    scores: dict[str, dict[str, float]] = {}
    best_source = "reference"
    best_score = (float("inf"), float("inf"))

    for candidate_name, candidate_white in candidates.items():
        adapted_reference_xyz = bradford_adapt_xyz(reference_xyz, reference_white, candidate_white)
        baseline = src_module.fit_white_preserving_3x3(
            measured_rgb,
            adapted_reference_xyz,
            white_index=white_index,
        )
        baseline_xyz = baseline.predict(measured_rgb)
        baseline_de00 = delta_e00_summary(src_module, baseline_xyz, adapted_reference_xyz, candidate_white)
        neutral_mean = float(np.mean(baseline_de00[scoring_indices]))
        overall_mean = float(np.mean(baseline_de00))
        scores[candidate_name] = {
            "neutral_mean_de00": neutral_mean,
            "overall_mean_de00": overall_mean,
            "cct": estimate_cct_from_xyz(candidate_white),
        }
        candidate_score = (neutral_mean, overall_mean)
        if candidate_score < best_score:
            best_source = candidate_name
            best_score = candidate_score

    return best_source, candidates[best_source], scores


def estimate_scene_white_from_camera_wb(
    camera_wb: tuple,
    daylight_wb: tuple,
    reference_white: np.ndarray,
) -> np.ndarray:
    """Estimate the actual scene illuminant XYZ from camera white-balance multipliers.

    Uses the ratio (daylight_wb / camera_wb) — normalised to G=1 — as a Bradford-space
    scaling factor relative to the reference illuminant (typically D65).  This avoids
    the inaccurate metadata rgb_xyz_matrix and relies only on WB ratios.

    Returns a Y-normalised XYZ white point for the measured scene illuminant.
    """
    cam = np.array([float(camera_wb[0]), float(camera_wb[1]), float(camera_wb[2])], dtype=np.float64)
    day = np.array([float(daylight_wb[0]), float(daylight_wb[1]), float(daylight_wb[2])], dtype=np.float64)
    if cam[1] == 0.0 or day[1] == 0.0:
        return np.asarray(reference_white, dtype=np.float64).copy()
    cam_norm = cam / cam[1]
    day_norm = day / day[1]
    # daylight_wb represents the D65 illuminant response; camera_wb the actual scene.
    # A lower camera R gain → more R light in scene → warmer than D65.
    # In Bradford space the illuminant scales as: scene = reference × (day_norm / cam_norm).
    wb_ratio = day_norm / cam_norm
    ref_white = np.asarray(reference_white, dtype=np.float64)
    ref_bradford = BRADFORD_MATRIX @ ref_white
    scene_bradford = ref_bradford * wb_ratio
    scene_white = np.linalg.inv(BRADFORD_MATRIX) @ scene_bradford
    scene_white = scene_white / scene_white[1]
    return scene_white


def find_raw_path(image_dir: Path) -> Path:
    candidates = []
    for pattern in ("*.NEF", "*.CR2", "*.CR3", "*.ARW", "*.RAF", "*.DNG"):
        candidates.extend(sorted(image_dir.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(f"No RAW file found in {image_dir}")
    return candidates[0]


def find_raw_paths(image_dir: Path, *, recursive: bool = False) -> list[Path]:
    from .raw import INPUT_SUFFIXES
    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
    seen: set[Path] = set()
    result = []
    for p in iterator:
        if p.is_file() and p.suffix.lower() in INPUT_SUFFIXES and p not in seen:
            seen.add(p)
            result.append(p)
    return sorted(result)


def _load_reference_triplets(path: Path, key: str) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data["CONFIG"]["TARGET"][key]
    triplets = np.asarray([values[str(i)] for i in range(1, 25)], dtype=np.float64)
    if triplets.shape != (24, 3):
        raise ValueError(f"Reference data '{key}' must contain 24 patches with 3 values each.")
    return triplets


def _load_reference_scalars(path: Path, key: str) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data["CONFIG"]["TARGET"][key]
    scalars = np.asarray([values[str(i)] for i in range(1, 25)], dtype=np.float64)
    if scalars.shape != (24,):
        raise ValueError(f"Reference data '{key}' must contain 24 scalar values.")
    return scalars


def load_reference_white_xyz(path: Path, reference_space: str) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data["CONFIG"]["TARGET"]["referenceWhiteXYZ"][reference_space]
    white_xyz = np.asarray(values, dtype=np.float64)
    if white_xyz.shape != (3,):
        raise ValueError(f"Reference white for '{reference_space}' must contain exactly 3 values.")
    return white_xyz


def load_reference_xyz(path: Path) -> np.ndarray:
    return _load_reference_triplets(path, "referenceXYZValues")


def load_reference_rgb(path: Path) -> np.ndarray:
    return _load_reference_triplets(path, "referenceRGBValues")


def load_reference_lab(path: Path) -> np.ndarray:
    return _load_reference_triplets(path, "referenceLabValues")


def load_reference_xyy(path: Path) -> np.ndarray:
    return _load_reference_triplets(path, "referencexyYValues")


def load_reference_chroma(path: Path) -> np.ndarray:
    return _load_reference_scalars(path, "referenceChromaValues")


def delta_e00_summary(src_module, predicted_xyz: np.ndarray, reference_xyz: np.ndarray, white_xyz: np.ndarray) -> np.ndarray:
    predicted_lab = src_module.xyz_to_lab(predicted_xyz, white_xyz)
    reference_lab = src_module.xyz_to_lab(reference_xyz, white_xyz)
    return src_module.delta_e_2000(predicted_lab, reference_lab)


def chroma_from_lab(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    return np.sqrt(np.square(lab[..., 1]) + np.square(lab[..., 2]))


def chroma_error_summary(src_module, predicted_xyz: np.ndarray, illuminant_white: np.ndarray, reference_chroma: np.ndarray) -> np.ndarray:
    predicted_lab = src_module.xyz_to_lab(predicted_xyz, illuminant_white)
    predicted_chroma = chroma_from_lab(predicted_lab)
    return predicted_chroma - np.asarray(reference_chroma, dtype=np.float64)


def bradford_adapt_xyz(xyz: np.ndarray, source_white: np.ndarray, target_white: np.ndarray) -> np.ndarray:
    source_rgb = BRADFORD_MATRIX @ source_white
    target_rgb = BRADFORD_MATRIX @ target_white
    scaling = np.diag(target_rgb / source_rgb)
    adaptation = np.linalg.inv(BRADFORD_MATRIX) @ scaling @ BRADFORD_MATRIX
    return xyz @ adaptation.T


_D65_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def identify_unreliable_patches(
    normalized_rgb: np.ndarray,
    white_index: int,
    min_level: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (reliable_indices, excluded_indices) based on per-channel signal level.

    A patch is considered unreliable when any normalized channel is <= min_level.
    This threshold is intentionally low (1e-4): it targets only patches where a channel
    has clipped below the sensor black level (raw value ≤ black offset), producing a
    zero or near-zero reading that makes the channel's hue contribution undefined.
    Patches with low but non-zero signal still carry valid hue information for HPPCC.

    The white patch is always retained. Excluded patches remain in delta-E evaluation.
    These indices are used to filter the HPPCC chromatic partition only; baseline and RPCC
    models are always trained on all patches for numerical stability.
    """
    reliable, excluded = [], []
    for i in range(normalized_rgb.shape[0]):
        if i == white_index or np.all(normalized_rgb[i] > min_level):
            reliable.append(i)
        else:
            excluded.append(i)
    return np.array(reliable, dtype=int), np.array(excluded, dtype=int)


def render_xyz_to_display(
    xyz: np.ndarray,
    scene_white_xyz: np.ndarray,
    output_colorspace: str,
) -> np.ndarray:
    """Convert scene XYZ to display RGB with reverse Bradford adaptation to D65.

    The HPPCC model maps camera RGB to XYZ expressed under the scene illuminant.
    Direct XYZ→sRGB conversion (which assumes D65) would produce a colour-cast output
    whenever the scene white differs from D65. This function first adapts the XYZ from
    scene_white_xyz back to D65, then converts to output RGB.
    """
    scene_white = np.asarray(scene_white_xyz, dtype=np.float64)
    adapted = (
        bradford_adapt_xyz(xyz, scene_white, _D65_WHITE)
        if not np.allclose(scene_white, _D65_WHITE, atol=1e-4)
        else np.asarray(xyz, dtype=np.float64)
    )
    return xyz_to_output_rgb(adapted, output_colorspace)


def reduce_cfa_values_to_rgb(values: tuple[int, int, int, int] | None, color_desc: str) -> np.ndarray | None:
    if values is None:
        return None
    values_array = np.asarray(values, dtype=np.float64)
    mapping: dict[str, list[float]] = {"R": [], "G": [], "B": []}
    for channel_name, value in zip(color_desc[: len(values_array)], values_array, strict=False):
        if channel_name in mapping:
            mapping[channel_name].append(float(value))
    if any(len(mapping[channel]) == 0 for channel in ("R", "G", "B")):
        raise ValueError("Cannot derive RGB channel metadata from CFA description.")
    return np.array([np.mean(mapping["R"]), np.mean(mapping["G"]), np.mean(mapping["B"])], dtype=np.float64)


def reduce_cfa_matrix_to_rgb(matrix: np.ndarray | None, color_desc: str) -> np.ndarray | None:
    if matrix is None:
        return None
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("Expected a CFA->XYZ matrix with shape (N, 3).")
    mapping: dict[str, list[np.ndarray]] = {"R": [], "G": [], "B": []}
    for channel_name, row in zip(color_desc[: matrix.shape[0]], matrix, strict=False):
        if channel_name in mapping:
            mapping[channel_name].append(row)
    if any(len(mapping[channel]) == 0 for channel in ("R", "G", "B")):
        raise ValueError("Cannot derive RGB matrix from CFA matrix and CFA description.")
    return np.vstack(
        [
            np.mean(mapping["R"], axis=0),
            np.mean(mapping["G"], axis=0),
            np.mean(mapping["B"], axis=0),
        ]
    )


def normalize_with_sensor_levels(
    rgb: np.ndarray,
    black_levels_rgb: np.ndarray,
    white_levels_rgb: np.ndarray,
) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    black_levels_rgb = np.asarray(black_levels_rgb, dtype=np.float64)
    white_levels_rgb = np.asarray(white_levels_rgb, dtype=np.float64)
    span = white_levels_rgb - black_levels_rgb
    if np.any(span <= 0):
        raise ValueError("Sensor white levels must exceed black levels for all RGB channels.")
    normalized = (rgb - black_levels_rgb) / span
    return np.clip(normalized, 0.0, 1.0)


def compute_white_field_falloff(white_rgb: np.ndarray) -> dict | None:
    # Fit a radial polynomial `atten(r²) = c0 + c1·r² + c2·r⁴` to the
    # light fall-off of a uniformly-white reference image. r is the pixel
    # distance from the image center, normalized so the corner is r = 1.
    # The polynomial is scale-invariant: the caller reconstructs the per-
    # pixel gain map from any image's dimensions at apply time.
    white_rgb = np.asarray(white_rgb, dtype=np.float64)
    if white_rgb.ndim != 3 or white_rgb.shape[2] < 3 or min(white_rgb.shape[:2]) < 32:
        return None

    h, w = white_rgb.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    diag2 = cx * cx + cy * cy

    brightness = np.mean(white_rgb[..., :3], axis=-1)
    brightness = cv2.GaussianBlur(brightness.astype(np.float32), (0, 0), sigmaX=15.0).astype(np.float64)

    yy, xx = np.indices((h, w))
    r2_px = (xx - cx) ** 2 + (yy - cy) ** 2
    r2_norm = r2_px / max(diag2, 1.0)

    center_radius_px = 0.05 * min(h, w)
    center_mask = r2_px < center_radius_px * center_radius_px
    if not center_mask.any():
        return None
    center_value = float(np.percentile(brightness[center_mask], 98))
    if center_value <= 1e-6:
        return None

    attenuation = brightness / center_value

    flat_r2 = r2_norm.flatten()
    flat_att = attenuation.flatten()
    if flat_r2.size > 200_000:
        step = flat_r2.size // 200_000
        flat_r2 = flat_r2[::step]
        flat_att = flat_att[::step]

    coeffs = np.polyfit(flat_r2, flat_att, 2)   # [c2 (r⁴), c1 (r²), c0]
    c2, c1, c0 = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
    corner_attenuation = c0 + c1 + c2

    return {
        "polynomial_in_r2": [c2, c1, c0],
        "center_brightness": center_value,
        "min_attenuation": float(np.min(attenuation)),
        "corner_attenuation": float(corner_attenuation),
    }


def _white_field_poly(params: dict, r2: np.ndarray) -> np.ndarray:
    c2, c1, c0 = (float(v) for v in params["polynomial_in_r2"])
    return c0 + c1 * r2 + c2 * r2 * r2


def apply_white_field_correction(rgb: np.ndarray, params: dict) -> np.ndarray:
    if not params:
        return rgb
    rgb = np.asarray(rgb, dtype=np.float64)
    h, w = rgb.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    diag2 = max(cx * cx + cy * cy, 1.0)
    yy, xx = np.indices((h, w))
    r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / diag2
    atten = np.clip(_white_field_poly(params, r2), 0.05, None)
    gain = 1.0 / atten
    return rgb * gain[..., None]


def apply_white_field_correction_to_patches(
    rgb_patches: np.ndarray,
    patch_centers: np.ndarray,
    image_shape: tuple,
    params: dict,
) -> np.ndarray:
    if not params:
        return rgb_patches
    rgb_patches = np.asarray(rgb_patches, dtype=np.float64)
    h, w = image_shape[:2]
    cy, cx = h / 2.0, w / 2.0
    diag2 = max(cx * cx + cy * cy, 1.0)
    centers = np.asarray(patch_centers, dtype=np.float64)
    r2 = ((centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2) / diag2
    atten = np.clip(_white_field_poly(params, r2), 0.05, None)
    gain = 1.0 / atten
    return rgb_patches * gain[:, None]


def desaturate_highlights(rgb: np.ndarray, *, threshold: float = 0.93) -> np.ndarray:
    # Above `threshold` we leave HPPCC's training range; the per-hue regions
    # extrapolate unreliably and demosaicing artefacts (false-color near
    # clipped pixels) push the hue toward magenta/green. Forcing channels
    # toward their per-pixel max smoothly removes those colored blobs from
    # specular highlights — at the cost of progressively flattening any
    # legitimate saturation above the threshold (sensor info there is
    # already partly lost to clipping anyway).
    rgb = np.asarray(rgb, dtype=np.float64)
    max_channel = np.max(rgb, axis=-1, keepdims=True)
    denom = max(1.0 - float(threshold), 1e-12)
    blend = np.clip((max_channel - float(threshold)) / denom, 0.0, 1.0)
    return rgb * (1.0 - blend) + max_channel * blend


def _extract_patch_pixels(
    image: np.ndarray,
    center: np.ndarray,
    patch_size: tuple[int, int],
) -> np.ndarray:
    width, height = patch_size
    x, y = [int(round(v)) for v in center]
    half_w = width // 2
    half_h = height // 2
    top = max(0, y - half_h)
    bottom = min(image.shape[0], y + half_h)
    left = max(0, x - half_w)
    right = min(image.shape[1], x + half_w)
    return np.asarray(image[top:bottom, left:right, :3], dtype=np.float64)


def estimate_noise_profile_from_patches(
    normalized_image: np.ndarray,
    absolute_patch_centers: np.ndarray,
    patch_size: tuple[int, int],
    neutral_indices: np.ndarray,
    min_value: float = 0.01,
) -> dict[str, object]:
    neutral_indices = np.asarray(neutral_indices, dtype=int)
    patch_means = []
    patch_variances = []
    patch_payload = []

    for patch_index in neutral_indices:
        patch = _extract_patch_pixels(normalized_image, absolute_patch_centers[patch_index], patch_size)
        if patch.size == 0:
            continue
        pixels = patch.reshape(-1, 3)
        mean_rgb = np.mean(pixels, axis=0)
        if np.any(mean_rgb <= min_value) or np.any(mean_rgb >= 0.99):
            continue
        variance_rgb = np.var(pixels, axis=0, ddof=1)
        patch_means.append(mean_rgb)
        patch_variances.append(variance_rgb)
        patch_payload.append(
            {
                "patch_index": int(patch_index),
                "mean_rgb": mean_rgb.tolist(),
                "variance_rgb": variance_rgb.tolist(),
            }
        )

    if not patch_means:
        sigma_rgb = np.array([0.003, 0.003, 0.003], dtype=np.float64)
        variance_rgb = sigma_rgb**2
        return {
            "sigma_rgb": sigma_rgb.tolist(),
            "variance_rgb": variance_rgb.tolist(),
            "patch_count": 0,
            "patches": [],
        }

    means = np.asarray(patch_means, dtype=np.float64)
    variances = np.asarray(patch_variances, dtype=np.float64)
    sigma_rgb = np.sqrt(np.median(np.clip(variances, 0.0, None), axis=0))
    return {
        "sigma_rgb": sigma_rgb.tolist(),
        "variance_rgb": np.median(np.clip(variances, 0.0, None), axis=0).tolist(),
        "patch_count": int(len(patch_payload)),
        "patches": patch_payload,
    }


def denoise_linear_rgb_bilateral(
    normalized_image: np.ndarray,
    noise_profile: dict[str, object],
    *,
    strength: float,
    diameter: int,
    sigma_space: float,
) -> np.ndarray:
    if diameter <= 0 or strength <= 0.0:
        return np.asarray(normalized_image, dtype=np.float64).copy()

    sigma_rgb = np.asarray(noise_profile["sigma_rgb"], dtype=np.float64)
    image = np.asarray(normalized_image, dtype=np.float64)
    denoised = np.empty_like(image, dtype=np.float32)
    for channel_index in range(3):
        sigma_color = max(1e-6, float(strength * sigma_rgb[channel_index]))
        denoised[..., channel_index] = cv2.bilateralFilter(
            image[..., channel_index].astype(np.float32),
            d=int(diameter),
            sigmaColor=sigma_color,
            sigmaSpace=float(sigma_space),
        )
    return np.asarray(denoised, dtype=np.float64)


def _haar_dwt2(image: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], tuple[int, int]]:
    height, width = image.shape
    even_height = height // 2 * 2
    even_width = width // 2 * 2
    base = np.asarray(image[:even_height, :even_width], dtype=np.float64)
    a = base[0::2, 0::2]
    b = base[0::2, 1::2]
    c = base[1::2, 0::2]
    d = base[1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, (lh, hl, hh), image.shape


def _haar_idwt2(
    ll: np.ndarray,
    bands: tuple[np.ndarray, np.ndarray, np.ndarray],
    original_shape: tuple[int, int],
) -> np.ndarray:
    lh, hl, hh = bands
    height_half, width_half = ll.shape
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    reconstructed = np.zeros((height_half * 2, width_half * 2), dtype=np.float64)
    reconstructed[0::2, 0::2] = a
    reconstructed[0::2, 1::2] = b
    reconstructed[1::2, 0::2] = c
    reconstructed[1::2, 1::2] = d
    if reconstructed.shape != original_shape:
        padded = np.zeros(original_shape, dtype=np.float64)
        padded[: reconstructed.shape[0], : reconstructed.shape[1]] = reconstructed
        return padded
    return reconstructed


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _denoise_channel_wavelet(
    channel: np.ndarray,
    sigma: float,
    *,
    strength: float,
    levels: int = 4,
) -> np.ndarray:
    channel = np.asarray(channel, dtype=np.float64)
    approximation = channel
    pyramid: list[tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[int, int]]] = []

    for _ in range(levels):
        if min(approximation.shape) < 2:
            break
        approximation, detail_bands, original_shape = _haar_dwt2(approximation)
        pyramid.append((detail_bands, original_shape))

    for level in reversed(range(len(pyramid))):
        detail_bands, original_shape = pyramid[level]
        scale_sigma = float(sigma) / (2**level)
        denoised_bands = []
        for band in detail_bands:
            observed_variance = float(np.var(band))
            signal_sigma = np.sqrt(max(observed_variance - scale_sigma**2, 0.0))
            if signal_sigma > 0.0:
                threshold = float(strength) * (scale_sigma**2 / max(signal_sigma, 1e-8))
            else:
                threshold = float(strength) * scale_sigma
            denoised_bands.append(_soft_threshold(band, threshold))
        approximation = _haar_idwt2(approximation, tuple(denoised_bands), original_shape)

    return np.clip(approximation, 0.0, None)


def denoise_linear_rgb(
    normalized_image: np.ndarray,
    noise_profile: dict[str, object],
    *,
    method: str,
    strength: float,
    diameter: int,
    sigma_space: float,
) -> np.ndarray:
    if method == "bilateral":
        return denoise_linear_rgb_bilateral(
            normalized_image,
            noise_profile,
            strength=strength,
            diameter=diameter,
            sigma_space=sigma_space,
        )
    if method != "wavelet":
        raise ValueError(f"Unsupported denoise method: {method}")

    image = np.asarray(normalized_image, dtype=np.float64)
    sigma_rgb = np.asarray(noise_profile["sigma_rgb"], dtype=np.float64)
    denoised = np.empty_like(image, dtype=np.float64)
    for channel_index in range(3):
        denoised[..., channel_index] = _denoise_channel_wavelet(
            image[..., channel_index],
            float(sigma_rgb[channel_index]),
            strength=strength,
            levels=4,
        )
    return denoised


def sharpen_adaptive_rgb(
    normalized_image: np.ndarray,
    noise_profile: dict[str, object] | None,
    *,
    amount: float,
    radius: float,
    threshold: float,
) -> np.ndarray:
    image = np.asarray(normalized_image, dtype=np.float64)
    if amount <= 0.0 or radius <= 0.0 or image.size == 0:
        return image.copy()

    ksize = max(3, int(round(radius * 6.0)) | 1)
    blurred = cv2.GaussianBlur(image.astype(np.float32), (ksize, ksize), float(radius))
    detail = image - np.asarray(blurred, dtype=np.float64)

    if noise_profile is not None and threshold > 0.0:
        sigma_rgb = np.asarray(noise_profile.get("sigma_rgb", [0.0, 0.0, 0.0]), dtype=np.float64)
        mask = np.zeros_like(detail)
        for channel_index in range(3):
            sigma_threshold = max(float(sigma_rgb[channel_index]) * float(threshold), 1e-12)
            mask[..., channel_index] = np.clip(
                (np.abs(detail[..., channel_index]) - sigma_threshold) / sigma_threshold,
                0.0,
                1.0,
            )
        detail = detail * mask

    return np.clip(image + float(amount) * detail, 0.0, None)


def _apply_display_transfer(rgb_linear: np.ndarray) -> np.ndarray:
    rgb_linear = np.clip(rgb_linear, 0.0, None)
    threshold = 0.0031308
    rgb = np.where(
        rgb_linear <= threshold,
        12.92 * rgb_linear,
        1.055 * np.power(rgb_linear, 1.0 / 2.4) - 0.055,
    )
    return np.clip(rgb, 0.0, 1.0)


def xyz_to_output_rgb(xyz: np.ndarray, output_colorspace: str) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    if output_colorspace == "sRGB":
        matrix = XYZ_TO_SRGB_MATRIX
    elif output_colorspace == "Display-P3":
        matrix = XYZ_TO_DISPLAY_P3_MATRIX
    else:
        raise ValueError(f"Unsupported output colorspace: {output_colorspace}")
    rgb_linear = xyz @ matrix.T
    return _apply_display_transfer(rgb_linear)


def linear_xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
    return xyz_to_output_rgb(xyz, "sRGB")


def output_extension(output_format: str) -> str:
    mapping = {
        "jpeg": ".jpg",
        "png": ".png",
        "tif": ".tif",
    }
    try:
        return mapping[output_format]
    except KeyError as exc:
        raise ValueError(f"Unsupported output format: {output_format}") from exc


def get_icc_profile_bytes(output_colorspace: str) -> bytes:
    candidates = ICC_PROFILE_CANDIDATES.get(output_colorspace)
    if candidates is None:
        raise ValueError(f"Unsupported output colorspace: {output_colorspace}")

    for profile_path in candidates:
        if profile_path.exists():
            return profile_path.read_bytes()

    if output_colorspace == "sRGB":
        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    raise FileNotFoundError(f"No ICC profile found for output colorspace {output_colorspace}.")


def apply_matrix_transform(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    return rgb @ matrix


def predict_hppcc(
    hppcc_model,
    rgb: np.ndarray,
    *,
    use_blending: bool,
    blend_width: float,
) -> np.ndarray:
    if use_blending:
        return hppcc_model.predict_blending(rgb, blend_width=blend_width)
    return hppcc_model.predict(rgb)


def predict_hppcc_rpcc(
    model: HPPCCRPCCModel,
    rgb: np.ndarray,
    *,
    use_blending: bool,
    blend_width: float,
) -> np.ndarray:
    if use_blending:
        return model.predict_blending(rgb, blend_width=blend_width)
    return model.predict(rgb)



def hppcc_rpcc_model_to_dict(model: HPPCCRPCCModel) -> dict[str, object]:
    return {
        "hppcc": hppcc_model_to_dict(model.hppcc),
        "rpcc_residual": rpcc_model_to_dict(model.rpcc_residual),
    }


def hppcc_rpcc_model_from_dict(payload: dict[str, object]) -> HPPCCRPCCModel:
    return HPPCCRPCCModel(
        hppcc=hppcc_model_from_dict(payload["hppcc"]),
        rpcc_residual=rpcc_model_from_dict(payload["rpcc_residual"]),
    )


def rpcc_model_to_dict(model: RPCCModel) -> dict[str, object]:
    return {
        "matrix": np.asarray(model.matrix, dtype=np.float64).tolist(),
        "white_rgb": np.asarray(model.white_rgb, dtype=np.float64).tolist(),
        "white_xyz": np.asarray(model.white_xyz, dtype=np.float64).tolist(),
    }


def rpcc_model_from_dict(payload: dict[str, object]) -> RPCCModel:
    return RPCCModel(
        matrix=np.asarray(payload["matrix"], dtype=np.float64),
        white_rgb=np.asarray(payload["white_rgb"], dtype=np.float64),
        white_xyz=np.asarray(payload["white_xyz"], dtype=np.float64),
    )



def linear_model_to_dict(model: LinearWhitePreservingModel) -> dict[str, object]:
    return {
        "matrix": np.asarray(model.matrix, dtype=np.float64).tolist(),
        "white_rgb": np.asarray(model.white_rgb, dtype=np.float64).tolist(),
        "white_xyz": np.asarray(model.white_xyz, dtype=np.float64).tolist(),
    }


def hppcc_model_to_dict(model: HPPCCModel) -> dict[str, object]:
    return {
        "matrices": np.asarray(model.matrices, dtype=np.float64).tolist(),
        "boundaries": np.asarray(model.boundaries, dtype=np.float64).tolist(),
        "white_rgb": np.asarray(model.white_rgb, dtype=np.float64).tolist(),
        "white_xyz": np.asarray(model.white_xyz, dtype=np.float64).tolist(),
    }


def linear_model_from_dict(payload: dict[str, object]) -> LinearWhitePreservingModel:
    return LinearWhitePreservingModel(
        matrix=np.asarray(payload["matrix"], dtype=np.float64),
        white_rgb=np.asarray(payload["white_rgb"], dtype=np.float64),
        white_xyz=np.asarray(payload["white_xyz"], dtype=np.float64),
    )


def hppcc_model_from_dict(payload: dict[str, object]) -> HPPCCModel:
    return HPPCCModel(
        matrices=np.asarray(payload["matrices"], dtype=np.float64),
        boundaries=np.asarray(payload["boundaries"], dtype=np.float64),
        white_rgb=np.asarray(payload["white_rgb"], dtype=np.float64),
        white_xyz=np.asarray(payload["white_xyz"], dtype=np.float64),
    )


def save_analysis_result(output_dir: Path, raw_path: Path, payload: dict[str, object]) -> Path:
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{raw_path.stem}_correction.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_analysis_result(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_exif_from_raw(source_raw_path: Path, target_image_path: Path) -> None:
    command = [
        "exiftool",
        "-m",                              # ignore minor warnings (otherwise exit code 1)
        "-charset", "filename=UTF8",       # interpret path args as UTF-8 (non-ASCII paths)
        "-charset", "utf8",                # tag values also UTF-8
        "-overwrite_original",
        "-TagsFromFile",
        str(source_raw_path),
        "-EXIF:all",
        "-XMP:all",
        "-IPTC:all",
        "-MakerNotes:all",
        str(target_image_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        stdout = completed.stdout.decode("utf-8", errors="replace").strip()
        detail = stderr or stdout or f"exit code {completed.returncode}, no output"
        raise RuntimeError(f"exiftool failed for {target_image_path}: {detail}")


def to_uint8_image(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).round().astype(np.uint8)


def summarize_model(
    src_module,
    measured_rgb: np.ndarray,
    reference_xyz: np.ndarray,
    illuminant_white: np.ndarray,
    *,
    white_index: int,
    chromatic_indices: np.ndarray,
    hppcc_regions: int,
    optimize_boundaries: bool = False,
    use_hppcc_blending: bool = False,
    hppcc_blend_width: float = 0.15,
    perform_nonlinear_corrections: bool = True,
    reference_chroma: np.ndarray | None = None,
    reliable_patch_indices: np.ndarray | None = None,
) -> dict[str, object]:
    # Baseline and RPCC train on all patches for numerical stability.
    all_indices = np.arange(measured_rgb.shape[0], dtype=int)
    measured_rgb_train = measured_rgb[all_indices]
    reference_xyz_train = reference_xyz[all_indices]
    white_index_train = white_index

    # HPPCC chromatic partition uses only reliable patches to avoid noise-floor contamination.
    if reliable_patch_indices is not None:
        reliable_set = set(int(i) for i in reliable_patch_indices)
        chromatic_indices_train = np.array(
            [ci for ci in chromatic_indices if int(ci) in reliable_set], dtype=int
        )
    else:
        chromatic_indices_train = np.asarray(chromatic_indices, dtype=int)

    baseline = src_module.fit_white_preserving_3x3(
        measured_rgb_train,
        reference_xyz_train,
        white_index=white_index_train,
    )
    baseline_xyz = baseline.predict(measured_rgb)
    baseline_de00 = delta_e00_summary(src_module, baseline_xyz, reference_xyz, illuminant_white)

    rpcc = src_module.fit_rpcc(
        measured_rgb_train,
        reference_xyz_train,
        white_index=white_index_train,
    )
    rpcc_xyz = rpcc.predict(measured_rgb)
    rpcc_de00 = delta_e00_summary(src_module, rpcc_xyz, reference_xyz, illuminant_white)

    result = {
        "baseline": baseline,
        "baseline_de00": baseline_de00,
        "rpcc": rpcc,
        "rpcc_de00": rpcc_de00,
    }

    if reference_chroma is not None:
        result["baseline_chroma_error"] = chroma_error_summary(src_module, baseline_xyz, illuminant_white, reference_chroma)
        result["rpcc_chroma_error"] = chroma_error_summary(src_module, rpcc_xyz, illuminant_white, reference_chroma)

    if len(chromatic_indices_train) < 2:
        # Too few reliable chromatic patches — HPPCC cannot be fit; fall back to RPCC.
        result["hppcc"] = rpcc
        result["hppcc_de00"] = rpcc_de00
        if reference_chroma is not None:
            result["hppcc_chroma_error"] = result["rpcc_chroma_error"]
        if perform_nonlinear_corrections:
            result["hppcc_rpcc"] = rpcc
            result["hppcc_rpcc_de00"] = rpcc_de00
            if reference_chroma is not None:
                result["hppcc_rpcc_chroma_error"] = result["rpcc_chroma_error"]
        return result

    hppcc = src_module.fit_hppcc(
        measured_rgb_train,
        reference_xyz_train,
        white_index=white_index_train,
        chromatic_indices=chromatic_indices_train,
        k_regions=hppcc_regions,
        optimize_boundaries=optimize_boundaries,
    )
    hppcc_xyz = predict_hppcc(hppcc, measured_rgb, use_blending=use_hppcc_blending, blend_width=hppcc_blend_width)
    hppcc_de00 = delta_e00_summary(src_module, hppcc_xyz, reference_xyz, illuminant_white)
    result["hppcc"] = hppcc
    result["hppcc_de00"] = hppcc_de00

    if reference_chroma is not None:
        result["hppcc_chroma_error"] = chroma_error_summary(src_module, hppcc_xyz, illuminant_white, reference_chroma)

    if perform_nonlinear_corrections:
        hppcc_rpcc = src_module.fit_hppcc_rpcc(
            measured_rgb_train,
            reference_xyz_train,
            white_index=white_index_train,
            chromatic_indices=chromatic_indices_train,
            k_regions=hppcc_regions,
            optimize_boundaries=optimize_boundaries,
        )
        hppcc_rpcc_xyz = predict_hppcc_rpcc(
            hppcc_rpcc, measured_rgb, use_blending=use_hppcc_blending, blend_width=hppcc_blend_width
        )
        hppcc_rpcc_de00 = delta_e00_summary(src_module, hppcc_rpcc_xyz, reference_xyz, illuminant_white)
        result["hppcc_rpcc"] = hppcc_rpcc
        result["hppcc_rpcc_de00"] = hppcc_rpcc_de00
        if reference_chroma is not None:
            result["hppcc_rpcc_chroma_error"] = chroma_error_summary(
                src_module, hppcc_rpcc_xyz, illuminant_white, reference_chroma
            )

    return result


def evaluate_illuminants(
    src_module,
    measured_rgb: np.ndarray,
    reference_xyz: np.ndarray,
    *,
    reference_illuminant: str,
    standard_whites: dict[str, np.ndarray],
    white_index: int,
    chromatic_indices: np.ndarray,
    hppcc_regions: int,
    training_indices: np.ndarray | None = None,
    use_hppcc_blending: bool = False,
    hppcc_blend_width: float = 0.15,
) -> dict[str, dict[str, object]]:
    if training_indices is None:
        training_indices = np.arange(measured_rgb.shape[0], dtype=int)
    training_indices = np.asarray(training_indices, dtype=int)

    measured_rgb_train = measured_rgb[training_indices]
    index_map = {int(index): position for position, index in enumerate(training_indices)}
    white_index_train = int(index_map[white_index])
    chromatic_indices_train = np.array(
        [index_map[int(index)] for index in chromatic_indices if int(index) in index_map],
        dtype=int,
    )

    illuminant_results = {}
    source_white = standard_whites[reference_illuminant]
    for illuminant_name, target_white in standard_whites.items():
        adapted_reference_xyz = bradford_adapt_xyz(reference_xyz, source_white, target_white)
        illuminant_results[illuminant_name] = summarize_model(
            src_module,
            measured_rgb_train,
            adapted_reference_xyz[training_indices],
            target_white,
            white_index=white_index_train,
            chromatic_indices=chromatic_indices_train,
            hppcc_regions=hppcc_regions,
            use_hppcc_blending=use_hppcc_blending,
            hppcc_blend_width=hppcc_blend_width,
        )
        illuminant_results[illuminant_name]["baseline_de00"] = delta_e00_summary(
            src_module,
            illuminant_results[illuminant_name]["baseline"].predict(measured_rgb),
            adapted_reference_xyz,
            target_white,
        )
        illuminant_results[illuminant_name]["rpcc_de00"] = delta_e00_summary(
            src_module,
            illuminant_results[illuminant_name]["rpcc"].predict(measured_rgb),
            adapted_reference_xyz,
            target_white,
        )
        illuminant_results[illuminant_name]["hppcc_de00"] = delta_e00_summary(
            src_module,
            predict_hppcc(
                illuminant_results[illuminant_name]["hppcc"],
                measured_rgb,
                use_blending=use_hppcc_blending,
                blend_width=hppcc_blend_width,
            ),
            adapted_reference_xyz,
            target_white,
        )
        illuminant_results[illuminant_name]["rpcc_hppcc_de00"] = delta_e00_summary(
            src_module,
            predict_rpcc_hppcc(
                illuminant_results[illuminant_name]["rpcc_hppcc"],
                measured_rgb,
                use_blending=use_hppcc_blending,
                blend_width=hppcc_blend_width,
            ),
            adapted_reference_xyz,
            target_white,
        )
    return illuminant_results
