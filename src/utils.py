import json
import subprocess
from pathlib import Path

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

    r_g = valid_rgb[:, 0] / valid_rgb[:, 1]
    b_g = valid_rgb[:, 2] / valid_rgb[:, 1]
    scene_cam = np.array([np.mean(r_g), 1.0, np.mean(b_g)], dtype=np.float64)

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


def find_raw_paths(image_dir: Path) -> list[Path]:
    candidates = []
    for pattern in ("*.NEF", "*.CR2", "*.CR3", "*.ARW", "*.RAF", "*.DNG"):
        candidates.extend(sorted(image_dir.glob(pattern)))
    return candidates


def load_reference_xyz(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data["CONFIG"]["TARGET"]["referenceXYZValues"]
    xyz = np.asarray([values[str(i)] for i in range(1, 25)], dtype=np.float64)
    if xyz.shape != (24, 3):
        raise ValueError("Reference XYZ data must contain 24 patches with 3 values each.")
    return xyz


def delta_e00_summary(src_module, predicted_xyz: np.ndarray, reference_xyz: np.ndarray, white_xyz: np.ndarray) -> np.ndarray:
    predicted_lab = src_module.xyz_to_lab(predicted_xyz, white_xyz)
    reference_lab = src_module.xyz_to_lab(reference_xyz, white_xyz)
    return src_module.delta_e_2000(predicted_lab, reference_lab)


def bradford_adapt_xyz(xyz: np.ndarray, source_white: np.ndarray, target_white: np.ndarray) -> np.ndarray:
    source_rgb = BRADFORD_MATRIX @ source_white
    target_rgb = BRADFORD_MATRIX @ target_white
    scaling = np.diag(target_rgb / source_rgb)
    adaptation = np.linalg.inv(BRADFORD_MATRIX) @ scaling @ BRADFORD_MATRIX
    return xyz @ adaptation.T


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
    return np.clip(normalized, 0.0, None)


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
    output_path = output_dir / f"result_{raw_path.stem}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_analysis_result(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_exif_from_raw(source_raw_path: Path, target_image_path: Path) -> None:
    command = [
        "exiftool",
        "-overwrite_original",
        "-TagsFromFile",
        str(source_raw_path),
        "-EXIF:all",
        "-XMP:all",
        "-IPTC:all",
        "-MakerNotes:all",
        str(target_image_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"exiftool failed for {target_image_path}: {completed.stderr.strip() or completed.stdout.strip()}"
        )


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
) -> dict[str, object]:
    training_indices = np.arange(measured_rgb.shape[0], dtype=int)
    measured_rgb_train = measured_rgb[training_indices]
    reference_xyz_train = reference_xyz[training_indices]
    white_index_train = int(np.where(training_indices == white_index)[0][0])
    index_map = {int(index): position for position, index in enumerate(training_indices)}
    chromatic_indices_train = np.array([index_map[int(index)] for index in chromatic_indices], dtype=int)

    baseline = src_module.fit_white_preserving_3x3(
        measured_rgb_train,
        reference_xyz_train,
        white_index=white_index_train,
    )
    baseline_xyz = baseline.predict(measured_rgb)
    baseline_de00 = delta_e00_summary(src_module, baseline_xyz, reference_xyz, illuminant_white)

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

    result = {
        "baseline": baseline,
        "baseline_de00": baseline_de00,
        "hppcc": hppcc,
        "hppcc_de00": hppcc_de00,
    }

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
