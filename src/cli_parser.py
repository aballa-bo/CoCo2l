import argparse
import json
from pathlib import Path

import numpy as np

from src.config import (
    ANALYSIS_DIR,
    CHROMATIC_INDICES,
    DENOISE_METHOD,
    DENOISE_DIAMETER,
    DENOISE_SIGMA_SPACE,
    DENOISE_STRENGTH,
    ENABLE_ADAPTIVE_SHARPEN,
    ENABLE_PATCH_VARIANCE_DENOISE,
    ENABLE_PROCESS_WHITE_FIELD,
    HPPCC_BLEND_WIDTH,
    HPPCC_GRADIENT,
    HPPCC_GRADIENT_HARMONICS,
    HPPCC_REGION_CANDIDATES,
    HPPCC_REGION_SMOOTHNESS,
    IMAGE_DIR,
    OUTPUT_COLORSPACE,
    OUTPUT_FORMAT,
    PROCESS_DIR,
    REFERENCE_ILLUMINANT,
    REFERENCE_PATH,
    REFERENCE_SPACE,
    SCENE_WHITE_SOURCE,
    SHARPEN_AMOUNT,
    SHARPEN_RADIUS,
    SHARPEN_THRESHOLD,
    SHOW_DETECTION_PREVIEW,
    SHOW_DEVELOPED_IMAGE_PREVIEW,
    STANDARD_WHITES,
    USE_HPPCC,
    USE_HPPCC_BLENDING,
    USE_METADATA_RGB_XYZ_BASELINE,
    USE_RPCC,
    WHITE_INDEX,
)


def parse_indices(value: str) -> np.ndarray:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("Expected at least one integer index.")
    try:
        return np.asarray([int(token) for token in tokens], dtype=int)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Indices must be comma-separated integers.") from exc


def parse_standard_whites(value: str) -> dict[str, np.ndarray]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("standard whites must be valid JSON.") from exc
    if not isinstance(payload, dict) or not payload:
        raise argparse.ArgumentTypeError("standard whites must be a non-empty JSON object.")

    parsed: dict[str, np.ndarray] = {}
    for name, white_xyz in payload.items():
        array = np.asarray(white_xyz, dtype=np.float64)
        if array.shape != (3,):
            raise argparse.ArgumentTypeError(
                f"Standard white '{name}' must contain exactly 3 numeric values."
            )
        parsed[str(name)] = array
    return parsed


def _add_analysis_config_arguments(parser: argparse.ArgumentParser, *, use_defaults: bool) -> None:
    parser.add_argument(
        "--white-index",
        type=int,
        default=WHITE_INDEX if use_defaults else None,
        help="Zero-based white patch index.",
    )
    parser.add_argument(
        "--chromatic-indices",
        type=parse_indices,
        default=CHROMATIC_INDICES.copy() if use_defaults else None,
        help="Comma-separated zero-based chromatic patch indices.",
    )
    parser.add_argument(
        "--hppcc-region-candidates",
        type=parse_indices,
        default=np.asarray(HPPCC_REGION_CANDIDATES, dtype=int) if use_defaults else None,
        help="Comma-separated HPPCC region counts to compare.",
    )
    parser.add_argument(
        "--reference-illuminant",
        type=str,
        default=REFERENCE_ILLUMINANT if use_defaults else None,
        help="Reference illuminant key.",
    )
    parser.add_argument(
        "--reference-space",
        choices=("xyz", "lab", "xyy"),
        default=REFERENCE_SPACE if use_defaults else None,
        help="Reference dataset representation used in the reference JSON.",
    )
    parser.add_argument(
        "--standard-whites-json",
        type=parse_standard_whites,
        default={key: value.copy() for key, value in STANDARD_WHITES.items()} if use_defaults else None,
        help='JSON object overriding standard whites, e.g. \'{"D65":[0.95047,1,1.08883]}\'',
    )
    parser.add_argument(
        "--scene-white-source",
        choices=("auto", "reference", "camera-wb", "neutral-patches"),
        default=SCENE_WHITE_SOURCE if use_defaults else None,
        help="Scene white estimate to use for reference adaptation.",
    )
    parser.add_argument(
        "--patch-variance-denoise",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_PATCH_VARIANCE_DENOISE if use_defaults else None,
        help="Estimate noise from chart patches and denoise linear RGB before fitting.",
    )
    parser.add_argument(
        "--denoise-method",
        choices=("wavelet", "bilateral"),
        default=DENOISE_METHOD if use_defaults else None,
        help="Denoising method used with the patch-derived noise profile.",
    )
    parser.add_argument(
        "--denoise-strength",
        type=float,
        default=DENOISE_STRENGTH if use_defaults else None,
        help="Strength parameter applied to the estimated per-channel noise sigma.",
    )
    parser.add_argument(
        "--denoise-diameter",
        type=int,
        default=DENOISE_DIAMETER if use_defaults else None,
        help="Neighborhood diameter for bilateral denoising.",
    )
    parser.add_argument(
        "--denoise-sigma-space",
        type=float,
        default=DENOISE_SIGMA_SPACE if use_defaults else None,
        help="Spatial sigma for bilateral denoising in pixels.",
    )
    parser.add_argument(
        "--adaptive-sharpen",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_ADAPTIVE_SHARPEN if use_defaults else None,
        help="Apply adaptive unsharp mask gated by the patch-derived noise profile.",
    )
    parser.add_argument(
        "--sharpen-amount",
        type=float,
        default=SHARPEN_AMOUNT if use_defaults else None,
        help="Strength multiplier on the unsharp-mask detail layer.",
    )
    parser.add_argument(
        "--sharpen-radius",
        type=float,
        default=SHARPEN_RADIUS if use_defaults else None,
        help="Gaussian sigma (pixels) used to build the unsharp-mask blur.",
    )
    parser.add_argument(
        "--sharpen-threshold",
        type=float,
        default=SHARPEN_THRESHOLD if use_defaults else None,
        help="Multiplier of per-channel noise sigma below which detail is suppressed.",
    )
    parser.add_argument(
        "--process-white-field",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_PROCESS_WHITE_FIELD if use_defaults else None,
        help="Fit a vignetting/light fall-off model from a white reference image and apply it before HPPCC.",
    )
    parser.add_argument(
        "--white-field-image",
        type=Path,
        default=None,
        help="Path to the white reference RAW. Required when --process-white-field is set (analyze only).",
    )
    parser.add_argument(
        "--use-metadata-rgb-xyz-baseline",
        action=argparse.BooleanOptionalAction,
        default=USE_METADATA_RGB_XYZ_BASELINE if use_defaults else None,
        help="Enable metadata rgb_xyz_matrix baseline report.",
    )
    parser.add_argument(
        "--use-hppcc-blending",
        action=argparse.BooleanOptionalAction,
        default=USE_HPPCC_BLENDING if use_defaults else None,
        help="Use predict_blending instead of hard HPPCC prediction.",
    )
    parser.add_argument(
        "--hppcc-blend-width",
        type=float,
        default=HPPCC_BLEND_WIDTH if use_defaults else None,
        help="Blend width fraction for HPPCC soft prediction.",
    )
    parser.add_argument(
        "--hppcc-region-smoothness",
        type=float,
        default=HPPCC_REGION_SMOOTHNESS if use_defaults else None,
        help="Tikhonov weight coupling adjacent HPPCC region matrices during "
             "the fit; higher = smoother hue transitions, 0 disables (analyze only).",
    )
    parser.add_argument(
        "--use-hppcc",
        action=argparse.BooleanOptionalAction,
        default=USE_HPPCC if use_defaults else None,
        help="Enable Hue-Plane Preserving Color Correction (HPPCC). "
             "--no-use-hppcc applies the linear baseline only.",
    )
    parser.add_argument(
        "--use-rpcc",
        action=argparse.BooleanOptionalAction,
        default=USE_RPCC if use_defaults else None,
        help="Enable Root-Polynomial residual stage on top of HPPCC (HPPCC+RPCC). "
             "Has no effect when --no-use-hppcc is set.",
    )
    parser.add_argument(
        "--hppcc-gradient",
        action=argparse.BooleanOptionalAction,
        default=HPPCC_GRADIENT if use_defaults else None,
        help="Fit a trigonometric-series HPPCC instead of the piecewise-constrained model.",
    )
    parser.add_argument(
        "--hppcc-gradient-harmonics",
        type=int,
        default=HPPCC_GRADIENT_HARMONICS if use_defaults else None,
        help="Number of Fourier harmonics for the gradient HPPCC (default: 2).",
    )
    parser.add_argument(
        "--show-detection-preview",
        action=argparse.BooleanOptionalAction,
        default=SHOW_DETECTION_PREVIEW if use_defaults else None,
        help="Show detection preview windows.",
    )
    parser.add_argument(
        "--show-developed-image-preview",
        action=argparse.BooleanOptionalAction,
        default=SHOW_DEVELOPED_IMAGE_PREVIEW if use_defaults else None,
        help="Show a preview window of the developed image after analysis.",
    )
    parser.add_argument(
        "--output-format",
        choices=("tif", "jpeg", "png"),
        default=OUTPUT_FORMAT if use_defaults else None,
        help="Output image format.",
    )
    parser.add_argument(
        "--output-colorspace",
        choices=("sRGB", "Display-P3"),
        default=OUTPUT_COLORSPACE if use_defaults else None,
        help="Output RGB colorspace.",
    )


def parse_roi(value: str) -> tuple[int, int, int, int]:
    """Parse --roi as 'x1,y1,x2,y2' pixel coordinates."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi must be four integers: x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--roi values must be integers") from exc
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("--roi requires x2>x1 and y2>y1")
    return x1, y1, x2, y2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HPPCC RAW analyzer.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze and correct a RAW file.")
    analyze.add_argument("--cc-image", type=Path, default=None, help="RAW file to analyze.")
    analyze.add_argument("--image-dir", type=Path, default=IMAGE_DIR, help="Directory used when --cc-image is omitted.")
    analyze.add_argument("--reference-path", type=Path, default=REFERENCE_PATH, help="Reference JSON path.")
    analyze.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR, help="Output directory for analysis files (overlays, JSON). Pass the same path as --process-dir to collect all outputs in one folder.")
    analyze.add_argument("--process-dir", type=Path, default=PROCESS_DIR, help="Output directory for the developed image. Pass the same path as --analysis-dir to collect all outputs in one folder.")
    analyze.add_argument("--roi", type=parse_roi, default=None, metavar="x1,y1,x2,y2",
                         help="Pixel crop applied to the RAW before color checker detection (e.g. 200,100,1800,1200).")
    _add_analysis_config_arguments(analyze, use_defaults=True)

    process = subparsers.add_parser("process", help="Process a folder of RAW files using a saved analysis result.")
    process.add_argument("result_json", type=Path, help="Path to <stem>_correction.json produced by analyze.")
    process.add_argument("folder_to_process", type=Path, help="Folder containing RAW files to process.")
    process.add_argument("--process-dir", type=Path, default=None, help="Output directory for developed images.")
    process.add_argument("--recursive", action="store_true", default=False,
                         help="Recurse into subdirectories when looking for RAW files.")
    process.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers. Default: min(available CPUs, number of RAW files).",
    )
    _add_analysis_config_arguments(process, use_defaults=False)
    return parser
