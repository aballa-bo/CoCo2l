import argparse
import json
from pathlib import Path

import numpy as np

from src.config import (
    CHROMATIC_INDICES,
    HPPCC_BLEND_WIDTH,
    HPPCC_REGION_CANDIDATES,
    IMAGE_DIR,
    OUTPUT_DIR,
    OUTPUT_COLORSPACE,
    OUTPUT_FORMAT,
    REFERENCE_ILLUMINANT,
    REFERENCE_PATH,
    SHOW_DETECTION_PREVIEW,
    STANDARD_WHITES,
    USE_HPPCC_BLENDING,
    USE_METADATA_RGB_XYZ_BASELINE,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HPPCC RAW analyzer.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze and correct a RAW file.")
    analyze.add_argument("--cc-image", type=Path, default=None, help="RAW file to analyze.")
    analyze.add_argument("--image-dir", type=Path, default=IMAGE_DIR, help="Directory used when --cc-image is omitted.")
    analyze.add_argument("--reference-path", type=Path, default=REFERENCE_PATH, help="Reference JSON path.")
    analyze.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    analyze.add_argument("--white-index", type=int, default=WHITE_INDEX, help="Zero-based white patch index.")
    analyze.add_argument(
        "--chromatic-indices",
        type=parse_indices,
        default=CHROMATIC_INDICES.copy(),
        help="Comma-separated zero-based chromatic patch indices.",
    )
    analyze.add_argument(
        "--hppcc-region-candidates",
        type=parse_indices,
        default=np.asarray(HPPCC_REGION_CANDIDATES, dtype=int),
        help="Comma-separated HPPCC region counts to compare.",
    )
    analyze.add_argument(
        "--reference-illuminant",
        type=str,
        default=REFERENCE_ILLUMINANT,
        help="Reference illuminant key.",
    )
    analyze.add_argument(
        "--standard-whites-json",
        type=parse_standard_whites,
        default={key: value.copy() for key, value in STANDARD_WHITES.items()},
        help='JSON object overriding standard whites, e.g. \'{"D65":[0.95047,1,1.08883]}\'',
    )
    analyze.add_argument(
        "--hppcc-blend-width",
        type=float,
        default=HPPCC_BLEND_WIDTH,
        help="Blend width fraction for HPPCC soft prediction.",
    )
    analyze.add_argument(
        "--use-metadata-rgb-xyz-baseline",
        action=argparse.BooleanOptionalAction,
        default=USE_METADATA_RGB_XYZ_BASELINE,
        help="Enable metadata rgb_xyz_matrix baseline report.",
    )
    analyze.add_argument(
        "--use-hppcc-blending",
        action=argparse.BooleanOptionalAction,
        default=USE_HPPCC_BLENDING,
        help="Use predict_blending instead of hard HPPCC prediction.",
    )
    analyze.add_argument(
        "--show-detection-preview",
        action=argparse.BooleanOptionalAction,
        default=SHOW_DETECTION_PREVIEW,
        help="Show detection preview windows.",
    )
    analyze.add_argument(
        "--output-format",
        choices=("tif", "jpeg", "png"),
        default=OUTPUT_FORMAT,
        help="Output image format.",
    )
    analyze.add_argument(
        "--output-colorspace",
        choices=("sRGB", "Display-P3"),
        default=OUTPUT_COLORSPACE,
        help="Output RGB colorspace.",
    )

    process = subparsers.add_parser("process", help="Process a folder of RAW files using a saved analysis result.")
    process.add_argument("result_json", type=Path, help="Path to result_<raw>.json produced by analyze.")
    process.add_argument("folder_to_process", type=Path, help="Folder containing RAW files to process.")
    process.add_argument("--output-dir", type=Path, default=None, help="Output directory for corrected images.")
    process.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers. Default: min(available CPUs, number of RAW files).",
    )
    return parser
