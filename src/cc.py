from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

import src
from src.colorchecker import CLASSIC_24_PATCH_NAMES
from src.cli_parser import build_parser
from src.colorchecker_detector import (
    detect_and_orient_colorchecker,
    make_json_grid_preview,
    show_detection_preview,
)
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
from src.report import (
    maybe_show_detection_preview,
    print_detection_summary,
    print_hppcc_candidate_report,
    print_hppcc_region_report,
    print_illuminant_report,
    print_metadata_matrix_report,
    print_model_report,
    print_overlay_summary,
    print_patch_correction_report,
    print_patch_delta_e_report,
    save_named_corrected_image,
    save_json_grid_preview,
    show_corrected_image,
)
from src.utils import (
    apply_matrix_transform,
    bradford_adapt_xyz,
    copy_exif_from_raw,
    delta_e00_summary,
    evaluate_illuminants,
    find_raw_path,
    find_raw_paths,
    hppcc_model_from_dict,
    linear_model_to_dict,
    load_reference_xyz,
    load_analysis_result,
    normalize_with_sensor_levels,
    output_extension,
    predict_hppcc,
    reduce_cfa_matrix_to_rgb,
    reduce_cfa_values_to_rgb,
    save_analysis_result,
    hppcc_model_to_dict,
    to_uint8_image,
    get_icc_profile_bytes,
    xyz_to_output_rgb,
)

def run_analysis(args) -> None:
    raw_path = args.cc_image if args.cc_image is not None else find_raw_path(args.image_dir)
    reference_xyz = load_reference_xyz(args.reference_path)

    raw = src.load_raw_linear_rgb(raw_path)
    detection = detect_and_orient_colorchecker(raw.rgb, white_index=args.white_index)
    sensor_black_levels_rgb = reduce_cfa_values_to_rgb(raw.black_level_per_channel, raw.color_desc)
    sensor_white_levels_rgb = reduce_cfa_values_to_rgb(raw.camera_white_level_per_channel, raw.color_desc)
    metadata_rgb_xyz_matrix = reduce_cfa_matrix_to_rgb(raw.rgb_xyz_matrix, raw.color_desc)
    if sensor_black_levels_rgb is None or sensor_white_levels_rgb is None:
        raise RuntimeError("Missing sensor black/white metadata required for radiometric normalization.")

    normalized_rgb = normalize_with_sensor_levels(
        detection.measured_rgb,
        sensor_black_levels_rgb,
        sensor_white_levels_rgb,
    )
    corrected_rgb = normalized_rgb
    training_indices = np.arange(normalized_rgb.shape[0], dtype=int)

    print_detection_summary(raw_path, detection)
    print_patch_correction_report(
        sensor_black_levels_rgb,
        sensor_white_levels_rgb,
        detection.measured_rgb,
        normalized_rgb,
    )

    save_json_grid_preview(args.output_dir, make_json_grid_preview, corrected_rgb)

    hppcc_candidate_results = []
    for k_regions in np.asarray(args.hppcc_region_candidates, dtype=int):
        illuminant_results = evaluate_illuminants(
            src,
            corrected_rgb,
            reference_xyz,
            reference_illuminant=args.reference_illuminant,
            standard_whites=args.standard_whites_json,
            white_index=args.white_index,
            chromatic_indices=np.asarray(args.chromatic_indices, dtype=int),
            hppcc_regions=int(k_regions),
            training_indices=training_indices,
            use_hppcc_blending=args.use_hppcc_blending,
            hppcc_blend_width=args.hppcc_blend_width,
        )
        best_illuminant = min(
            illuminant_results,
            key=lambda name: np.mean(illuminant_results[name]["hppcc_de00"]),
        )
        hppcc_candidate_results.append(
            {
                "k": int(k_regions),
                "illuminant_results": illuminant_results,
                "best_illuminant": best_illuminant,
                "best": illuminant_results[best_illuminant],
            }
        )

    best_candidate_index = print_hppcc_candidate_report(
        hppcc_candidate_results,
        use_hppcc_blending=args.use_hppcc_blending,
        hppcc_blend_width=args.hppcc_blend_width,
    )
    selected_candidate = hppcc_candidate_results[best_candidate_index]
    illuminant_results = selected_candidate["illuminant_results"]
    best_illuminant = selected_candidate["best_illuminant"]
    best = selected_candidate["best"]
    selected_k_regions = int(selected_candidate["k"])

    print_illuminant_report(args.reference_illuminant, illuminant_results)
    if args.use_metadata_rgb_xyz_baseline and metadata_rgb_xyz_matrix is not None:
        adapted_reference_xyz = bradford_adapt_xyz(
            reference_xyz,
            args.standard_whites_json[args.reference_illuminant],
            args.standard_whites_json[best_illuminant],
        )
        metadata_xyz = apply_matrix_transform(corrected_rgb, metadata_rgb_xyz_matrix)
        metadata_de00 = delta_e00_summary(
            src,
            metadata_xyz,
            adapted_reference_xyz,
            args.standard_whites_json[best_illuminant],
        )
        print_metadata_matrix_report("Metadata rgb_xyz_matrix", metadata_de00)

    normalized_full_rgb = normalize_with_sensor_levels(
        raw.rgb,
        sensor_black_levels_rgb,
        sensor_white_levels_rgb,
    )
    corrected_full_xyz = predict_hppcc(
        best["hppcc"],
        normalized_full_rgb,
        use_blending=args.use_hppcc_blending,
        blend_width=args.hppcc_blend_width,
    )
    corrected_full_output_rgb = xyz_to_output_rgb(corrected_full_xyz, args.output_colorspace)
    corrected_full_uint8 = to_uint8_image(corrected_full_output_rgb)
    analysis_output_path = args.output_dir / f"corrected_hppcc{output_extension(args.output_format)}"
    save_named_corrected_image(
        analysis_output_path,
        corrected_full_uint8,
        get_icc_profile_bytes(args.output_colorspace),
    )
    result_json_path = save_analysis_result(
        args.output_dir,
        raw_path,
        {
            "analysis_raw_path": str(raw_path),
            "settings": {
                "white_index": int(args.white_index),
                "chromatic_indices": np.asarray(args.chromatic_indices, dtype=int).tolist(),
                "hppcc_region_candidates": np.asarray(args.hppcc_region_candidates, dtype=int).tolist(),
                "reference_illuminant": args.reference_illuminant,
                "standard_whites": {
                    name: np.asarray(white_xyz, dtype=np.float64).tolist()
                    for name, white_xyz in args.standard_whites_json.items()
                },
                "use_metadata_rgb_xyz_baseline": bool(args.use_metadata_rgb_xyz_baseline),
                "use_hppcc_blending": bool(args.use_hppcc_blending),
                "hppcc_blend_width": float(args.hppcc_blend_width),
                "output_format": args.output_format,
                "output_colorspace": args.output_colorspace,
            },
            "selection": {
                "best_illuminant": best_illuminant,
                "selected_k_regions": selected_k_regions,
            },
            "models": {
                "baseline": linear_model_to_dict(best["baseline"]),
                "hppcc": hppcc_model_to_dict(best["hppcc"]),
            },
        },
    )

    print_overlay_summary(args.output_dir)
    print(f"Analysis result JSON written to: {result_json_path}")
    maybe_show_detection_preview(
        args.show_detection_preview,
        show_detection_preview,
        detection,
        corrected_rgb,
    )
    print_model_report(
        best,
        selected_k_regions,
        use_hppcc_blending=args.use_hppcc_blending,
        hppcc_blend_width=args.hppcc_blend_width,
    )
    hppcc_region_indices = np.searchsorted(
        best["hppcc"].boundaries,
        src.hue_angle_from_rgb(corrected_rgb),
        side="right",
    ) % len(best["hppcc"].boundaries)
    print_hppcc_region_report(
        CLASSIC_24_PATCH_NAMES,
        best["hppcc"].boundaries,
        hppcc_region_indices,
        np.asarray(args.chromatic_indices, dtype=int),
    )
    print_patch_delta_e_report(
        CLASSIC_24_PATCH_NAMES,
        best["baseline_de00"],
        best["hppcc_de00"],
    )
    show_corrected_image(corrected_full_uint8)


def run_process(args) -> None:
    payload = load_analysis_result(args.result_json)
    settings = payload["settings"]
    hppcc_payload = payload["models"]["hppcc"]

    raw_paths = find_raw_paths(args.folder_to_process)
    if not raw_paths:
        raise FileNotFoundError(f"No RAW file found in {args.folder_to_process}")

    output_dir = args.output_dir if args.output_dir is not None else args.folder_to_process / "corrected"
    output_dir.mkdir(exist_ok=True)

    worker_count = args.workers
    if worker_count is None:
        try:
            import os

            worker_count = min(len(raw_paths), max(1, os.cpu_count() or 1))
        except Exception:
            worker_count = 1
    if worker_count < 1:
        raise ValueError("--workers must be at least 1.")

    if worker_count == 1:
        for raw_path in raw_paths:
            output_path = _process_single_raw(raw_path, output_dir, settings, hppcc_payload)
            print(f"Processed: {raw_path} -> {output_path}")
        return

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_process_single_raw, raw_path, output_dir, settings, hppcc_payload): raw_path
            for raw_path in raw_paths
        }
        for future in as_completed(futures):
            raw_path = futures[future]
            output_path = future.result()
            print(f"Processed: {raw_path} -> {output_path}")


def _process_single_raw(
    raw_path: Path,
    output_dir: Path,
    settings: dict[str, object],
    hppcc_payload: dict[str, object],
) -> Path:
    hppcc_model = hppcc_model_from_dict(hppcc_payload)
    raw = src.load_raw_linear_rgb(raw_path)
    sensor_black_levels_rgb = reduce_cfa_values_to_rgb(raw.black_level_per_channel, raw.color_desc)
    sensor_white_levels_rgb = reduce_cfa_values_to_rgb(raw.camera_white_level_per_channel, raw.color_desc)
    if sensor_black_levels_rgb is None or sensor_white_levels_rgb is None:
        raise RuntimeError(f"Missing sensor black/white metadata required for radiometric normalization: {raw_path}")

    normalized_full_rgb = normalize_with_sensor_levels(
        raw.rgb,
        sensor_black_levels_rgb,
        sensor_white_levels_rgb,
    )
    corrected_full_xyz = predict_hppcc(
        hppcc_model,
        normalized_full_rgb,
        use_blending=bool(settings["use_hppcc_blending"]),
        blend_width=float(settings["hppcc_blend_width"]),
    )
    corrected_full_output_rgb = xyz_to_output_rgb(
        corrected_full_xyz,
        settings.get("output_colorspace", OUTPUT_COLORSPACE),
    )
    corrected_full_uint8 = to_uint8_image(corrected_full_output_rgb)
    output_path = output_dir / f"{raw_path.stem}{output_extension(settings.get('output_format', OUTPUT_FORMAT))}"
    save_named_corrected_image(
        output_path,
        corrected_full_uint8,
        get_icc_profile_bytes(settings.get("output_colorspace", OUTPUT_COLORSPACE)),
    )
    copy_exif_from_raw(raw_path, output_path)
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze":
        run_analysis(args)
        return
    if args.command == "process":
        run_process(args)
        return
    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
