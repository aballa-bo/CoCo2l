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
    PERFORM_NONLINEAR_CORRECTIONS,
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
    print_metadata_matrix_report,
    print_model_report,
    print_overlay_summary,
    print_patch_correction_report,
    print_patch_delta_e_report,
    print_scene_white_report,
    save_named_corrected_image,
    save_json_grid_preview,
    show_corrected_image,
)
from src.utils import (
    apply_matrix_transform,
    bradford_adapt_xyz,
    copy_exif_from_raw,
    delta_e00_summary,
    estimate_scene_white_from_camera_wb,
    estimate_scene_white_from_neutral_patches,
    find_raw_path,
    find_raw_paths,
    hppcc_model_from_dict,
    hppcc_model_to_dict,
    hppcc_rpcc_model_from_dict,
    hppcc_rpcc_model_to_dict,
    linear_model_to_dict,
    load_reference_xyz,
    load_analysis_result,
    normalize_with_sensor_levels,
    output_extension,
    predict_hppcc,
    predict_hppcc_rpcc,
    reduce_cfa_matrix_to_rgb,
    reduce_cfa_values_to_rgb,
    rpcc_model_to_dict,
    save_analysis_result,
    summarize_model,
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

    reference_illuminant_white = args.standard_whites_json[args.reference_illuminant]
    chromatic_index_set = set(int(i) for i in args.chromatic_indices)
    neutral_indices = np.array(
        [i for i in range(normalized_rgb.shape[0]) if i not in chromatic_index_set], dtype=int
    )

    camera_wb_white = None
    neutral_patch_white = None
    if raw.camera_whitebalance is not None and raw.daylight_whitebalance is not None:
        camera_wb_white = estimate_scene_white_from_camera_wb(
            raw.camera_whitebalance,
            raw.daylight_whitebalance,
            reference_illuminant_white,
        )
        neutral_patch_white = estimate_scene_white_from_neutral_patches(
            normalized_rgb,
            neutral_indices,
            raw.daylight_whitebalance,
            reference_illuminant_white,
        )
        scene_white_xyz = neutral_patch_white
    else:
        scene_white_xyz = reference_illuminant_white
    adapted_reference_xyz = bradford_adapt_xyz(reference_xyz, reference_illuminant_white, scene_white_xyz)

    print_detection_summary(raw_path, detection)
    print_patch_correction_report(
        sensor_black_levels_rgb,
        sensor_white_levels_rgb,
        detection.measured_rgb,
        normalized_rgb,
    )
    print_scene_white_report(reference_illuminant_white, scene_white_xyz, camera_wb_white, neutral_patch_white)

    save_json_grid_preview(args.output_dir, make_json_grid_preview, corrected_rgb)

    hppcc_candidate_results = []
    for k_regions in np.asarray(args.hppcc_region_candidates, dtype=int):
        result = summarize_model(
            src,
            corrected_rgb,
            adapted_reference_xyz,
            scene_white_xyz,
            white_index=args.white_index,
            chromatic_indices=np.asarray(args.chromatic_indices, dtype=int),
            hppcc_regions=int(k_regions),
            optimize_boundaries=True,
            use_hppcc_blending=args.use_hppcc_blending,
            hppcc_blend_width=args.hppcc_blend_width,
            perform_nonlinear_corrections=args.perform_nonlinear_corrections,
        )
        hppcc_candidate_results.append({"k": int(k_regions), "best": result})

    best_candidate_index = print_hppcc_candidate_report(
        hppcc_candidate_results,
        perform_nonlinear_corrections=args.perform_nonlinear_corrections,
        use_hppcc_blending=args.use_hppcc_blending,
        hppcc_blend_width=args.hppcc_blend_width,
    )
    selected_candidate = hppcc_candidate_results[best_candidate_index]
    best = selected_candidate["best"]
    selected_k_regions = int(selected_candidate["k"])

    if args.use_metadata_rgb_xyz_baseline and metadata_rgb_xyz_matrix is not None:
        metadata_xyz = apply_matrix_transform(corrected_rgb, metadata_rgb_xyz_matrix)
        metadata_de00 = delta_e00_summary(src, metadata_xyz, adapted_reference_xyz, scene_white_xyz)
        print_metadata_matrix_report("Metadata rgb_xyz_matrix", metadata_de00)

    normalized_full_rgb = normalize_with_sensor_levels(
        raw.rgb,
        sensor_black_levels_rgb,
        sensor_white_levels_rgb,
    )
    if args.perform_nonlinear_corrections:
        corrected_full_xyz = predict_hppcc_rpcc(
            best["hppcc_rpcc"],
            normalized_full_rgb,
            use_blending=args.use_hppcc_blending,
            blend_width=args.hppcc_blend_width,
        )
        output_label = "hppcc_rpcc"
    else:
        corrected_full_xyz = predict_hppcc(
            best["hppcc"],
            normalized_full_rgb,
            use_blending=args.use_hppcc_blending,
            blend_width=args.hppcc_blend_width,
        )
        output_label = "hppcc"
    corrected_full_output_rgb = xyz_to_output_rgb(corrected_full_xyz, args.output_colorspace)
    corrected_full_uint8 = to_uint8_image(corrected_full_output_rgb)
    analysis_output_path = args.output_dir / f"corrected_{output_label}{output_extension(args.output_format)}"
    save_named_corrected_image(
        analysis_output_path,
        corrected_full_uint8,
        get_icc_profile_bytes(args.output_colorspace),
    )
    models_payload: dict[str, object] = {
        "baseline": linear_model_to_dict(best["baseline"]),
        "hppcc": hppcc_model_to_dict(best["hppcc"]),
    }
    if args.perform_nonlinear_corrections:
        models_payload["hppcc_rpcc"] = hppcc_rpcc_model_to_dict(best["hppcc_rpcc"])
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
                "scene_white_xyz": np.asarray(scene_white_xyz, dtype=np.float64).tolist(),
                "use_metadata_rgb_xyz_baseline": bool(args.use_metadata_rgb_xyz_baseline),
                "use_hppcc_blending": bool(args.use_hppcc_blending),
                "hppcc_blend_width": float(args.hppcc_blend_width),
                "perform_nonlinear_corrections": bool(args.perform_nonlinear_corrections),
                "output_format": args.output_format,
                "output_colorspace": args.output_colorspace,
            },
            "selection": {"selected_k_regions": selected_k_regions},
            "models": models_payload,
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
    primary_model = best["hppcc_rpcc"] if args.perform_nonlinear_corrections else best["hppcc"]
    print_model_report(
        best,
        selected_k_regions,
        perform_nonlinear_corrections=args.perform_nonlinear_corrections,
        use_hppcc_blending=args.use_hppcc_blending,
        hppcc_blend_width=args.hppcc_blend_width,
    )
    hppcc_region_indices = np.searchsorted(
        primary_model.boundaries,
        src.hue_angle_from_rgb(corrected_rgb),
        side="right",
    ) % len(primary_model.boundaries)
    print_hppcc_region_report(
        CLASSIC_24_PATCH_NAMES,
        primary_model.boundaries,
        hppcc_region_indices,
        np.asarray(args.chromatic_indices, dtype=int),
    )
    print_patch_delta_e_report(
        CLASSIC_24_PATCH_NAMES,
        best["baseline_de00"],
        best["hppcc_de00"],
        best.get("hppcc_rpcc_de00"),
    )
    show_corrected_image(corrected_full_uint8)


def run_process(args) -> None:
    payload = load_analysis_result(args.result_json)
    settings = payload["settings"]
    models_payload = payload["models"]
    use_nonlinear = bool(settings.get("perform_nonlinear_corrections", "hppcc_rpcc" in models_payload))

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
            output_path = _process_single_raw(raw_path, output_dir, settings, models_payload, use_nonlinear)
            print(f"Processed: {raw_path} -> {output_path}")
        return

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_process_single_raw, raw_path, output_dir, settings, models_payload, use_nonlinear): raw_path
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
    models_payload: dict[str, object],
    use_nonlinear: bool,
) -> Path:
    use_blending = bool(settings["use_hppcc_blending"])
    blend_width = float(settings["hppcc_blend_width"])
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
    if use_nonlinear:
        model = hppcc_rpcc_model_from_dict(models_payload["hppcc_rpcc"])
        corrected_full_xyz = predict_hppcc_rpcc(
            model, normalized_full_rgb, use_blending=use_blending, blend_width=blend_width
        )
    else:
        model = hppcc_model_from_dict(models_payload["hppcc"])
        corrected_full_xyz = predict_hppcc(
            model, normalized_full_rgb, use_blending=use_blending, blend_width=blend_width
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
