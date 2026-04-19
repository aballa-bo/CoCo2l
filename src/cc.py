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
    make_scene_preview,
    make_json_grid_preview,
    retrieve_rgb_from_image,
    show_detection_preview,
)
from src.config import (
    ANALYSIS_DIR,
    CHROMATIC_INDICES,
    DENOISE_METHOD,
    DENOISE_DIAMETER,
    DENOISE_SIGMA_SPACE,
    DENOISE_STRENGTH,
    ENABLE_PATCH_VARIANCE_DENOISE,
    HPPCC_BLEND_WIDTH,
    HPPCC_REGION_CANDIDATES,
    IMAGE_DIR,
    OUTPUT_COLORSPACE,
    OUTPUT_FORMAT,
    PERFORM_NONLINEAR_CORRECTIONS,
    PROCESS_DIR,
    REFERENCE_ILLUMINANT,
    REFERENCE_PATH,
    REFERENCE_SPACE,
    SHOW_DETECTION_PREVIEW,
    SHOW_DEVELOPED_IMAGE_PREVIEW,
    STANDARD_WHITES,
    USE_HPPCC_BLENDING,
    USE_METADATA_RGB_XYZ_BASELINE,
    WHITE_INDEX,
)
from src.report import (
    maybe_show_detection_preview,
    print_detection_summary,
    print_hppcc_candidate_report,
    print_chroma_report,
    print_hppcc_region_report,
    print_metadata_matrix_report,
    print_neutral_gradient_report,
    print_noise_profile_report,
    print_model_report,
    print_overlay_summary,
    print_patch_correction_report,
    print_patch_delta_e_report,
    print_scene_white_report,
    save_detection_overlay_preview,
    save_named_corrected_image,
    save_json_grid_preview,
    show_corrected_image,
)
from src.utils import (
    apply_matrix_transform,
    analyze_neutral_illuminant_gradient,
    bradford_adapt_xyz,
    copy_exif_from_raw,
    denoise_linear_rgb,
    delta_e00_summary,
    estimate_scene_white_from_camera_wb,
    estimate_scene_white_from_neutral_patches,
    estimate_noise_profile_from_patches,
    find_raw_path,
    find_raw_paths,
    hppcc_model_from_dict,
    hppcc_model_to_dict,
    hppcc_rpcc_model_from_dict,
    hppcc_rpcc_model_to_dict,
    identify_unreliable_patches,
    linear_model_to_dict,
    load_reference_chroma,
    load_reference_white_xyz,
    load_reference_xyz,
    load_analysis_result,
    normalize_with_sensor_levels,
    output_extension,
    predict_hppcc,
    predict_hppcc_rpcc,
    reduce_cfa_matrix_to_rgb,
    reduce_cfa_values_to_rgb,
    render_xyz_to_display,
    rpcc_model_from_dict,
    rpcc_model_to_dict,
    save_analysis_result,
    select_scene_white_source,
    summarize_model,
    to_uint8_image,
    get_icc_profile_bytes,
    xyz_to_output_rgb,
)


PROCESS_SETTING_NAMES = (
    "white_index",
    "chromatic_indices",
    "hppcc_region_candidates",
    "reference_illuminant",
    "reference_space",
    "standard_whites_json",
    "scene_white_source",
    "patch_variance_denoise",
    "denoise_method",
    "denoise_strength",
    "denoise_diameter",
    "denoise_sigma_space",
    "use_metadata_rgb_xyz_baseline",
    "use_hppcc_blending",
    "hppcc_blend_width",
    "perform_nonlinear_corrections",
    "show_detection_preview",
    "show_developed_image_preview",
    "output_format",
    "output_colorspace",
)


def _merge_process_settings(saved_settings: dict[str, object], args) -> dict[str, object]:
    merged = dict(saved_settings)
    for setting_name in PROCESS_SETTING_NAMES:
        override_value = getattr(args, setting_name, None)
        if override_value is None:
            continue
        if isinstance(override_value, np.ndarray):
            merged[setting_name] = override_value.tolist()
        else:
            merged[setting_name] = override_value
    return merged


def run_analysis(args) -> None:
    raw_path = args.cc_image if args.cc_image is not None else find_raw_path(args.image_dir)
    reference_illuminant_white = args.standard_whites_json[args.reference_illuminant]
    native_reference_white = load_reference_white_xyz(args.reference_path, args.reference_space)
    if args.reference_space == "lab":
        reference_xyz = src.lab_to_xyz(src.load_reference_lab(args.reference_path), native_reference_white)
    elif args.reference_space == "xyy":
        reference_xyz = src.xyy_to_xyz(src.load_reference_xyy(args.reference_path))
    else:
        reference_xyz = load_reference_xyz(args.reference_path)
    reference_chroma = load_reference_chroma(args.reference_path)

    raw = src.load_raw_linear_rgb(raw_path)
    detection = detect_and_orient_colorchecker(raw.rgb, white_index=args.white_index)
    save_detection_overlay_preview(args.analysis_dir, make_scene_preview, detection)
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

    reliable_patch_indices, excluded_patch_indices = identify_unreliable_patches(
        normalized_rgb, int(args.white_index)
    )
    if len(excluded_patch_indices) > 0:
        print(f"\nExcluded {len(excluded_patch_indices)} patch(es) from fitting (near noise floor):")
        for idx in excluded_patch_indices:
            print(f"  patch {idx + 1:02d}  norm={normalized_rgb[idx]}")

    chromatic_index_set = set(int(i) for i in args.chromatic_indices)
    neutral_indices = np.array(
        [i for i in range(normalized_rgb.shape[0]) if i not in chromatic_index_set], dtype=int
    )
    noise_profile = None
    scene_linear_rgb = normalize_with_sensor_levels(
        detection.scene_image,
        sensor_black_levels_rgb,
        sensor_white_levels_rgb,
    )
    if args.patch_variance_denoise:
        noise_profile = estimate_noise_profile_from_patches(
            scene_linear_rgb,
            detection.absolute_patch_centers,
            detection.measurement_patch_size,
            neutral_indices,
        )
        scene_linear_rgb = denoise_linear_rgb(
            scene_linear_rgb,
            noise_profile,
            method=args.denoise_method,
            strength=float(args.denoise_strength),
            diameter=int(args.denoise_diameter),
            sigma_space=float(args.denoise_sigma_space),
        )
        corrected_rgb = retrieve_rgb_from_image(
            scene_linear_rgb,
            detection.absolute_patch_centers,
            detection.measurement_patch_size,
        )
    else:
        corrected_rgb = normalized_rgb

    camera_wb_white = None
    neutral_patch_white = None
    scene_white_source = "reference"
    scene_white_selection_scores = {}
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
        scene_white_source, scene_white_xyz, scene_white_selection_scores = select_scene_white_source(
            src,
            normalized_rgb,
            reference_xyz,
            native_reference_white,
            white_index=int(args.white_index),
            chromatic_indices=np.asarray(args.chromatic_indices, dtype=int),
            neutral_indices=neutral_indices,
            requested_source=args.scene_white_source,
            camera_wb_white=camera_wb_white,
            neutral_patch_white=neutral_patch_white,
        )
    else:
        scene_white_xyz = reference_illuminant_white
    adapted_reference_xyz = bradford_adapt_xyz(reference_xyz, native_reference_white, scene_white_xyz)

    print_detection_summary(raw_path, detection)
    print_patch_correction_report(
        sensor_black_levels_rgb,
        sensor_white_levels_rgb,
        detection.measured_rgb,
        normalized_rgb,
    )
    print_scene_white_report(
        reference_illuminant_white,
        native_reference_white,
        args.reference_space,
        args.reference_illuminant,
        scene_white_xyz,
        scene_white_source,
        camera_wb_white,
        neutral_patch_white,
        scene_white_selection_scores,
    )
    print_noise_profile_report(
        CLASSIC_24_PATCH_NAMES,
        noise_profile,
        enabled=bool(args.patch_variance_denoise),
        method=args.denoise_method,
        strength=float(args.denoise_strength),
        diameter=int(args.denoise_diameter),
        sigma_space=float(args.denoise_sigma_space),
    )
    neutral_gradient_report = None
    if raw.daylight_whitebalance is not None:
        neutral_gradient_report = analyze_neutral_illuminant_gradient(
            normalized_rgb,
            neutral_indices,
            detection.absolute_patch_centers,
            raw.daylight_whitebalance,
            reference_illuminant_white,
        )
    print_neutral_gradient_report(CLASSIC_24_PATCH_NAMES, neutral_gradient_report)

    save_json_grid_preview(args.analysis_dir, make_json_grid_preview, corrected_rgb)

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
            reference_chroma=reference_chroma,
            reliable_patch_indices=reliable_patch_indices,
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
    if args.patch_variance_denoise and noise_profile is not None:
        normalized_full_rgb = denoise_linear_rgb(
            normalized_full_rgb,
            noise_profile,
            method=args.denoise_method,
            strength=float(args.denoise_strength),
            diameter=int(args.denoise_diameter),
            sigma_space=float(args.denoise_sigma_space),
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
    corrected_full_output_rgb = render_xyz_to_display(corrected_full_xyz, scene_white_xyz, args.output_colorspace)
    corrected_full_uint8 = to_uint8_image(corrected_full_output_rgb)
    ext = output_extension(args.output_format)
    icc_bytes = get_icc_profile_bytes(args.output_colorspace)
    args.process_dir.mkdir(parents=True, exist_ok=True)
    analysis_output_path = args.process_dir / f"{raw_path.stem}_analysis{ext}"
    save_named_corrected_image(analysis_output_path, corrected_full_uint8, icc_bytes)
    try:
        copy_exif_from_raw(raw_path, analysis_output_path)
    except RuntimeError as exc:
        print(f"Warning: {exc}")
    rpcc_full_xyz = best["rpcc"].predict(normalized_full_rgb)
    rpcc_full_output_rgb = render_xyz_to_display(rpcc_full_xyz, scene_white_xyz, args.output_colorspace)
    rpcc_full_uint8 = to_uint8_image(rpcc_full_output_rgb)
    rpcc_output_path = args.process_dir / f"{raw_path.stem}_analysis_rpcc{ext}"
    save_named_corrected_image(rpcc_output_path, rpcc_full_uint8, icc_bytes)
    try:
        copy_exif_from_raw(raw_path, rpcc_output_path)
    except RuntimeError as exc:
        print(f"Warning: {exc}")
    models_payload: dict[str, object] = {
        "baseline": linear_model_to_dict(best["baseline"]),
        "rpcc": rpcc_model_to_dict(best["rpcc"]),
        "hppcc": hppcc_model_to_dict(best["hppcc"]),
    }
    if args.perform_nonlinear_corrections:
        models_payload["hppcc_rpcc"] = hppcc_rpcc_model_to_dict(best["hppcc_rpcc"])
    result_json_path = save_analysis_result(
        args.analysis_dir,
        raw_path,
        {
            "analysis_raw_path": str(raw_path),
            "settings": {
                "white_index": int(args.white_index),
                "chromatic_indices": np.asarray(args.chromatic_indices, dtype=int).tolist(),
                "hppcc_region_candidates": np.asarray(args.hppcc_region_candidates, dtype=int).tolist(),
                "reference_illuminant": args.reference_illuminant,
                "reference_space": args.reference_space,
                "reference_native_white_xyz": np.asarray(native_reference_white, dtype=np.float64).tolist(),
                "scene_white_source": scene_white_source,
                "scene_white_xyz": np.asarray(scene_white_xyz, dtype=np.float64).tolist(),
                "patch_variance_denoise": bool(args.patch_variance_denoise),
                "denoise_method": args.denoise_method,
                "denoise_strength": float(args.denoise_strength),
                "denoise_diameter": int(args.denoise_diameter),
                "denoise_sigma_space": float(args.denoise_sigma_space),
                "use_metadata_rgb_xyz_baseline": bool(args.use_metadata_rgb_xyz_baseline),
                "use_hppcc_blending": bool(args.use_hppcc_blending),
                "hppcc_blend_width": float(args.hppcc_blend_width),
                "perform_nonlinear_corrections": bool(args.perform_nonlinear_corrections),
                "output_format": args.output_format,
                "output_colorspace": args.output_colorspace,
            },
            "selection": {"selected_k_regions": selected_k_regions},
            "diagnostics": {
                "neutral_gradient": neutral_gradient_report,
                "noise_profile": noise_profile,
            },
            "models": models_payload,
        },
    )

    print_overlay_summary(args.analysis_dir)
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
        best["rpcc_de00"],
        best["hppcc_de00"],
        best.get("hppcc_rpcc_de00"),
    )
    print_chroma_report(
        CLASSIC_24_PATCH_NAMES,
        best["baseline_chroma_error"],
        best["rpcc_chroma_error"],
        best["hppcc_chroma_error"],
        best.get("hppcc_rpcc_chroma_error"),
    )
    if args.show_developed_image_preview:
        show_corrected_image(corrected_full_uint8)


def run_process(args) -> None:
    payload = load_analysis_result(args.result_json)
    settings = _merge_process_settings(payload["settings"], args)
    diagnostics = payload.get("diagnostics", {})
    models_payload = payload["models"]
    use_nonlinear = bool(settings.get("perform_nonlinear_corrections", "hppcc_rpcc" in models_payload))
    if use_nonlinear and "hppcc_rpcc" not in models_payload:
        raise ValueError("The saved analysis result does not contain an HPPCC+RPCC model, so --perform-nonlinear-corrections cannot be enabled.")

    raw_paths = find_raw_paths(args.folder_to_process)
    if not raw_paths:
        raise FileNotFoundError(f"No RAW file found in {args.folder_to_process}")

    output_dir = args.process_dir if args.process_dir is not None else args.folder_to_process / "corrected"
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
            output_paths = _process_single_raw(raw_path, output_dir, settings, diagnostics, models_payload, use_nonlinear)
            outputs_text = ", ".join(str(path) for path in output_paths)
            print(f"Processed: {raw_path} -> {outputs_text}")
        return

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_process_single_raw, raw_path, output_dir, settings, diagnostics, models_payload, use_nonlinear): raw_path
            for raw_path in raw_paths
        }
        for future in as_completed(futures):
            raw_path = futures[future]
            output_paths = future.result()
            outputs_text = ", ".join(str(path) for path in output_paths)
            print(f"Processed: {raw_path} -> {outputs_text}")


def _process_single_raw(
    raw_path: Path,
    output_dir: Path,
    settings: dict[str, object],
    diagnostics: dict[str, object],
    models_payload: dict[str, object],
    use_nonlinear: bool,
) -> list[Path]:
    use_blending = bool(settings["use_hppcc_blending"])
    blend_width = float(settings["hppcc_blend_width"])
    output_colorspace = settings.get("output_colorspace", OUTPUT_COLORSPACE)
    scene_white_xyz = np.asarray(settings["scene_white_xyz"], dtype=np.float64)
    output_paths: list[Path] = []
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
    noise_profile = diagnostics.get("noise_profile")
    if bool(settings.get("patch_variance_denoise", False)) and noise_profile is not None:
        normalized_full_rgb = denoise_linear_rgb(
            normalized_full_rgb,
            noise_profile,
            method=str(settings.get("denoise_method", DENOISE_METHOD)),
            strength=float(settings.get("denoise_strength", DENOISE_STRENGTH)),
            diameter=int(settings.get("denoise_diameter", DENOISE_DIAMETER)),
            sigma_space=float(settings.get("denoise_sigma_space", DENOISE_SIGMA_SPACE)),
        )
    extension = output_extension(settings.get("output_format", OUTPUT_FORMAT))
    icc_profile_bytes = get_icc_profile_bytes(settings.get("output_colorspace", OUTPUT_COLORSPACE))

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
    corrected_full_output_rgb = render_xyz_to_display(corrected_full_xyz, scene_white_xyz, output_colorspace)
    corrected_full_uint8 = to_uint8_image(corrected_full_output_rgb)
    output_path = output_dir / f"{raw_path.stem}{extension}"
    save_named_corrected_image(output_path, corrected_full_uint8, icc_profile_bytes)
    try:
        copy_exif_from_raw(raw_path, output_path)
    except RuntimeError as exc:
        print(f"Warning: {exc}")
    output_paths.append(output_path)
    return output_paths


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
