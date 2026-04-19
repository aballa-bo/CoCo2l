from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .utils import estimate_cct_from_xyz


def print_detection_summary(raw_path: Path, detection) -> None:
    print("raw path:", raw_path)
    print("quadrilateral shape:", detection.quadrilateral.shape)
    print("colour_checker shape:", detection.colour_checker_shape)
    print("swatch_masks shape:", detection.swatch_masks_shape)
    print("swatch_colours shape:", detection.measured_rgb.shape)
    print(
        "orientation:",
        f"rotation_steps={detection.orientation_steps}",
        f"mirrored={detection.orientation_mirrored}",
        f"white_patch={detection.detected_white_index + 1}",
        f"black_patch={detection.detected_black_index + 1}",
    )


def print_patch_correction_report(
    sensor_black_levels_rgb: np.ndarray,
    sensor_white_levels_rgb: np.ndarray,
    measured_rgb: np.ndarray,
    normalized_rgb: np.ndarray,
) -> None:
    print()
    print("Pre-fit patch normalization")
    print(
        "Sensor black RGB:",
        np.array2string(np.asarray(sensor_black_levels_rgb, dtype=np.float64), precision=6),
    )
    print(
        "Sensor white RGB:",
        np.array2string(np.asarray(sensor_white_levels_rgb, dtype=np.float64), precision=6),
    )
    for patch_index in (18, 21, 23):
        print(
            f"Patch {patch_index + 1} raw:",
            np.array2string(np.asarray(measured_rgb[patch_index], dtype=np.float64), precision=6),
        )
        print(
            f"Patch {patch_index + 1} norm:",
            np.array2string(np.asarray(normalized_rgb[patch_index], dtype=np.float64), precision=6),
        )


def save_json_grid_preview(output_dir: Path, make_json_grid_preview, measured_rgb: np.ndarray) -> Path:
    output_dir.mkdir(exist_ok=True)
    preview = make_json_grid_preview(measured_rgb)
    output_path = output_dir / "checker_json_order.png"
    cv2.imwrite(str(output_path), preview)
    return output_path


def save_detection_overlay_preview(output_dir: Path, make_scene_preview, detection) -> Path:
    output_dir.mkdir(exist_ok=True)
    preview = make_scene_preview(
        detection.scene_image,
        detection.quadrilateral,
        detection.absolute_patch_centers,
    )
    output_path = output_dir / "checker_detection_overlay.png"
    cv2.imwrite(str(output_path), preview)
    return output_path


def save_corrected_image(output_dir: Path, corrected_rgb_uint8: np.ndarray, extension: str) -> Path:
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"corrected_hppcc{extension}"
    raise RuntimeError("save_corrected_image requires ICC profile bytes. Use save_named_corrected_image instead.")


def save_named_corrected_image(output_path: Path, corrected_rgb_uint8: np.ndarray, icc_profile_bytes: bytes) -> Path:
    output_path.parent.mkdir(exist_ok=True)
    image = Image.fromarray(corrected_rgb_uint8, mode="RGB")
    suffix = output_path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        image.save(output_path, format="JPEG", quality=95, subsampling=0, icc_profile=icc_profile_bytes)
    elif suffix == ".png":
        image.save(output_path, format="PNG", icc_profile=icc_profile_bytes)
    elif suffix in (".tif", ".tiff"):
        image.save(output_path, format="TIFF", compression="tiff_lzw", icc_profile=icc_profile_bytes)
    else:
        raise ValueError(f"Unsupported output image extension: {suffix}")
    return output_path


def maybe_show_detection_preview(
    show_detection_preview: bool,
    show_detection_preview_fn,
    detection,
    measured_rgb: np.ndarray,
) -> None:
    if not show_detection_preview:
        return
    show_detection_preview_fn(
        detection.scene_image,
        detection.quadrilateral,
        detection.absolute_patch_centers,
        detection.patch_images,
        measured_rgb,
    )


def show_corrected_image(corrected_rgb_uint8: np.ndarray, scale: float = 0.4) -> None:
    preview = cv2.cvtColor(corrected_rgb_uint8, cv2.COLOR_RGB2BGR)
    preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow("HPPCC Corrected Image", preview)
    print("Immagine corretta aperta. Premi un tasto nella finestra per chiudere.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_overlay_summary(output_dir: Path) -> None:
    print()
    print(f"Overlay images written to: {output_dir}")


def print_scene_white_report(
    reference_white: np.ndarray,
    native_reference_white: np.ndarray,
    reference_space: str,
    reference_illuminant: str,
    scene_white: np.ndarray,
    scene_white_source: str,
    camera_wb_white: np.ndarray | None = None,
    neutral_patch_white: np.ndarray | None = None,
    auto_scores: dict[str, dict[str, float]] | None = None,
) -> None:
    print()
    print("Scene illuminant estimation")
    print(
        f"  reference dataset white ({reference_space}) XYZ: "
        f"[{native_reference_white[0]:.5f}, {native_reference_white[1]:.5f}, {native_reference_white[2]:.5f}]"
    )
    print(
        f"  illuminant anchor ({reference_illuminant}) XYZ:   "
        f"[{reference_white[0]:.5f}, {reference_white[1]:.5f}, {reference_white[2]:.5f}]"
    )
    if camera_wb_white is not None:
        cct = estimate_cct_from_xyz(camera_wb_white)
        cct_str = f"  ~{cct:.0f} K" if cct is not None else ""
        print(
            f"  camera WB estimate:  [{camera_wb_white[0]:.5f}, {camera_wb_white[1]:.5f}, {camera_wb_white[2]:.5f}]{cct_str}"
        )
    if neutral_patch_white is not None:
        cct = estimate_cct_from_xyz(neutral_patch_white)
        cct_str = f"  ~{cct:.0f} K" if cct is not None else ""
        print(
            f"  neutral patches:     [{neutral_patch_white[0]:.5f}, {neutral_patch_white[1]:.5f}, {neutral_patch_white[2]:.5f}]{cct_str}"
        )
    if auto_scores:
        print("  auto candidate scores (baseline fit)")
        for source_name, score in auto_scores.items():
            cct = score.get("cct")
            cct_str = f", ~{cct:.0f} K" if cct is not None else ""
            print(
                f"    {source_name:<15} neutral mean dE00={score['neutral_mean_de00']:.4f} "
                f"overall mean dE00={score['overall_mean_de00']:.4f}{cct_str}"
            )
    cct = estimate_cct_from_xyz(scene_white)
    cct_str = f"  ~{cct:.0f} K" if cct is not None else ""
    print(
        f"  used ({scene_white_source}):   [{scene_white[0]:.5f}, {scene_white[1]:.5f}, {scene_white[2]:.5f}]{cct_str}"
    )


def print_neutral_gradient_report(
    patch_names: list[str],
    gradient_report: dict[str, object] | None,
) -> None:
    if gradient_report is None:
        return

    print()
    print("Neutral-patch illuminant gradient")
    print("  geometry: Classic 24 neutrals support a horizontal-only test")
    print(f"  severity: {gradient_report['severity']}")
    print(f"  left-right xy span:  {gradient_report['horizontal_xy_span']:.5f}")
    cct_span = gradient_report["horizontal_cct_span"]
    if isinstance(cct_span, float) and np.isfinite(cct_span):
        print(f"  left-right CCT span: {cct_span:.0f} K")
    print(f"  r/g span:            {gradient_report['r_over_g_span']:.5f}")
    print(f"  b/g span:            {gradient_report['b_over_g_span']:.5f}")
    print(
        "  weighted slopes:    "
        f"r/g={gradient_report['weighted_rg_slope_per_chart_width']:.5f} "
        f"b/g={gradient_report['weighted_bg_slope_per_chart_width']:.5f} "
        f"x={gradient_report['weighted_x_slope_per_chart_width']:.5f} "
        f"y={gradient_report['weighted_y_slope_per_chart_width']:.5f}"
    )
    print("  local estimates by neutral patch")
    for patch in gradient_report["patches"]:
        patch_name = patch_names[int(patch["patch_index"])]
        cct = patch["local_white_cct"]
        cct_str = f" ~{cct:.0f} K" if cct is not None else ""
        print(
            f"    {int(patch['patch_index']) + 1:02d} {patch_name:<15} "
            f"x={patch['center_xy'][0]:.1f} "
            f"r/g={patch['r_over_g']:.5f} "
            f"b/g={patch['b_over_g']:.5f} "
            f"xy=({patch['local_white_xy'][0]:.5f}, {patch['local_white_xy'][1]:.5f}){cct_str}"
        )


def print_noise_profile_report(
    patch_names: list[str],
    noise_profile: dict[str, object] | None,
    *,
    enabled: bool,
    method: str,
    strength: float,
    diameter: int,
    sigma_space: float,
) -> None:
    print()
    print("Patch-variance denoise")
    print(f"  enabled: {enabled}")
    if not enabled or noise_profile is None:
        return
    sigma_rgb = np.asarray(noise_profile["sigma_rgb"], dtype=np.float64)
    variance_rgb = np.asarray(noise_profile["variance_rgb"], dtype=np.float64)
    print(f"  profile patches: {int(noise_profile['patch_count'])}")
    print(f"  sigma RGB:       [{sigma_rgb[0]:.6f}, {sigma_rgb[1]:.6f}, {sigma_rgb[2]:.6f}]")
    print(f"  variance RGB:    [{variance_rgb[0]:.8f}, {variance_rgb[1]:.8f}, {variance_rgb[2]:.8f}]")
    if method == "bilateral":
        print(f"  method:          bilateral strength={strength:.3f} diameter={diameter} sigma_space={sigma_space:.3f}")
    else:
        print(f"  method:          {method} strength={strength:.3f}")
    for patch in noise_profile["patches"]:
        patch_name = patch_names[int(patch["patch_index"])]
        mean_rgb = np.asarray(patch["mean_rgb"], dtype=np.float64)
        variance_patch = np.asarray(patch["variance_rgb"], dtype=np.float64)
        print(
            f"    {int(patch['patch_index']) + 1:02d} {patch_name:<15} "
            f"mean=({mean_rgb[0]:.5f}, {mean_rgb[1]:.5f}, {mean_rgb[2]:.5f}) "
            f"var=({variance_patch[0]:.8f}, {variance_patch[1]:.8f}, {variance_patch[2]:.8f})"
        )


def print_metadata_matrix_report(
    metadata_matrix_name: str,
    de00: np.ndarray,
) -> None:
    print()
    print(f"{metadata_matrix_name} baseline")
    print("deltaE00 mean:", float(np.mean(de00)))
    print("deltaE00 median:", float(np.median(de00)))
    print("deltaE00 max:", float(np.max(de00)))


def print_patch_delta_e_report(
    patch_names: list[str],
    baseline_de00: np.ndarray,
    rpcc_de00: np.ndarray,
    hppcc_de00: np.ndarray,
    hppcc_rpcc_de00: np.ndarray | None,
) -> None:
    print()
    print("deltaE00 by patch")
    for patch_index, (patch_name, bl, rp, hp) in enumerate(
        zip(patch_names, baseline_de00, rpcc_de00, hppcc_de00, strict=False),
        start=1,
    ):
        line = (
            f"{patch_index:02d} {patch_name:<15} "
            f"baseline={float(bl):.4f} "
            f"rpcc={float(rp):.4f} "
            f"hppcc={float(hp):.4f}"
        )
        if hppcc_rpcc_de00 is not None:
            hr = hppcc_rpcc_de00[patch_index - 1]
            line += f" hppcc+rpcc={float(hr):.4f}"
        print(line)


def print_chroma_report(
    patch_names: list[str],
    baseline_chroma_error: np.ndarray,
    rpcc_chroma_error: np.ndarray,
    hppcc_chroma_error: np.ndarray,
    hppcc_rpcc_chroma_error: np.ndarray | None,
) -> None:
    print()
    print("deltaC* by patch")
    print("summary")
    print(
        f"  baseline mean|dC*|={float(np.mean(np.abs(baseline_chroma_error))):.4f} "
        f"bias={float(np.mean(baseline_chroma_error)):+.4f}"
    )
    print(
        f"  rpcc     mean|dC*|={float(np.mean(np.abs(rpcc_chroma_error))):.4f} "
        f"bias={float(np.mean(rpcc_chroma_error)):+.4f}"
    )
    print(
        f"  hppcc    mean|dC*|={float(np.mean(np.abs(hppcc_chroma_error))):.4f} "
        f"bias={float(np.mean(hppcc_chroma_error)):+.4f}"
    )
    if hppcc_rpcc_chroma_error is not None:
        print(
            f"  hppcc+rpcc mean|dC*|={float(np.mean(np.abs(hppcc_rpcc_chroma_error))):.4f} "
            f"bias={float(np.mean(hppcc_rpcc_chroma_error)):+.4f}"
        )
    print("per patch")
    for patch_index, (patch_name, bl, rp, hp) in enumerate(
        zip(patch_names, baseline_chroma_error, rpcc_chroma_error, hppcc_chroma_error, strict=False),
        start=1,
    ):
        line = (
            f"{patch_index:02d} {patch_name:<15} "
            f"baseline={float(bl):+.4f} "
            f"rpcc={float(rp):+.4f} "
            f"hppcc={float(hp):+.4f}"
        )
        if hppcc_rpcc_chroma_error is not None:
            hr = hppcc_rpcc_chroma_error[patch_index - 1]
            line += f" hppcc+rpcc={float(hr):+.4f}"
        print(line)


def print_hppcc_region_report(
    patch_names: list[str],
    boundaries: np.ndarray,
    region_indices: np.ndarray,
    chromatic_indices: np.ndarray,
) -> None:
    boundaries = np.asarray(boundaries, dtype=np.float64)
    region_indices = np.asarray(region_indices, dtype=int)
    chromatic_index_set = {int(index) for index in np.asarray(chromatic_indices, dtype=int)}

    print()
    print("HPPCC boundaries (radians)")
    print(np.array2string(boundaries, precision=6))
    print()
    print("HPPCC region intervals")
    wrapped_boundaries = np.concatenate([boundaries, [2.0 * np.pi]])
    for region in range(len(boundaries)):
        start = wrapped_boundaries[region]
        end = wrapped_boundaries[region + 1]
        width = end - start
        patch_labels = [
            f"{patch_index + 1:02d} {patch_names[patch_index]}"
            for patch_index in range(len(patch_names))
            if patch_index in chromatic_index_set and region_indices[patch_index] == region
        ]
        patch_text = ", ".join(patch_labels) if patch_labels else "-"
        print(
            f"region={region} "
            f"start={start:.6f} rad ({np.degrees(start):.2f} deg) "
            f"end={end:.6f} rad ({np.degrees(end):.2f} deg) "
            f"width={width:.6f} rad ({np.degrees(width):.2f} deg)"
        )
        print(f"patches: {patch_text}")
    print()
    print("HPPCC patch assignment")
    for patch_index, (patch_name, region_index) in enumerate(
        zip(patch_names, region_indices, strict=False),
        start=1,
    ):
        print(f"{patch_index:02d} {patch_name:<15} region={int(region_index)}")


def print_hppcc_candidate_report(
    candidate_results: list[dict[str, object]],
    *,
    perform_nonlinear_corrections: bool,
    use_hppcc_blending: bool,
    hppcc_blend_width: float,
) -> int:
    print()
    print("Multi-k model comparison (scene-measured white, optimised boundaries)")
    if use_hppcc_blending:
        print(f"Prediction mode: blending (blend_width={hppcc_blend_width:.3f})")
    else:
        print("Prediction mode: hard")
    best_index = 0
    best_value = float("inf")
    for index, result in enumerate(candidate_results):
        best = result["best"]
        baseline_mean = float(np.mean(best["baseline_de00"]))
        rpcc_mean = float(np.mean(best["rpcc_de00"]))
        hppcc_mean = float(np.mean(best["hppcc_de00"]))
        primary_mean = hppcc_mean
        line = (
            f"k={int(result['k'])}  "
            f"baseline={baseline_mean:.4f}  "
            f"rpcc={rpcc_mean:.4f}  "
            f"hppcc={hppcc_mean:.4f}"
        )
        if perform_nonlinear_corrections and "hppcc_rpcc_de00" in best:
            hppcc_rpcc_mean = float(np.mean(best["hppcc_rpcc_de00"]))
            line += f"  hppcc+rpcc={hppcc_rpcc_mean:.4f}"
            primary_mean = hppcc_rpcc_mean
        print(line)
        if primary_mean < best_value:
            best_value = primary_mean
            best_index = index
    print()
    print(f"Selected k: {int(candidate_results[best_index]['k'])}")
    return best_index


def print_model_report(
    best: dict[str, object],
    hppcc_regions: int,
    *,
    perform_nonlinear_corrections: bool,
    use_hppcc_blending: bool,
    hppcc_blend_width: float,
) -> None:
    print()
    print("Baseline white-preserving 3x3")
    print("matrix shape:", best["baseline"].matrix.shape)
    print("deltaE00 mean:", float(np.mean(best["baseline_de00"])))
    print("deltaE00 median:", float(np.median(best["baseline_de00"])))
    print("deltaE00 max:", float(np.max(best["baseline_de00"])))

    print()
    print("Global RPCC")
    print("matrix shape:", best["rpcc"].matrix.shape)
    print("deltaE00 mean:", float(np.mean(best["rpcc_de00"])))
    print("deltaE00 median:", float(np.median(best["rpcc_de00"])))
    print("deltaE00 max:", float(np.max(best["rpcc_de00"])))

    print()
    print(f"HPPCC ({hppcc_regions} regions)")
    if use_hppcc_blending:
        print(f"prediction mode: blending (blend_width={hppcc_blend_width:.3f})")
    else:
        print("prediction mode: hard")
    print("matrices shape:", best["hppcc"].matrices.shape)
    print("deltaE00 mean:", float(np.mean(best["hppcc_de00"])))
    print("deltaE00 median:", float(np.median(best["hppcc_de00"])))
    print("deltaE00 max:", float(np.max(best["hppcc_de00"])))

    if perform_nonlinear_corrections and "hppcc_rpcc" in best:
        print()
        print(f"HPPCC + RPCC residual ({hppcc_regions} regions)")
        if use_hppcc_blending:
            print(f"prediction mode: blending (blend_width={hppcc_blend_width:.3f})")
        else:
            print("prediction mode: hard")
        print("matrices shape:", best["hppcc_rpcc"].matrices.shape)
        print("deltaE00 mean:", float(np.mean(best["hppcc_rpcc_de00"])))
        print("deltaE00 median:", float(np.median(best["hppcc_rpcc_de00"])))
        print("deltaE00 max:", float(np.max(best["hppcc_rpcc_de00"])))
