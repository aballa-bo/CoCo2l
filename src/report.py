import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .utils import estimate_cct_from_xyz, xyz_to_xy


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
    *,
    hppcc_de00: np.ndarray | None = None,
    hppcc_rpcc_de00: np.ndarray | None = None,
    hppcc_label: str = "hppcc",
) -> None:
    print()
    print("deltaE00 by patch")
    for patch_index, (patch_name, bl) in enumerate(
        zip(patch_names, baseline_de00, strict=False),
        start=1,
    ):
        line = f"{patch_index:02d} {patch_name:<15} baseline={float(bl):.4f}"
        if hppcc_de00 is not None:
            line += f" {hppcc_label}={float(hppcc_de00[patch_index - 1]):.4f}"
        if hppcc_rpcc_de00 is not None:
            line += f" {hppcc_label}+rpcc={float(hppcc_rpcc_de00[patch_index - 1]):.4f}"
        print(line)


def print_chroma_report(
    patch_names: list[str],
    baseline_chroma_error: np.ndarray,
    *,
    hppcc_chroma_error: np.ndarray | None = None,
    hppcc_rpcc_chroma_error: np.ndarray | None = None,
    hppcc_label: str = "hppcc",
) -> None:
    print()
    print("deltaC* by patch")
    print("summary")
    print(
        f"  baseline   mean|dC*|={float(np.mean(np.abs(baseline_chroma_error))):.4f} "
        f"bias={float(np.mean(baseline_chroma_error)):+.4f}"
    )
    if hppcc_chroma_error is not None:
        print(
            f"  {hppcc_label:<10} mean|dC*|={float(np.mean(np.abs(hppcc_chroma_error))):.4f} "
            f"bias={float(np.mean(hppcc_chroma_error)):+.4f}"
        )
    if hppcc_rpcc_chroma_error is not None:
        print(
            f"  {hppcc_label}+rpcc mean|dC*|={float(np.mean(np.abs(hppcc_rpcc_chroma_error))):.4f} "
            f"bias={float(np.mean(hppcc_rpcc_chroma_error)):+.4f}"
        )
    print("per patch")
    for patch_index, (patch_name, bl) in enumerate(
        zip(patch_names, baseline_chroma_error, strict=False),
        start=1,
    ):
        line = f"{patch_index:02d} {patch_name:<15} baseline={float(bl):+.4f}"
        if hppcc_chroma_error is not None:
            line += f" {hppcc_label}={float(hppcc_chroma_error[patch_index - 1]):+.4f}"
        if hppcc_rpcc_chroma_error is not None:
            line += f" {hppcc_label}+rpcc={float(hppcc_rpcc_chroma_error[patch_index - 1]):+.4f}"
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


def print_hppcc_gradient_contribution_report(
    patch_names: list[str],
    model,
    rgb: np.ndarray,
    chromatic_indices: np.ndarray,
) -> None:
    from .metrics import hue_angle_from_rgb

    rgb = np.asarray(rgb, dtype=np.float64)
    n_patches = rgb.shape[0]
    chromatic_index_set = {int(i) for i in np.asarray(chromatic_indices, dtype=int)}

    angles = hue_angle_from_rgb(rgb)
    basis = model._basis(angles)  # (N, n_basis)
    n_basis = model.coeffs.shape[0]
    n_harmonics = (n_basis - 1) // 2

    # Per-component XYZ contribution: component_xyz[i, k] = basis[i,k] * (rgb[i] @ coeffs[k])
    rgb_corr = np.einsum("ij,kjl->ikl", rgb, model.coeffs)  # (N, n_basis, 3)
    component_xyz = basis[:, :, np.newaxis] * rgb_corr       # (N, n_basis, 3)

    # Group components into DC + harmonic pairs H1, H2, ...
    n_groups = 1 + n_harmonics
    group_xyz = np.zeros((n_patches, n_groups, 3), dtype=np.float64)
    group_xyz[:, 0] = component_xyz[:, 0]
    for k in range(1, n_harmonics + 1):
        group_xyz[:, k] = component_xyz[:, 2 * k - 1] + component_xyz[:, 2 * k]

    group_mag = np.linalg.norm(group_xyz, axis=2)            # (N, n_groups)
    total_mag = group_mag.sum(axis=1, keepdims=True).clip(1e-12)
    group_pct = group_mag / total_mag * 100                   # (N, n_groups)

    group_labels = ["DC"] + [f"H{k}" for k in range(1, n_harmonics + 1)]
    header = " ".join(f"{lbl:>7}" for lbl in group_labels)

    print()
    print(f"HPPCC gradient harmonic contributions (DC + {n_harmonics} harmonic{'s' if n_harmonics != 1 else ''})")
    print(f"{'patch':<20} {'hue(deg)':>8}  {header}")
    for i, patch_name in enumerate(patch_names):
        if i not in chromatic_index_set:
            continue
        pct_str = " ".join(f"{group_pct[i, k]:>6.1f}%" for k in range(n_groups))
        print(f"{i + 1:02d} {patch_name:<17} {np.degrees(angles[i]):>8.1f}  {pct_str}")


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
    use_simple_linear: bool = False,
    chart_underexposed: bool = False,
    use_hppcc_gradient: bool = False,
) -> None:
    print()
    print("Baseline white-preserving 3x3")
    print("deltaE00 mean:", float(np.mean(best["baseline_de00"])))
    print("deltaE00 median:", float(np.median(best["baseline_de00"])))
    print("deltaE00 max:", float(np.max(best["baseline_de00"])))

    if use_simple_linear or chart_underexposed:
        return

    print()
    if use_hppcc_gradient:
        print("HPPCC gradient")
    else:
        print(f"HPPCC ({hppcc_regions} regions)")
        if use_hppcc_blending:
            print(f"prediction mode: blending (blend_width={hppcc_blend_width:.3f})")
        else:
            print("prediction mode: hard")
    print("deltaE00 mean:", float(np.mean(best["hppcc_de00"])))
    print("deltaE00 median:", float(np.median(best["hppcc_de00"])))
    print("deltaE00 max:", float(np.max(best["hppcc_de00"])))

    if perform_nonlinear_corrections and "hppcc_rpcc" in best:
        print()
        if use_hppcc_gradient:
            print("HPPCC gradient + RPCC residual")
        else:
            print(f"HPPCC + RPCC residual ({hppcc_regions} regions)")
            if use_hppcc_blending:
                print(f"prediction mode: blending (blend_width={hppcc_blend_width:.3f})")
            else:
                print("prediction mode: hard")
        print("deltaE00 mean:", float(np.mean(best["hppcc_rpcc_de00"])))
        print("deltaE00 median:", float(np.median(best["hppcc_rpcc_de00"])))
        print("deltaE00 max:", float(np.max(best["hppcc_rpcc_de00"])))


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _de00_quality(mean_de: float) -> str:
    if mean_de < 1.0:
        return "excellent (perceptually perfect)"
    if mean_de < 2.0:
        return "very good"
    if mean_de < 4.0:
        return "good"
    if mean_de < 7.0:
        return "acceptable"
    if mean_de < 10.0:
        return "marginal"
    return "poor"


def _dc_quality(mean_abs_dc: float, bias: float) -> str:
    direction = "over-saturated" if bias > 0.5 else ("under-saturated" if bias < -0.5 else "neutral")
    if mean_abs_dc < 1.0:
        return f"excellent ({direction})"
    if mean_abs_dc < 3.0:
        return f"good ({direction})"
    if mean_abs_dc < 6.0:
        return f"acceptable ({direction})"
    return f"poor ({direction})"


def _de00_block(label: str, de00: np.ndarray) -> list[str]:
    return [
        f"  {label}",
        f"    mean   = {float(np.mean(de00)):.4f}   {_de00_quality(float(np.mean(de00)))}",
        f"    median = {float(np.median(de00)):.4f}",
        f"    max    = {float(np.max(de00)):.4f}",
        f"    p95    = {float(np.percentile(de00, 95)):.4f}",
    ]


def _dc_block(label: str, dc: np.ndarray) -> list[str]:
    mean_abs = float(np.mean(np.abs(dc)))
    bias = float(np.mean(dc))
    return [
        f"  {label}",
        f"    mean|dC*| = {mean_abs:.4f}   {_dc_quality(mean_abs, bias)}",
        f"    bias      = {bias:+.4f}  ({'over' if bias > 0 else 'under'}-saturated)",
        f"    max|dC*|  = {float(np.max(np.abs(dc))):.4f}",
    ]


def save_analysis_text_report(
    output_path: Path,
    *,
    app_version: str,
    raw_path: Path,
    analysis_image_path: Path,
    result_json_path: Path,
    patch_names: list[str],
    reference_illuminant: str,
    scene_white_source: str,
    scene_white_xyz: np.ndarray,
    native_reference_white: np.ndarray,
    white_patch_level: float,
    chart_underexposed: bool,
    training_max_rgb: float,
    use_simple_linear: bool,
    use_hppcc_gradient: bool,
    perform_nonlinear_corrections: bool,
    selected_k_regions: int,
    output_label: str,
    neutral_gradient: dict | None,
    best: dict,
    hppcc_label: str,
    linear_only: bool,
    chromatic_indices: np.ndarray,
) -> None:
    L: list[str] = []
    sep = "=" * 72
    thin = "-" * 72

    def _s(*args) -> None:
        L.append(" ".join(str(a) for a in args))

    # ------------------------------------------------------------------ header
    _s(sep)
    _s(f"CoCo2l v{app_version} — Colour Correction Analysis Report")
    _s(f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _s(f"RAW file  : {raw_path}")
    _s(f"Output    : {analysis_image_path}")
    _s(f"JSON      : {result_json_path}")
    _s(sep)

    # --------------------------------------------------------------- settings
    _s("")
    _s("SETTINGS")
    _s(thin)
    _s(f"Reference illuminant : {reference_illuminant}")
    _s(f"Scene white source   : {scene_white_source}")
    cct = estimate_cct_from_xyz(np.asarray(scene_white_xyz))
    xy = xyz_to_xy(np.asarray(scene_white_xyz))
    _s(f"Scene white XYZ      : [{scene_white_xyz[0]:.5f}, {scene_white_xyz[1]:.5f}, {scene_white_xyz[2]:.5f}]")
    _s(f"Scene white xy       : ({xy[0]:.5f}, {xy[1]:.5f})   CCT ~ {cct:.0f} K" if cct else "")
    _s(f"White patch level    : {white_patch_level * 100:.1f}% of full scale")
    if chart_underexposed:
        _s(f"  WARNING: chart under-exposed (< {25:.0f}% threshold) — linear-only mode")
    _s(f"Training max RGB     : {training_max_rgb:.5f}")
    _s(f"HPPCC gradient       : {'yes' if use_hppcc_gradient else 'no'}")
    if not use_hppcc_gradient:
        _s(f"HPPCC regions        : {selected_k_regions}")
    _s(f"Nonlinear (RPCC)     : {'yes' if perform_nonlinear_corrections else 'no'}")
    _s(f"Output method        : {output_label}")

    # --------------------------------------------------------- neutral gradient
    if neutral_gradient is not None:
        _s("")
        _s("NEUTRAL GRADIENT")
        _s(thin)
        sev = neutral_gradient.get("severity", "unknown")
        span = neutral_gradient.get("horizontal_xy_span", float("nan"))
        cct_span = neutral_gradient.get("horizontal_cct_span", float("nan"))
        _s(f"Severity             : {sev}")
        _s(f"Horizontal xy span   : {span:.5f}")
        _s(f"Horizontal CCT span  : {cct_span:.1f} K")
        wl = neutral_gradient.get("weighted_line", {})
        rg = wl.get("r_over_g", {})
        bg = wl.get("b_over_g", {})
        if rg and bg:
            _s(f"Fitted R/G slope     : {rg.get('slope', 0.0):+.6f} / chart width")
            _s(f"Fitted B/G slope     : {bg.get('slope', 0.0):+.6f} / chart width")

    # -------------------------------------------------- per-model summary table
    _s("")
    _s("MODEL SUMMARY — deltaE00")
    _s(thin)
    _s(f"  {'method':<22}  {'mean':>7}  {'median':>7}  {'max':>7}  {'p95':>7}  quality")
    _s(f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  -------")

    def _row(label: str, de: np.ndarray) -> None:
        m, med, mx, p95 = (float(np.mean(de)), float(np.median(de)),
                           float(np.max(de)), float(np.percentile(de, 95)))
        _s(f"  {label:<22}  {m:7.4f}  {med:7.4f}  {mx:7.4f}  {p95:7.4f}  {_de00_quality(m)}")

    _row("baseline", best["baseline_de00"])
    if not linear_only:
        _row(hppcc_label, best["hppcc_de00"])
        if perform_nonlinear_corrections and "hppcc_rpcc_de00" in best:
            _row(f"{hppcc_label}+rpcc", best["hppcc_rpcc_de00"])

    # -------------------------------------------------- per-model chroma table
    _s("")
    _s("MODEL SUMMARY — deltaC* (chroma error)")
    _s(thin)
    _s(f"  {'method':<22}  {'mean|dC*|':>9}  {'bias':>7}  {'max|dC*|':>9}  quality")
    _s(f"  {'-'*22}  {'-'*9}  {'-'*7}  {'-'*9}  -------")

    def _crow(label: str, dc: np.ndarray) -> None:
        ma = float(np.mean(np.abs(dc)))
        bias = float(np.mean(dc))
        mx = float(np.max(np.abs(dc)))
        _s(f"  {label:<22}  {ma:9.4f}  {bias:+7.4f}  {mx:9.4f}  {_dc_quality(ma, bias)}")

    _crow("baseline", best["baseline_chroma_error"])
    if not linear_only:
        if "hppcc_chroma_error" in best:
            _crow(hppcc_label, best["hppcc_chroma_error"])
        if perform_nonlinear_corrections and "hppcc_rpcc_chroma_error" in best:
            _crow(f"{hppcc_label}+rpcc", best["hppcc_rpcc_chroma_error"])

    # --------------------------------------------------------- per-patch dE00
    _s("")
    _s("deltaE00 PER PATCH")
    _s(thin)
    chromatic_set = set(int(i) for i in chromatic_indices)
    header = f"  {'#':>2}  {'patch':<17}  {'type':<8}  {'baseline':>8}"
    if not linear_only:
        header += f"  {hppcc_label:>14}"
        if perform_nonlinear_corrections and "hppcc_rpcc_de00" in best:
            header += f"  {hppcc_label+'+rpcc':>19}"
    _s(header)
    _s("  " + "-" * (len(header) - 2))
    for i, name in enumerate(patch_names):
        ptype = "chrom" if i in chromatic_set else "neutral"
        bl = float(best["baseline_de00"][i])
        row = f"  {i + 1:02d}  {name:<17}  {ptype:<8}  {bl:8.4f}"
        if not linear_only:
            row += f"  {float(best['hppcc_de00'][i]):14.4f}"
            if perform_nonlinear_corrections and "hppcc_rpcc_de00" in best:
                row += f"  {float(best['hppcc_rpcc_de00'][i]):19.4f}"
        _s(row)

    # --------------------------------------------------------- per-patch dC*
    _s("")
    _s("deltaC* PER PATCH")
    _s(thin)
    header_c = f"  {'#':>2}  {'patch':<17}  {'type':<8}  {'baseline':>9}"
    if not linear_only and "hppcc_chroma_error" in best:
        header_c += f"  {hppcc_label:>14}"
        if perform_nonlinear_corrections and "hppcc_rpcc_chroma_error" in best:
            header_c += f"  {hppcc_label+'+rpcc':>19}"
    _s(header_c)
    _s("  " + "-" * (len(header_c) - 2))
    for i, name in enumerate(patch_names):
        ptype = "chrom" if i in chromatic_set else "neutral"
        bl = float(best["baseline_chroma_error"][i])
        row = f"  {i + 1:02d}  {name:<17}  {ptype:<8}  {bl:+9.4f}"
        if not linear_only and "hppcc_chroma_error" in best:
            row += f"  {float(best['hppcc_chroma_error'][i]):+14.4f}"
            if perform_nonlinear_corrections and "hppcc_rpcc_chroma_error" in best:
                row += f"  {float(best['hppcc_rpcc_chroma_error'][i]):+19.4f}"
        _s(row)

    # ---------------------------------------------------------------- summary
    _s("")
    _s(sep)
    _s("FINAL EVALUATION")
    _s(sep)
    _s("")

    used_de = best["baseline_de00"]
    used_dc = best["baseline_chroma_error"]
    used_label = "baseline"
    if not linear_only:
        if perform_nonlinear_corrections and "hppcc_rpcc_de00" in best:
            used_de = best["hppcc_rpcc_de00"]
            used_dc = best.get("hppcc_rpcc_chroma_error", used_dc)
            used_label = f"{hppcc_label}+rpcc"
        else:
            used_de = best["hppcc_de00"]
            used_dc = best.get("hppcc_chroma_error", used_dc)
            used_label = hppcc_label

    _s(f"Output method applied to image: {used_label}")
    _s("")

    _s("deltaE00 (CIEDE2000 — perceptual colour distance):")
    for line in _de00_block(used_label, used_de):
        _s(line)
    if not linear_only:
        _s("  (reference — uncorrected baseline)")
        for line in _de00_block("baseline", best["baseline_de00"]):
            _s(line)
    _s("")

    _s("deltaC* (chroma / saturation error, signed):")
    for line in _dc_block(used_label, used_dc):
        _s(line)
    if not linear_only:
        _s("  (reference — uncorrected baseline)")
        for line in _dc_block("baseline", best["baseline_chroma_error"]):
            _s(line)
    _s("")

    # interpretation
    mean_de_used = float(np.mean(used_de))
    mean_dc_used = float(np.mean(np.abs(used_dc)))
    bias_dc_used = float(np.mean(used_dc))
    _s("Interpretation:")
    _s(f"  Colour accuracy (dE00): {_de00_quality(mean_de_used)}  (mean {mean_de_used:.2f})")
    _s(f"  Saturation accuracy  : {_dc_quality(mean_dc_used, bias_dc_used)}  (mean|dC*| {mean_dc_used:.2f}, bias {bias_dc_used:+.2f})")

    if chart_underexposed:
        _s("")
        _s("  NOTE: the ColorChecker was under-exposed during analysis. The HPPCC/RPCC")
        _s("  models were computed but not applied; only the linear baseline (or the")
        _s("  adapted metadata matrix if it scored lower) was used for the output image.")
        _s("  For best results, re-shoot the chart at 60-90% sensor full scale.")

    _s("")
    _s(sep)

    output_path.write_text("\n".join(L) + "\n", encoding="utf-8")
