from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def print_detection_summary(raw_path: Path, detection) -> None:
    print("raw path:", raw_path)
    print("quadrilateral shape:", detection.quadrilateral.shape)
    print("colour_checker shape:", detection.colour_checker_shape)
    print("swatch_masks shape:", detection.swatch_masks_shape)
    print("swatch_colours shape:", detection.measured_rgb.shape)


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


def _cct_from_xyz(xyz: np.ndarray) -> float | None:
    xyz_sum = float(np.sum(xyz))
    if xyz_sum <= 0:
        return None
    x = float(xyz[0]) / xyz_sum
    y = float(xyz[1]) / xyz_sum
    n = (x - 0.3320) / (0.1858 - y)
    return 449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33


def print_scene_white_report(
    reference_white: np.ndarray,
    scene_white: np.ndarray,
    camera_wb_white: np.ndarray | None = None,
    neutral_patch_white: np.ndarray | None = None,
) -> None:
    print()
    print("Scene illuminant estimation")
    print(
        f"  reference white XYZ: [{reference_white[0]:.5f}, {reference_white[1]:.5f}, {reference_white[2]:.5f}]"
    )
    if camera_wb_white is not None:
        cct = _cct_from_xyz(camera_wb_white)
        cct_str = f"  ~{cct:.0f} K" if cct is not None else ""
        print(
            f"  camera WB estimate:  [{camera_wb_white[0]:.5f}, {camera_wb_white[1]:.5f}, {camera_wb_white[2]:.5f}]{cct_str}"
        )
    if neutral_patch_white is not None:
        cct = _cct_from_xyz(neutral_patch_white)
        cct_str = f"  ~{cct:.0f} K" if cct is not None else ""
        print(
            f"  neutral patches:     [{neutral_patch_white[0]:.5f}, {neutral_patch_white[1]:.5f}, {neutral_patch_white[2]:.5f}]{cct_str}"
        )
    cct = _cct_from_xyz(scene_white)
    cct_str = f"  ~{cct:.0f} K" if cct is not None else ""
    print(
        f"  used (scene white):  [{scene_white[0]:.5f}, {scene_white[1]:.5f}, {scene_white[2]:.5f}]{cct_str}"
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
    hppcc_de00: np.ndarray,
    hppcc_rpcc_de00: np.ndarray | None,
) -> None:
    print()
    print("deltaE00 by patch")
    for patch_index, (patch_name, bl, hp) in enumerate(
        zip(patch_names, baseline_de00, hppcc_de00, strict=False),
        start=1,
    ):
        line = (
            f"{patch_index:02d} {patch_name:<15} "
            f"baseline={float(bl):.4f} "
            f"hppcc={float(hp):.4f}"
        )
        if hppcc_rpcc_de00 is not None:
            hr = hppcc_rpcc_de00[patch_index - 1]
            line += f" hppcc+rpcc={float(hr):.4f}"
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
        hppcc_mean = float(np.mean(best["hppcc_de00"]))
        primary_mean = hppcc_mean
        line = (
            f"k={int(result['k'])}  "
            f"baseline={baseline_mean:.4f}  "
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
