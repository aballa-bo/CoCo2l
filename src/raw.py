"""Linear RAW decoding helpers built on top of rawpy/libraw."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rawpy


@dataclass(frozen=True)
class RawLinearImage:
    """Linear demosaiced camera RGB and relevant capture metadata."""

    rgb: np.ndarray
    black_level_per_channel: tuple[int, int, int, int]
    camera_white_level_per_channel: tuple[int, int, int, int] | None
    camera_whitebalance: tuple[float, float, float, float] | None
    daylight_whitebalance: tuple[float, float, float, float] | None
    white_level: int | None
    rgb_xyz_matrix: np.ndarray | None
    color_desc: str
    raw_pattern: np.ndarray | None


def _normalize_wb(user_wb: Iterable[float] | None) -> list[float] | None:
    if user_wb is None:
        return None
    values = [float(v) for v in user_wb]
    if len(values) != 4:
        raise ValueError("user_wb must contain exactly 4 multipliers for R, G1, B, G2.")
    return values


def _effective_rgb_xyz_matrix(raw: rawpy.RawPy) -> np.ndarray | None:
    """Return the best available Camera→XYZ matrix.

    rawpy exposes two sources:
    - rgb_xyz_matrix: Camera→XYZ (D65), pre-computed by libraw. Present for most DSLRs/mirrorless.
      For DNG files libraw leaves it as an all-zero matrix.
    - color_matrix: XYZ_D50→Camera (DNG ColorMatrix tag). Present in DNG files.
      Inverting its 3×3 sub-block gives Camera→XYZ_D50.

    When rgb_xyz_matrix is degenerate (all zeros or missing), the inverted color_matrix
    is returned as fallback. Note that it references D50 rather than D65, but for the
    metadata-baseline comparison this is an acceptable approximation.
    """
    if raw.rgb_xyz_matrix is not None:
        m = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)
        if np.any(m != 0.0):
            return m

    if hasattr(raw, "color_matrix") and raw.color_matrix is not None:
        cm = np.asarray(raw.color_matrix, dtype=np.float64)
        # color_matrix shape is (num_colors, 4) in rawpy; only the first 3 columns are XYZ
        cm3 = cm[:3, :3]
        if np.linalg.matrix_rank(cm3) == 3:
            try:
                # cm3 is XYZ→Camera; invert to get Camera→XYZ rows
                cam_to_xyz = np.linalg.inv(cm3).T  # shape (3, 3): each row is a camera channel
                # Expand to (3, 3) in the same convention as rgb_xyz_matrix (N_cfa_channels × 3)
                return cam_to_xyz
            except np.linalg.LinAlgError:
                pass

    return None


def load_raw_linear_rgb(
    path: str | Path,
    *,
    use_camera_wb: bool = False,
    user_wb: Iterable[float] | None = None,
    demosaic_algorithm: rawpy.DemosaicAlgorithm | None = None,
    half_size: bool = False,
) -> RawLinearImage:
    """Decode a RAW file to linear demosaiced camera RGB.

    Scientific defaults:
    - camera color space is preserved (`output_color=raw`)
    - no auto brightening
    - no gamma encoding
    - no automatic scaling
    - no denoising or median filtering

    If neither `use_camera_wb` nor `user_wb` is provided, an identity white balance is used.
    This keeps the decoded data closest to camera space; a separate white-balance step can then be
    applied explicitly in the calibration pipeline.
    """

    user_wb_values = _normalize_wb(user_wb)
    with rawpy.imread(str(path)) as raw:
        params = rawpy.Params(
            demosaic_algorithm=demosaic_algorithm,
            half_size=half_size,
            four_color_rgb=False,
            dcb_iterations=0,
            dcb_enhance=False,
            fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
            noise_thr=None,
            median_filter_passes=0,
            use_camera_wb=use_camera_wb,
            use_auto_wb=False,
            user_wb=user_wb_values if user_wb_values is not None else [1.0, 1.0, 1.0, 1.0],
            output_color=rawpy.ColorSpace.raw,
            output_bps=16,
            no_auto_bright=True,
            bright=1.0,
            no_auto_scale=True,
            gamma=(1.0, 1.0),
            chromatic_aberration=(1.0, 1.0),
        )
        rgb = raw.postprocess(params=params).astype(np.float64)
        color_desc = raw.color_desc.decode("ascii", errors="ignore")
        raw_pattern = None if raw.raw_pattern is None else np.array(raw.raw_pattern, copy=True)
        # LibRaw subtracts black_level internally during postprocess(), so the output
        # pixel range is [0, white - black].  Store adjusted levels so that downstream
        # normalize_with_sensor_levels(value, black=0, white=white-black) is correct.
        black_per_ch = tuple(int(v) for v in raw.black_level_per_channel)
        if raw.camera_white_level_per_channel is not None:
            raw_white = tuple(int(v) for v in raw.camera_white_level_per_channel)
        elif raw.white_level is not None:
            raw_white = tuple(int(raw.white_level) for _ in black_per_ch)
        else:
            raw_white = None
        white_level_per_channel = (
            tuple(w - b for w, b in zip(raw_white, black_per_ch))
            if raw_white is not None else None
        )
        return RawLinearImage(
            rgb=rgb,
            black_level_per_channel=tuple(0 for _ in black_per_ch),
            camera_white_level_per_channel=white_level_per_channel,
            camera_whitebalance=tuple(float(v) for v in raw.camera_whitebalance)
            if raw.camera_whitebalance is not None
            else None,
            daylight_whitebalance=tuple(float(v) for v in raw.daylight_whitebalance)
            if raw.daylight_whitebalance is not None
            else None,
            white_level=None if raw.white_level is None else int(raw.white_level),
            rgb_xyz_matrix=_effective_rgb_xyz_matrix(raw),
            color_desc=color_desc,
            raw_pattern=raw_pattern,
        )
