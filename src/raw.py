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
        return RawLinearImage(
            rgb=rgb,
            black_level_per_channel=tuple(int(v) for v in raw.black_level_per_channel),
            camera_white_level_per_channel=tuple(int(v) for v in raw.camera_white_level_per_channel)
            if raw.camera_white_level_per_channel is not None
            else None,
            camera_whitebalance=tuple(float(v) for v in raw.camera_whitebalance)
            if raw.camera_whitebalance is not None
            else None,
            daylight_whitebalance=tuple(float(v) for v in raw.daylight_whitebalance)
            if raw.daylight_whitebalance is not None
            else None,
            white_level=None if raw.white_level is None else int(raw.white_level),
            rgb_xyz_matrix=None if raw.rgb_xyz_matrix is None else np.asarray(raw.rgb_xyz_matrix, dtype=np.float64),
            color_desc=color_desc,
            raw_pattern=raw_pattern,
        )
