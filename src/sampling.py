"""Patch sampling helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _crop_inset(box: Sequence[float], inset: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in box]
    if not (0.0 <= inset < 0.5):
        raise ValueError("inset must be in [0, 0.5).")
    width = x1 - x0
    height = y1 - y0
    dx = width * inset
    dy = height * inset
    left = int(round(x0 + dx))
    top = int(round(y0 + dy))
    right = int(round(x1 - dx))
    bottom = int(round(y1 - dy))
    if left >= right or top >= bottom:
        raise ValueError("The requested inset leaves an empty patch region.")
    return left, top, right, bottom


def sample_patch_means(
    image: np.ndarray,
    patch_boxes: Iterable[Sequence[float]],
    *,
    inset: float = 0.2,
    reject_clipped: bool = False,
    clip_value: float | None = None,
) -> np.ndarray:
    """Return mean RGB values for rectangular patch boxes.

    Boxes are in `(x0, y0, x1, y1)` image coordinates. Sampling uses a central inset by default to
    avoid patch borders and common edge/shadow contamination.
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3).")
    means = []
    for box in patch_boxes:
        left, top, right, bottom = _crop_inset(box, inset)
        patch = image[top:bottom, left:right]
        if patch.size == 0:
            raise ValueError("Patch crop is empty.")
        if reject_clipped:
            if clip_value is None:
                raise ValueError("clip_value must be provided when reject_clipped=True.")
            valid = np.all(patch < clip_value, axis=-1)
            if not np.any(valid):
                raise ValueError("All patch pixels are clipped.")
            patch = patch[valid]
        means.append(np.mean(patch, axis=(0, 1)))
    return np.asarray(means, dtype=np.float64)


def sample_patch_means_from_masks(
    image: np.ndarray,
    swatch_masks: Iterable[Sequence[float]],
    *,
    reject_clipped: bool = False,
    clip_value: float | None = None,
) -> np.ndarray:
    """Return mean RGB values from ``colour-checker-detection`` swatch masks.

    The upstream library defines masks in `(y0, y1, x0, x1)` order on the rectified
    `colour_checker` image returned by the detector.
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3).")
    means = []
    for mask in swatch_masks:
        top, bottom, left, right = [int(round(float(v))) for v in mask]
        if top >= bottom or left >= right:
            raise ValueError("Invalid swatch mask.")
        patch = image[top:bottom, left:right]
        if patch.size == 0:
            raise ValueError("Swatch mask crop is empty.")
        if reject_clipped:
            if clip_value is None:
                raise ValueError("clip_value must be provided when reject_clipped=True.")
            valid = np.all(patch < clip_value, axis=-1)
            if not np.any(valid):
                raise ValueError("All swatch mask pixels are clipped.")
            patch = patch[valid]
        means.append(np.mean(patch, axis=(0, 1)))
    return np.asarray(means, dtype=np.float64)
