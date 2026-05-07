"""Lens geometric undistortion via lensfunpy + exiftool EXIF extraction."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def _read_lens_exif(raw_path: Path) -> dict:
    """Return Make, Model, LensModel, FocalLength from EXIF via exiftool."""
    try:
        result = subprocess.run(
            ["exiftool", "-json", "-Make", "-Model", "-LensModel", "-FocalLength",
             str(raw_path)],
            capture_output=True, text=True, timeout=15, check=False,
        )
        data = json.loads(result.stdout)
        return data[0] if data else {}
    except Exception:
        return {}


def _parse_focal_length(value: object) -> float:
    """Parse '35.0 mm' → 35.0, or numeric → float."""
    try:
        return float(str(value).split()[0])
    except (ValueError, IndexError):
        return 0.0


def try_undistort(
    linear_rgb: np.ndarray,
    raw_path: Path,
) -> tuple[np.ndarray, dict | None]:
    """Attempt geometric undistortion using the Lensfun database.

    Returns:
        (undistorted_rgb, lens_info_dict)  if undistortion was applied
        (original_rgb,    None)            if skipped (no DB match / lensfunpy missing)
    """
    try:
        import lensfunpy  # noqa: PLC0415
    except ImportError:
        return linear_rgb, None

    exif = _read_lens_exif(raw_path)
    make = str(exif.get("Make", "")).strip()
    model = str(exif.get("Model", "")).strip()
    lens_model = str(exif.get("LensModel", "")).strip()
    focal_length = _parse_focal_length(exif.get("FocalLength", 0))

    if not make or not model:
        return linear_rgb, None

    db = lensfunpy.Database()

    cameras = db.find_cameras(make, model)
    if not cameras:
        return linear_rgb, None
    cam = cameras[0]

    lenses = db.find_lenses(cam, lens_model)
    if not lenses:
        return linear_rgb, None
    lens = lenses[0]

    h, w = linear_rgb.shape[:2]
    mod = lensfunpy.Modifier(lens, cam.crop_factor, w, h)
    fl = focal_length if focal_length > 0 else lens.min_focal
    mod.initialize(fl, 8.0, pixel_format=np.float64)

    undist_coords = mod.apply_geometry_distortion()
    if undist_coords is None:
        return linear_rgb, None

    map_x = undist_coords[:, :, 0].astype(np.float32)
    map_y = undist_coords[:, :, 1].astype(np.float32)

    undistorted = cv2.remap(
        linear_rgb.astype(np.float32), map_x, map_y, cv2.INTER_LANCZOS4
    ).astype(linear_rgb.dtype)

    lens_info = {
        "applied": True,
        "camera_make": make,
        "camera_model": model,
        "lens_model": str(lens.model),
        "focal_length_mm": fl,
        "crop_factor": float(cam.crop_factor),
    }
    return undistorted, lens_info
