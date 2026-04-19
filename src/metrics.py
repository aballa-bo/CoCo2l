"""Color science math implemented with NumPy only."""

from __future__ import annotations

import numpy as np


def rg_chromaticity(rgb: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    denom = np.sum(rgb, axis=-1, keepdims=True)
    denom = np.maximum(denom, eps)
    rg = rgb[..., :2] / denom
    return rg


def hue_angle_from_rgb(rgb: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    rg = rg_chromaticity(rgb, eps=eps)
    u = rg[..., 0] - (1.0 / 3.0)
    v = rg[..., 1] - (1.0 / 3.0)
    angle = np.arctan2(v, u)
    return np.mod(angle, 2.0 * np.pi)


def xyz_to_lab(xyz: np.ndarray, white_xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    white_xyz = np.asarray(white_xyz, dtype=np.float64)
    if white_xyz.shape != (3,) and white_xyz.shape != xyz.shape:
        raise ValueError("white_xyz must have shape (3,) or match xyz shape.")

    xyz_n = xyz / white_xyz
    delta = 6.0 / 29.0
    delta3 = delta**3
    linear_scale = 1.0 / (3.0 * delta * delta)
    linear_offset = 4.0 / 29.0

    f = np.where(xyz_n > delta3, np.cbrt(xyz_n), linear_scale * xyz_n + linear_offset)
    l = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([l, a, b], axis=-1)


def lab_to_xyz(lab: np.ndarray, white_xyz: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    white_xyz = np.asarray(white_xyz, dtype=np.float64)
    if white_xyz.shape != (3,) and white_xyz.shape != lab.shape:
        raise ValueError("white_xyz must have shape (3,) or match lab shape.")

    delta = 6.0 / 29.0
    linear_scale = 3.0 * delta * delta

    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    xyz_n = np.where(f > delta, f**3, linear_scale * (f - 4.0 / 29.0))
    return xyz_n * white_xyz


def xyy_to_xyz(xyy: np.ndarray) -> np.ndarray:
    xyy = np.asarray(xyy, dtype=np.float64)
    x = xyy[..., 0]
    y = xyy[..., 1]
    Y = xyy[..., 2]
    safe_y = np.where(np.abs(y) < 1e-12, 1e-12, y)
    X = x * Y / safe_y
    Z = (1.0 - x - y) * Y / safe_y
    return np.stack([X, Y, Z], axis=-1)


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Vectorized CIEDE2000 implementation."""

    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)

    l1, a1, b1 = np.moveaxis(lab1, -1, 0)
    l2, a2, b2 = np.moveaxis(lab2, -1, 0)

    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)
    c_bar = 0.5 * (c1 + c2)
    c_bar7 = c_bar**7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + 25.0**7 + 1e-12)))

    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = np.sqrt(a1p * a1p + b1 * b1)
    c2p = np.sqrt(a2p * a2p + b2 * b2)

    h1p = np.mod(np.arctan2(b1, a1p), 2.0 * np.pi)
    h2p = np.mod(np.arctan2(b2, a2p), 2.0 * np.pi)

    dl = l2 - l1
    dc = c2p - c1p

    dh = h2p - h1p
    dh = np.where(dh > np.pi, dh - 2.0 * np.pi, dh)
    dh = np.where(dh < -np.pi, dh + 2.0 * np.pi, dh)
    dh = np.where((c1p * c2p) == 0.0, 0.0, dh)
    d_hp = 2.0 * np.sqrt(c1p * c2p) * np.sin(dh / 2.0)

    l_bar_p = 0.5 * (l1 + l2)
    c_bar_p = 0.5 * (c1p + c2p)

    h_sum = h1p + h2p
    h_bar_p = np.where(
        (c1p * c2p) == 0.0,
        h_sum,
        np.where(
            np.abs(h1p - h2p) <= np.pi,
            0.5 * h_sum,
            np.where(h_sum < 2.0 * np.pi, 0.5 * (h_sum + 2.0 * np.pi), 0.5 * (h_sum - 2.0 * np.pi)),
        ),
    )

    t = (
        1.0
        - 0.17 * np.cos(h_bar_p - np.deg2rad(30.0))
        + 0.24 * np.cos(2.0 * h_bar_p)
        + 0.32 * np.cos(3.0 * h_bar_p + np.deg2rad(6.0))
        - 0.20 * np.cos(4.0 * h_bar_p - np.deg2rad(63.0))
    )

    delta_theta = np.deg2rad(30.0) * np.exp(-(((np.rad2deg(h_bar_p) - 275.0) / 25.0) ** 2))
    r_c = 2.0 * np.sqrt((c_bar_p**7) / (c_bar_p**7 + 25.0**7 + 1e-12))
    s_l = 1.0 + (0.015 * ((l_bar_p - 50.0) ** 2)) / np.sqrt(20.0 + ((l_bar_p - 50.0) ** 2))
    s_c = 1.0 + 0.045 * c_bar_p
    s_h = 1.0 + 0.015 * c_bar_p * t
    r_t = -np.sin(2.0 * delta_theta) * r_c

    d_l_term = dl / s_l
    d_c_term = dc / s_c
    d_h_term = d_hp / s_h
    return np.sqrt(d_l_term**2 + d_c_term**2 + d_h_term**2 + r_t * d_c_term * d_h_term)
