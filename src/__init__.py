"""Scientific core for hue-plane preserving color correction."""

__version__ = "0.2.5"

from .colorchecker import (
    CLASSIC_24_CHROMATIC_NAMES,
    CLASSIC_24_NEUTRAL_NAMES,
    CLASSIC_24_PATCH_NAMES,
)
from .metrics import delta_e_2000, hue_angle_from_rgb, lab_to_xyz, xyy_to_xyz, xyz_to_lab
from .models import (
    HPPCCModel,
    HPPCCRPCCModel,
    LinearWhitePreservingModel,
    RPCCModel,
    fit_hppcc,
    fit_hppcc_rpcc,
    fit_rpcc,
    fit_white_preserving_3x3,
)
from .raw import (
    INPUT_SUFFIXES,
    IMAGE_SUFFIXES,
    RAW_SUFFIXES,
    RawLinearImage,
    load_image_linear_rgb,
    load_raw_linear_rgb,
)
from .sampling import sample_patch_means, sample_patch_means_from_masks
from .utils import load_reference_chroma, load_reference_lab, load_reference_rgb, load_reference_white_xyz, load_reference_xyy, load_reference_xyz

__all__ = [
    "CLASSIC_24_CHROMATIC_NAMES",
    "CLASSIC_24_NEUTRAL_NAMES",
    "CLASSIC_24_PATCH_NAMES",
    "HPPCCModel",
    "HPPCCRPCCModel",
    "LinearWhitePreservingModel",
    "RPCCModel",
    "INPUT_SUFFIXES",
    "IMAGE_SUFFIXES",
    "RAW_SUFFIXES",
    "RawLinearImage",
    "delta_e_2000",
    "fit_hppcc",
    "fit_hppcc_rpcc",
    "fit_rpcc",
    "fit_white_preserving_3x3",
    "hue_angle_from_rgb",
    "lab_to_xyz",
    "load_image_linear_rgb",
    "load_raw_linear_rgb",
    "load_reference_chroma",
    "load_reference_lab",
    "load_reference_rgb",
    "load_reference_white_xyz",
    "load_reference_xyy",
    "load_reference_xyz",
    "sample_patch_means",
    "sample_patch_means_from_masks",
    "xyy_to_xyz",
    "xyz_to_lab",
]
