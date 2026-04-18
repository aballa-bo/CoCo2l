"""Scientific core for hue-plane preserving color correction."""

from .colorchecker import (
    CLASSIC_24_CHROMATIC_NAMES,
    CLASSIC_24_NEUTRAL_NAMES,
    CLASSIC_24_PATCH_NAMES,
)
from .metrics import delta_e_2000, hue_angle_from_rgb, xyz_to_lab
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
from .raw import RawLinearImage, load_raw_linear_rgb
from .sampling import sample_patch_means, sample_patch_means_from_masks

__all__ = [
    "CLASSIC_24_CHROMATIC_NAMES",
    "CLASSIC_24_NEUTRAL_NAMES",
    "CLASSIC_24_PATCH_NAMES",
    "HPPCCModel",
    "HPPCCRPCCModel",
    "LinearWhitePreservingModel",
    "RPCCModel",
    "RawLinearImage",
    "delta_e_2000",
    "fit_hppcc",
    "fit_hppcc_rpcc",
    "fit_rpcc",
    "fit_white_preserving_3x3",
    "hue_angle_from_rgb",
    "load_raw_linear_rgb",
    "sample_patch_means",
    "sample_patch_means_from_masks",
    "xyz_to_lab",
]
