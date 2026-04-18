from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "img"
REFERENCE_PATH = PROJECT_ROOT / "assets" / "reference.json"
OUTPUT_DIR = PROJECT_ROOT / "output"

WHITE_INDEX = 18
USE_METADATA_RGB_XYZ_BASELINE = True
CHROMATIC_INDICES = np.arange(18)
HPPCC_REGION_CANDIDATES = [3, 4]
USE_HPPCC_BLENDING = False
HPPCC_BLEND_WIDTH = 0.15
REFERENCE_ILLUMINANT = "D65"
PERFORM_NONLINEAR_CORRECTIONS = True
SHOW_DETECTION_PREVIEW = False
OUTPUT_FORMAT = "jpeg"
OUTPUT_COLORSPACE = "sRGB"

STANDARD_WHITES = {
    "D50": np.array([0.96422, 1.0, 0.82521], dtype=np.float64),
    "D55": np.array([0.95682, 1.0, 0.92149], dtype=np.float64),
    "D65": np.array([0.95047, 1.0, 1.08883], dtype=np.float64),
}
