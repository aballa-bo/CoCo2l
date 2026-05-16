import sys
from pathlib import Path

import numpy as np


# Path layout. In a PyInstaller frozen build, read-only bundled data (assets,
# reference values) lives under sys._MEIPASS while the executable and adjacent
# files (exiftool.exe, config.ini, user output) sit alongside sys.executable.
# In dev mode the two coincide with the repository root.
if getattr(sys, "frozen", False):
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", str(Path(sys.executable).resolve().parent))).resolve()
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    BUNDLE_DIR = Path(__file__).resolve().parent.parent
    PROJECT_ROOT = BUNDLE_DIR

IMAGE_DIR = PROJECT_ROOT / "img"
REFERENCE_PATH = BUNDLE_DIR / "assets" / "reference.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
ANALYSIS_DIR = PROJECT_ROOT / "output" / "analysis"
PROCESS_DIR = PROJECT_ROOT / "output" / "process"

WHITE_INDEX = 18
USE_METADATA_RGB_XYZ_BASELINE = True
CHROMATIC_INDICES = np.arange(18)
HPPCC_REGION_CANDIDATES = [3, 4]
USE_HPPCC_BLENDING = True
HPPCC_BLEND_WIDTH = 0.15
REFERENCE_ILLUMINANT = "D65"
REFERENCE_SPACE = "lab"
SCENE_WHITE_SOURCE = "auto"
ENABLE_PATCH_VARIANCE_DENOISE = False
DENOISE_METHOD = "wavelet"
DENOISE_STRENGTH = 6.0
DENOISE_DIAMETER = 5
DENOISE_SIGMA_SPACE = 2.0
ENABLE_ADAPTIVE_SHARPEN = False
SHARPEN_AMOUNT = 0.6
SHARPEN_RADIUS = 1.0
SHARPEN_THRESHOLD = 1.5
ENABLE_PROCESS_WHITE_FIELD = False
PERFORM_NONLINEAR_CORRECTIONS = True
SHOW_DETECTION_PREVIEW = False
SHOW_DEVELOPED_IMAGE_PREVIEW = False
OUTPUT_FORMAT = "jpeg"
OUTPUT_COLORSPACE = "sRGB"

STANDARD_WHITES = {
    "D50": np.array([0.96422, 1.0, 0.82521], dtype=np.float64),
    "D55": np.array([0.95682, 1.0, 0.92149], dtype=np.float64),
    "D65": np.array([0.95047, 1.0, 1.08883], dtype=np.float64),
}
