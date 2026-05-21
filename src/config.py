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

# Minimum exposure of the ColorChecker white patch, as a fraction of sensor
# full scale (brightest channel). The white patch is the brightest neutral
# training sample, so it bounds the tonal range the HPPCC/RPCC model is fitted
# on. Below this level the model only learns the shadows; analyze warns and
# the per-pixel linear fallback (below) carries the out-of-range tones.
MIN_WHITE_PATCH_LEVEL = 0.25

# HPPCC region-matrix smoothness. Tikhonov weight penalising the difference
# between adjacent regions' 3x3 matrices during the fit. Without it each region
# over-fits its handful of ColorChecker patches and the matrices diverge, so
# even a perfectly smooth hue blend still shows a residual colour transition.
# Higher = adjacent matrices kept closer (smoother images) at a small cost in
# chart accuracy; 0 disables it. Exposed in the GUI via HPPCC > HPPCC settings.
HPPCC_REGION_SMOOTHNESS = 0.0

# Out-of-training-range fallback. HPPCC/RPCC extrapolate badly above the
# brightest fitted patch value; the white-preserving linear matrix degrades
# gracefully. Per pixel the output ramps from HPPCC/RPCC to the linear
# baseline between `train_max` and `train_max * LINEAR_FALLBACK_RANGE_FACTOR`.
LINEAR_FALLBACK_RANGE_FACTOR = 2.0

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
