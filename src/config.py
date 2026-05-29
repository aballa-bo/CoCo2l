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
# Maximum white-patch level before the chart is considered over-exposed.
# When one or more channels of the white patch are clipped (saturated), the
# measured white_rgb is biased and the derived correction matrices are
# unreliable.  The analysis falls back to the linear baseline in that case.
MAX_WHITE_PATCH_LEVEL = 0.95

# HPPCC region-matrix smoothness. Tikhonov weight penalising the difference
# between adjacent regions' 3x3 matrices during the fit. Without it each region
# over-fits its handful of ColorChecker patches and the matrices diverge, so
# even a perfectly smooth hue blend still shows a residual colour transition.
# Higher = adjacent matrices kept closer (smoother images) at a small cost in
# chart accuracy; 0 disables it. Exposed in the GUI via HPPCC > HPPCC settings.
HPPCC_REGION_SMOOTHNESS = 0.0
HPPCC_GRADIENT = False
HPPCC_GRADIENT_HARMONICS = 2
# Spatial Gaussian sigma (pixels) applied to the rg-chromaticity maps before
# computing hue angles in HPPCCGradientModel.predict() for full images.
# Suppresses per-pixel shot-noise in the hue angle (σ_θ ≈ σ_noise/ρ), which
# is otherwise amplified for low-saturation hues like yellow and causes grain.
# Set to 0 to disable. Has no effect on patch-mean predictions (N×3 inputs).
HPPCC_GRADIENT_SMOOTH_SIGMA = 2.0
# Chromaticity-magnitude thresholds for the gradient-model hue fallback.
# The Fourier correction varies continuously with hue angle θ = arctan2(…).
# For near-neutral pixels (r≈g≈b) the chromaticity vector ||(r/S-⅓, g/S-⅓)||
# is tiny, so shot noise rotates θ by tens of degrees, making hue-dependent
# corrections wildly unstable.  Below CHROMA_LOW the harmonic terms are zeroed
# out and only the angle-independent DC matrix is applied; above CHROMA_HIGH
# the full Fourier series is used; between the two a smoothstep blends in the
# harmonics gradually.  Units: rg-chromaticity distance from (⅓,⅓).
# A typical saturated patch has magnitude 0.05–0.15; a near-neutral (or the
# magenta CC patch in sensor space) sits around 0.01–0.02.
# Chromaticity-magnitude threshold for near-neutral patch detection used when
# HPPCC gradient mode is active.  A chromatic patch whose rg-chromaticity
# magnitude ||(r/S-1/3, g/S-1/3)|| falls below this value is considered
# near-neutral in sensor space: arctan2 noise dominates the hue estimate and
# the gradient model produces severe spatial grain on those image areas.
# In that case the analysis falls back to the linear baseline (same behaviour
# as for an under-exposed chart).  Typical saturated patches: 0.05-0.15.
HPPCC_GRADIENT_NEAR_NEUTRAL_CHROMA_THRESHOLD = 0.02

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
# Primary method controls.  USE_HPPCC enables the hue-plane model; USE_RPCC
# adds a global Root-Polynomial residual stage on top of HPPCC.  Setting
# USE_HPPCC=False is equivalent to the old --simple-linear flag.
USE_HPPCC = True
USE_RPCC = True
# Tikhonov (ridge) regularization weight for Ridge-RPCC.  A small value (1e-3)
# shrinks the polynomial coefficients toward zero, reducing overfitting when
# the training set is sparse.
RPCC_RIDGE_LAMBDA = 1e-3
# Number of equal-width hue sectors for the Hue-Linear Color Correction model.
# Must be ≥ 2; 4 matches the default HPPCC region count.
HLCC_SECTORS = 4
SHOW_DETECTION_PREVIEW = False
SHOW_DEVELOPED_IMAGE_PREVIEW = False
OUTPUT_FORMAT = "jpeg"
OUTPUT_COLORSPACE = "sRGB"

STANDARD_WHITES = {
    "D50": np.array([0.96422, 1.0, 0.82521], dtype=np.float64),
    "D55": np.array([0.95682, 1.0, 0.92149], dtype=np.float64),
    "D65": np.array([0.95047, 1.0, 1.08883], dtype=np.float64),
}
