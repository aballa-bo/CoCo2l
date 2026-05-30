# coco2 — Color Correction Tool

GUI + CLI tool for ColorChecker-calibrated color correction of RAW and developed
images, built around a **visual processing pipeline**: you assemble an ordered
sequence of preprocessing and color-correction operations, tune each one, and run
it against an X-Rite ColorChecker Classic 24.


A range of correction models is available as interchangeable building blocks —
linear and locally-linear (baseline 3×3, Wiener, PCA, ΔE00-optimised, HPPCC,
HLCC, LWCC) and non-linear (RPCC, ridge-RPCC, HPPCC+RPCC, thin-plate spline) —
none privileged over the others; you choose which to apply.


## Features

- **Visual pipeline** — drag operations from a palette into an ordered list and
  reorder them by drag & drop. The palette is grouped into:
  - **Preprocessing**: lens undistortion, devignetting, denoise, sharpening.
  - **Linear / locally-linear corrections**: `baseline`, `wiener`, `pca`,
    `de00_opt`, `hppcc`, `hlcc`, `lwcc`.
  - **Non-linear corrections**: `rpcc`, `rpcc_ridge`, `hppcc_rpcc`, `tps`.
- **Per-step settings** — every operation that has tunable parameters carries a
  ⚙ gear button opening its own settings dialog (e.g. TPS smoothing, ridge λ,
  Wiener SNR, PCA components, HLCC sectors, HPPCC regions/blending, denoise
  method/strength…). Each dialog has a **Revert to default** button. Settings are
  per pipeline instance and are remembered between sessions.
- **Sequential residual cascade** — correction steps run top-to-bottom on a
  single working image; after each step the ColorChecker is re-measured so the
  next step is fit on the updated values, refining the previous step's residual.
- **Live per-step feedback** — each step shows an hourglass while pending, a
  green check with its mean & median **ΔE₀₀** once applied, or a red mark if it
  was skipped. A summary of the pipeline **actually applied** is printed at the
  end, and a warning is raised if execution differed from what you arranged
  (e.g. corrections skipped because the chart was under/over-exposed).
- RAW decoding via `rawpy` (Nikon NEF, Canon CR2/CR3, Sony ARW, Fuji RAF, Adobe
  DNG) and developed-image input (JPEG/PNG/TIFF/BMP/WebP) with sRGB-style
  linearization.
- Automatic ColorChecker detection with manual ROI fallback.
- Highlight recovery (`rawpy.HighlightMode.Blend`) and blown-highlight
  neutralisation; optional single-image (blind) vignetting / white-field
  correction.
- Three GUI tabs:
  - **Analysis** — calibrate from a single shot containing the chart.
  - **Processing** — batch a folder of RAWs against an existing correction file.
  - **Batch** — chain multiple (chart, target folder) pairings, run sequentially.
- Authoritative `*_correction.json`: the ordered pipeline and every step's
  parameters are baked at analysis time and replayed during processing, so the
  results are reproducible across a folder.

## Setup

### 1. Python and dependencies

Requires Python 3.11+ (tested with 3.11). Create a virtual environment and install
dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
```

### 2. ExifTool (required to copy EXIF from RAW to developed images)

ExifTool is **not bundled with the repository** (it is an external, separately
distributed dependency). Install it for your platform:

**Windows** — download the "Windows Executable" from <https://exiftool.org/>,
then place it in the **project root folder** (the folder containing `coco2.py`):
- rename `exiftool(-k).exe` → **`exiftool.exe`**
- also copy the **`exiftool_files/`** folder next to it (required by recent
  distributions — it holds `perl5*.dll` and the Perl modules ExifTool needs at
  runtime).

**macOS** — `brew install exiftool` (or download the macOS package from
<https://exiftool.org/>). It lands on your `PATH` as `exiftool`.

**Linux** — install from your package manager, e.g. `sudo apt install
libimage-exiftool-perl` (Debian/Ubuntu) or `sudo dnf install perl-Image-ExifTool`
(Fedora). It lands on your `PATH` as `exiftool`.

The app resolves ExifTool in this order:

1. `<project_root>/exiftool.exe` (Windows) or `<project_root>/exiftool` (macOS/Linux).
2. Path declared in `config.ini` (see below).
3. System `PATH` (generic `exiftool` lookup) — the usual case on macOS/Linux.

### 3. Optional path override: `config.ini`

If you prefer ExifTool installed elsewhere (e.g. a shared install on a workstation),
create a `config.ini` file in the project root:

```ini
[paths]
exiftool = C:\Tools\exiftool\exiftool.exe
```

`config.ini` is gitignored — it is a per-machine local preference.

## Usage

### GUI

```bash
python coco2.py
```

In the **Analysis** tab:

1. Load (or drag in) a RAW/developed image that contains the ColorChecker. If
   detection fails, draw a ROI on the preview to outline the chart.
2. Build the pipeline: drag operations from the palette into the pipeline column
   and order them. Click a step's ⚙ to adjust its parameters.
3. Click **Run analysis**. Each step reports its status and ΔE₀₀ as it runs, the
   developed image appears on the right, and a `*_correction.json` is written.

Use the **Processing** / **Batch** tabs to apply a saved `*_correction.json` to a
folder (or many folders) of RAWs.

### CLI

```bash
# Analyze with an explicit pipeline (ordered op ids)
python coco2.py analyze --cc-image path/to/IMG_chart.NEF --analysis-dir output/ \
    --pipeline undistort,hppcc,tps

# Analyze with per-step parameters (JSON spec; supersedes --pipeline)
python coco2.py analyze --cc-image path/to/IMG_chart.NEF --analysis-dir output/ \
    --pipeline-spec '[{"op":"undistort","params":{"method":"lensfun"}},
                      {"op":"hppcc","params":{"k_regions":3}},
                      {"op":"tps","params":{"smoothing":0.05}}]'

# Batch process a folder using an existing correction
python coco2.py process output/IMG_chart_correction.json path/to/raw_folder/ --workers 4
```

When neither `--pipeline` nor `--pipeline-spec` is given, the legacy single-model
path (`--output-label`) is used. All CLI flags are documented through
`python coco2.py analyze --help` and `python coco2.py process --help`.

## Troubleshooting

- **`Warning: exiftool failed ... exit code 4294967295`** — ExifTool is unreachable or
  cannot start. Verify `exiftool.exe` is in the project root and that `exiftool_files/`
  sits next to it.
- **`No ColorChecker detected`** — switch to the ROI tool in Analysis to outline the
  chart area manually, then run again.
- **Correction steps were skipped / a warning says the pipeline ran differently** —
  the chart was under- or over-exposed (the white patch must sit roughly 25–95% of
  full scale), so the correction models would be unreliable and the robust linear
  baseline was applied to the whole image instead. Re-shoot the chart with the white
  patch around 60–90% of full scale, without clipping, and re-run.
- **Magenta/pink tint on over-exposed areas** — mitigated in three stages:
  `rawpy.HighlightMode.Blend` (raw decode) → `desaturate_highlights` (linear sensor
  RGB, before correction) → `neutralize_blown_highlights` (display RGB, after colour
  correction). The last stage removes the cast a nonlinear correction can re-introduce
  when it extrapolates clipped highlights past its training range: a blown-pixel
  weight is computed from the *sensor* RGB by `highlight_blowout_weight` (any channel
  near the clip ceiling) and the matching pixels are forced toward white in the final
  image. If the artefact persists, lower the `threshold`/`full` parameters of
  `highlight_blowout_weight` in `src/utils.py` (e.g. `threshold=0.88`); if bright
  saturated colours look washed out, raise `threshold`.

## Project layout

```
coco2.py              # GUI / CLI dispatcher entry point
src/                  # scientific library (correction models, raw decode, utils)
  cc.py               # CLI script spawned as subprocess by the GUI
  models.py           # correction model fits (linear, RPCC, HPPCC, HLCC, TPS, …)
  raw.py              # RAW decoding + sRGB linearization for non-RAW inputs
  utils.py            # normalize, denoise, sharpen, white field, cascade, exif copy
  ...
assets/reference.json # ColorChecker Classic 24 reference values
exiftool.exe          # (gitignored) — see setup
exiftool_files/       # (gitignored) — see setup
config.ini            # (gitignored) — optional exiftool path override
settings.json         # (gitignored) — persisted GUI state in %APPDATA%/coco
```
