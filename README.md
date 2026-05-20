# coco2 — Color Correction Tool

GUI + CLI tool for freely inspired HPPCC (Hue-Plane Preserving Color Correction - https://opg.optica.org/josaa/fulltext.cfm?uri=josaa-33-11-2166) 
+ optional RPCC color correction on RAW and developed images, 
+ calibrated against an X-Rite ColorChecker Classic 24.

## Features

- RAW decoding via `rawpy` (Nikon NEF, Canon CR2/CR3, Sony ARW, Fuji RAF, Adobe DNG).
- Developed image input (JPEG/PNG/TIFF/BMP/WebP) with sRGB-style linearization.
- Automatic ColorChecker detection with manual ROI fallback.
- freeely inspired HPPCC fit (piecewise-linear by hue region) plus optional RPCC nonlinear residual.
- Optional pipeline stages: wavelet/bilateral denoise, adaptive sharpening, white-field
  (vignetting) correction, highlight recovery (`rawpy.HighlightMode.Blend`), lens
  undistortion via `lensfunpy`.
- Three GUI tabs:
  - **Analysis** — calibrate from a single shot containing the chart.
  - **Processing** — batch a folder of RAWs against an existing correction file.
  - **Batch** — chain multiple (chart, target folder) pairings, run sequentially.
- Authoritative `*_correction.json`: during processing the GUI cannot override settings
  baked at analysis time.

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
distributed dependency). Download it from the official site and place it in the
**project root folder** (the same folder that contains `coco2.py`):

1. Download the "Windows Executable" from <https://exiftool.org/>.
2. Extract the archive and copy into the project root:
   - `exiftool(-k).exe` → rename to **`exiftool.exe`**
   - The **`exiftool_files/`** folder (required by recent distributions — it contains
     `perl5*.dll` and the Perl modules ExifTool needs at runtime)

The app resolves `exiftool` in this order:

1. `<project_root>/exiftool.exe` (recommended).
2. Path declared in `config.ini` (see below).
3. System `PATH` (generic `exiftool` lookup).

On macOS / Linux: use the name without extension (`exiftool`), or install it via your
package manager and let `PATH` resolve it.

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

### CLI

```bash
# Analyze (produces *_correction.json)
python coco2.py analyze --cc-image path/to/IMG_chart.NEF --analysis-dir output/

# Batch process a folder using an existing correction
python coco2.py process output/IMG_chart_correction.json path/to/raw_folder/ --workers 4
```

All CLI flags are documented through `python coco2.py analyze --help` and
`python coco2.py process --help`.

## Troubleshooting

- **`Warning: exiftool failed ... exit code 4294967295`** — ExifTool is unreachable or
  cannot start. Verify `exiftool.exe` is in the project root and that `exiftool_files/`
  sits next to it.
- **`No ColorChecker detected`** — switch to the ROI tool in Analysis to outline the
  chart area manually, then run again.
- **Magenta/pink tint on over-exposed areas** — mitigated in three stages:
  `rawpy.HighlightMode.Blend` (raw decode) → `desaturate_highlights` (linear sensor RGB,
  before HPPCC) → `neutralize_blown_highlights` (display RGB, after colour correction).
  The last stage removes the cast that HPPCC/RPCC re-introduces when it extrapolates
  clipped highlights past its training range: a blown-pixel weight is computed from the
  *sensor* RGB by `highlight_blowout_weight` (any channel near the clip ceiling) and the
  matching pixels are forced toward white in the final image. If the artefact still
  persists, lower the `threshold`/`full` parameters of `highlight_blowout_weight` in
  `src/utils.py` (e.g. `threshold=0.88`); if instead bright saturated colours look
  washed out, raise `threshold`. Note that raising the HPPCC blend width does **not**
  help here — it only smooths hue-region boundaries, not highlight extrapolation.

## Project layout

```
coco2.py              # GUI / CLI dispatcher entry point
src/                  # scientific library (HPPCC, RPCC, raw decode, utils)
  cc.py               # CLI script spawned as subprocess by the GUI
  raw.py              # RAW decoding + sRGB linearization for non-RAW inputs
  utils.py            # normalize, denoise, sharpen, white field, exif copy
  ...
assets/reference.json # ColorChecker Classic 24 reference values
exiftool.exe          # (gitignored) — see setup
exiftool_files/       # (gitignored) — see setup
config.ini            # (gitignored) — optional exiftool path override
settings.json         # (gitignored) — persisted GUI state in %APPDATA%/coco
```
