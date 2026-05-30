# PyInstaller spec for coco2 (GUI + CLI dispatcher).
# Build:  pyinstaller coco2.spec
# Result: dist/coco2/coco2.exe + adjacent DLLs and data folder.
#
# Note: ExifTool is NOT bundled. After building, copy `exiftool.exe` and
# `exiftool_files/` next to dist/coco2/coco2.exe (same rule as dev mode).

import sys
from pathlib import Path

PROJECT_ROOT = Path(SPECPATH).resolve()

a = Analysis(
    [str(PROJECT_ROOT / 'coco2.py')],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[
        (str(PROJECT_ROOT / 'assets'), 'assets'),
        (str(PROJECT_ROOT / 'README.md'), '.'),  # accessed via BUNDLE_DIR/README.md
    ],
    hiddenimports=[
        # `src` package and its modules are imported dynamically via
        # `from src.cc import main` and through cc.main()'s own imports.
        'src',
        'src.cc',
        'src.cli_parser',
        'src.colorchecker',
        'src.colorchecker_detector',
        'src.config',
        'src.lens',
        'src.metrics',
        'src.models',
        'src.raw',
        'src.report',
        'src.sampling',
        'src.utils',
        # colour-checker-detection pulls submodules at runtime that the
        # static analyser sometimes misses.
        'colour_checker_detection',
        'colour_checker_detection.detection',
        'colour_checker_detection.detection.segmentation',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['matplotlib', 'tkinter', 'pytest', 'IPython', 'jupyter'],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

# PyInstaller's Splash screen is supported only on Windows and Linux — skip it
# on macOS, where Splash() would abort the build.
splash = None
if sys.platform != 'darwin':
    splash = Splash(
        str(PROJECT_ROOT / 'assets' / 'cocoico1.png'),
        binaries=a.binaries,
        datas=a.datas,
        text_pos=None,             # no progress text — just the artwork
        always_on_top=True,
        minify_script=True,
    )

exe = EXE(
    pyz,
    a.scripts,
    *([splash] if splash else []),   # splash launcher embedded in the exe (onedir mode)
    [],
    exclude_binaries=True,
    name='coco2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,             # GUI app; subprocess stdout is captured via QProcess pipes
    disable_windowed_traceback=False,
    # .ico is Windows-only; macOS wants .icns and Linux ignores the icon.
    icon=str(PROJECT_ROOT / 'assets' / 'cocoicobn.ico') if sys.platform == 'win32' else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    *([splash.binaries] if splash else []),   # splash native lib next to the exe
    strip=False,
    upx=False,
    name='coco2',
)
