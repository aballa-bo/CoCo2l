"""HPPCC GUI — PyQt6 interface for color correction analysis and processing."""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit,
    QFileDialog, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QRubberBand, QMessageBox, QGroupBox, QSizePolicy, QSplitter,
    QDialog, QDialogButtonBox, QScrollArea, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QTextBrowser, QWidgetAction, QListWidget, QListWidgetItem, QToolButton,
)
from PyQt6.QtCore import Qt, QProcess, QThread, pyqtSignal, QRect, QSize, QPoint, QEvent
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QAction

if getattr(sys, "frozen", False):
    # PyInstaller bundle: the executable itself dispatches to cc.main() when
    # called with CLI args (see main() at the bottom), so we re-spawn ourselves
    # instead of looking for a Python interpreter + the cc.py script.
    PROJECT_ROOT = Path(sys.executable).resolve().parent
    PYTHON = sys.executable
    _CC_PREFIX: list[str] = []
else:
    PROJECT_ROOT = Path(__file__).resolve().parent
    PYTHON = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
    CC_SCRIPT = str(PROJECT_ROOT / "src" / "cc.py")
    _CC_PREFIX = [CC_SCRIPT]


def _cc_args(*sub_args: str) -> list[str]:
    """Build the args list for QProcess.start(PYTHON, ...).

    Dev: [<src/cc.py>, <sub_args...>]
    Frozen: [<sub_args...>] (the bundled exe dispatches to cc.main() itself).
    """
    return [*_CC_PREFIX, *sub_args]

_EXT_MAP = {"jpeg": ".jpg", "tif": ".tif", "png": ".png"}

# Single source of truth for accepted input extensions. Importing from `src.raw`
# would pull in numpy/rawpy at import time of coco2.py — keep it inline.
_RAW_EXTS = {".nef", ".cr2", ".cr3", ".arw", ".raf", ".dng"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
_INPUT_EXTS = _RAW_EXTS | _IMAGE_EXTS

_INPUT_FILE_FILTER = (
    "Images (*.nef *.cr2 *.cr3 *.arw *.raf *.dng *.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp "
    "*.NEF *.CR2 *.CR3 *.ARW *.RAF *.DNG *.JPG *.JPEG *.PNG *.TIF *.TIFF *.BMP *.WEBP);;"
    "RAW (*.nef *.cr2 *.cr3 *.arw *.raf *.dng *.NEF *.CR2 *.CR3 *.ARW *.RAF *.DNG);;"
    "All files (*)"
)

_APP_DIR_NAME = "coco"


def _settings_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / _APP_DIR_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / _APP_DIR_NAME
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / _APP_DIR_NAME


def _settings_path() -> Path:
    return _settings_dir() / "settings.json"


def load_settings() -> dict:
    path = _settings_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    path = _settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        # Persistence is best-effort; never block app close on disk errors.
        pass


def _widget_value(widget):
    from PyQt6.QtWidgets import QCheckBox, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QLineEdit):
        return widget.text()
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        return widget.value()
    raise TypeError(f"Unsupported widget type for persistence: {type(widget).__name__}")


def _set_widget_value(widget, value) -> None:
    if value is None:
        return
    from PyQt6.QtWidgets import QCheckBox, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox
    try:
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QComboBox):
            widget.setCurrentText(str(value))
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
    except (TypeError, ValueError):
        # Ignore malformed persisted values — keep default.
        pass
def _count_raw_files(folder: Path, recursive: bool) -> int:
    if not folder.is_dir():
        return 0
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sum(1 for p in iterator if p.is_file() and p.suffix.lower() in _INPUT_EXTS)


# Windows-only process-tree control via ctypes (no extra deps).
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.windll.kernel32
    _ntdll = ctypes.windll.ntdll
    _PROCESS_SUSPEND_RESUME = 0x0800
    _PROCESS_TERMINATE = 0x0001
    _TH32CS_SNAPPROCESS = 0x00000002

    class _PROCESSENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_void_p),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_char * 260),
        ]

    def _enumerate_children(parent_pid: int) -> list[int]:
        snapshot = _kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
        if snapshot in (None, 0) or snapshot == ctypes.c_void_p(-1).value:
            return []
        try:
            entry = _PROCESSENTRY32()
            entry.dwSize = ctypes.sizeof(_PROCESSENTRY32)
            children: list[int] = []
            if _kernel32.Process32First(snapshot, ctypes.byref(entry)):
                while True:
                    if int(entry.th32ParentProcessID) == parent_pid:
                        children.append(int(entry.th32ProcessID))
                    if not _kernel32.Process32Next(snapshot, ctypes.byref(entry)):
                        break
            return children
        finally:
            _kernel32.CloseHandle(snapshot)

    def _enumerate_tree(root_pid: int) -> list[int]:
        tree = [int(root_pid)]
        idx = 0
        while idx < len(tree):
            tree.extend(_enumerate_children(tree[idx]))
            idx += 1
        return tree

    def _process_op(pid: int, access: int, op) -> None:
        h = _kernel32.OpenProcess(access, False, pid)
        if h:
            try:
                op(h)
            finally:
                _kernel32.CloseHandle(h)

    def suspend_process_tree(root_pid: int) -> None:
        for pid in _enumerate_tree(root_pid):
            _process_op(pid, _PROCESS_SUSPEND_RESUME, _ntdll.NtSuspendProcess)

    def resume_process_tree(root_pid: int) -> None:
        for pid in reversed(_enumerate_tree(root_pid)):
            _process_op(pid, _PROCESS_SUSPEND_RESUME, _ntdll.NtResumeProcess)

    def kill_process_tree(root_pid: int) -> None:
        for pid in _enumerate_tree(root_pid):
            _process_op(pid, _PROCESS_TERMINATE, lambda h: _kernel32.TerminateProcess(h, 1))
else:
    def suspend_process_tree(root_pid: int) -> None: return None
    def resume_process_tree(root_pid: int) -> None: return None
    def kill_process_tree(root_pid: int) -> None: return None


# ─────────────────────────────────────────────────────────────────────────────
# Background RAW thumbnail loader
# ─────────────────────────────────────────────────────────────────────────────

class RawPreviewLoader(QThread):
    ready = pyqtSignal(np.ndarray, int, int)   # rgb_uint8, orig_w, orig_h
    failed = pyqtSignal(str)

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path

    def run(self) -> None:
        try:
            suffix = Path(self._path).suffix.lower()
            if suffix in _RAW_EXTS:
                import rawpy  # type: ignore
                with rawpy.imread(self._path) as raw:
                    orig_h, orig_w = raw.raw_image_visible.shape[:2]
                    params = rawpy.Params(
                        use_camera_wb=True,
                        half_size=True,
                        output_color=rawpy.ColorSpace.sRGB,
                        output_bps=8,
                        no_auto_bright=False,
                        bright=1.0,
                        gamma=(2.222, 4.5),
                    )
                    rgb = raw.postprocess(params=params)
                self.ready.emit(rgb, orig_w, orig_h)
                return
            from PIL import Image  # local import: avoid PIL on RAW-only sessions
            with Image.open(self._path) as img:
                img_rgb = img.convert("RGB")
                rgb = np.asarray(img_rgb, dtype=np.uint8)
            self.ready.emit(rgb, rgb.shape[1], rgb.shape[0])
        except Exception as exc:
            self.failed.emit(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Zoomable preview base — QScrollArea + inner QLabel, mouse-wheel zoom,
# scroll-position signals for cross-preview synchronization
# ─────────────────────────────────────────────────────────────────────────────

class _ZoomableImagePreview(QWidget):
    zoom_changed = pyqtSignal(float)
    scroll_changed = pyqtSignal(float, float)   # h_norm, v_norm in [0, 1]

    MIN_ZOOM = 0.5
    MAX_ZOOM = 32.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background:#1a1a1a; border:1px solid #555;")

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.viewport().setStyleSheet("background:#1a1a1a;")

        self._inner = QLabel()
        self._inner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._inner.setStyleSheet("background:#1a1a1a; color:#ccc;")
        self._inner.setWordWrap(True)
        self._scroll.setWidget(self._inner)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll)

        self._pixmap_src: QPixmap | None = None
        self._orig_w = 1
        self._orig_h = 1
        self._zoom = 1.0   # 1.0 == fit-to-viewport
        self._scale = 1.0  # current orig→screen pixel scale
        self._suppress_sync = False

        self._scroll.viewport().installEventFilter(self)
        self._scroll.horizontalScrollBar().valueChanged.connect(self._on_scrolled)
        self._scroll.verticalScrollBar().valueChanged.connect(self._on_scrolled)

    # ── public API ────────────────────────────────────────────────────────

    def setText(self, text: str) -> None:
        self._inner.setText(text)
        self._inner.setPixmap(QPixmap())
        self._pixmap_src = None
        self._inner.resize(self._scroll.viewport().size())

    def set_zoom_external(self, zoom: float) -> None:
        if self._pixmap_src is None:
            return
        zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, float(zoom)))
        if abs(zoom - self._zoom) < 1e-3:
            return
        self._suppress_sync = True
        try:
            self._zoom = zoom
            self._redraw()
        finally:
            self._suppress_sync = False

    def set_scroll_external(self, h_norm: float, v_norm: float) -> None:
        if self._pixmap_src is None:
            return
        self._suppress_sync = True
        try:
            hbar = self._scroll.horizontalScrollBar()
            vbar = self._scroll.verticalScrollBar()
            if hbar.maximum() > 0:
                hbar.setValue(int(round(h_norm * hbar.maximum())))
            if vbar.maximum() > 0:
                vbar.setValue(int(round(v_norm * vbar.maximum())))
        finally:
            self._suppress_sync = False

    # ── event handling ────────────────────────────────────────────────────

    def eventFilter(self, obj, event):
        if obj is self._scroll.viewport():
            if event.type() == QEvent.Type.Wheel:
                self._handle_wheel(event)
                return True
            if event.type() == QEvent.Type.Resize:
                if self._pixmap_src is None:
                    self._inner.resize(self._scroll.viewport().size())
                else:
                    self._redraw()
        return super().eventFilter(obj, event)

    # ── rendering ─────────────────────────────────────────────────────────

    def _fit_scale(self) -> float:
        v = self._scroll.viewport().size()
        return min(max(v.width(), 1) / self._orig_w, max(v.height(), 1) / self._orig_h)

    def _redraw(self) -> None:
        if self._pixmap_src is None:
            self._inner.resize(self._scroll.viewport().size())
            return
        scale = self._fit_scale() * self._zoom
        self._scale = scale
        sw = max(1, int(round(self._orig_w * scale)))
        sh = max(1, int(round(self._orig_h * scale)))
        scaled = self._pixmap_src.scaled(
            sw, sh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        canvas = self._compose_canvas(scaled, scale)
        self._inner.setPixmap(canvas)
        self._inner.resize(canvas.size())

    def _compose_canvas(self, scaled_pixmap: QPixmap, scale: float) -> QPixmap:
        return scaled_pixmap

    # ── interaction ───────────────────────────────────────────────────────

    def _handle_wheel(self, event) -> None:
        if self._pixmap_src is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 1.0 / 1.25
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._zoom * factor))
        if abs(new_zoom - self._zoom) < 1e-3:
            return

        cursor = event.position()
        hbar = self._scroll.horizontalScrollBar()
        vbar = self._scroll.verticalScrollBar()
        viewport_size = self._scroll.viewport().size()
        old_off_x = max(0, (viewport_size.width() - self._inner.width()) // 2)
        old_off_y = max(0, (viewport_size.height() - self._inner.height()) // 2)
        old_scale = self._scale if self._scale > 0 else 1.0
        img_x = (hbar.value() + cursor.x() - old_off_x) / old_scale
        img_y = (vbar.value() + cursor.y() - old_off_y) / old_scale

        self._zoom = new_zoom
        self._redraw()

        new_off_x = max(0, (viewport_size.width() - self._inner.width()) // 2)
        new_off_y = max(0, (viewport_size.height() - self._inner.height()) // 2)
        target_h = int(round(img_x * self._scale - cursor.x() + new_off_x))
        target_v = int(round(img_y * self._scale - cursor.y() + new_off_y))
        self._suppress_sync = True
        try:
            hbar.setValue(max(0, min(target_h, hbar.maximum())))
            vbar.setValue(max(0, min(target_v, vbar.maximum())))
        finally:
            self._suppress_sync = False

        self.zoom_changed.emit(self._zoom)
        self._emit_scroll()
        event.accept()

    def _on_scrolled(self) -> None:
        if self._suppress_sync:
            return
        self._emit_scroll()

    def _emit_scroll(self) -> None:
        hbar = self._scroll.horizontalScrollBar()
        vbar = self._scroll.verticalScrollBar()
        h_norm = hbar.value() / hbar.maximum() if hbar.maximum() > 0 else 0.0
        v_norm = vbar.value() / vbar.maximum() if vbar.maximum() > 0 else 0.0
        self.scroll_changed.emit(h_norm, v_norm)


# ─────────────────────────────────────────────────────────────────────────────
# RAW preview widget — rubber-band ROI selection + drag & drop
# ─────────────────────────────────────────────────────────────────────────────

class ImagePreview(_ZoomableImagePreview):
    roi_changed = pyqtSignal(int, int, int, int)   # x1,y1,x2,y2 in original pixels
    roi_cleared = pyqtSignal()
    file_dropped = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._inner.setText('Drop a RAW file here or use "Browse"')
        self._inner.setCursor(Qt.CursorShape.CrossCursor)
        self.setAcceptDrops(True)

        self._roi: tuple[int, int, int, int] | None = None
        self._rubber_band: QRubberBand | None = None
        self._rb_origin = QPoint()

        self._inner.installEventFilter(self)

    def set_from_array(self, rgb: np.ndarray, orig_w: int, orig_h: int) -> None:
        h, w = rgb.shape[:2]
        img = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        self._pixmap_src = QPixmap.fromImage(img)
        self._orig_w = int(orig_w)
        self._orig_h = int(orig_h)
        self._zoom = 1.0
        self._inner.setText("")
        self._redraw()
        self._inner.setToolTip("Click and drag to select the color checker ROI")

    def _compose_canvas(self, scaled_pixmap: QPixmap, scale: float) -> QPixmap:
        if not self._roi:
            return scaled_pixmap
        canvas = QPixmap(scaled_pixmap)
        p = QPainter(canvas)
        x1, y1, x2, y2 = self._roi
        rx = int(x1 * scale)
        ry = int(y1 * scale)
        rw = int((x2 - x1) * scale)
        rh = int((y2 - y1) * scale)
        p.setPen(QPen(QColor(255, 100, 0), 2, Qt.PenStyle.DashLine))
        p.drawRect(rx, ry, rw, rh)
        p.setPen(QPen(QColor(255, 100, 0)))
        p.setFont(QFont("Consolas", 9))
        p.drawText(rx + 4, ry - 4, f"{x1},{y1} -> {x2},{y2}")
        p.end()
        return canvas

    def _to_image(self, pt: QPoint) -> tuple[int, int]:
        scale = self._scale if self._scale > 0 else 1.0
        ix = int(pt.x() / scale)
        iy = int(pt.y() / scale)
        return max(0, min(self._orig_w, ix)), max(0, min(self._orig_h, iy))

    def eventFilter(self, obj, event):
        if obj is self._inner:
            t = event.type()
            if t == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                if self._pixmap_src is not None:
                    self._rb_origin = event.position().toPoint()
                    if self._rubber_band is None:
                        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self._inner)
                    self._rubber_band.setGeometry(QRect(self._rb_origin, QSize()))
                    self._rubber_band.show()
                return True
            if t == QEvent.Type.MouseMove:
                if self._rubber_band and self._rubber_band.isVisible():
                    self._rubber_band.setGeometry(
                        QRect(self._rb_origin, event.position().toPoint()).normalized()
                    )
                return False
            if t == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                if self._rubber_band and self._rubber_band.isVisible():
                    self._rubber_band.hide()
                    rect = QRect(self._rb_origin, event.position().toPoint()).normalized()
                    if rect.width() >= 8 and rect.height() >= 8:
                        x1, y1 = self._to_image(rect.topLeft())
                        x2, y2 = self._to_image(rect.bottomRight())
                        if x2 > x1 and y2 > y1:
                            self._roi = (x1, y1, x2, y2)
                            self._redraw()
                            self.roi_changed.emit(x1, y1, x2, y2)
                return True
        return super().eventFilter(obj, event)

    def set_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        self._roi = roi
        self._redraw()

    def clear_roi(self) -> None:
        self._roi = None
        self._redraw()
        self.roi_cleared.emit()

    def request_roi(self) -> None:
        self.setStyleSheet("background:#1a1a1a; border:2px solid #e06000;")

    def acknowledge_roi(self) -> None:
        self.setStyleSheet("background:#1a1a1a; border:1px solid #555;")

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if urls:
            self.file_dropped.emit(urls[0].toLocalFile())


# ─────────────────────────────────────────────────────────────────────────────
# Developed image preview — read-only, loads from a file path
# ─────────────────────────────────────────────────────────────────────────────

class DevelopedPreview(_ZoomableImagePreview):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._reset_text()

    def _reset_text(self) -> None:
        self.setText("Developed image will appear here after analysis")

    def set_from_file(self, path: str) -> None:
        px = QPixmap(path)
        if px.isNull():
            self._pixmap_src = None
            self.setText(f"Cannot load image:\n{path}")
        else:
            self._pixmap_src = px
            self._orig_w = max(1, px.width())
            self._orig_h = max(1, px.height())
            self._inner.setText("")
            self._redraw()

    def clear(self) -> None:
        self._pixmap_src = None
        self._inner.setPixmap(QPixmap())
        self._reset_text()


# ─────────────────────────────────────────────────────────────────────────────
# Operation pipeline (palette + ordered drop target)
# ─────────────────────────────────────────────────────────────────────────────

# Single source of truth for op ids, mirrored on the backend by
# src.utils.CORRECTION_LABELS and the --pipeline op id set. Preprocessing ops
# always run before correction ops (their intra-group order is fixed by the
# maths); the pipeline governs *which* preprocessing ops run and the full
# correction-stage order.
PREPROCESSING_OPS = ("undistort", "devignetting", "denoise", "sharpen")
# Correction ops split by linearity for the palette. Strictly-linear 3x3 models
# first, then locally-linear ones; the genuinely non-linear models form their
# own group. `CORRECTION_OPS` keeps the combined set for membership checks
# (order is irrelevant there).
CORRECTION_LINEAR_OPS = ("baseline", "wiener", "pca", "de00_opt", "hppcc", "hlcc", "lwcc")
CORRECTION_NONLINEAR_OPS = ("rpcc", "rpcc_ridge", "hppcc_rpcc", "tps")
CORRECTION_OPS = CORRECTION_LINEAR_OPS + CORRECTION_NONLINEAR_OPS
ALL_OPS = PREPROCESSING_OPS + CORRECTION_OPS
OP_LABELS = {
    "undistort": "Undistort",
    "devignetting": "Devignetting",
    "denoise": "Denoise",
    "sharpen": "Sharpen",
    "hppcc_rpcc": "hppcc_rpcc",
    "hppcc": "hppcc",
    "baseline": "baseline",
    "rpcc": "rpcc",
    "rpcc_ridge": "rpcc_ridge",
    "hlcc": "hlcc",
    "tps": "tps",
    "lwcc": "lwcc",
    "de00_opt": "de00_opt",
    "wiener": "wiener",
    "pca": "pca",
}
LABEL_TO_OP = {label: op for op, label in OP_LABELS.items()}

# Per-op default parameters. The keys mirror the backend's stage params
# (src.utils.fit_correction_stage / cc.py preprocessing). Ops absent here
# (baseline, rpcc, de00_opt) have no tunable parameters, so they get no gear.
OP_DEFAULTS: dict[str, dict] = {
    "undistort": {"method": "lensfun"},
    "devignetting": {"method": "zheng"},
    "denoise": {"method": "wavelet", "strength": 6.0, "diameter": 5, "sigma_space": 2.0},
    "sharpen": {"amount": 0.6, "radius": 1.0, "threshold": 1.5},
    "hppcc": {
        "k_regions": 4, "optimize_boundaries": True, "region_smoothness": 0.0,
        "use_blending": True, "blend_width": 0.15, "gradient": False, "gradient_harmonics": 2,
    },
    "hppcc_rpcc": {
        "k_regions": 4, "optimize_boundaries": True, "region_smoothness": 0.0,
        "use_blending": True, "blend_width": 0.15, "gradient": False, "gradient_harmonics": 2,
    },
    "rpcc_ridge": {"lambda_ridge": 1e-3},
    "hlcc": {"k_sectors": 4},
    "tps": {"smoothing": 0.0},
    "lwcc": {"bandwidth": None},   # None == auto
    "wiener": {"snr": 100.0},
    "pca": {"n_components": 3},
}
OP_HAS_PARAMS = set(OP_DEFAULTS)


def _make_op_item(op: str) -> QListWidgetItem:
    item = QListWidgetItem(OP_LABELS.get(op, op))
    item.setData(Qt.ItemDataRole.UserRole, op)
    return item


class OperationPalette(QWidget):
    """Two stacked drag-only lists: Preprocessing and Correction.

    Each item stores its op id in ``Qt.ItemDataRole.UserRole``. The lists are
    drag-only and copy on drag, so the source items stay put while a copy is
    dropped into the :class:`PipelineList`.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        v.addWidget(QLabel("<b>Operations</b>"))
        v.addWidget(QLabel("Preprocessing"))
        self._pre = self._make_list(PREPROCESSING_OPS)
        v.addWidget(self._pre)
        v.addWidget(QLabel("Linear / locally linear corrections"))
        self._corr_linear = self._make_list(CORRECTION_LINEAR_OPS)
        v.addWidget(self._corr_linear, 1)
        v.addWidget(QLabel("Non-linear corrections"))
        self._corr_nonlinear = self._make_list(CORRECTION_NONLINEAR_OPS)
        v.addWidget(self._corr_nonlinear, 1)

    @staticmethod
    def _make_list(ops: tuple[str, ...]) -> QListWidget:
        lst = QListWidget()
        lst.setDragEnabled(True)
        lst.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        lst.setDefaultDropAction(Qt.DropAction.CopyAction)
        lst.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for op in ops:
            lst.addItem(_make_op_item(op))
        return lst


class _PipelineRow(QWidget):
    """Row widget for a pipeline entry: status icon + op label + ΔE info +
    (optional) gear button."""

    def __init__(self, op: str, on_gear) -> None:
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(6, 2, 6, 2)
        h.setSpacing(6)
        self._status = QLabel("")
        self._status.setFixedWidth(16)
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h.addWidget(self._status)
        h.addWidget(QLabel(OP_LABELS.get(op, op)))
        self._info = QLabel("")
        self._info.setStyleSheet("color:#9aa0a6;")
        h.addWidget(self._info, 1)
        if op in OP_HAS_PARAMS:
            gear = QToolButton()
            gear.setText("⚙")   # ⚙
            gear.setAutoRaise(True)
            gear.setCursor(Qt.CursorShape.PointingHandCursor)
            gear.setToolTip("Edit settings for this step")
            gear.clicked.connect(on_gear)
            h.addWidget(gear)

    def set_status(self, icon: str, color: str = "", tooltip: str = "") -> None:
        self._status.setText(icon)
        self._status.setStyleSheet(f"color:{color};" if color else "")
        self._status.setToolTip(tooltip)

    def set_info(self, text: str) -> None:
        self._info.setText(text)


class PipelineList(QListWidget):
    """Ordered drop target with per-instance operation settings.

    A Python list of ``{"op", "params"}`` entries is the single source of truth;
    each row renders the op label plus a gear button (for ops with tunable
    params) that opens its settings dialog. Accepts copies dropped from the
    palette and supports internal drag reordering; Delete removes the selected
    entry and double-click opens its settings. Duplicates are allowed.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setToolTip(
            "Drag operations here and reorder them. Correction steps run "
            "top-to-bottom as a residual cascade; preprocessing always runs "
            "first regardless of position. Click the gear to edit a step's "
            "settings; press Delete to remove it."
        )
        self._entries: list[dict] = []
        self._raw_path_provider = None   # set by AnalyzeTab; feeds Undistort dialog
        self.itemDoubleClicked.connect(lambda item: self._open_settings(self.row(item)))

    # ── source of truth ─────────────────────────────────────────────────────
    def ops(self) -> list[str]:
        return [e["op"] for e in self._entries]

    def entries(self) -> list[dict]:
        return [{"op": e["op"], "params": deepcopy(e.get("params", {}))} for e in self._entries]

    def set_entries(self, entries) -> None:
        self._entries = []
        for e in (entries or []):
            if isinstance(e, str):          # legacy persisted format (op ids only)
                op, params = e, {}
            elif isinstance(e, dict) and e.get("op"):
                op, params = e["op"], dict(e.get("params") or {})
            else:
                continue
            if op in ALL_OPS:
                merged = deepcopy(OP_DEFAULTS.get(op, {}))
                merged.update(params)
                self._entries.append({"op": op, "params": merged})
        self._rebuild()

    # ``set_ops`` kept for call sites that only deal in op ids.
    def set_ops(self, ops) -> None:
        self.set_entries(list(ops or []))

    def clear(self) -> None:
        self._entries = []
        super().clear()

    # ── per-step run status (driven by the analysis subprocess) ──────────────
    def _row_widget(self, index: int):
        item = self.item(index)
        return self.itemWidget(item) if item is not None else None

    def mark_all_pending(self) -> None:
        """Put an hourglass on every step before a run starts."""
        for i in range(self.count()):
            w = self._row_widget(i)
            if w is not None:
                w.set_status("⏳", "#e0a000", "Pending")
                w.set_info("")

    def clear_status(self) -> None:
        for i in range(self.count()):
            w = self._row_widget(i)
            if w is not None:
                w.set_status("")
                w.set_info("")

    def set_step_status(self, index: int, status: str, *, mean=None, median=None,
                        reason: str = "", effective: str = "") -> None:
        w = self._row_widget(index)
        if w is None:
            return
        if status == "ok":
            tip = f"Applied as {effective}" if effective else "Applied"
            w.set_status("✓", "#2ecc71", tip)
            if mean is not None and median is not None:
                # ΔE₀₀ with Greek delta and subscript zeros.
                w.set_info(f"ΔE₀₀ mean {mean:.2f} / median {median:.2f}")
        elif status == "skipped":
            w.set_status("❗", "#e74c3c", reason or "Skipped")
            w.set_info(reason or "skipped")

    # ── rendering ────────────────────────────────────────────────────────────
    def _rebuild(self, select: int = -1) -> None:
        self.blockSignals(True)
        QListWidget.clear(self)   # clear the view only; keep self._entries
        for i, entry in enumerate(self._entries):
            op = entry["op"]
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, op)
            self.addItem(item)
            row = _PipelineRow(op, lambda _checked=False, idx=i: self._open_settings(idx))
            item.setSizeHint(row.sizeHint())
            self.setItemWidget(item, row)
        self.blockSignals(False)
        if 0 <= select < self.count():
            self.setCurrentRow(select)

    def _open_settings(self, idx: int) -> None:
        if not (0 <= idx < len(self._entries)):
            return
        entry = self._entries[idx]
        dlg_cls = OP_DIALOGS.get(entry["op"])
        if dlg_cls is None:
            return
        raw_path = self._raw_path_provider() if self._raw_path_provider else ""
        dlg = dlg_cls(entry.get("params", {}), parent=self, raw_path=raw_path)
        if dlg.exec():
            entry["params"] = dlg.get_settings()

    # ── drag & drop ──────────────────────────────────────────────────────────
    @staticmethod
    def _drag_ok(src) -> bool:
        return isinstance(src, QListWidget)

    def dragEnterEvent(self, event) -> None:
        if self._drag_ok(event.source()):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if self._drag_ok(event.source()):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def _drop_row(self, event) -> int:
        idx = self.indexAt(event.position().toPoint())
        if not idx.isValid():
            return len(self._entries)
        row = idx.row()
        if self.dropIndicatorPosition() == QAbstractItemView.DropIndicatorPosition.BelowItem:
            row += 1
        return row

    def dropEvent(self, event) -> None:
        src = event.source()
        target = self._drop_row(event)
        if src is self:
            source_row = self.currentRow()
            if not (0 <= source_row < len(self._entries)):
                event.ignore(); return
            entry = self._entries.pop(source_row)
            if source_row < target:
                target -= 1
            target = max(0, min(target, len(self._entries)))
            self._entries.insert(target, entry)
            event.accept()
            self._rebuild(select=target)
        elif isinstance(src, QListWidget):
            item = src.currentItem()
            if item is None:
                event.ignore(); return
            op = item.data(Qt.ItemDataRole.UserRole) or LABEL_TO_OP.get(item.text(), item.text())
            if op not in ALL_OPS:
                event.ignore(); return
            target = max(0, min(target, len(self._entries)))
            self._entries.insert(target, {"op": op, "params": deepcopy(OP_DEFAULTS.get(op, {}))})
            event.accept()
            self._rebuild(select=target)
        else:
            event.ignore()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            row = self.currentRow()
            if 0 <= row < len(self._entries):
                self._entries.pop(row)
                self._rebuild(select=min(row, len(self._entries) - 1))
            return
        super().keyPressEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Denoise settings dialog
# ─────────────────────────────────────────────────────────────────────────────

class OpSettingsDialog(QDialog):
    """Base for per-instance op settings dialogs.

    Subclasses set ``OP``/``TITLE`` and implement ``_build(form)`` (add field
    rows), ``_load(params)`` (populate widgets) and ``get_settings()``. The base
    wires OK / Cancel plus a **Revert to default** button that reloads
    ``OP_DEFAULTS[OP]`` into the fields.
    """

    OP = ""
    TITLE = "Settings"
    MIN_WIDTH = 340

    def __init__(self, params: dict | None = None, parent: QWidget | None = None,
                 raw_path: str = "") -> None:
        super().__init__(parent)
        self._raw_path = raw_path
        self.setWindowTitle(self.TITLE)
        self.setMinimumWidth(self.MIN_WIDTH)

        form = QFormLayout()
        form.setSpacing(8)
        self._build(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.RestoreDefaults
            | QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        revert = buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults)
        revert.setText("Revert to default")
        revert.clicked.connect(lambda: self._load(dict(OP_DEFAULTS.get(self.OP, {}))))

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(buttons)

        merged = dict(OP_DEFAULTS.get(self.OP, {}))
        if params:
            merged.update(params)
        self._load(merged)

    # ── subclass hooks ──────────────────────────────────────────────────────
    def _build(self, form: QFormLayout) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _load(self, params: dict) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def get_settings(self) -> dict:  # pragma: no cover - overridden
        raise NotImplementedError


class DenoiseSettingsDialog(OpSettingsDialog):
    OP = "denoise"
    TITLE = "Denoise settings"

    def _build(self, form: QFormLayout) -> None:
        self._method = QComboBox()
        self._method.addItems(["wavelet", "bilateral", "adaptive_bilateral"])
        form.addRow("Method:", self._method)
        self._strength = QDoubleSpinBox()
        self._strength.setRange(0.1, 50.0); self._strength.setSingleStep(0.5); self._strength.setDecimals(1)
        form.addRow("Strength:", self._strength)
        self._diameter = QSpinBox(); self._diameter.setRange(1, 30)
        form.addRow("Diameter (bilateral):", self._diameter)
        self._sigma_space = QDoubleSpinBox()
        self._sigma_space.setRange(0.1, 20.0); self._sigma_space.setSingleStep(0.1); self._sigma_space.setDecimals(1)
        form.addRow("Sigma space (bilateral):", self._sigma_space)

    def _load(self, p: dict) -> None:
        self._method.setCurrentText(str(p.get("method", "wavelet")))
        self._strength.setValue(float(p.get("strength", 6.0)))
        self._diameter.setValue(int(p.get("diameter", 5)))
        self._sigma_space.setValue(float(p.get("sigma_space", 2.0)))

    def get_settings(self) -> dict:
        return {
            "method": self._method.currentText(),
            "strength": self._strength.value(),
            "diameter": self._diameter.value(),
            "sigma_space": self._sigma_space.value(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Sharpen settings dialog
# ─────────────────────────────────────────────────────────────────────────────

class SharpenSettingsDialog(OpSettingsDialog):
    OP = "sharpen"
    TITLE = "Sharpen settings"

    def _build(self, form: QFormLayout) -> None:
        self._amount = QDoubleSpinBox()
        self._amount.setRange(0.0, 5.0); self._amount.setSingleStep(0.1); self._amount.setDecimals(2)
        form.addRow("Amount:", self._amount)
        self._radius = QDoubleSpinBox()
        self._radius.setRange(0.1, 5.0); self._radius.setSingleStep(0.1); self._radius.setDecimals(2)
        form.addRow("Radius (px):", self._radius)
        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.0, 10.0); self._threshold.setSingleStep(0.1); self._threshold.setDecimals(2)
        form.addRow("Threshold (× σ):", self._threshold)

    def _load(self, p: dict) -> None:
        self._amount.setValue(float(p.get("amount", 0.6)))
        self._radius.setValue(float(p.get("radius", 1.0)))
        self._threshold.setValue(float(p.get("threshold", 1.5)))

    def get_settings(self) -> dict:
        return {
            "amount": self._amount.value(),
            "radius": self._radius.value(),
            "threshold": self._threshold.value(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# HPPCC settings dialog
# ─────────────────────────────────────────────────────────────────────────────

class HPPCCSettingsDialog(OpSettingsDialog):
    OP = "hppcc"
    TITLE = "HPPCC settings"
    MIN_WIDTH = 360

    def _build(self, form: QFormLayout) -> None:
        self._k_regions = QSpinBox(); self._k_regions.setRange(2, 8)
        self._k_regions.setToolTip("Number of hue regions for the piecewise-linear fit.")
        form.addRow("Hue regions (k):", self._k_regions)

        self._optimize = QCheckBox("Optimize region boundaries")
        form.addRow("", self._optimize)

        self._region_smoothness = QDoubleSpinBox()
        self._region_smoothness.setRange(0.0, 1.0); self._region_smoothness.setSingleStep(0.005)
        self._region_smoothness.setDecimals(3)
        self._region_smoothness.setToolTip(
            "Couples adjacent hue-region matrices during analysis.\n"
            "Higher = smoother hue transitions but flatter colour detail; "
            "0 disables it. Applied when analysing a chart."
        )
        form.addRow("Region smoothness:", self._region_smoothness)

        self._use_blending = QCheckBox("Blend across hue-region boundaries")
        form.addRow("", self._use_blending)
        self._blend_width = QDoubleSpinBox()
        self._blend_width.setRange(0.01, 1.0); self._blend_width.setSingleStep(0.05); self._blend_width.setDecimals(2)
        form.addRow("Blend width:", self._blend_width)

        self._gradient = QCheckBox("Use gradient (Fourier) HPPCC")
        self._gradient.setToolTip(
            "Fit a trigonometric-series HPPCC instead of the piecewise-constrained "
            "model (region count / smoothness are ignored when enabled)."
        )
        form.addRow("", self._gradient)
        self._harmonics = QSpinBox(); self._harmonics.setRange(1, 8)
        form.addRow("Gradient harmonics:", self._harmonics)

    def _load(self, p: dict) -> None:
        self._k_regions.setValue(int(p.get("k_regions", 4)))
        self._optimize.setChecked(bool(p.get("optimize_boundaries", True)))
        self._region_smoothness.setValue(float(p.get("region_smoothness", 0.0)))
        self._use_blending.setChecked(bool(p.get("use_blending", True)))
        self._blend_width.setValue(float(p.get("blend_width", 0.15)))
        self._gradient.setChecked(bool(p.get("gradient", False)))
        self._harmonics.setValue(int(p.get("gradient_harmonics", 2)))

    def get_settings(self) -> dict:
        return {
            "k_regions": self._k_regions.value(),
            "optimize_boundaries": self._optimize.isChecked(),
            "region_smoothness": self._region_smoothness.value(),
            "use_blending": self._use_blending.isChecked(),
            "blend_width": self._blend_width.value(),
            "gradient": self._gradient.isChecked(),
            "gradient_harmonics": self._harmonics.value(),
        }


class HPPCCRPCCSettingsDialog(HPPCCSettingsDialog):
    OP = "hppcc_rpcc"
    TITLE = "HPPCC+RPCC settings"


class DevignettingSettingsDialog(OpSettingsDialog):
    OP = "devignetting"
    TITLE = "Devignetting settings"
    MIN_WIDTH = 360

    def _build(self, form: QFormLayout) -> None:
        self._method = QComboBox()
        self._method.addItems(["zheng", "goldman", "kim"])
        self._method.setToolTip(
            "zheng: Zheng et al. (2009), radial consistency (Default)\n"
            "goldman: Goldman (2010), gradient distribution symmetry\n"
            "kim: Kim & Pollefeys (2008), radiometric calibration"
        )
        form.addRow("Method:", self._method)

    def _load(self, p: dict) -> None:
        self._method.setCurrentText(str(p.get("method", "zheng")))

    def get_settings(self) -> dict:
        return {"method": self._method.currentText()}


class UndistortSettingsDialog(OpSettingsDialog):
    OP = "undistort"
    TITLE = "Undistort settings"
    MIN_WIDTH = 380

    def _build(self, form: QFormLayout) -> None:
        self._method = QComboBox()
        self._method.addItems(["lensfun", "devernay", "aleman"])
        self._method.setToolTip(
            "lensfun: Use Lensfun database (Default)\n"
            "devernay: Devernay and Faugeras (2001)\n"
            "aleman: Aleman-Flores et al. (2014)"
        )
        form.addRow("Method:", self._method)

        self._check_btn = QPushButton("Check Lensfun DB")
        self._check_btn.setToolTip("Check whether the loaded RAW's camera/lens is in the Lensfun database.")
        self._check_btn.clicked.connect(self._check_lensfun)
        form.addRow("", self._check_btn)

    def _load(self, p: dict) -> None:
        self._method.setCurrentText(str(p.get("method", "lensfun")))

    def get_settings(self) -> dict:
        return {"method": self._method.currentText()}

    def _check_lensfun(self) -> None:
        path = (self._raw_path or "").strip()
        if not path or not Path(path).is_file():
            QMessageBox.information(self, "Check Lensfun DB", "Please load a valid RAW image first.")
            return
        from src.lens import is_lens_in_db
        found, msg = is_lens_in_db(Path(path))
        title = "Lensfun DB - Found" if found else "Lensfun DB - Not Found"
        QMessageBox.information(self, title, msg)


class RPCCRidgeSettingsDialog(OpSettingsDialog):
    OP = "rpcc_ridge"
    TITLE = "Ridge-RPCC settings"

    def _build(self, form: QFormLayout) -> None:
        self._lambda = QDoubleSpinBox()
        self._lambda.setRange(0.0, 1.0); self._lambda.setDecimals(6); self._lambda.setSingleStep(1e-4)
        self._lambda.setToolTip("Tikhonov (ridge) regularisation strength for the root-polynomial fit.")
        form.addRow("Ridge lambda:", self._lambda)

    def _load(self, p: dict) -> None:
        self._lambda.setValue(float(p.get("lambda_ridge", 1e-3)))

    def get_settings(self) -> dict:
        return {"lambda_ridge": self._lambda.value()}


class HLCCSettingsDialog(OpSettingsDialog):
    OP = "hlcc"
    TITLE = "HLCC settings"

    def _build(self, form: QFormLayout) -> None:
        self._k_sectors = QSpinBox(); self._k_sectors.setRange(2, 12)
        self._k_sectors.setToolTip("Number of hue sectors for the hue-linear fit.")
        form.addRow("Hue sectors (k):", self._k_sectors)

    def _load(self, p: dict) -> None:
        self._k_sectors.setValue(int(p.get("k_sectors", 4)))

    def get_settings(self) -> dict:
        return {"k_sectors": self._k_sectors.value()}


class TPSSettingsDialog(OpSettingsDialog):
    OP = "tps"
    TITLE = "TPS settings"

    def _build(self, form: QFormLayout) -> None:
        self._smoothing = QDoubleSpinBox()
        self._smoothing.setRange(0.0, 10.0); self._smoothing.setDecimals(4); self._smoothing.setSingleStep(0.001)
        self._smoothing.setToolTip("Thin-plate-spline smoothing (0 = exact interpolation through the patches).")
        form.addRow("Smoothing:", self._smoothing)

    def _load(self, p: dict) -> None:
        self._smoothing.setValue(float(p.get("smoothing", 0.0)))

    def get_settings(self) -> dict:
        return {"smoothing": self._smoothing.value()}


class LWCCSettingsDialog(OpSettingsDialog):
    OP = "lwcc"
    TITLE = "LWCC settings"

    def _build(self, form: QFormLayout) -> None:
        self._auto = QCheckBox("Auto bandwidth")
        self._auto.toggled.connect(lambda on: self._bandwidth.setEnabled(not on))
        form.addRow("", self._auto)
        self._bandwidth = QDoubleSpinBox()
        self._bandwidth.setRange(0.01, 5.0); self._bandwidth.setDecimals(3); self._bandwidth.setSingleStep(0.05)
        self._bandwidth.setToolTip("Kernel bandwidth for locally-weighted regression. Uncheck Auto to set it.")
        form.addRow("Bandwidth:", self._bandwidth)

    def _load(self, p: dict) -> None:
        bw = p.get("bandwidth", None)
        is_auto = bw in (None, "", "auto")
        self._auto.setChecked(is_auto)
        self._bandwidth.setEnabled(not is_auto)
        self._bandwidth.setValue(0.5 if is_auto else float(bw))

    def get_settings(self) -> dict:
        if self._auto.isChecked():
            return {"bandwidth": None}
        return {"bandwidth": self._bandwidth.value()}


class WienerSettingsDialog(OpSettingsDialog):
    OP = "wiener"
    TITLE = "Wiener settings"

    def _build(self, form: QFormLayout) -> None:
        self._snr = QDoubleSpinBox()
        self._snr.setRange(1.0, 10000.0); self._snr.setDecimals(1); self._snr.setSingleStep(10.0)
        self._snr.setToolTip("Assumed signal-to-noise ratio for the Wiener-regularised linear fit.")
        form.addRow("SNR:", self._snr)

    def _load(self, p: dict) -> None:
        self._snr.setValue(float(p.get("snr", 100.0)))

    def get_settings(self) -> dict:
        return {"snr": self._snr.value()}


class PCASettingsDialog(OpSettingsDialog):
    OP = "pca"
    TITLE = "PCA settings"

    def _build(self, form: QFormLayout) -> None:
        self._n = QSpinBox(); self._n.setRange(1, 3)
        self._n.setToolTip("Number of PCA components retained for the spectral linear fit.")
        form.addRow("Components:", self._n)

    def _load(self, p: dict) -> None:
        self._n.setValue(int(p.get("n_components", 3)))

    def get_settings(self) -> dict:
        return {"n_components": self._n.value()}


# op id -> settings dialog class (only ops with tunable params appear here).
OP_DIALOGS: dict[str, type] = {
    "undistort": UndistortSettingsDialog,
    "devignetting": DevignettingSettingsDialog,
    "denoise": DenoiseSettingsDialog,
    "sharpen": SharpenSettingsDialog,
    "hppcc": HPPCCSettingsDialog,
    "hppcc_rpcc": HPPCCRPCCSettingsDialog,
    "rpcc_ridge": RPCCRidgeSettingsDialog,
    "hlcc": HLCCSettingsDialog,
    "tps": TPSSettingsDialog,
    "lwcc": LWCCSettingsDialog,
    "wiener": WienerSettingsDialog,
    "pca": PCASettingsDialog,
}


# ─────────────────────────────────────────────────────────────────────────────
# Help — How to (renders README.md) and About dialogs
# ─────────────────────────────────────────────────────────────────────────────

class HowToDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("How to — coco2")
        self.resize(820, 620)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(True)
        from src.config import BUNDLE_DIR
        readme_path = BUNDLE_DIR / "README.md"
        if readme_path.is_file():
            try:
                self._browser.setMarkdown(readme_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self._browser.setPlainText(f"Could not render README.md: {exc}")
        else:
            self._browser.setPlainText(
                "README.md is not bundled with this build. See the project repository."
            )
        layout.addWidget(self._browser, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        # The Close standard button is mapped to RejectRole; both rejected/accepted close.
        buttons.clicked.connect(lambda _btn: self.accept())
        layout.addWidget(buttons)


class AboutDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About coco2")
        self.setFixedSize(420, 340)

        from src import __version__
        from src.config import BUNDLE_DIR

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        icon_label = QLabel()
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_path = BUNDLE_DIR / "assets" / "cocoicobn.png"
        if icon_path.is_file():
            pixmap = QPixmap(str(icon_path)).scaled(
                96, 96,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            icon_label.setPixmap(pixmap)
        layout.addWidget(icon_label)

        text = QLabel(
            f"<div style='text-align:center'>"
            f"<h2 style='margin:0'>coco2</h2>"
            f"<p style='margin:4px 0; color:#aaa'>Color Correction Tool</p>"
            f"<p style='margin:6px 0'><b>Version {__version__}</b></p>"
            f"<p style='margin:10px 0; font-size:11px'>"
            f"HPPCC + RPCC color calibration from X-Rite<br>"
            f"ColorChecker Classic 24, with RAW decoding,<br>"
            f"highlight recovery, denoise, sharpening and<br>"
            f"vignetting correction."
            f"</p>"
            f"</div>"
        )
        text.setTextFormat(Qt.TextFormat.RichText)
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text.setWordWrap(True)
        layout.addWidget(text)

        layout.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


# ─────────────────────────────────────────────────────────────────────────────
# Analyze tab
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeTab(QWidget):
    correction_ready = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._raw_path = ""
        self._roi: tuple[int, int, int, int] | None = None
        self._loader: RawPreviewLoader | None = None
        self._proc: QProcess | None = None
        self._out_buf = ""   # line buffer for parsing @@STEP@@/@@PIPELINE@@ markers
        self._divergences: list[str] = []   # steps that ran differently than planned
        self._applied_pipeline_text = ""    # formatted summary of what actually ran
        # Injected by MainWindow; encodes the ordered pipeline plus all
        # method/param flags (see MainWindow.get_pipeline_args).
        self.pipeline_args_provider: callable = lambda: []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # ── file row (always visible, outside splitter) ───────────────────
        file_row = QHBoxLayout()
        self._raw_edit = QLineEdit()
        self._raw_edit.setPlaceholderText("RAW or developed image with color checker...")
        self._raw_edit.textChanged.connect(self._on_path_changed)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_raw)
        file_row.addWidget(QLabel("RAW image:"))
        file_row.addWidget(self._raw_edit, 1)
        file_row.addWidget(browse)
        root.addLayout(file_row)

        # ── output folder row (always visible, outside splitter) ──────────
        out_row = QHBoxLayout()
        self._output_dir = QLineEdit(str(PROJECT_ROOT / "output"))
        self._output_dir.setPlaceholderText("Output folder for all analysis results...")
        browse_out = QPushButton("Browse...")
        browse_out.clicked.connect(lambda: self._pick_dir(self._output_dir))
        out_row.addWidget(QLabel("Output folder:"))
        out_row.addWidget(self._output_dir, 1)
        out_row.addWidget(browse_out)
        root.addLayout(out_row)

        # ── outer vertical splitter ───────────────────────────────────────
        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.setChildrenCollapsible(False)

        # ── Panel 1: horizontal splitter — RAW preview | developed preview ─
        hsplit = QSplitter(Qt.Orientation.Horizontal)
        hsplit.setChildrenCollapsible(False)

        # Left — RAW preview + ROI row
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(4)
        self._preview = ImagePreview()
        self._preview.roi_changed.connect(self._on_roi_changed)
        self._preview.roi_cleared.connect(self._on_roi_cleared)
        self._preview.file_dropped.connect(self._set_raw)
        lv.addWidget(self._preview)
        roi_row = QHBoxLayout()
        roi_row.addWidget(QLabel("ROI (x1,y1,x2,y2):"))
        self._roi_edit = QLineEdit()
        self._roi_edit.setPlaceholderText("Draw on the preview above or type manually")
        self._roi_edit.textEdited.connect(self._on_roi_text)
        clear_roi = QPushButton("Clear ROI")
        clear_roi.clicked.connect(self._preview.clear_roi)
        roi_row.addWidget(self._roi_edit, 1)
        roi_row.addWidget(clear_roi)
        lv.addLayout(roi_row)
        hsplit.addWidget(left)

        # Middle-left — draggable operation palette
        self._palette = OperationPalette()
        hsplit.addWidget(self._palette)

        # Middle-right — ordered pipeline (drop target) + Clear button
        pipe_w = QWidget()
        pipe_v = QVBoxLayout(pipe_w)
        pipe_v.setContentsMargins(4, 0, 4, 0)
        pipe_v.setSpacing(4)
        pipe_v.addWidget(QLabel("<b>Pipeline</b>"))
        self._pipeline = PipelineList()
        # The Undistort settings dialog's "Check Lensfun DB" button needs the
        # currently loaded RAW path.
        self._pipeline._raw_path_provider = lambda: self._raw_path
        # Default mirrors prior defaults (undistort on, hppcc_rpcc output);
        # overridden by restored state when a saved pipeline exists.
        self._pipeline.set_ops(["undistort", "hppcc_rpcc"])
        pipe_v.addWidget(self._pipeline, 1)
        clear_pipe = QPushButton("Clear")
        clear_pipe.clicked.connect(self._pipeline.clear)
        pipe_v.addWidget(clear_pipe)
        hsplit.addWidget(pipe_w)

        # Right — "Show developed" checkbox + developed preview
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 0, 0, 0)
        rv.setSpacing(4)
        self._show_developed = QCheckBox("Show developed image after analysis")
        self._show_developed.setChecked(True)
        rv.addWidget(self._show_developed)
        self._developed = DevelopedPreview()
        rv.addWidget(self._developed)
        hsplit.addWidget(right)

        # Synchronize zoom/scroll between RAW and developed previews
        self._preview.zoom_changed.connect(self._developed.set_zoom_external)
        self._preview.scroll_changed.connect(self._developed.set_scroll_external)
        self._developed.zoom_changed.connect(self._preview.set_zoom_external)
        self._developed.scroll_changed.connect(self._preview.set_scroll_external)

        hsplit.setSizes([440, 150, 190, 420])
        vsplit.addWidget(hsplit)

        # ── Panel 2: parameters + run button ─────────────────────────────
        panel2 = QWidget()
        p2 = QVBoxLayout(panel2)
        p2.setContentsMargins(0, 4, 0, 0)
        p2.setSpacing(6)

        # White field reference (vignetting correction) — must precede the
        # format/colorspace row so the path field stays visible on resize.
        white_row = QHBoxLayout()
        self._white_field = QCheckBox("Process white field")
        self._white_field.setChecked(False)
        self._white_field.toggled.connect(self._on_white_field_toggled)
        white_row.addWidget(self._white_field)
        self._white_field_path = QLineEdit()
        self._white_field_path.setPlaceholderText("Path to white reference RAW (required if checked)...")
        self._white_field_path.setEnabled(False)
        white_row.addWidget(self._white_field_path, 1)
        self._white_field_browse = QPushButton("Browse...")
        self._white_field_browse.setEnabled(False)
        self._white_field_browse.clicked.connect(self._browse_white_field)
        white_row.addWidget(self._white_field_browse)
        p2.addLayout(white_row)

        # Format / colorspace / output method options
        opts = QHBoxLayout()
        opts.addWidget(QLabel("Format:"))
        self._fmt = QComboBox(); self._fmt.addItems(["jpeg", "tif", "png"])
        opts.addWidget(self._fmt)
        opts.addSpacing(12)
        opts.addWidget(QLabel("Color space:"))
        self._cs = QComboBox(); self._cs.addItems(["sRGB", "Display-P3"])
        opts.addWidget(self._cs)
        opts.addStretch()
        p2.addLayout(opts)

        # Pre-processing and correction are now arranged in the visual pipeline
        # (palette → ordered list above). Method/parameter settings for each op
        # still live in the menu bar (Undistort/Devignetting/Denoise/Sharpen/
        # HPPCC → Settings…).

        run_row = QHBoxLayout()
        run_row.setSpacing(8)
        self._led = QLabel()
        self._led.setFixedSize(14, 14)
        self._set_led_idle()
        run_row.addWidget(self._led)
        self._run_btn = QPushButton("▶  Run analysis")
        self._run_btn.setFixedHeight(34)
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)
        run_row.addWidget(self._run_btn, 1)
        p2.addLayout(run_row)
        p2.addStretch()
        vsplit.addWidget(panel2)

        # ── Panel 3: log ──────────────────────────────────────────────────
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        vsplit.addWidget(self._log)

        vsplit.setSizes([520, 160, 180])
        root.addWidget(vsplit)

    # ── helpers ───────────────────────────────────────────────────────────

    def _browse_raw(self) -> None:
        current = self._raw_edit.text().strip()
        start_dir = str(Path(current).parent) if current and Path(current).exists() else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select input image", start_dir, _INPUT_FILE_FILTER,
        )
        if path:
            if not self._output_dir.text().strip():
                self._output_dir.setText(str(Path(path).parent / "out"))
            self._set_raw(path)

    def _set_raw(self, path: str) -> None:
        self._raw_edit.setText(path)

    def _on_path_changed(self, text: str) -> None:
        self._raw_path = text.strip()
        self._run_btn.setEnabled(bool(self._raw_path))
        p = Path(self._raw_path)
        if p.is_file():
            self._load_preview(str(p))

    def _load_preview(self, path: str) -> None:
        if self._loader and self._loader.isRunning():
            self._loader.terminate()
            self._loader.wait()
        self._preview.setText("Loading preview...")
        self._loader = RawPreviewLoader(path)
        self._loader.ready.connect(self._preview.set_from_array)
        self._loader.failed.connect(lambda e: self._preview.setText(f"Preview unavailable:\n{e}"))
        self._loader.start()

    def _on_roi_changed(self, x1, y1, x2, y2) -> None:
        self._roi = (x1, y1, x2, y2)
        self._roi_edit.setText(f"{x1},{y1},{x2},{y2}")
        self._preview.acknowledge_roi()

    def _on_roi_cleared(self) -> None:
        self._roi = None
        self._roi_edit.clear()

    def _on_roi_text(self, text: str) -> None:
        parts = text.replace(" ", "").split(",")
        if len(parts) == 4:
            try:
                x1, y1, x2, y2 = (int(p) for p in parts)
                if x2 > x1 and y2 > y1:
                    self._roi = (x1, y1, x2, y2)
                    self._preview.set_roi(self._roi)
                    return
            except ValueError:
                pass
        self._roi = None
        self._preview.set_roi(None)

    def _pick_dir(self, edit: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select folder", edit.text())
        if d:
            edit.setText(d)

    def _on_white_field_toggled(self, checked: bool) -> None:
        self._white_field_path.setEnabled(checked)
        self._white_field_browse.setEnabled(checked)

    def _set_led_idle(self) -> None:
        self._led.setStyleSheet(
            "background-color: #2ecc71; border: 1px solid #1e8449; border-radius: 7px;"
        )
        self._led.setToolTip("Idle")

    def _set_led_busy(self) -> None:
        self._led.setStyleSheet(
            "background-color: #e74c3c; border: 1px solid #922b21; border-radius: 7px;"
        )
        self._led.setToolTip("Analyzing")

    def _browse_white_field(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select white reference image", "", _INPUT_FILE_FILTER,
        )
        if path:
            self._white_field_path.setText(path)

    # ── run ───────────────────────────────────────────────────────────────

    def _run(self) -> None:
        if not self._raw_path:
            return
        if self._white_field.isChecked() and not self._white_field_path.text().strip():
            QMessageBox.warning(
                self, "Missing white field",
                "'Process white field' is enabled but no reference RAW was provided.\n"
                "Pick a white reference image or uncheck the option.",
            )
            return
        if not any(op in CORRECTION_OPS for op in self._pipeline.ops()):
            QMessageBox.warning(
                self, "Empty pipeline",
                "Add at least one correction operation to the pipeline before "
                "running — there is nothing to render otherwise.",
            )
            return
        self._log.clear()
        self._developed.clear()
        self._out_buf = ""
        self._divergences = []
        self._applied_pipeline_text = ""
        self._pipeline.mark_all_pending()
        self._run_btn.setEnabled(False)
        self._set_led_busy()

        try:
            out = self._output_dir.text().strip() or str(PROJECT_ROOT / "output")
            args = _cc_args(
                "analyze",
                "--cc-image", self._raw_path,
                "--analysis-dir", out,
                "--process-dir", out,
                "--output-format", self._fmt.currentText(),
                "--output-colorspace", self._cs.currentText(),
                "--no-show-detection-preview",
                "--no-show-developed-image-preview",
            )

            args += self.pipeline_args_provider()

            if self._white_field.isChecked():
                args += [
                    "--process-white-field",
                    "--white-field-image", self._white_field_path.text().strip(),
                ]
            else:
                args.append("--no-process-white-field")
            if self._roi:
                args += ["--roi", f"{self._roi[0]},{self._roi[1]},{self._roi[2]},{self._roi[3]}"]

            self._proc = QProcess(self)
            self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            self._proc.readyReadStandardOutput.connect(self._on_output)
            self._proc.finished.connect(self._on_done)
            self._proc.start(PYTHON, args)
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            QMessageBox.critical(self, "GUI Crash", f"An internal GUI error occurred:\n\n{err_msg}")
            self._run_btn.setEnabled(True)
            self._set_led_idle()


    def _on_output(self) -> None:
        raw = self._proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self._out_buf += raw
        while "\n" in self._out_buf:
            line, self._out_buf = self._out_buf.split("\n", 1)
            self._handle_line(line + "\n")

    def _handle_line(self, line: str) -> None:
        stripped = line.strip()
        if stripped.startswith("@@STEP@@"):
            self._apply_step_marker(stripped[len("@@STEP@@"):].strip())
            return
        if stripped.startswith("@@PIPELINE@@"):
            self._apply_pipeline_marker(stripped[len("@@PIPELINE@@"):].strip())
            return
        self._log.insertPlainText(line)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _apply_step_marker(self, payload: str) -> None:
        try:
            data = json.loads(payload)
        except ValueError:
            return
        status = str(data.get("status", ""))
        reason = str(data.get("reason", ""))
        effective = str(data.get("effective", ""))
        op = str(data.get("op", "")) or f"step {data.get('index', '?')}"
        self._pipeline.set_step_status(
            int(data.get("index", -1)),
            status,
            mean=data.get("mean_de00"),
            median=data.get("median_de00"),
            reason=reason,
            effective=effective,
        )
        # Track any departure from the planned pipeline so we can warn at the end.
        if status == "skipped":
            self._divergences.append(
                f"• '{op}' was not applied" + (f" — {reason}." if reason else ".")
            )
        elif status == "ok" and effective and effective != op:
            self._divergences.append(
                f"• '{op}' was substituted with '{effective}'"
                + (f" — {reason}." if reason else ".")
            )

    def _apply_pipeline_marker(self, payload: str) -> None:
        try:
            data = json.loads(payload)
        except ValueError:
            return
        applied = data.get("applied", [])
        if not applied:
            return

        def _fmt_val(v):
            if isinstance(v, float):
                return f"{v:g}"
            return str(v)

        lines = ["Applied pipeline (what was actually used):"]
        for i, step in enumerate(applied, 1):
            op = step.get("op", "")
            eff = step.get("effective", "")
            name = f"{op} → {eff}" if eff and eff != op else op
            params = step.get("params") or {}
            param_txt = ", ".join(f"{k}={_fmt_val(v)}" for k, v in params.items())
            line = f"  {i}. {name}"
            if param_txt:
                line += f"  [{param_txt}]"
            if step.get("mean_de00") is not None and step.get("median_de00") is not None:
                line += f"  →  ΔE₀₀ mean {step['mean_de00']:.2f} / median {step['median_de00']:.2f}"
            if step.get("reason"):
                line += f"  ({step['reason']})"
            lines.append(line)
        self._applied_pipeline_text = "\n".join(lines)
        self._log.append("\n" + self._applied_pipeline_text)

    def _on_done(self, code: int, _) -> None:
        # Flush any trailing partial line left in the buffer.
        if self._out_buf:
            self._handle_line(self._out_buf)
            self._out_buf = ""
        self._run_btn.setEnabled(True)
        self._set_led_idle()
        if code == 0:
            corr_path = ""
            for line in self._log.toPlainText().splitlines():
                if "_correction.json" in line and "written" in line.lower():
                    corr_path = line.split(":", 1)[-1].strip()
                    self.correction_ready.emit(corr_path)
                    break
            self._log.append("\n✔ Analysis complete.")
            if self._show_developed.isChecked():
                img_path = None
                for line in self._log.toPlainText().splitlines():
                    if "Analysis image written to:" in line:
                        img_path = Path(line.split(":", 1)[-1].strip())
                        break
                if img_path and img_path.exists():
                    self._developed.set_from_file(str(img_path))
                else:
                    self._developed.setText("Developed image not found.")
            # The analysis succeeded but the pipeline ran differently than the
            # user arranged (steps skipped or models substituted) — make it
            # obvious instead of silently producing a different result.
            if self._divergences:
                message = (
                    "The analysis completed, but some steps did not run as you "
                    "arranged them:\n\n"
                    + "\n".join(self._divergences)
                )
                if self._applied_pipeline_text:
                    message += "\n\n" + self._applied_pipeline_text
                QMessageBox.warning(self, "Pipeline executed differently than planned", message)
        else:
            log_text = self._log.toPlainText()
            if "No ColorChecker detected" in log_text:
                self._log.append(
                    "\n✖ Color checker not detected.\n"
                    "   Draw a ROI on the preview to indicate where the color checker is, then retry."
                )
                self._preview.request_roi()
                QMessageBox.information(
                    self,
                    "Color checker not detected",
                    "The color checker could not be found automatically.\n\n"
                    "Draw a rectangle on the RAW preview to indicate where the color checker is,\n"
                    "then run the analysis again.",
                )
            else:
                self._log.append(f"\n✖ Error (exit code {code}).")


# ─────────────────────────────────────────────────────────────────────────────
# Process tab
# ─────────────────────────────────────────────────────────────────────────────

class ProcessTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._processed_count = 0
        self._total_count = 0
        self._output_buffer = ""
        self._paused = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        form = QFormLayout()
        form.setSpacing(6)

        def path_row(placeholder: str, browse_slot) -> tuple[QLineEdit, QHBoxLayout]:
            edit = QLineEdit(); edit.setPlaceholderText(placeholder)
            btn = QPushButton("..."); btn.setFixedWidth(28); btn.clicked.connect(browse_slot)
            row = QHBoxLayout(); row.addWidget(edit); row.addWidget(btn)
            return edit, row

        self._corr_edit, cr = path_row("*_correction.json file...", self._browse_corr)
        form.addRow("Correction file:", cr)

        self._src_edit, sr = path_row("Source RAW folder...", self._browse_src)
        form.addRow("Source folder:", sr)

        self._out_edit, or_ = path_row("Output folder for developed images...", self._browse_out)
        form.addRow("Output folder:", or_)

        root.addLayout(form)

        opts = QHBoxLayout()
        self._recursive = QCheckBox("Include subdirectories")
        opts.addWidget(self._recursive)
        opts.addStretch()
        opts.addWidget(QLabel("Format:"))
        self._fmt = QComboBox(); self._fmt.addItems(["jpeg", "tif", "png"])
        opts.addWidget(self._fmt)
        opts.addSpacing(12)
        opts.addWidget(QLabel("Color space:"))
        self._cs = QComboBox(); self._cs.addItems(["sRGB", "Display-P3"])
        opts.addWidget(self._cs)
        opts.addSpacing(12)
        opts.addWidget(QLabel("Workers:"))
        self._workers = QSpinBox(); self._workers.setRange(1, 32); self._workers.setValue(4)
        opts.addWidget(self._workers)
        root.addLayout(opts)

        progress_row = QHBoxLayout()
        progress_row.setSpacing(6)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("%v / %m files (%p%)")
        self._progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress.setFixedHeight(20)
        self._progress.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        progress_row.addWidget(self._progress, 4)
        progress_row.addStretch(4)
        root.addLayout(progress_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        self._led = QLabel()
        self._led.setFixedSize(14, 14)
        self._set_led_idle()
        action_row.addWidget(self._led)
        self._run_btn = QPushButton("▶  Run processing")
        self._run_btn.setFixedHeight(34)
        self._run_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._run_btn.clicked.connect(self._run)
        action_row.addWidget(self._run_btn, 4)
        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setFixedHeight(34)
        self._stop_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        action_row.addWidget(self._stop_btn, 2)
        self._pause_btn = QPushButton("⏸  Pause")
        self._pause_btn.setFixedHeight(34)
        self._pause_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._pause_btn.setEnabled(False)
        self._pause_btn.clicked.connect(self._on_pause_toggle)
        if sys.platform != "win32":
            self._pause_btn.setToolTip("Pause/Resume is supported only on Windows in this build.")
        action_row.addWidget(self._pause_btn, 2)
        root.addLayout(action_row)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        root.addWidget(self._log)

    # ── LED helpers ────────────────────────────────────────────────────────

    def _set_led_idle(self) -> None:
        self._led.setStyleSheet(
            "background-color: #2ecc71; border: 1px solid #1e8449; border-radius: 7px;"
        )
        self._led.setToolTip("Idle")

    def _set_led_busy(self) -> None:
        self._led.setStyleSheet(
            "background-color: #e74c3c; border: 1px solid #922b21; border-radius: 7px;"
        )
        self._led.setToolTip("Processing")

    def set_correction(self, path: str) -> None:
        self._corr_edit.setText(path)

    def _browse_corr(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Correction file", "", "JSON (*.json);;All files (*)")
        if p:
            self._corr_edit.setText(p)

    def _browse_src(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Source folder", self._src_edit.text())
        if d:
            self._src_edit.setText(d)

    def _browse_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder", self._out_edit.text())
        if d:
            self._out_edit.setText(d)

    def _run(self) -> None:
        corr = self._corr_edit.text().strip()
        src = self._src_edit.text().strip()
        if not corr or not src:
            QMessageBox.warning(self, "Missing fields",
                                "Please specify both the correction file and the source folder.")
            return

        src_path = Path(src)
        recursive = self._recursive.isChecked()
        total = _count_raw_files(src_path, recursive)
        if total == 0:
            QMessageBox.warning(
                self, "No RAW files",
                f"No RAW files found in:\n{src}\n\n"
                "Check the source folder and the 'Include subdirectories' option.",
            )
            return

        self._log.clear()
        self._processed_count = 0
        self._total_count = total
        self._output_buffer = ""
        self._paused = False
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        self._set_led_busy()
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._pause_btn.setEnabled(sys.platform == "win32")
        self._pause_btn.setText("⏸  Pause")

        args = _cc_args(
            "process",
            corr, src,
            "--output-format", self._fmt.currentText(),
            "--output-colorspace", self._cs.currentText(),
            "--workers", str(self._workers.value()),
        )
        out = self._out_edit.text().strip()
        if out:
            args += ["--process-dir", out]
        if recursive:
            args.append("--recursive")

        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._on_output)
        self._proc.finished.connect(self._on_done)
        self._proc.start(PYTHON, args)

    def _on_stop(self) -> None:
        if self._proc is None or self._proc.state() == QProcess.ProcessState.NotRunning:
            return
        pid = int(self._proc.processId())
        if self._paused and sys.platform == "win32":
            # A suspended process can't be killed cleanly — wake it up first.
            resume_process_tree(pid)
            self._paused = False
        if sys.platform == "win32":
            kill_process_tree(pid)
        else:
            self._proc.kill()
        self._log.append("\n■ Processing stopped by user.")

    def _on_pause_toggle(self) -> None:
        if self._proc is None or self._proc.state() == QProcess.ProcessState.NotRunning:
            return
        if sys.platform != "win32":
            return
        pid = int(self._proc.processId())
        if self._paused:
            resume_process_tree(pid)
            self._paused = False
            self._pause_btn.setText("⏸  Pause")
            self._log.append("\n▶ Resumed.")
        else:
            suspend_process_tree(pid)
            self._paused = True
            self._pause_btn.setText("▶  Resume")
            self._log.append("\n⏸ Paused.")

    def _on_output(self) -> None:
        raw = self._proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self._log.insertPlainText(raw)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())
        self._output_buffer += raw
        while "\n" in self._output_buffer:
            line, self._output_buffer = self._output_buffer.split("\n", 1)
            if line.startswith("Processed:"):
                self._processed_count += 1
                self._progress.setValue(min(self._processed_count, self._total_count))

    def _on_done(self, code: int, _) -> None:
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._pause_btn.setEnabled(False)
        self._pause_btn.setText("⏸  Pause")
        self._paused = False
        self._set_led_idle()
        msg = "\n✔ Processing complete." if code == 0 else f"\n✖ Error (exit code {code})."
        self._log.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Batch tab — multiple (chart-image, target-folder) jobs run sequentially
# ─────────────────────────────────────────────────────────────────────────────

class BatchTab(QWidget):
    COL_CC = 0
    COL_FOLDER = 1
    COL_DEST = 2
    COL_PROGRESS = 3
    COL_STATUS = 4

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._current_row = -1
        self._current_stage = "idle"        # idle | analyzing | processing
        self._current_correction = ""
        self._output_buffer = ""
        self._stopping = False
        # Providers injected by MainWindow so the batch uses the same denoise/
        # sharpen/HPPCC menu state as the Analyze tab.
        self.pipeline_args_provider: callable = lambda: []
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # Top row: row management + output settings
        top = QHBoxLayout()
        self._add_btn = QPushButton("➕  Add row")
        # Lambda absorbs the bool from QPushButton.clicked(bool) so it doesn't
        # become cc_path=False (which would crash setText via SIP coercion).
        self._add_btn.clicked.connect(lambda: self._add_row())
        top.addWidget(self._add_btn)
        self._remove_btn = QPushButton("➖  Remove selected")
        self._remove_btn.clicked.connect(self._remove_selected_rows)
        top.addWidget(self._remove_btn)
        self._remove_all_btn = QPushButton("🗑  Remove all")
        self._remove_all_btn.clicked.connect(lambda: self._remove_all_rows())
        top.addWidget(self._remove_all_btn)
        top.addStretch()
        top.addWidget(QLabel("Format:"))
        self._fmt = QComboBox(); self._fmt.addItems(["jpeg", "tif", "png"])
        top.addWidget(self._fmt)
        top.addSpacing(12)
        top.addWidget(QLabel("Color space:"))
        self._cs = QComboBox(); self._cs.addItems(["sRGB", "Display-P3"])
        top.addWidget(self._cs)
        top.addSpacing(12)
        top.addWidget(QLabel("Workers:"))
        self._workers = QSpinBox(); self._workers.setRange(1, 32); self._workers.setValue(4)
        top.addWidget(self._workers)
        root.addLayout(top)

        # Table
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Color checker image", "Folder to process", "Destination folder", "Progress", "Status"]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.verticalHeader().setDefaultSectionSize(34)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(self.COL_CC, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self.COL_FOLDER, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self.COL_DEST, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self.COL_PROGRESS, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(self.COL_STATUS, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(self.COL_PROGRESS, 180)
        self._table.setColumnWidth(self.COL_STATUS, 220)
        root.addWidget(self._table, 1)

        # Action row
        action = QHBoxLayout()
        action.setSpacing(6)
        self._led = QLabel()
        self._led.setFixedSize(14, 14)
        self._set_led_idle()
        action.addWidget(self._led)
        self._run_btn = QPushButton("▶  Run batch")
        self._run_btn.setFixedHeight(34)
        self._run_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._run_btn.clicked.connect(self._run_batch)
        action.addWidget(self._run_btn, 4)
        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setFixedHeight(34)
        self._stop_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        action.addWidget(self._stop_btn, 2)
        root.addLayout(action)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setFixedHeight(160)
        root.addWidget(self._log)

    # ── LED helpers ────────────────────────────────────────────────────────

    def _set_led_idle(self) -> None:
        self._led.setStyleSheet(
            "background-color: #2ecc71; border: 1px solid #1e8449; border-radius: 7px;"
        )
        self._led.setToolTip("Idle")

    def _set_led_busy(self) -> None:
        self._led.setStyleSheet(
            "background-color: #e74c3c; border: 1px solid #922b21; border-radius: 7px;"
        )
        self._led.setToolTip("Running batch")

    # ── Row management ─────────────────────────────────────────────────────

    def _make_path_cell(self, placeholder: str, picker) -> tuple[QWidget, QLineEdit]:
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(2, 2, 2, 2)
        h.setSpacing(4)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        btn = QPushButton("...")
        btn.setFixedWidth(28)
        btn.clicked.connect(lambda _checked=False, e=edit: picker(e))
        h.addWidget(edit, 1)
        h.addWidget(btn)
        return container, edit

    def _pick_cc_image(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select color checker image", edit.text() or "", _INPUT_FILE_FILTER,
        )
        if path:
            edit.setText(path)

    def _pick_folder(self, edit: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select folder", edit.text())
        if d:
            edit.setText(d)

    def _add_row(self, cc_path: str = "", folder_path: str = "", dest_path: str = "") -> None:
        cc_path = str(cc_path) if isinstance(cc_path, str) else ""
        folder_path = str(folder_path) if isinstance(folder_path, str) else ""
        dest_path = str(dest_path) if isinstance(dest_path, str) else ""

        row = self._table.rowCount()
        self._table.insertRow(row)

        cc_widget, cc_edit = self._make_path_cell("Color checker image...", self._pick_cc_image)
        cc_edit.setText(cc_path)
        self._table.setCellWidget(row, self.COL_CC, cc_widget)

        # Destination cell needs to be built first so the folder cell's autofill
        # callback can reference it.
        dest_widget, dest_edit = self._make_path_cell(
            "Destination folder (defaults to <Folder>/Corrected)...", self._pick_folder,
        )

        # Folder cell built inline (instead of via _make_path_cell) so we can
        # hook autofill into both editingFinished (typing) and Browse click.
        folder_widget = QWidget()
        folder_layout = QHBoxLayout(folder_widget)
        folder_layout.setContentsMargins(2, 2, 2, 2)
        folder_layout.setSpacing(4)
        folder_edit = QLineEdit()
        folder_edit.setPlaceholderText("Folder to process...")

        def _autofill_dest() -> None:
            if dest_edit.text().strip():
                return
            folder_val = folder_edit.text().strip()
            if folder_val:
                dest_edit.setText(str(Path(folder_val) / "Corrected"))

        folder_edit.editingFinished.connect(_autofill_dest)
        folder_btn = QPushButton("...")
        folder_btn.setFixedWidth(28)
        folder_btn.clicked.connect(
            lambda _checked=False, e=folder_edit: (self._pick_folder(e), _autofill_dest())
        )
        folder_layout.addWidget(folder_edit, 1)
        folder_layout.addWidget(folder_btn)
        folder_edit.setText(folder_path)
        self._table.setCellWidget(row, self.COL_FOLDER, folder_widget)

        dest_edit.setText(dest_path)
        self._table.setCellWidget(row, self.COL_DEST, dest_widget)
        # If the row is created with a folder but no explicit destination, fill it now.
        if folder_path and not dest_path:
            _autofill_dest()

        progress = QProgressBar()
        progress.setRange(0, 1)
        progress.setValue(0)
        progress.setFormat("%v / %m")
        progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setCellWidget(row, self.COL_PROGRESS, progress)

        status = QLabel("Pending")
        status.setStyleSheet("color: #aaa; padding: 0 6px;")
        status.setToolTip("")
        self._table.setCellWidget(row, self.COL_STATUS, status)

    def _remove_selected_rows(self) -> None:
        if self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning:
            return  # don't allow row editing during a run
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()}, reverse=True)
        for r in rows:
            self._table.removeRow(r)

    def _remove_all_rows(self) -> None:
        if self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning:
            return  # don't allow row editing during a run
        while self._table.rowCount() > 0:
            self._table.removeRow(0)

    # ── Row accessors ──────────────────────────────────────────────────────

    def _row_cc(self, row: int) -> str:
        widget = self._table.cellWidget(row, self.COL_CC)
        return widget.findChild(QLineEdit).text().strip() if widget else ""

    def _row_folder(self, row: int) -> str:
        widget = self._table.cellWidget(row, self.COL_FOLDER)
        return widget.findChild(QLineEdit).text().strip() if widget else ""

    def _row_dest(self, row: int) -> str:
        widget = self._table.cellWidget(row, self.COL_DEST)
        return widget.findChild(QLineEdit).text().strip() if widget else ""

    def _row_progress(self, row: int) -> QProgressBar:
        return self._table.cellWidget(row, self.COL_PROGRESS)

    def _set_row_status(self, row: int, text: str, color: str, tooltip: str = "") -> None:
        label: QLabel = self._table.cellWidget(row, self.COL_STATUS)
        if label is None:
            return
        label.setText(text)
        label.setStyleSheet(f"color: {color}; padding: 0 6px;")
        label.setToolTip(tooltip or text)

    def collect_rows(self) -> list[dict]:
        return [
            {
                "cc_image": self._row_cc(r),
                "folder": self._row_folder(r),
                "dest": self._row_dest(r),
            }
            for r in range(self._table.rowCount())
        ]

    def restore_rows(self, rows: list) -> None:
        if not isinstance(rows, list):
            return
        # Clear current rows first
        while self._table.rowCount() > 0:
            self._table.removeRow(0)
        for entry in rows:
            if not isinstance(entry, dict):
                continue
            self._add_row(
                cc_path=str(entry.get("cc_image", "")),
                folder_path=str(entry.get("folder", "")),
                dest_path=str(entry.get("dest", "")),
            )

    # ── Batch run ──────────────────────────────────────────────────────────

    def _run_batch(self) -> None:
        if self._table.rowCount() == 0:
            QMessageBox.information(self, "Batch", "Add at least one row before running.")
            return
        # Reset progress/status on all rows that don't already say OK.
        for r in range(self._table.rowCount()):
            label: QLabel = self._table.cellWidget(r, self.COL_STATUS)
            if label is None or label.text() != "OK":
                self._set_row_status(r, "Pending", "#aaa")
                pb = self._row_progress(r)
                pb.setRange(0, 1)
                pb.setValue(0)
        self._log.clear()
        self._stopping = False
        self._current_row = -1
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._add_btn.setEnabled(False)
        self._remove_btn.setEnabled(False)
        self._set_led_busy()
        self._start_next_row()

    def _start_next_row(self) -> None:
        if self._stopping:
            self._on_batch_finished()
            return
        # Advance to next non-OK row
        self._current_row += 1
        while self._current_row < self._table.rowCount():
            status_label: QLabel = self._table.cellWidget(self._current_row, self.COL_STATUS)
            if status_label is None or status_label.text() != "OK":
                break
            self._current_row += 1
        if self._current_row >= self._table.rowCount():
            self._on_batch_finished()
            return
        self._launch_analyze()

    def _launch_analyze(self) -> None:
        cc_path = self._row_cc(self._current_row)
        folder = self._row_folder(self._current_row)
        if not cc_path or not Path(cc_path).is_file():
            self._set_row_status(self._current_row, "Invalid color checker path", "#e74c3c")
            self._start_next_row()
            return
        if not folder or not Path(folder).is_dir():
            self._set_row_status(self._current_row, "Invalid target folder", "#e74c3c")
            self._start_next_row()
            return

        analysis_dir = str(Path(cc_path).parent / "analysis")
        args = _cc_args(
            "analyze",
            "--cc-image", cc_path,
            "--analysis-dir", analysis_dir,
            "--process-dir", analysis_dir,
            "--output-format", self._fmt.currentText(),
            "--output-colorspace", self._cs.currentText(),
            "--no-show-detection-preview",
            "--no-show-developed-image-preview",
        )
        args += self.pipeline_args_provider()

        self._current_stage = "analyzing"
        self._current_correction = ""
        self._output_buffer = ""
        self._set_row_status(self._current_row, "Analyzing...", "#e09e00")
        self._log.append(f"\n── [row {self._current_row + 1}] analyze {cc_path}")

        self._launch_proc(args)

    def _launch_process(self, correction_path: str) -> None:
        folder = self._row_folder(self._current_row)
        total = _count_raw_files(Path(folder), recursive=False)
        if total == 0:
            self._set_row_status(self._current_row, "No input files in target folder", "#e74c3c")
            self._start_next_row()
            return
        pb = self._row_progress(self._current_row)
        pb.setRange(0, total)
        pb.setValue(0)

        # Destination: user override on the row, falling back to <folder>/Corrected.
        dest_override = self._row_dest(self._current_row)
        output_dir = dest_override or str(Path(folder) / "Corrected")
        args = _cc_args(
            "process",
            correction_path, folder,
            "--process-dir", output_dir,
            "--output-format", self._fmt.currentText(),
            "--output-colorspace", self._cs.currentText(),
            "--workers", str(self._workers.value()),
        )

        self._current_stage = "processing"
        self._output_buffer = ""
        self._set_row_status(self._current_row, f"Processing 0/{total}", "#e09e00")
        self._log.append(f"── [row {self._current_row + 1}] process {folder} → {output_dir}")
        self._launch_proc(args)

    def _launch_proc(self, args: list[str]) -> None:
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._on_output)
        self._proc.finished.connect(self._on_proc_finished)
        self._proc.start(PYTHON, args)

    # ── Subprocess output ──────────────────────────────────────────────────

    def _on_output(self) -> None:
        raw = self._proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self._log.insertPlainText(raw)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())
        self._output_buffer += raw
        while "\n" in self._output_buffer:
            line, self._output_buffer = self._output_buffer.split("\n", 1)
            if self._current_stage == "analyzing":
                if "_correction.json" in line and "written" in line.lower():
                    self._current_correction = line.split(":", 1)[-1].strip()
            elif self._current_stage == "processing" and line.startswith("Processed:"):
                pb = self._row_progress(self._current_row)
                pb.setValue(min(pb.value() + 1, pb.maximum()))
                self._set_row_status(
                    self._current_row,
                    f"Processing {pb.value()}/{pb.maximum()}",
                    "#e09e00",
                )

    def _on_proc_finished(self, code: int, _status) -> None:
        if self._stopping:
            self._on_batch_finished()
            return

        if self._current_stage == "analyzing":
            if code != 0:
                log = self._log.toPlainText()
                if "No ColorChecker detected" in log:
                    self._set_row_status(
                        self._current_row, "Color checker not detected", "#e74c3c"
                    )
                else:
                    self._set_row_status(
                        self._current_row, f"Analyze error (exit {code})", "#e74c3c"
                    )
                self._start_next_row()
                return
            if not self._current_correction:
                self._set_row_status(
                    self._current_row, "Correction file not found in log", "#e74c3c"
                )
                self._start_next_row()
                return
            self._launch_process(self._current_correction)
            return

        if self._current_stage == "processing":
            if code != 0:
                self._set_row_status(
                    self._current_row, f"Process error (exit {code})", "#e74c3c"
                )
            else:
                pb = self._row_progress(self._current_row)
                pb.setValue(pb.maximum())
                self._set_row_status(self._current_row, "OK", "#2ecc71")
            self._start_next_row()
            return

    # ── Stop ───────────────────────────────────────────────────────────────

    def _on_stop(self) -> None:
        if self._proc is None or self._proc.state() == QProcess.ProcessState.NotRunning:
            return
        self._stopping = True
        pid = int(self._proc.processId())
        if sys.platform == "win32":
            kill_process_tree(pid)
        else:
            self._proc.kill()
        if 0 <= self._current_row < self._table.rowCount():
            self._set_row_status(self._current_row, "Stopped", "#e74c3c")
        self._log.append("\n■ Batch stopped by user.")

    def _on_batch_finished(self) -> None:
        self._current_stage = "idle"
        self._stopping = False
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._add_btn.setEnabled(True)
        self._remove_btn.setEnabled(True)
        self._set_led_idle()


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CoCo2l - Color Correction Tool")
        self.setMinimumSize(1180, 860)

        # Per-op settings now live on each pipeline entry (edited via its gear
        # button), so there are no global per-op config dicts here anymore.
        self._tabs = QTabWidget()
        self._analyze = AnalyzeTab()
        self._process = ProcessTab()
        self._batch = BatchTab()
        self._tabs.addTab(self._analyze, "  Analysis  ")
        self._tabs.addTab(self._process, "  Processing  ")
        self._tabs.addTab(self._batch, "  Batch  ")

        self._analyze.correction_ready.connect(self._on_correction_ready)

        # The pipeline (ordered ops) plus the per-op method/param menu settings
        # are encoded together by get_pipeline_args and wired into Analyze and
        # Batch (the analyze step inside batch jobs uses the same menu settings).
        # During Process the values saved in the correction JSON take priority
        # over GUI settings, so ProcessTab is not wired.
        self._analyze.pipeline_args_provider = self.get_pipeline_args
        self._batch.pipeline_args_provider = self.get_pipeline_args

        self._build_menu()
        self.setCentralWidget(self._tabs)

        # Restore saved UI state (after menu is built so checkbox-driven menu
        # sync signals fire correctly during restore).
        self._restore_state(load_settings())

    # ── persistence ──────────────────────────────────────────────────────

    def _persisted_widgets(self) -> dict:
        # Single registry — add a line here for every new persistable control.
        # Keys are namespaced "<tab>.<name>" so the JSON stays human-readable.
        a = self._analyze
        p = self._process
        return {
            "analyze.raw_path":       a._raw_edit,
            "analyze.output_dir":     a._output_dir,
            "analyze.roi":            a._roi_edit,
            "analyze.show_developed": a._show_developed,
            "analyze.format":         a._fmt,
            "analyze.colorspace":     a._cs,
            "analyze.white_field":    a._white_field,
            "analyze.white_field_path": a._white_field_path,
            "process.correction":     p._corr_edit,
            "process.source":         p._src_edit,
            "process.output":         p._out_edit,
            "process.recursive":      p._recursive,
            "process.format":         p._fmt,
            "process.colorspace":     p._cs,
            "process.workers":        p._workers,
            "batch.format":           self._batch._fmt,
            "batch.colorspace":       self._batch._cs,
            "batch.workers":          self._batch._workers,
        }

    def _persisted_state(self) -> dict:
        widgets = self._persisted_widgets()
        out = {key: _widget_value(w) for key, w in widgets.items()}
        # The pipeline (ordered ops + per-instance params) is the single source
        # of truth for op settings; it is persisted as a list of {op, params}.
        out["analyze_pipeline"] = self._analyze._pipeline.entries()
        out["batch_rows"] = self._batch.collect_rows()
        return out

    def _restore_state(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        for key, widget in self._persisted_widgets().items():
            if key in data:
                _set_widget_value(widget, data[key])
        pipeline = data.get("analyze_pipeline")
        if isinstance(pipeline, list) and pipeline:
            # set_entries tolerates both the new {op, params} dicts and the
            # legacy list-of-op-id-strings format.
            self._analyze._pipeline.set_entries(pipeline)
        batch_rows = data.get("batch_rows")
        if isinstance(batch_rows, list):
            self._batch.restore_rows(batch_rows)

    def closeEvent(self, event) -> None:
        save_settings(self._persisted_state())
        super().closeEvent(event)

    # ── menu ─────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        # All per-op settings now live on the pipeline entries (gear buttons),
        # so the only menu left is Help.
        mb = self.menuBar()
        help_menu = mb.addMenu("Help")

        howto_action = QAction("How to", self)
        howto_action.triggered.connect(self._open_howto)
        help_menu.addAction(howto_action)

        about_action = QAction("About", self)
        about_action.triggered.connect(self._open_about)
        help_menu.addAction(about_action)

    def _open_howto(self) -> None:
        dlg = HowToDialog(self)
        dlg.exec()

    def _open_about(self) -> None:
        dlg = AboutDialog(self)
        dlg.exec()

    def get_pipeline_args(self) -> list[str]:
        """Encode the ordered pipeline with each entry's per-instance params as a
        single ``--pipeline-spec`` JSON argument.

        ``QProcess.start(prog, args)`` passes argv as a list (no shell), so the
        JSON string travels as one element with no quoting concerns. The backend
        derives op enablement from membership and reads each op's params from the
        spec.
        """
        entries = self._analyze._pipeline.entries()
        return ["--pipeline-spec", json.dumps(entries)]

    # ── slots ─────────────────────────────────────────────────────────────

    def _on_correction_ready(self, path: str) -> None:
        # Pre-fill the correction path on the Process tab so the user can jump
        # there manually, but stay on Analysis — auto-switching is disruptive.
        self._process.set_correction(path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — GUI when run directly, CLI passthrough when args are given
# ─────────────────────────────────────────────────────────────────────────────

import traceback

def global_exception_hook(exctype, value, tb):
    with open("crash.log", "w") as f:
        traceback.print_exception(exctype, value, tb, file=f)
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = global_exception_hook

def main() -> None:
    # Force UTF-8 on stdio so that error messages containing non-ASCII path
    # components (e.g. "Università") or the U+FFFD replacement character
    # don't crash with UnicodeEncodeError when stdout is on cp1252 (default
    # locale on Italian Windows). `errors='replace'` is the safety net.
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
            except (AttributeError, OSError):
                pass

    if len(sys.argv) > 1:
        from src.cc import main as cc_main
        cc_main()
        return

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Application icon. Looked up in BUNDLE_DIR/assets so it works for both dev
    # runs (BUNDLE_DIR == project root) and frozen builds (BUNDLE_DIR == _MEIPASS).
    from src.config import BUNDLE_DIR
    from PyQt6.QtGui import QIcon
    icon_path = BUNDLE_DIR / "assets" / "cocoicobn.png"
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))

    from PyQt6.QtGui import QPalette
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(45, 45, 45))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    pal.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Button, QColor(60, 60, 60))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(pal)

    win = MainWindow()
    win.show()

    # If launched from a PyInstaller bundle with a Splash screen, close it
    # now that the main window is visible. Harmless no-op in dev runs.
    try:
        import pyi_splash  # type: ignore[import-not-found]
        if pyi_splash.is_alive():
            pyi_splash.close()
    except (ImportError, Exception):
        pass

    sys.exit(app.exec())


if __name__ == "__main__":
    # Required for PyInstaller-frozen builds so ProcessPoolExecutor workers
    # don't re-run main() when they spawn. No-op on dev runs.
    import multiprocessing
    multiprocessing.freeze_support()
    main()
