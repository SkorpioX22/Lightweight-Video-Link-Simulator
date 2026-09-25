"""PySide6 GUI demo for LVLS — Lightweight Video Link Simulation.

Run: python demo/gui/main.py
"""
import os
import queue
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

try:
    import cv2
except Exception as exc:
    cv2 = None
    CV2_IMPORT_ERROR = str(exc)
else:
    CV2_IMPORT_ERROR = None

try:
    from engine import AnalogVideoBreakupEngine, PRESETS
except Exception as exc:
    AnalogVideoBreakupEngine = None
    PRESETS = {}
    ENGINE_IMPORT_ERROR = str(exc)
else:
    ENGINE_IMPORT_ERROR = None

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

WINDOW_TITLE = "LVLS — Lightweight Video Link Simulation"
SAMPLE_PATH = os.path.join(ROOT, "demo", "gui", "demo_video1.mp4")
SAMPLE_FALLBACK = os.path.join(ROOT, "demo", "sample_media", "sample.mp4")
if not os.path.isfile(SAMPLE_PATH) and os.path.isfile(SAMPLE_FALLBACK):
    SAMPLE_PATH = SAMPLE_FALLBACK
TIMER_MS = 10          # ~100 Hz playback cap (80 fps target needs <=12.5 ms)
REPROC_DEBOUNCE_MS = 16  # slider scrub coalescing
MAX_PANE_WIDTH = 960
PARAM_KEYS = ("signalStrength", "multipath", "rfInterference")
SLIDER_TITLES = ("Signal Strength", "Multipath", "RF Interference")

_TIMER_BEGIN_BALANCE = 0


def _request_high_res_timer():
    """Request ~1 ms system timer resolution (safe to call repeatedly).

    Numba's JIT resets the Windows timer quantum to ~15.6 ms process-wide,
    which caps QTimer-driven playback at ~64 fps. winmm.timeBeginPeriod no
    longer sticks after the JIT, but NtSetTimerResolution does — call this
    at startup and again after warmup compiles.
    """
    global _TIMER_BEGIN_BALANCE
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
        _TIMER_BEGIN_BALANCE += 1
    except Exception:
        pass
    try:
        ntdll = ctypes.windll.ntdll
        ntdll.NtSetTimerResolution.argtypes = [
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        ntdll.NtSetTimerResolution.restype = ctypes.c_long
        current = ctypes.c_ulong()
        # 1 ms expressed in 100 ns units; second arg = request resolution ON.
        ntdll.NtSetTimerResolution(10000, 1, ctypes.byref(current))
    except Exception:
        pass


def _release_high_res_timer():
    global _TIMER_BEGIN_BALANCE
    if sys.platform != "win32" or _TIMER_BEGIN_BALANCE <= 0:
        return
    import ctypes

    try:
        for _ in range(_TIMER_BEGIN_BALANCE):
            ctypes.windll.winmm.timeEndPeriod(1)
    except Exception:
        pass
    _TIMER_BEGIN_BALANCE = 0

STYLESHEET = """
QWidget { background-color: #202225; color: #d8d8d8; font-size: 13px; }
QLabel { background: transparent; }
QPushButton {
    background-color: #3c4043; border: 1px solid #5f6368;
    border-radius: 4px; padding: 6px 14px; min-height: 22px;
}
QPushButton:hover { background-color: #4c5054; }
QPushButton:pressed { background-color: #2d3134; }
QPushButton:disabled { color: #777; background-color: #2a2c2e; }
QSlider::groove:horizontal { height: 6px; background: #3c4043; border-radius: 3px; }
QSlider::handle:horizontal {
    width: 14px; margin: -5px 0; border-radius: 7px; background: #4fc3f7;
}
QSlider::sub-page:horizontal { background: #2f6f8f; border-radius: 3px; }
QComboBox {
    background-color: #3c4043; border: 1px solid #5f6368;
    border-radius: 4px; padding: 5px 10px; min-width: 170px;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #2a2c2e; selection-background-color: #2f6f8f;
}
"""


def error_label(msg):
    lab = QLabel(msg)
    lab.setAlignment(Qt.AlignCenter)
    lab.setWordWrap(True)
    lab.setStyleSheet(
        "color: #ff5252; font-size: 15px; background-color: #1b1d1f;"
        " border: 1px solid #ff5252; padding: 18px;"
    )
    return lab


class FrameReader:
    """Background H.264 decoder with a small frame queue.

    Decode runs off the GUI thread so occasional codec spikes (up to ~10 ms)
    never stall the render loop; the queue acts as jitter absorption.
    """

    def __init__(self, cap, bufsize=8):
        self.cap = cap
        self.q = queue.Queue(maxsize=bufsize)
        self.gen = 0
        self._stop = False
        self._cap_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            with self._cap_lock:
                if self._stop:
                    return
                gen = self.gen
                ok, fr = self.cap.read()
                if not ok:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, fr = self.cap.read()
                pos = (
                    int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1 if ok else -1
                )
            if not ok:
                time.sleep(0.05)
                continue
            # blocks when the queue is full — natural backpressure, no drops
            self.q.put((gen, pos, fr))

    def read(self, timeout=0.25):
        """Return (frame_no, bgr) for the current generation, or None."""
        deadline = time.monotonic() + timeout
        while True:
            remain = deadline - time.monotonic()
            if remain <= 0:
                return None
            try:
                gen, pos, fr = self.q.get(timeout=min(remain, 0.05))
            except queue.Empty:
                continue
            if gen == self.gen:
                return pos, fr
            # stale frame from before a seek — drop and keep waiting

    def seek(self, n):
        with self._cap_lock:
            self.gen += 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def stop(self):
        self._stop = True
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1100, 900)
        self.setMinimumSize(800, 600)

        self.eng = None
        self.cap = None
        self.reader = None
        self.video_ok = False
        self.video_error = None
        self.playing = False
        self.frame_no = 0
        self.total_frames = 0
        self.native_fps = 30.0
        self.frame_interval = 1.0 / 30.0
        self._next_deadline = None
        self.current_bgr = None
        self.orig_bgr = None
        self.proc_rgb = None
        self._orig_shown = None
        self._orig_size = None
        self.proc_ms_ema = 0.0
        self.fps_ema = 0.0
        self._last_render_t = None
        self._last_label_t = 0.0
        self._rgb_buf = None
        self._out_buf = None
        self._scale_buf_orig = None
        self._scale_buf_proc = None
        self._suspend_reprocess = False
        self.sliders = []
        self.value_labels = []
        self.lbl_orig = None
        self.lbl_proc = None

        if ENGINE_IMPORT_ERROR is None:
            self.eng = AnalogVideoBreakupEngine(seed=12345)
            self._init_video()

        # coalesce slider storms into one reprocess per debounce interval
        # (created before _build_ui so handlers can always reach it)
        self._reproc_timer = QTimer(self)
        self._reproc_timer.setTimerType(Qt.PreciseTimer)
        self._reproc_timer.setSingleShot(True)
        self._reproc_timer.setInterval(REPROC_DEBOUNCE_MS)
        self._reproc_timer.timeout.connect(self._process_and_present)

        self._build_ui()

        if self.video_ok:
            self._process_and_present()

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.setInterval(TIMER_MS)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        # JIT every kernel on a throwaway engine in the background so the
        # first slider drag never hits a multi-second compile mid-drag
        self._warm_eng = None
        self._warm_frame = None
        self._warm_queue = [
            ({"signalStrength": 99, "multipath": 0, "rfInterference": 0},
             "JIT: signal kernels (first run only)..."),
            ({"signalStrength": 0, "multipath": 99, "rfInterference": 0},
             "JIT: multipath kernels (first run only)..."),
            ({"signalStrength": 0, "multipath": 0, "rfInterference": 99},
             "JIT: interference kernels (first run only)..."),
            ({"signalStrength": 99, "multipath": 99, "rfInterference": 99},
             "JIT: geometry + composite (first run only)..."),
        ]
        QTimer.singleShot(0, self._warm_step)

    def _init_video(self):
        if CV2_IMPORT_ERROR is not None:
            self.video_error = (
                "OpenCV (cv2) failed to import:\n"
                f"{CV2_IMPORT_ERROR}\n"
                "Install it with: pip install opencv-python"
            )
            return
        if not os.path.isfile(SAMPLE_PATH):
            self.video_error = (
                "Sample video not found:\n"
                f"{SAMPLE_PATH}\n"
                "Expected demo/gui/demo_video1.mp4 (or regenerate the "
                "synthetic fallback: python demo/sample_media/generate_sample.py)"
            )
            return
        self.cap = cv2.VideoCapture(SAMPLE_PATH)
        if not self.cap.isOpened():
            self.video_error = f"Could not open video:\n{SAMPLE_PATH}"
            self.cap = None
            return
        ok, fr = self.cap.read()
        if not ok:
            self.cap.release()
            self.cap = None
            self.video_error = f"Could not decode first frame:\n{SAMPLE_PATH}"
            return
        self.video_ok = True
        self.current_bgr = fr
        self.frame_no = 0
        self.total_frames = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        # playback must match the content's native rate (not free-run)
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if not (0.0 < fps <= 240.0):
            fps = 30.0
        self.native_fps = fps
        self.frame_interval = 1.0 / fps
        # hand the open capture to the background decoder (its read position
        # is already 1, so the next read() delivers frame 1 — matching the
        # old synchronous _advance behaviour)
        self.reader = FrameReader(self.cap)
        self.cap = None

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QLabel("LVLS — LIGHTWEIGHT VIDEO LINK SIMULATION")
        header.setAlignment(Qt.AlignCenter)
        font = header.font()
        font.setPointSize(15)
        font.setBold(True)
        header.setFont(font)
        root.addWidget(header)

        if ENGINE_IMPORT_ERROR is not None:
            root.addWidget(
                error_label(f"Failed to import engine:\n{ENGINE_IMPORT_ERROR}"), 1
            )
            return

        video_box = QVBoxLayout()
        video_box.setSpacing(4)
        if self.video_error is not None:
            video_box.addWidget(error_label(self.video_error), 1)
        else:
            video_box.addWidget(self._make_caption("ORIGINAL"), 0, Qt.AlignHCenter)
            self.lbl_orig = self._make_pane()
            video_box.addWidget(self.lbl_orig, 1, Qt.AlignHCenter)
            video_box.addWidget(self._make_caption("PROCESSED"), 0, Qt.AlignHCenter)
            self.lbl_proc = self._make_pane()
            video_box.addWidget(self.lbl_proc, 1, Qt.AlignHCenter)
        root.addLayout(video_box, 1)

        for title in SLIDER_TITLES:
            row, slider, vlab = self._make_slider_row(title)
            self.sliders.append(slider)
            self.value_labels.append(vlab)
            slider.valueChanged.connect(
                lambda _v, s=slider, v=vlab: self._on_slider(s, v)
            )
            root.addLayout(row)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.combo = QComboBox()
        self.combo.addItems(list(PRESETS.keys()))
        self.combo.currentTextChanged.connect(self._on_preset)
        preset_row.addWidget(self.combo)
        self.lbl_preset_value = QLabel("CLEAN")
        self.lbl_preset_value.setStyleSheet("font-weight: bold; color: #8ab4f8;")
        preset_row.addWidget(self.lbl_preset_value, 1)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._on_reset)
        preset_row.addWidget(self.btn_reset)
        root.addLayout(preset_row)

        transport_row = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self._on_play)
        self.btn_restart = QPushButton("Restart")
        self.btn_restart.clicked.connect(self._on_restart)
        self.btn_step = QPushButton("Step Frame")
        self.btn_step.clicked.connect(self._on_step)
        transport_row.addWidget(self.btn_play)
        transport_row.addWidget(self.btn_restart)
        transport_row.addWidget(self.btn_step)
        transport_row.addStretch(1)
        root.addLayout(transport_row)

        status_row = QHBoxLayout()
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_ms = QLabel("Proc: -- ms/frame")
        self.lbl_frame = QLabel("Frame: - / -")
        self.lbl_eng = QLabel("engine.frame_index: 0")
        for lab, width in (
            (self.lbl_fps, 110),
            (self.lbl_ms, 170),
            (self.lbl_frame, 150),
            (self.lbl_eng, 200),
        ):
            lab.setFixedWidth(width)
            lab.setStyleSheet("color: #b0b3b8;")
            status_row.addWidget(lab)
        status_row.addStretch(1)
        root.addLayout(status_row)

        if not self.video_ok:
            self.btn_play.setEnabled(False)
            self.btn_restart.setEnabled(False)
            self.btn_step.setEnabled(False)
            self.btn_reset.setEnabled(False)

    @staticmethod
    def _make_caption(text):
        cap = QLabel(text)
        cap.setAlignment(Qt.AlignCenter)
        cap.setStyleSheet("color: #9aa0a6; font-weight: bold; letter-spacing: 2px;")
        return cap

    @staticmethod
    def _make_pane():
        pane = QLabel()
        pane.setAlignment(Qt.AlignCenter)
        pane.setMinimumSize(320, 180)
        pane.setMaximumWidth(MAX_PANE_WIDTH)
        pane.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pane.setStyleSheet("background-color: #000000; border: 1px solid #444444;")
        return pane

    def _make_slider_row(self, title):
        row = QHBoxLayout()
        name = QLabel(title)
        name.setFixedWidth(130)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 99)
        slider.setSingleStep(1)
        vlab = QLabel("0")
        vlab.setFixedWidth(40)
        vlab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        vlab.setStyleSheet("font-weight: bold; color: #4fc3f7;")
        row.addWidget(name)
        row.addWidget(slider, 1)
        row.addWidget(vlab)
        return row, slider, vlab

    def _params(self):
        return {key: slider.value() for key, slider in zip(PARAM_KEYS, self.sliders)}

    def _warm_step(self):
        """One warmup compile per event-loop turn; progress shown in status."""
        if AnalogVideoBreakupEngine is None:
            return
        if self._warm_eng is None:
            self._warm_eng = AnalogVideoBreakupEngine(seed=1)
            self._warm_frame = np.full((64, 64, 3), 96, dtype=np.uint8)
        if self._warm_queue:
            params, label = self._warm_queue.pop(0)
            self.lbl_ms.setText(label)
            QApplication.processEvents()
            try:
                self._warm_eng.process(self._warm_frame, params, advance=False)
            except Exception:
                self._warm_queue = []
            QTimer.singleShot(0, self._warm_step)
            return
        # a few temporal frames so event machines compile their branches too
        for _ in range(4):
            self._warm_eng.process(
                self._warm_frame,
                {"signalStrength": 99, "multipath": 99, "rfInterference": 99},
                advance=True,
            )
        self._warm_eng = None
        self._warm_frame = None
        # Numba's JIT dropped the Windows timer quantum; restore it now that
        # compilation is done (otherwise QTimer playback caps at ~64 fps).
        _request_high_res_timer()
        # A QTimer created against the old 15.6 ms quantum keeps its degraded
        # cadence even after stop/start — rebuild it so playback runs at TIMER_MS.
        was_active = self.timer.isActive()
        self.timer.stop()
        self.timer.deleteLater()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.setInterval(TIMER_MS)
        self.timer.timeout.connect(self._tick)
        if was_active:
            self.timer.start()
        self.lbl_ms.setText(f"Proc: {self.proc_ms_ema:.1f} ms/frame")

    def _on_slider(self, slider, vlab):
        vlab.setText(str(slider.value()))
        if self._suspend_reprocess:
            return
        self._sync_preset_display()
        if self.video_ok and not self.playing and self.current_bgr is not None:
            # leading-edge throttle: schedule soon if idle, otherwise let the
            # already-pending timer fire (it reads the latest slider values)
            if not self._reproc_timer.isActive():
                self._reproc_timer.start()

    def _sync_preset_display(self):
        cur = self._params()
        match = None
        for name, preset in PRESETS.items():
            if all(int(preset[k]) == cur[k] for k in PARAM_KEYS):
                match = name
                break
        if match is not None:
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(list(PRESETS).index(match))
            self.combo.blockSignals(False)
            self.lbl_preset_value.setText(match)
        else:
            self.lbl_preset_value.setText("CUSTOM")

    def _on_preset(self, name):
        preset = PRESETS.get(name)
        if preset is None:
            return
        self._suspend_reprocess = True
        try:
            for slider, key in zip(self.sliders, PARAM_KEYS):
                slider.setValue(int(preset[key]))
                self.value_labels[self.sliders.index(slider)].setText(
                    str(slider.value())
                )
        finally:
            self._suspend_reprocess = False
        self.lbl_preset_value.setText(name)
        if self.video_ok and not self.playing and self.current_bgr is not None:
            if not self._reproc_timer.isActive():
                self._reproc_timer.start()

    def _on_reset(self):
        self._suspend_reprocess = True
        try:
            for slider, vlab in zip(self.sliders, self.value_labels):
                slider.setValue(0)
                vlab.setText("0")
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(0)
            self.combo.blockSignals(False)
        finally:
            self._suspend_reprocess = False
        self.lbl_preset_value.setText("CLEAN")
        if self.eng is not None:
            self.eng.reset()
        if self.video_ok:
            self._seek(0)

    def _on_play(self):
        if not self.video_ok:
            return
        self._set_playing(not self.playing)

    def _set_playing(self, playing):
        was = self.playing
        self.playing = playing
        self.btn_play.setText("Pause" if playing else "Play")
        self._last_render_t = None
        if playing and not was:
            # self-rescheduling pump paced to the content's native FPS
            # (a fixed TIMER_MS interval slips against the Windows quantum
            # when work exceeds it, and a 0-delay chain free-runs too fast)
            self._next_deadline = None
            self.timer.stop()
            QTimer.singleShot(0, self._tick)
        elif not playing and was:
            if not self.timer.isActive():
                self.timer.start()

    def _on_restart(self):
        if not self.video_ok:
            return
        if self.eng is not None:
            self.eng.reset()
        self._seek(0)

    def _on_step(self):
        if not self.video_ok:
            return
        if self.playing:
            self._set_playing(False)
        self._advance()

    def _tick(self):
        if self.playing and self.video_ok:
            now = time.perf_counter()
            if self._next_deadline is None:
                self._next_deadline = now
            self._advance()
            if self.playing:  # EOF inside _advance clears the flag
                # pace to native fps: next frame at previous deadline + interval
                self._next_deadline += self.frame_interval
                delay = self._next_deadline - time.perf_counter()
                if delay <= 0.0:
                    # behind schedule (work > frame budget): don't burst-catch-up
                    self._next_deadline = time.perf_counter()
                    delay_ms = 0
                else:
                    delay_ms = int(delay * 1000.0 + 0.5)
                QTimer.singleShot(delay_ms, self._tick)
            else:
                self._next_deadline = None
                if not self.timer.isActive():
                    self.timer.start()
            return
        # frame/eng labels while paused (playback path throttles its own)
        if not self.playing:
            if self.video_ok:
                self.lbl_frame.setText(
                    f"Frame: {self.frame_no} / {self.total_frames}"
                )
            if self.eng is not None:
                self.lbl_eng.setText(
                    f"engine.frame_index: {self.eng.frame_index}"
                )

    def _read_frame(self):
        if self.reader is None:
            return None, None
        got = self.reader.read()
        if got is None:
            return None, None
        return got

    def _advance(self):
        got = self._read_frame()
        if got is None:
            self._set_playing(False)
            return
        pos, fr = got
        self.current_bgr = fr
        self.frame_no = max(0, pos)
        self._process_and_present()

    def _seek(self, n):
        if self.reader is None:
            self._set_playing(False)
            return
        self.reader.seek(n)
        got = self.reader.read(timeout=1.0)
        if got is None:
            self._set_playing(False)
            return
        pos, fr = got
        self.current_bgr = fr
        self.frame_no = max(0, pos)
        self._process_and_present()

    def _process_and_present(self):
        if not self.video_ok or self.current_bgr is None or self.eng is None:
            return
        self._reproc_timer.stop()  # direct call supersedes a pending debounce
        if (
            self._rgb_buf is None
            or self._rgb_buf.shape != self.current_bgr.shape
        ):
            self._rgb_buf = np.empty_like(self.current_bgr)
        cv2.cvtColor(
            self.current_bgr, cv2.COLOR_BGR2RGB, dst=self._rgb_buf
        )
        rgb = self._rgb_buf
        params = self._params()
        if self._out_buf is None or self._out_buf.shape != rgb.shape:
            self._out_buf = np.empty_like(rgb)
        t0 = time.perf_counter()
        try:
            out_rgb = self.eng.process(rgb, params, out=self._out_buf)
        except Exception as exc:
            self._set_playing(False)
            self.lbl_ms.setText(f"Proc error: {exc}")
            return
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if self.proc_ms_ema == 0.0:
            self.proc_ms_ema = dt_ms
        else:
            self.proc_ms_ema = self.proc_ms_ema * 0.9 + dt_ms * 0.1
        now = time.perf_counter()
        if self._last_render_t is not None:
            dt = now - self._last_render_t
            if dt > 1e-6:
                inst_fps = 1.0 / dt
                if self.fps_ema == 0.0:
                    self.fps_ema = inst_fps
                else:
                    self.fps_ema = self.fps_ema * 0.85 + inst_fps * 0.15
        self._last_render_t = now
        self.orig_bgr = self.current_bgr
        self.proc_rgb = out_rgb  # presented directly as QImage.Format_RGB888
        self._present()
        # status labels only need ~10 Hz — setText each frame adds up
        if now - self._last_label_t >= 0.1:
            self._last_label_t = now
            self.lbl_ms.setText(f"Proc: {self.proc_ms_ema:.1f} ms/frame")
            if self.fps_ema > 0.0:
                self.lbl_fps.setText(f"FPS: {self.fps_ema:.1f}")
            self.lbl_eng.setText(
                f"engine.frame_index: {self.eng.frame_index}"
            )
            self.lbl_frame.setText(
                f"Frame: {self.frame_no} / {self.total_frames}"
            )

    def _present(self):
        if not self.video_ok or self.orig_bgr is None:
            return
        if self.lbl_orig is not None:
            # the ORIGINAL pane only changes when the source frame changes
            # (i.e. never during a paused slider scrub) — skip redundant work
            size = (self.lbl_orig.width(), self.lbl_orig.height())
            if self._orig_shown is not self.orig_bgr or self._orig_size != size:
                self._show_frame(
                    self.lbl_orig, self.orig_bgr, QImage.Format_BGR888
                )
                self._orig_shown = self.orig_bgr
                self._orig_size = size
        if self.lbl_proc is not None and self.proc_rgb is not None:
            self._show_frame(
                self.lbl_proc, self.proc_rgb, QImage.Format_RGB888
            )

    def _show_frame(self, label, arr, fmt=QImage.Format_BGR888):
        # cv2.INTER_LINEAR pre-scale is ~5x cheaper than QPixmap.scaled
        # SmoothTransformation and keeps present under the 80 fps budget.
        h, w = arr.shape[0], arr.shape[1]
        tw = max(1, label.width())
        th = max(1, label.height())
        scale = min(tw / w, th / h)
        dw = max(1, int(round(w * scale)))
        dh = max(1, int(round(h * scale)))
        buf_attr = (
            "_scale_buf_orig" if label is self.lbl_orig else "_scale_buf_proc"
        )
        buf = getattr(self, buf_attr)
        if buf is None or buf.shape[0] != dh or buf.shape[1] != dw:
            buf = np.empty((dh, dw, 3), np.uint8)
            setattr(self, buf_attr, buf)
        if arr.flags["C_CONTIGUOUS"]:
            cv2.resize(arr, (dw, dh), dst=buf, interpolation=cv2.INTER_LINEAR)
        else:
            cv2.resize(
                np.ascontiguousarray(arr),
                (dw, dh),
                dst=buf,
                interpolation=cv2.INTER_LINEAR,
            )
        image = QImage(buf.data, dw, dh, buf.strides[0], fmt)
        label.setPixmap(QPixmap.fromImage(image))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._present()

    def showEvent(self, event):
        super().showEvent(event)
        self._present()


def main():
    # Windows default timer quantum (~15.6 ms) would cap QTimer-driven
    # playback well below 80 fps; request 1 ms ticks for this process.
    _request_high_res_timer()
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    # run the real event loop (benchmark harnesses can install their own
    # stop conditions by calling app.quit via a single-shot timer)
    code = app.exec()
    if window.reader is not None:
        window.reader.stop()
    _release_high_res_timer()
    sys.exit(code)


if __name__ == "__main__":
    main()
