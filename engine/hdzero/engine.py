"""HDZeroEngine — HDZero digital FPV link model.

Public API mirrors AnalogVideoBreakupEngine (same three 0..99 params,
same process/set/get contract) so the demo GUI can switch engines
unchanged. Digital visual model (docs/research/hdzero.md):

  clean RGB
    -> codec chroma quant  (always-on codec look, gated on signal > 0)
    -> corrupt 8x8 blocks  (scattered DCT garbage + clustered punch-outs)
    -> white speckle       (single-bit errors)
    -> takeover events     (band / rainbow / lines / black bursts)
    -> signal-loss screen  (real no-signal image: growing 8x8 patches
                            ramp into the full screen during loss runs;
                            signal 99 = dead link: the full screen only)
    -> breakup-to-black    (extreme signal, never at 99)

0,0,0 fast-path returns the input unchanged (bit-identical).
Deterministic: all randomness derives from (seed, frame_index).
"""
import numpy as np

from ..core.params import clamp_param, normalize_params, raw
from .kernels import (
    apply_speckle,
    black_hit,
    chroma_quant,
    compute_bmap,
    fill_black,
    fill_blocks,
    fill_loss,
    fill_loss_patches,
    render_event,
    update_hz_state,
)
from .params import (
    hdz_interference_stages,
    hdz_multipath_stages,
    hdz_signal_stages,
)
from .state import (
    HZ_DRIFT,
    HZ_ENV,
    HZ_EVT_ACTIVE,
    HZ_EVT_CD,
    HZ_EVT_H,
    HZ_EVT_TIMER,
    HZ_EVT_TYPE,
    HZ_EVT_Y0,
    HZ_LOSS,
    HZ_LOSS_SEED,
    HZ_LOSS_T,
    reset_hz_state,
)

# shipped no-signal screen (real HDZero capture); loaded lazily once
_NOSIGNAL_SRC = None
_NOSIGNAL_TRIED = False


def _load_no_signal_src():
    """Load the bundled no-signal screen as RGB uint8, or None.

    Tries no_signal.npy first, then no_signal.png (needs OpenCV — the
    engine itself only requires numpy + numba, so a missing cv2 or file
    silently falls back to the procedural loss screen in fill_loss).
    """
    global _NOSIGNAL_SRC, _NOSIGNAL_TRIED
    if _NOSIGNAL_TRIED:
        return _NOSIGNAL_SRC
    _NOSIGNAL_TRIED = True
    import os

    d = os.path.dirname(os.path.abspath(__file__))
    try:
        npy = os.path.join(d, "no_signal.npy")
        png = os.path.join(d, "no_signal.png")
        if os.path.exists(npy):
            a = np.load(npy)
            if a.ndim == 3 and a.shape[2] == 3 and a.dtype == np.uint8:
                _NOSIGNAL_SRC = np.ascontiguousarray(a)
        elif os.path.exists(png):
            import cv2

            bgr = cv2.imread(png, cv2.IMREAD_COLOR)
            if bgr is not None:
                _NOSIGNAL_SRC = np.ascontiguousarray(bgr[:, :, ::-1])
    except Exception:
        _NOSIGNAL_SRC = None
    return _NOSIGNAL_SRC


def _fit_no_signal(src, w, h):
    """Center-crop the no-signal screen to the frame aspect, scale nearest."""
    sh, sw = src.shape[0], src.shape[1]
    target = w / float(h)
    if sw / float(sh) > target:
        nw = max(1, int(round(sh * target)))
        x0 = (sw - nw) // 2
        sub = src[:, x0:x0 + nw]
    else:
        nh = max(1, int(round(sw / target)))
        y0 = (sh - nh) // 2
        sub = src[y0:y0 + nh, :]
    ys = (np.arange(h, dtype=np.int64) * sub.shape[0]) // h
    xs = (np.arange(w, dtype=np.int64) * sub.shape[1]) // w
    return np.ascontiguousarray(sub[ys][:, xs])


class HDZeroEngine:
    """Digital (HDZero) video-link degradation model for FPV simulators.

    Parameters (0..99, 0 = no effect, 99 = extreme):
      signalStrength  - bit errors: corrupt blocks -> speckle -> blackout
      multipath       - burst errors: clustered punch-out patches
      rfInterference  - takeovers: band scramble / rainbow / lines / black
    """

    VERSION = "1.0.0"

    def __init__(self, seed=12345, width=None, height=None):
        self.seed = int(seed) & 0xFFFFFFFF
        self.frame_index = 0
        self._hz = reset_hz_state()
        self._w = 0
        self._h = 0
        self._bmap = None
        self._loss_full = None
        if width and height:
            self._ensure(int(width), int(height))
        self.last_params = normalize_params(None)
        self.last_stages = None
        self.in_loss = False

    # ---------------- public parameter API ----------------
    def setSignalStrength(self, v):
        self.last_params["signalStrength"] = clamp_param(v)

    def setMultipath(self, v):
        self.last_params["multipath"] = clamp_param(v)

    def setRFInterference(self, v):
        self.last_params["rfInterference"] = clamp_param(v)

    def setParams(self, params):
        self.last_params = normalize_params(params)
        return self.last_params

    def getParams(self):
        return dict(self.last_params)

    def reset(self):
        """Clear temporal state (event machine, cluster drift/envelope)."""
        self._hz = reset_hz_state()
        self.frame_index = 0
        self.in_loss = False

    def setSeed(self, seed):
        self.seed = int(seed) & 0xFFFFFFFF

    # ---------------- internals ----------------
    def _ensure(self, w, h):
        if self._bmap is not None and self._w == w and self._h == h:
            return
        self._w, self._h = w, h
        bh = (h + 7) // 8
        bw = (w + 7) // 8
        self._bmap = np.zeros((bh, bw), dtype=np.uint8)
        # frame-sized real no-signal screen (asset -> fallback: procedural)
        src = _load_no_signal_src()
        self._loss_full = _fit_no_signal(src, w, h) if src is not None else None

    def _resolve(self, params):
        p = normalize_params(params) if params is not None else normalize_params(self.last_params)
        self.last_params = p
        rs, rm, ri = raw(p["signalStrength"]), raw(p["multipath"]), raw(p["rfInterference"])
        stages = {
            "signal": hdz_signal_stages(rs),
            "multipath": hdz_multipath_stages(rm),
            "interference": hdz_interference_stages(ri),
            "raw": {"signal": rs, "multipath": rm, "interference": ri},
        }
        self.last_stages = stages
        return p, stages

    # ---------------- frame processing ----------------
    def process(self, frame, params=None, advance=True, out=None):
        """Degrade one RGB uint8 frame (H,W,3). Returns new uint8 array.

        params: optional dict {signalStrength, multipath, rfInterference}
                (0..99 each). Omitted/None keys are treated as 0 for this
                call (defaults always merge from a clean base).
        advance: bump frame_index after processing (set False for idempotent tests).
        out: optional uint8 (H,W,3) destination — skips the per-frame
             allocation when the caller reuses a buffer (GUI playback path).
        """
        if frame is None:
            raise ValueError("frame is None")
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"expected (H,W,3) RGB frame, got shape {arr.shape}")
        if arr.dtype != np.uint8:
            raise ValueError(f"expected uint8 frame, got dtype {arr.dtype}")

        p, stages = self._resolve(params)
        h, w = arr.shape[0], arr.shape[1]
        self._ensure(w, h)

        if out is not None:
            if out.shape != (h, w, 3) or out.dtype != np.uint8:
                raise ValueError(
                    f"out must be uint8 ({h},{w},3), got {out.dtype} {out.shape}"
                )
            if out is arr:
                raise ValueError("out must not alias the input frame")

        all_zero = (
            stages["raw"]["signal"] == 0.0
            and stages["raw"]["multipath"] == 0.0
            and stages["raw"]["interference"] == 0.0
        )
        if all_zero:
            if out is None:
                out = arr.copy()
            else:
                out[:] = arr
            self.in_loss = False
            if advance:
                self.frame_index += 1
            return out

        fi = self.frame_index
        seed = self.seed
        hz = self._hz
        sig = stages["signal"]
        mp = stages["multipath"]
        it = stages["interference"]
        rs = stages["raw"]["signal"]
        ri = stages["raw"]["interference"]
        # signal 99 = dead link: the no-signal screen, nothing else — no
        # patch ramp (full screen from the first frame) and no blackouts
        dead = rs >= 0.995

        # zero interference: kill any running takeover, then let the machine
        # idle (duty = 0 -> no new events can start)
        if ri <= 0.0:
            hz[HZ_EVT_ACTIVE] = 0.0
            hz[HZ_EVT_CD] = 0.0
        # zero signal: drop out of the signal-loss state immediately
        if rs <= 0.0:
            hz[HZ_LOSS] = 0.0
            hz[HZ_LOSS_T] = 0.0
        # dead link: re-assert the loss latch every frame so the run never
        # ends while signal stays at 99
        if dead and hz[HZ_LOSS] < 2.0:
            hz[HZ_LOSS] = 2.0
        update_hz_state(
            hz,
            it["duty"], it["dur_lo"], it["dur_hi"], it["gap_lo"], it["gap_hi"],
            it["w_rain"], it["w_lines"], it["w_black"],
            it["band_speed"], it["band_frac"], mp["env_rate"], mp["mp_speed"],
            sig["loss_p"], sig["loss_dur_lo"], sig["loss_dur_hi"],
            seed, fi,
        )

        # base copy (digital links keep valid pixels clean)
        if out is None:
            out = np.empty((h, w, 3), dtype=np.uint8)
        out[:] = arr

        # 1) codec chroma quant (rate-drop color blockiness; signal-gated)
        chroma_quant(out, sig["chroma_q"] if rs > 0.0 else 0.0)

        # 2) corrupt 8x8 blocks: scattered signal errors + cluster punch-outs
        compute_bmap(
            self._bmap,
            sig["block_err"], mp["cl_cov"], mp["cl_boost"], sig["white_blk"],
            hz[HZ_ENV], hz[HZ_DRIFT], seed, fi,
        )
        fill_blocks(out, arr, self._bmap, seed, fi)

        # 3) white speckle (single-bit errors)
        apply_speckle(out, sig["speckle"], seed, fi)

        # 4) takeover events (band / rainbow / lines / black)
        render_event(
            out, arr,
            hz[HZ_EVT_TYPE], hz[HZ_EVT_ACTIVE], hz[HZ_EVT_Y0], hz[HZ_EVT_H],
            it["density"], seed, fi,
        )

        # 5) signal-loss screen: random 8x8 patches of the real no-signal
        # image accumulate during the run's ramp, then the full screen
        # takes over (procedural rainbow fallback without the asset);
        # at signal 99 the full screen is forced every frame
        loss_on = hz[HZ_LOSS] > 0.0
        if loss_on:
            if dead:
                cov = 1.0
            else:
                ramp = sig["loss_ramp"]
                cov = hz[HZ_LOSS_T] / ramp if ramp > 0.0 else 1.0
                if cov > 1.0:
                    cov = 1.0
            if self._loss_full is None:
                fill_loss(out, 1.0, seed, fi)
            elif cov >= 0.995:
                out[:] = self._loss_full
            else:
                fill_loss_patches(
                    out, self._loss_full, cov,
                    hz[HZ_LOSS_SEED], seed, fi)
        self.in_loss = loss_on

        # 6) breakup-to-black at extreme signal (overrides the loss screen;
        # still evaluated at 99 so the kernel stays compiled, but the dead
        # link shows the no-signal screen only)
        hit = black_hit(sig["black_frame"], seed, fi) if rs > 0.0 else False
        if dead:
            hit = False
        fill_black(out, 1.0 if hit else 0.0)

        if advance:
            self.frame_index += 1
        return out

    # spec-style alias
    def processFrame(self, frame, params=None):
        return self.process(frame, params)

    def warmup(self, width=64, height=64):
        """JIT-compile every kernel with a tiny throwaway frame.

        First use of each effect path otherwise costs a multi-second compile
        mid-interaction. Safe to call repeatedly (already-compiled kernels
        return instantly); with numba cache=True the compile also persists
        to disk across process launches.
        """
        import time as _time

        t0 = _time.perf_counter()
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = 96
        # each family separately so every branch compiles, then all together
        for p in (
            {"signalStrength": 99, "multipath": 0, "rfInterference": 0},
            {"signalStrength": 0, "multipath": 99, "rfInterference": 0},
            {"signalStrength": 0, "multipath": 0, "rfInterference": 99},
            {"signalStrength": 99, "multipath": 99, "rfInterference": 99},
        ):
            self.process(frame, p, advance=False)
        # run several frames so temporal machines cross more branches
        for _ in range(8):
            self.process(
                frame,
                {"signalStrength": 99, "multipath": 99, "rfInterference": 99},
                advance=True,
            )
        # force every takeover event type + blackout so those branches compile
        # regardless of what the random event machine happened to pick
        for et in range(4):
            self._hz[HZ_EVT_ACTIVE] = 1.0
            self._hz[HZ_EVT_TYPE] = float(et)
            self._hz[HZ_EVT_TIMER] = 2.0
            self._hz[HZ_EVT_Y0] = 0.30
            self._hz[HZ_EVT_H] = 0.20
            self.process(frame, {"signalStrength": 99, "rfInterference": 99}, advance=True)
        # force a partial-coverage loss frame so fill_loss_patches (or the
        # procedural fallback) compiles — signal 97, because 99 is the dead
        # link (always the full screen, a plain array copy)
        self._hz[HZ_LOSS] = 6.0
        self._hz[HZ_LOSS_T] = 1.0
        self.process(frame, {"signalStrength": 97}, advance=True)
        fill_black(frame, 1.0)
        self.reset()
        return _time.perf_counter() - t0


__all__ = ["HDZeroEngine"]
