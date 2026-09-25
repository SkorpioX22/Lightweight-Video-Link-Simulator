"""AnalogVideoBreakupEngine — public API + frame pipeline.

Pipeline (per docs/research compositing order):
  clean RGB
    -> planar YUV
    -> multipath   (structural: echoes / comb / displacement / bands / nulls)
    -> weak signal (snow / chroma kill / sparklies; accumulates jitter+tear)
    -> interference(burst bands / slices / fringe; accumulates false-sync)
    -> geometry    (row_shift + vertical roll + roll-seam noise)
    -> RGB

0,0,0 fast-path returns the input unchanged (bit-identical).
Deterministic: all randomness derives from (seed, frame_index).
"""
import numpy as np
from numba import njit, prange

from ..core.color import rgb_to_yuv_planar, yuv_planar_to_rgb
from ..core.params import (
    PRESETS,
    clamp_param,
    interference_stages,
    multipath_stages,
    normalize_params,
    raw,
    signal_stages,
)
from ..core.rng import hash2_01
from ..core.state import init_multipath_defaults, reset_state
from ..interference import interference as interference_mod
from ..multipath import multipath as multipath_mod
from ..signal import weak_signal as signal_mod


@njit(parallel=True, fastmath=True, cache=True)
def apply_geometry(y_src, u_src, v_src, y_dst, u_dst, v_dst, row_shift, roll_rows,
                   height, width, seed, frame_idx, seam_on):
    """Apply accumulated horizontal row shifts + vertical roll.

    out(y, x) = src((y + roll) mod h, x - row_shift[src_row])
    Reads src / writes dst (must NOT alias — in-place row shifts corrupt).
    Roll seam rows get injected snow (blanking-interval noise).
    """
    es = seed ^ (frame_idx * 0x9E3779B9)
    seam_row = (height - int(roll_rows)) % height if (seam_on and roll_rows != 0) else -10
    for j in prange(height):
        sj = j + int(roll_rows)
        sj = sj % height
        shift = row_shift[sj]
        isi = int(np.floor(shift))
        frac = shift - isi
        seam = abs(j - seam_row) < 3

        # unshifted row, no seam: plain copy (skips bilinear taps)
        if isi == 0 and frac == 0.0 and not seam:
            for i in range(width):
                y_dst[j, i] = y_src[sj, i]
                u_dst[j, i] = u_src[sj, i]
                v_dst[j, i] = v_src[sj, i]
            continue

        js = sj * 7919
        # integer shift, no seam: direct index (frac terms are 0/1)
        if frac == 0.0 and not seam:
            for i in range(width):
                xa = i - isi
                if xa < 0:
                    xa = 0
                elif xa >= width:
                    xa = width - 1
                y_dst[j, i] = y_src[sj, xa]
                u_dst[j, i] = u_src[sj, xa]
                v_dst[j, i] = v_src[sj, xa]
            continue

        for i in range(width):
            xa = i - isi
            xb = xa - 1
            if xa < 0:
                xa = 0
            elif xa >= width:
                xa = width - 1
            if xb < 0:
                xb = 0
            elif xb >= width:
                xb = width - 1
            yy = y_src[sj, xa] * (1.0 - frac) + y_src[sj, xb] * frac
            uu = u_src[sj, xa] * (1.0 - frac) + u_src[sj, xb] * frac
            vv = v_src[sj, xa] * (1.0 - frac) + v_src[sj, xb] * frac
            if seam:
                n = hash2_01(i, js ^ frame_idx, es ^ 0x5EA)
                yy = 0.03 + n * n * 0.5
                uu *= 0.1
                vv *= 0.1
            y_dst[j, i] = yy
            u_dst[j, i] = uu
            v_dst[j, i] = vv


class AnalogVideoBreakupEngine:
    """RF-to-video degradation model for FPV simulators.

    Parameters (0..99, 0 = no effect, 99 = extreme):
      signalStrength  - weak link: snow -> color kill -> tear/roll -> static
      multipath       - reflections: ghosts -> comb smear -> blocks -> nulls
      rfInterference  - other RF: burst noise bars -> slices -> fringe
    """

    VERSION = "1.0.0"

    def __init__(self, seed=12345, width=None, height=None):
        self.seed = int(seed) & 0xFFFFFFFF
        self.frame_index = 0
        self._state = reset_state()
        init_multipath_defaults(self._state["multipath"], width or 0)
        self._w = 0
        self._h = 0
        self._buf = None
        if width and height:
            self._ensure(int(width), int(height))
        self.last_params = normalize_params(None)
        self.last_stages = None

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
        """Clear temporal state (roll events, killer latch, burst machine...)."""
        self._state = reset_state()
        if self._w:
            init_multipath_defaults(self._state["multipath"], self._w)
            rs = np.zeros(self._h, dtype=np.float32)
            self._state["row_shift"] = rs
            self._buf["row_shift"] = rs
        self.frame_index = 0

    def setSeed(self, seed):
        self.seed = int(seed) & 0xFFFFFFFF

    # ---------------- internals ----------------
    def _ensure(self, w, h):
        if self._buf is not None and self._w == w and self._h == h:
            return
        self._w, self._h = w, h
        self._buf = {
            "y": np.empty((h, w), dtype=np.float32),
            "u": np.empty((h, w), dtype=np.float32),
            "v": np.empty((h, w), dtype=np.float32),
            "y2": np.empty((h, w), dtype=np.float32),
            "u2": np.empty((h, w), dtype=np.float32),
            "v2": np.empty((h, w), dtype=np.float32),
            "yblur": np.empty((h, w), dtype=np.float32),
            "row_shift": np.zeros(h, dtype=np.float32),
        }
        self._state["row_shift"] = np.zeros(h, dtype=np.float32)

    def _resolve(self, params):
        p = normalize_params(params) if params is not None else normalize_params(self.last_params)
        self.last_params = p
        rs, rm, ri = raw(p["signalStrength"]), raw(p["multipath"]), raw(p["rfInterference"])
        stages = {
            "signal": signal_stages(rs),
            "multipath": multipath_stages(rm),
            "interference": interference_stages(ri),
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
            if advance:
                self.frame_index += 1
            return out

        b = self._buf
        st = self._state
        sig, mp, it = st["signal"], st["multipath"], st["interference"]
        row_shift = b["row_shift"]
        row_shift[:] = 0.0
        fi = self.frame_index
        seed = self.seed

        # RGB uint8 -> planar float YUV (scale fused in kernel)
        rgb_to_yuv_planar(arr, b["y"], b["u"], b["v"])

        ss = stages["signal"]
        ms = stages["multipath"]
        isf = stages["interference"]

        # -------- 1) multipath (structural; reads src writes dst) --------
        do_mp = ms["ghost1_a"] > 0 or ms["comb_k"] > 0 or ms["smear_px"] > 0 or \
                ms["block_shift"] > 0 or ms["band_ripple"] > 0 or ms["null_rate"] > 0 or \
                ms["flash_rate"] > 0
        if do_mp:
            multipath_mod.update_state(
                mp, ms["null_rate"], seed, fi,
                ms["ghost1_d"], ms["ghost2_d"], ms["ghost1_d"] * 0.45,
                ms["ghost1_a"], ms["ghost2_a"], ms["ghost3_a"],
                ms["null_rate"], ms["flash_rate"], w,
            )
            smear_r = int(round(ms["smear_px"]))
            if smear_r < 1:
                smear_r = 0
            if smear_r > 0:
                multipath_mod.box_blur_h(b["y"], b["yblur"], smear_r, h, w)
            multipath_mod.apply_multipath(
                b["y"], b["u"], b["v"],
                b["y2"], b["u2"], b["v2"],
                row_shift, mp,
                ms["comb_k"], ms["smear_px"], ms["ghost1_a"],
                ms["block_shift"], ms["band_ripple"], ms["edge_spike"],
                ms["desat_bands"], ms["null_snow"],
                h, w, seed, fi,
                b["yblur"],
            )
            b["y"], b["y2"] = b["y2"], b["y"]
            b["u"], b["u2"] = b["u2"], b["u"]
            b["v"], b["v2"] = b["v2"], b["v"]
        else:
            # do_mp False: stages all-off; state machine does not tick (params
            # mid-range changes that turn mp off later reset via engine.reset())
            pass

        # -------- 2) weak signal (in place) --------
        do_sig = ss["luma_noise"] > 0 or ss["sparklie"] > 0 or ss["h_jitter"] > 0 or \
                 ss["tear"] > 0 or ss["roll"] > 0 or ss["sat_loss"] > 0 or \
                 ss["static_kill"] > 0 or ss["chroma_noise"] > 0
        if do_sig:
            signal_mod.update_state(sig, ss["roll"] * 0.12, seed, fi)
            # killer thresholds live in raw space (params.py); sat_loss =
            # smoothstep(0.18, 0.50, raw), so invert them into sat_loss space
            killer_on_sat = _raw_to_sat(ss["killer_on_th"])
            killer_off_sat = _raw_to_sat(ss["killer_off_th"])
            signal_mod.apply_signal(
                b["y"], b["u"], b["v"],
                row_shift, sig,
                ss["luma_noise"], ss["chroma_noise"], ss["sat_loss"],
                ss["hue_wander"], killer_on_sat, killer_off_sat,
                ss["sparklie"], ss["h_jitter"], ss["tear"], ss["static_kill"],
                ss["black_lift"],
                h, w, seed, fi,
            )

        # -------- 3) RF interference (in place) --------
        do_int = isf["band_amp"] > 0 or isf["fringe"] > 0 or isf["slice"] > 0 or \
                 isf["desense"] > 0 or isf["sync_false"] > 0
        if do_int:
            interference_mod.update_state(
                it, isf["duty"], isf["scroll"], isf["desense"], isf["long_event"],
                seed, fi,
            )
            interference_mod.apply_interference(
                b["y"], b["u"], b["v"],
                row_shift, it,
                isf["band_amp"], isf["tint"], isf["fringe"], isf["slice"],
                isf["impulse"], isf["desense"], isf["sync_false"], isf["band_count"],
                isf["band_thr"],
                h, w, seed, fi,
            )

        # -------- 4) geometry: row shifts + vertical roll --------
        roll = float(int(sig[signal_mod.S_ROLL_POS]))
        seam_on = sig[signal_mod.S_ROLL_ACTIVE] > 0.5
        if roll != 0.0 or np.any(row_shift) or seam_on:
            apply_geometry(
                b["y"], b["u"], b["v"], b["y2"], b["u2"], b["v2"],
                row_shift, roll, h, w, seed, fi, seam_on,
            )
            b["y"], b["y2"] = b["y2"], b["y"]
            b["u"], b["u2"] = b["u2"], b["u"]
            b["v"], b["v2"] = b["v2"], b["v"]

        # -------- 5) YUV -> RGB uint8 (clamp fused in kernel) --------
        if out is None:
            out = np.empty((h, w, 3), dtype=np.uint8)
        yuv_planar_to_rgb(b["y"], b["u"], b["v"], out)

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
        # (includes heavy paths: roll/geometry, nulls, takeover events)
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
        self.reset()
        return _time.perf_counter() - t0


def _raw_to_sat(raw_v):
    """Invert sat_loss = smoothstep(0.18, 0.50, raw) for threshold raw_v."""
    from ..core.params import smoothstep

    # monotonic on [0.18,0.5]; binary search fine and cheap
    if raw_v <= 0.18:
        return 0.0
    if raw_v >= 0.50:
        return 1.0
    lo, hi = 0.18, 0.50
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if smoothstep(0.18, 0.50, mid) < raw_v:
            lo = mid
        else:
            hi = mid
    return smoothstep(0.18, 0.50, 0.5 * (lo + hi))


__all__ = ["AnalogVideoBreakupEngine", "PRESETS"]
