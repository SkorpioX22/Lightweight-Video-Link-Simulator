"""RF interference degradation: burst-gated scrolling noise bands, slice
corruption, heterodyne fringe, impulses, desense hangover, false-sync shifts.

Represents other RF sources (Wi-Fi, nearby VTXs, broadband noise).
Signature: BURSTY (clean frames between bursts, sharp onset/offset) +
BANDED/STRUCTURED (horizontal lines, slices) — distinct from continuous
snow (weak signal) and structural echoes (multipath).
In-place on planar YUV; accumulates false-sync row shifts.
"""
import numpy as np
from numba import njit, prange

from ..core.rng import hash_c1, hash2_01, hash3_01, value_noise1f, value_noise2f

# float32[16] state indices
I_BURST = 0
I_TIMER = 1
I_SCROLL = 2
I_FRINGE = 3
I_HANG = 4
I_LONG = 5
I_LONG_TIMER = 6
I_CLUSTER = 7
I_LONG_CD = 8

EFFECT_ID = 0x1F4E


@njit(cache=True)
def update_state(it, duty, scroll_speed, desense, long_event, seed, frame_idx):
    """Advance interference temporal state: packet bursts + sessions + takeover."""
    es = seed ^ (frame_idx * 0x9E3779B9) ^ EFFECT_ID

    # --- traffic session cluster (slow envelope on duty cycle) ---
    cluster = value_noise1f(frame_idx * 0.07, es ^ 0xC1)
    it[I_CLUSTER] = cluster
    eff_duty = duty * (0.35 + 1.3 * cluster)
    if eff_duty > 0.75:
        eff_duty = 0.75

    # --- long takeover event (power-up / sustained co-channel) ---
    it[I_LONG] = 0.0  # default this frame; re-set below
    # keep state across frames via I_LONG flag stored: recompute properly
    long_flag = it[I_LONG]
    # NOTE: I_LONG stored in state; reset each frame only after reading
    # We treat I_LONG as persistent flag; use separate temp:
    # (handled below with timer machine)

    # read persistent long flag from slot 5 (don't zero it here)
    # -- machine:
    # We need the flag BEFORE zeroing; state machine uses timer as truth.
    if it[I_LONG_TIMER] > 0.0:
        it[I_LONG_TIMER] -= 1.0
        it[I_LONG] = 1.0
        if it[I_LONG_TIMER] <= 0.0:
            it[I_LONG] = 0.0
            it[I_LONG_CD] = 60.0 + hash2_01(frame_idx, 61, es) * 120.0
    else:
        it[I_LONG] = 0.0
        if it[I_LONG_CD] > 0.0:
            it[I_LONG_CD] -= 1.0
        elif long_event > 0.0 and hash2_01(frame_idx, 67, es ^ 0xEE) < long_event:
            it[I_LONG] = 1.0
            it[I_LONG_TIMER] = 9.0 + hash2_01(frame_idx, 71, es) * 66.0  # 0.3-2.5 s

    # --- packet burst machine (sharp on/off, ms-scale at 30fps ~ frames) ---
    if it[I_TIMER] > 0.0:
        it[I_TIMER] -= 1.0
    else:
        if it[I_BURST] > 0.5:
            # burst just ended -> AGC desense hangover
            it[I_BURST] = 0.0
            if desense > 0.0:
                it[I_HANG] = desense
            # gap: short, shrinks as traffic duty rises (clean frames between)
            g = hash2_01(frame_idx, 73, es ^ 0x0F)
            it[I_TIMER] = 1.0 + g * g * 9.0 * (1.0 - 0.6 * eff_duty)
        else:
            it[I_BURST] = 1.0
            on = 1.0 + hash2_01(frame_idx, 79, es ^ 0xF0) * 4.0
            # during heavy traffic sessions bursts chain, but leave gaps
            if hash2_01(frame_idx, 83, es) < eff_duty * 0.35:
                on += 2.0
            it[I_TIMER] = on

    # forced ON during long takeover
    if it[I_LONG] > 0.5:
        it[I_BURST] = 1.0

    # --- desense hangover decay ---
    if it[I_HANG] > 0.0:
        it[I_HANG] *= 0.78
        if it[I_HANG] < 0.01:
            it[I_HANG] = 0.0

    # --- scroll & fringe phases ---
    it[I_SCROLL] += scroll_speed
    it[I_FRINGE] += 0.7 + 1.4 * value_noise1f(frame_idx * 0.09, es ^ 0x2B)


@njit(parallel=True, fastmath=True, cache=True)
def apply_interference(
    y, u, v,
    row_shift,
    it,
    band_amp,
    tint,
    fringe,
    slice_p,
    impulse,
    desense,
    sync_false,
    band_count,
    band_thr,
    height,
    width,
    seed,
    frame_idx,
):
    """In-place bursty banded interference."""
    es = seed ^ (frame_idx * 0x9E3779B9) ^ EFFECT_ID
    burst = it[I_BURST] > 0.5
    long_on = it[I_LONG] > 0.5
    hang = it[I_HANG]
    scroll = it[I_SCROLL]
    fringe_ph = it[I_FRINGE]

    # fringe runs even between bursts (narrowband/CW-like interferer)
    fringe_on = fringe > 1e-4

    slice_seed = es ^ (frame_idx * 0x5BD1)
    slice_h = 6 + int(hash2_01(frame_idx, 91, slice_seed) * 30)

    # precompute fringe constants
    f_k = 0.055 + hash2_01(frame_idx, 97, es ^ 0x33) * 0.11   # bands across height
    f_k2 = f_k * 1.13                                          # beating partner

    for j in prange(height):
        js = j * 7919

        # ---------- false H-sync row shifts (during strong bursts) ----------
        if (burst or long_on) and sync_false > 0.0:
            bid = j // 5
            if hash3_01(bid, frame_idx, 0x5F, es) < 0.03 * sync_false / 18.0 + 0.015 * (sync_false / 18.0):
                sh = hash_c1(bid ^ frame_idx, es ^ 0x5A) * sync_false
                row_shift[j] += sh

        # ---------- scroll-phased band mask along y ----------
        band_gate = 0.0
        if (burst or long_on) and band_amp > 0.0:
            # band_count = number of discrete bars across the frame height
            yy = (j - scroll) * band_count / max(1, height)
            bn = value_noise1f(yy, es ^ 0xBB)
            # threshold falls with severity -> denser bars at high I (QA: flat 75..99)
            if bn > band_thr:
                band_gate = (bn - band_thr) * 3.0
                if band_gate > 1.0:
                    band_gate = 1.0
                band_gate *= band_amp
            # long takeover raises the floor a little (NOT full coverage)
            if long_on:
                band_gate = max(band_gate, band_amp * 0.25)

        # ---------- slice corruption (sharp line boundaries) ----------
        slice_on = False
        sid = j // slice_h
        if (burst or long_on) and slice_p > 0.0:
            if hash2_01(sid, frame_idx, slice_seed ^ 0x99) < slice_p * 0.22:
                slice_on = True

        # per-slice tint polarity (green vs purple): coin per band-run so the
        # split stays ~50/50 (row-local hash skewed green in QA round 2)
        tint_id = sid if slice_on else (j // 48)
        tint_sign = 1.0 if hash2_01(tint_id, frame_idx, es ^ 0x71) < 0.5 else -1.0

        # fringe term is constant along x — evaluate once per row
        if fringe_on:
            fringe_s = fringe * 0.5 * (
                np.sin(j * f_k + fringe_ph) + np.sin(j * f_k2 - fringe_ph * 0.83)
            )
        else:
            fringe_s = 0.0
        # slice inversion mode is per-slice, not per-pixel
        slice_mode = -1.0
        if slice_on:
            slice_mode = hash2_01(sid, frame_idx, es ^ 0x61)

        for i in range(width):
            yy = y[j, i]
            uu = u[j, i]
            vv = v[j, i]

            # ---------- heterodyne fringe (continuous luminance sine + beat) ----------
            if fringe_on:
                yy += fringe_s

            # ---------- scrolling noise bands ----------
            if band_gate > 0.0:
                # streaky (8px horizontal correlation) = analog, not digital dither
                ib = i >> 3
                n1 = hash2_01(ib, js ^ frame_idx, es ^ 0xB1)
                n2 = hash2_01(ib + 1, js ^ frame_idx, es ^ 0xB1)
                streak = (n1 + n2 - 1.0) * 1.9
                replacement = 0.5 + streak * 0.55
                yy = yy * (1.0 - band_gate) + replacement * band_gate
                if tint > 0.0:
                    # replace (not add to) the scene's U/V inside the bar so the
                    # tint coin alone decides green vs purple polarity
                    uu *= (1.0 - band_gate)
                    vv *= (1.0 - band_gate)
                    tt = band_gate * tint * 0.30
                    if tint_sign > 0.0:
                        uu += tt          # purple: U+, V+
                        vv += tt * 0.7
                    else:
                        uu -= tt          # green: U-, V-
                        vv -= tt * 0.6
                # impulses inside bands
                if impulse > 0.0:
                    if hash2_01(i, js ^ (frame_idx << 2), es ^ 0xB2) < impulse * 0.07:
                        yy = 1.0 if hash2_01(i, js, es ^ 0xB3) < 0.5 else 0.0

            # ---------- slice corruption ----------
            if slice_on:
                sn = hash2_01(i >> 1, js ^ (frame_idx * 3), es ^ 0x51)
                sn2 = hash2_01((i >> 1) + 1, js ^ (frame_idx * 3), es ^ 0x51)
                static = 0.5 + (sn + sn2 - 1.0) * 1.4
                if slice_mode < 0.25:
                    yy = 1.0 - yy  # inverted strip
                else:
                    yy = static
                if tint > 0.2:
                    uu += tint * 0.10 * tint_sign
                    vv -= tint * 0.08 * tint_sign

            # ---------- desense hangover: extra fine snow after bursts ----------
            if hang > 0.001:
                hn = hash2_01(i, js ^ (frame_idx * 7), es ^ 0x41)
                hn2 = hash2_01(i + 1, js ^ (frame_idx * 7), es ^ 0x41)
                yy += (hn + hn2 - 1.0) * desense * hang * 2.4

            if yy < 0.0:
                yy = 0.0
            elif yy > 1.0:
                yy = 1.0
            y[j, i] = yy
            u[j, i] = uu
            v[j, i] = vv
