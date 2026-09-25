"""Multipath (reflected RF) degradation: ghosts, comb smear, block shifts,
white bands, FM edge spikes, white-flash / deep-null dropouts.

Structural-dominant, low snow (except click-noise during nulls). Ghost
delay is smooth in time (geometry); ghost amplitude/polarity flickers fast
(Doppler @5.8GHz). Line-coherent within a frame.

Reads source planes, writes destination planes (echoes need the clean
source of this stage). Also accumulates row displacement into row_shift.
"""
import numpy as np
from numba import njit, prange

from ..core.rng import hash2_01, hash3_01, value_noise1f, value_noise2f

# float32[16] state indices
M_A1 = 0
M_A2 = 1
M_A3 = 2
M_D1 = 3
M_D2 = 4
M_D3 = 5
M_PHI = 6
M_NULL_ACTIVE = 7
M_NULL_TIMER = 8
M_FLASH = 9
M_BAND_PHI = 10
M_NULL_CD = 11
M_FLASH_CD = 12

EFFECT_ID = 0x3A77

BLOCK_H = 24  # rows per displacement/monochrome block


@njit(cache=True)
def update_state(mp, g, seed, frame_idx, d1_base, d2_base, d3_base, a1_max, a2_max, a3_max,
                 null_rate, flash_rate, width):
    """Advance multipath temporal state. Fast amp flicker + slow delay drift."""
    es = seed ^ (frame_idx * 0x9E3779B9) ^ EFFECT_ID

    # --- ghost amplitudes: guaranteed visibility envelope + fast polarity flips ---
    # Real multipath flickers fast (Doppler @5.8 GHz) but a still frame should
    # almost always show SOME ghost: env in [0.65, 1.0], independent sign flips.
    env1 = 0.65 + 0.35 * value_noise1f(frame_idx * 0.9, es ^ 0xE1)
    env2 = 0.65 + 0.35 * value_noise1f(frame_idx * 1.3 + 31.0, es ^ 0xE2)
    env3 = 0.65 + 0.35 * value_noise1f(frame_idx * 1.7 + 61.0, es ^ 0xE3)
    # sign: random walk with occasional flips (phase inversion of reflection)
    sg1 = 1.0 if hash2_01(frame_idx, 101, es ^ 0xF1) < 0.78 else -1.0
    sg2 = 1.0 if hash2_01(frame_idx, 103, es ^ 0xF2) < 0.72 else -1.0
    sg3 = 1.0 if hash2_01(frame_idx, 107, es ^ 0xF3) < 0.65 else -1.0
    # second fast modulation so ghosted<->clean flicker remains visible
    # floor raised: ghost must stay correlatable in most frames
    fl1 = 0.78 + 0.22 * (value_noise1f(frame_idx * 2.3, es ^ 0xA1) * 2.0 - 1.0)
    fl2 = 0.78 + 0.22 * (value_noise1f(frame_idx * 3.1 + 50.0, es ^ 0xA2) * 2.0 - 1.0)
    fl3 = 0.78 + 0.22 * (value_noise1f(frame_idx * 3.9 + 90.0, es ^ 0xA3) * 2.0 - 1.0)
    mp[M_A1] = a1_max * env1 * fl1 * sg1
    mp[M_A2] = a2_max * env2 * fl2 * sg2
    mp[M_A3] = a3_max * env3 * fl3 * sg3

    # --- ghost delays: slow geometric drift (smooth, NOT per-frame random) ---
    drift1 = 1.0 + 0.22 * (value_noise1f(frame_idx * 0.035, es ^ 0xD1) * 2.0 - 1.0)
    drift2 = 1.0 + 0.18 * (value_noise1f(frame_idx * 0.028 + 17.0, es ^ 0xD2) * 2.0 - 1.0)
    drift3 = 1.0 + 0.15 * (value_noise1f(frame_idx * 0.022 + 41.0, es ^ 0xD3) * 2.0 - 1.0)
    mp[M_D1] = d1_base * drift1
    mp[M_D2] = d2_base * drift2
    mp[M_D3] = d3_base * drift3

    mp[M_PHI] += 0.09
    mp[M_BAND_PHI] += 0.05 + 0.11 * value_noise1f(frame_idx * 0.05, es ^ 0xBD)

    # --- white AGC flash events (1 frame, sharp) ---
    mp[M_FLASH] = 0.0
    if mp[M_FLASH_CD] > 0.0:
        mp[M_FLASH_CD] -= 1.0
    elif hash2_01(frame_idx, 29, es ^ 0xF1) < flash_rate:
        mp[M_FLASH] = 1.0
        mp[M_FLASH_CD] = 4.0 + hash2_01(frame_idx, 31, es) * 14.0

    # --- deep null dropout (1..4 frames of cancellation collapse) ---
    # must read as pocket dropouts, never 0.5s static windows (QA defect)
    if mp[M_NULL_ACTIVE] > 0.5:
        mp[M_NULL_TIMER] -= 1.0
        if mp[M_NULL_TIMER] <= 0.0:
            mp[M_NULL_ACTIVE] = 0.0
            mp[M_NULL_CD] = 20.0 + hash2_01(frame_idx, 37, es) * 60.0
    elif mp[M_NULL_CD] > 0.0:
        mp[M_NULL_CD] -= 1.0
    elif hash2_01(frame_idx, 41, es ^ 0xF2) < null_rate:
        mp[M_NULL_ACTIVE] = 1.0
        mp[M_NULL_TIMER] = 1.0 + hash2_01(frame_idx, 43, es) * 2.0


@njit(parallel=True, fastmath=True, cache=True)
def box_blur_h(src, dst, radius, height, width):
    """Per-row horizontal box blur of radius r (running sum; O(H*W))."""
    for j in prange(height):
        acc = 0.0
        cnt = 0
        for k in range(-radius, radius + 1):
            if 0 <= k < width:
                acc += src[j, k]
                cnt += 1
        for i in range(width):
            dst[j, i] = acc / cnt
            drop = i - radius
            if 0 <= drop < width:
                acc -= src[j, drop]
                cnt -= 1
            add = i + radius + 1
            if 0 <= add < width:
                acc += src[j, add]
                cnt += 1


@njit(parallel=True, fastmath=True, cache=True)
def apply_multipath(
    ys, us, vs,
    yd, ud, vd,
    row_shift,
    mp,
    comb_k,
    smear_px,
    ghost1_a_amp,   # signed amp range already scaled; used as |a| base
    block_shift,
    band_ripple,
    edge_spike,
    desat_bands,
    null_snow,
    height,
    width,
    seed,
    frame_idx,
    blur_y,
):
    """Write multipath-degraded planes from source planes (distinct buffers)."""
    es = seed ^ (frame_idx * 0x9E3779B9) ^ EFFECT_ID

    d1 = mp[M_D1]
    d2 = mp[M_D2]
    d3 = mp[M_D3]
    a1 = mp[M_A1]
    a2 = mp[M_A2]
    a3 = mp[M_A3]
    if comb_k > 0.0:
        a0 = comb_k
    else:
        a0 = 0.0

    null_on = mp[M_NULL_ACTIVE] > 0.5
    flash_on = mp[M_FLASH] > 0.5
    band_ph = mp[M_BAND_PHI]

    di1 = int(round(d1))
    di2 = int(round(d2))
    di3 = int(round(d3))
    # fractional part of d1 for sub-pixel comb tap
    frac1 = d1 - float(di1)

    i1 = di1 if di1 > 0 else 1          # comb tap ~1px (sub-pixel halo)
    radius = int(round(smear_px))
    if radius < 1:
        radius = 0

    use_ghosts = (abs(a1) > 1e-4) or (abs(a2) > 1e-4) or (abs(a3) > 1e-4)
    use_a1 = abs(a1) > 1e-4
    use_a2 = abs(a2) > 1e-4
    use_a3 = abs(a3) > 1e-4
    use_comb = a0 > 1e-4
    use_smear = radius > 0
    edge_sp = 3 + int(hash2_01(frame_idx, 55, es) * 5)  # const per frame

    # normalize DC gain of the echo mix
    dc = 1.0 + abs(a1) + abs(a2) + abs(a3)  # mix-out form below keeps DC=1 anyway

    for j in prange(height):
        js = j * 7919

        # ---- block displacement (row plateaus, smooth drift in time) ----
        blk = j // BLOCK_H
        if block_shift > 0.0:
            r = hash2_01(blk, 0xB1, es ^ 0xD1)
            if r < 0.35:
                nfield = value_noise2f(blk * 0.8, frame_idx * 0.02, es ^ 0xB2) * 2.0 - 1.0
                row_shift[j] += nfield * block_shift

        # ---- monochrome blocks (color-burst echo cancellation on some bands) ----
        mono = False
        if desat_bands > 0.0:
            r2 = hash2_01(blk, 0xC1, es ^ 0xD2)
            if r2 < desat_bands * 0.30:
                mono = True

        # ---- white horizontal ripple band (multipath envelope) ----
        band_add = 0.0
        if band_ripple > 0.0:
            bn = value_noise1f(j * 0.045 + band_ph, es ^ 0xBD2)
            if bn > 0.62:
                band_add = band_ripple * (bn - 0.62) * 2.8

        for i in range(width):
            base = ys[j, i]

            # ---------- echo mix (ghosts + sub-pixel comb) ----------
            # out = base*(1-sum a) + sum a*echo  => DC gain 1 for aligned taps
            val = base
            if use_comb:
                # sub-pixel tap at ~1px: lerp between x-i1 and x-i1-1
                xa = i - i1
                xb = xa - 1
                if xa < 0:
                    xa = 0
                if xb < 0:
                    xb = 0
                e0 = ys[j, xa] * (1.0 - frac1) + ys[j, xb] * frac1
                val += a0 * (e0 - base)
            if use_ghosts:
                if use_a1:
                    xa = i - di1
                    if xa < 0:
                        xa = 0
                    val += a1 * (ys[j, xa] - base)
                if use_a2:
                    xa = i - di2
                    if xa < 0:
                        xa = 0
                    val += a2 * (ys[j, xa] - base)
                if use_a3:
                    xa = i - di3
                    if xa < 0:
                        xa = 0
                    val += a3 * (ys[j, xa] - base)

            # ---------- horizontal smear (frequency-selective fade proxy) ----------
            # Blend (never replace): echoes must remain correlatable — cap weight
            # so ghost taps always contribute (QA: smear was drowning echoes).
            # blur_y is a precomputed per-row box blur (box_blur_h).
            if use_smear:
                blur = blur_y[j, i]
                smear_w = 0.30 + 0.35 * min(1.0, smear_px / 3.5)   # max 0.65
                val = val * (1.0 - smear_w) + blur * smear_w

            # ---------- FM edge spikes (bright/dark stripe right of strong edges) ----------
            if edge_spike > 0.0:
                xa = i - edge_sp
                if xa >= 1:
                    g = ys[j, xa] - ys[j, xa - 1]
                    if g > 0.28:
                        val += edge_spike * min(g, 1.0) * 0.55
                    elif g < -0.28:
                        val -= edge_spike * min(-g, 1.0) * 0.55

            # ---------- white ripple band ----------
            val += band_add

            # ---------- deep null: image collapses into MONOCHROME click-noise ----------
            if null_on:
                nn1 = hash2_01(i, js ^ frame_idx, es ^ 0xE1)
                nn2 = hash2_01(i + 1, js ^ frame_idx, es ^ 0xE1)
                noise_field = (nn1 + nn2 - 1.0) * 1.9
                staticv = 0.08 + abs(noise_field) * null_snow
                keep = 0.10
                val = val * keep + staticv * (1.0 - keep)
                val += noise_field * null_snow * 0.30

            # ---------- white AGC flash (content stays visible through wash) ----------
            if flash_on:
                val = val * 1.06 + 0.28

            if val < 0.0:
                val = 0.0
            elif val > 1.0:
                val = 1.0
            yd[j, i] = val

            uu = us[j, i]
            vv = vs[j, i]
            if mono:
                uu = 0.0
                vv = 0.0
            elif null_on:
                # burst lost -> color killer engaged: fully monochrome static
                uu = 0.0
                vv = 0.0
            elif flash_on:
                uu *= 0.2
                vv *= 0.2
            ud[j, i] = uu
            vd[j, i] = vv
