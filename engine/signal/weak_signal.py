"""Weak-signal degradation: snow, chroma loss/color-killer, sparklies, sync.

Represents decreasing received RF signal strength (distance / obstacles /
antenna orientation). Progressive staged breakdown per docs/research:
grain -> snow -> hue wander -> color kill -> heavy snow + sparklies ->
line jitter / tearing -> roll events -> near-total static.

Owns temporal snow envelope, color-killer hysteresis and the roll event
machine. Geometry contributions go into row_shift; roll comes from sig[]
and is applied by the renderer once per frame.
"""
import numpy as np
from numba import njit, prange

from ..core.rng import hash_c1, hash2_01, hash3_01, value_noise1c, value_noise2f

# float32[16] state indices
S_ROLL_ACTIVE = 0
S_ROLL_POS = 1
S_ROLL_TIMER = 2
S_ROLL_COOLDOWN = 3
S_ROLL_DIR = 4
S_KILLER = 5
S_HUE = 6
S_NOISE_ENV = 7

EFFECT_ID = 0x51A1


@njit(cache=True)
def update_state(sig, roll_p, seed, frame_idx):
    """Advance weak-signal temporal state once per frame."""
    es = seed ^ (frame_idx * 0x9E3779B9)

    # slow snow envelope (fast fading layered on the range fade)
    env = 0.72 + 0.56 * value_noise2f(frame_idx * 0.13, 3.7, es ^ EFFECT_ID)
    sig[S_NOISE_ENV] = env

    # slow hue wander phase
    sig[S_HUE] += 0.028 + 0.09 * hash_c1(frame_idx, es ^ 0xABC)

    # vertical roll event machine (bursty with recovery)
    if sig[S_ROLL_ACTIVE] > 0.5:
        speed = 12.0 + hash2_01(frame_idx, 7, es) * 26.0
        sig[S_ROLL_POS] += sig[S_ROLL_DIR] * speed
        sig[S_ROLL_TIMER] -= 1.0
        if sig[S_ROLL_TIMER] <= 0.0:
            sig[S_ROLL_ACTIVE] = 0.0
            sig[S_ROLL_POS] = 0.0  # snap: V-sync re-locks
            sig[S_ROLL_COOLDOWN] = 30.0 + hash2_01(frame_idx, 11, es) * 90.0
    elif sig[S_ROLL_COOLDOWN] > 0.0:
        sig[S_ROLL_COOLDOWN] -= 1.0
    elif hash2_01(frame_idx, 13, es ^ EFFECT_ID) < roll_p:
        sig[S_ROLL_ACTIVE] = 1.0
        sig[S_ROLL_TIMER] = 6.0 + hash2_01(frame_idx, 17, es) * 54.0  # 0.2-2 s @30fps
        sig[S_ROLL_DIR] = 1.0 if hash2_01(frame_idx, 19, es) < 0.5 else -1.0
        sig[S_ROLL_POS] = 0.0


@njit(parallel=True, fastmath=True, cache=True)
def apply_signal(
    y, u, v,
    row_shift,
    sig,
    luma_noise,
    chroma_noise,
    sat_loss,
    hue_wander,
    killer_on_th,
    killer_off_th,
    sparklie,
    h_jitter,
    tear,
    static_kill,
    black_lift,
    height,
    width,
    seed,
    frame_idx,
):
    """In-place weak-signal corruption of planar YUV (float32, Y in [0,1])."""
    es = seed ^ (frame_idx * 0x9E3779B9) ^ EFFECT_ID
    env = sig[S_NOISE_ENV]
    hue_ph = sig[S_HUE]

    # color-killer hysteresis (sat_loss monotonic in raw severity)
    if sig[S_KILLER] > 0.5:
        if sat_loss < killer_off_th:
            sig[S_KILLER] = 0.0
    else:
        if sat_loss >= killer_on_th:
            sig[S_KILLER] = 1.0
    killer = sig[S_KILLER]

    sat_gain = 0.0 if killer > 0.5 else max(0.0, 1.0 - sat_loss)

    sigma = luma_noise * env
    n_gain = sigma * 2.45                 # triangular approx -> target std sigma
    c_gain = chroma_noise * env
    img_keep = 1.0 - static_kill * 0.93
    static_mix = static_kill

    tear_block = frame_idx >> 2           # tear pattern holds ~4 frames
    band_seed = es ^ (tear_block * 0x2545)
    band_h = 4 + int(hash2_01(tear_block, 3, band_seed) * 36)

    for j in prange(height):
        js = j * 7919

        # per-line hue: slow wander + burst-phase jitter per line/frame
        if hue_wander != 0.0:
            line_ang = hue_wander * (
                np.sin(hue_ph + j * 0.11)
                + 0.5 * hash_c1(js ^ frame_idx, es ^ 0xBEEF)
            )
            ca = np.cos(line_ang)
            sa = np.sin(line_ang)
        else:
            line_ang = 0.0
            ca = 1.0
            sa = 0.0

        # geometry: H jitter + slow AFC wander + tear bands (row_shift)
        jx = hash_c1(js ^ (frame_idx << 3), es ^ 0x1111) * h_jitter
        if h_jitter > 0.0:
            jx += value_noise1c(j * 0.013 + frame_idx * 0.05, es ^ 0x2222) * h_jitter * 0.55
        if tear > 0.0:
            bid = j // band_h
            r = hash3_01(bid, tear_block, 0x7E, band_seed)
            if r < 0.16 * tear:
                amp = tear * (10.0 + 70.0 * hash2_01(bid, tear_block, es ^ 0x3333))
                if hash2_01(bid, tear_block, es ^ 0x4444) < 0.5:
                    jx += amp
                else:
                    jx -= amp
        row_shift[j] += jx

        # occasionally a run of lines is fully replaced by static
        line_static = False
        if tear > 0.0:
            seg = j // 3
            if hash3_01(seg, frame_idx, 0x55, es) < 0.004 * tear:
                line_static = True

        for i in range(width):
            # ---------- geometry ---------- (nothing per-pixel; row_shift applied later)

            # ---------- multiplicative picture crush (near-total loss) ----------
            yy = y[j, i] * img_keep

            # ---------- snow: coherent flow blobs + fine per-frame grain ----------
            # ~60% temporally-coherent flow noise, ~40% fine grain (anti-cheese)
            fx = i * 0.055 + frame_idx * 1.9
            fy = j * 0.055 + frame_idx * 0.7
            flow = value_noise2f(fx, fy, es ^ 0x333) * 2.0 - 1.0      # ~[-1,1]
            g1 = hash2_01(i, js, es ^ 0x77)
            g2 = hash2_01(i + 1, js, es ^ 0x77)
            grain = (g1 + g2 - 1.0)                                  # ~[-1,1], 2px x-corr
            n = n_gain * (0.50 * flow + 0.50 * grain * 1.9)

            # near-total static replaces picture
            if static_mix > 0.0:
                static_field = n_gain * 2.6 * (0.5 * flow + 0.5 * grain * 2.2) + 0.03
                yy = yy * (1.0 - static_mix) + static_field * static_mix

            if line_static:
                yy = 0.04 + abs(grain) * sigma * 4.0

            yy = yy + n
            # snow lifts the black level (receivers never reach true 0)
            if black_lift > 0.0:
                yy += black_lift * (1.0 - yy)
            if yy < 0.0:
                yy = 0.0
            elif yy > 1.0:
                yy = 1.0
            y[j, i] = yy

            # ---------- chroma: coarse blotchy noise + hue rotate + sat + killer ----------
            uu = u[j, i]
            vv = v[j, i]
            if line_ang != 0.0:
                ru = uu * ca - vv * sa
                rv = uu * sa + vv * ca
                uu, vv = ru, rv

            if killer < 0.5 and c_gain > 0.0:
                # chroma noise is COARSE (~1/8 resolution lattice), colored blotches
                # dark-weighted: specks live in shadows, highlights stay clean
                cx = i * 0.12
                cy = j * 0.12 + frame_idx * 0.9
                cn1 = value_noise2f(cx, cy, es ^ 0x901) * 2.0 - 1.0
                cn2 = value_noise2f(cx + 37.0, cy + 11.0, es ^ 0x902) * 2.0 - 1.0
                dark = 0.35 + 0.65 * (1.0 - yy)
                uu += cn1 * c_gain * dark
                vv += cn2 * c_gain * dark

            if sat_gain < 1.0:
                uu *= sat_gain
                vv *= sat_gain
            if killer > 0.5:
                uu = 0.0
                vv = 0.0
            u[j, i] = uu
            v[j, i] = vv

            # ---------- FM threshold sparklies: hard b/w dots & short dashes ----------
            if sparklie > 0.0:
                dash_len = 2 + int(hash2_01(j, frame_idx, es ^ 0xD0) * 10)
                gx = i // dash_len
                sp = hash3_01(gx, j, frame_idx, es ^ 0xD1)
                if sp < sparklie * 0.055:
                    # bipolar impulse, held for the dash
                    if hash2_01(gx, j, es ^ 0xD2) < 0.5:
                        y[j, i] = 1.0
                    else:
                        y[j, i] = 0.0

    # export killer state
    return killer
