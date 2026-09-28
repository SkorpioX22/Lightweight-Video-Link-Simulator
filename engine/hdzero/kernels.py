"""HDZero kernels: block-error field, DCT-garbage fills, takeover events.

Visual model (docs/research/hdzero.md):
  - corrupt 8x8 DCT blocks -> confetti / solid / block-noise / shifted fills
    (the "rainbow block" and mis-decoded-block signatures; valid pixels stay
    clean — never analog-style global discoloration)
  - white speckle = single-bit errors
  - multipath -> spatially clustered punch-out patches (blob field)
  - interference -> bursty full-frame/band takeovers: band tear, rainbow
    screen, colored-line gibberish, black screen
  - extreme signal -> breakup-to-black (full black frames)

All randomness is hash(seed, frame_index, coordinates): deterministic.
Every kernel is called every non-zero frame with a flag/level argument so
one call compiles all branches (no first-use compile mid-interaction).
"""
import numpy as np
from numba import njit, prange

from ..core.rng import hash2_01, hash3_01, value_noise1f, value_noise2f
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
)

# block fill types
T_CLEAN = 0
T_CONFETTI = 1
T_SOLID = 2
T_DCT = 3
T_SHIFTED = 4
T_WHITE = 5

# takeover event types
EV_BAND = 0
EV_RAINBOW = 1
EV_LINES = 2
EV_BLACK = 3

BS = 8  # DCT block edge (px)


@njit(inline="always")
def _boost(r, g, b, k):
    m = (r + g + b) * (1.0 / 3.0)
    r = m + (r - m) * k
    g = m + (g - m) * k
    b = m + (b - m) * k
    if r < 0.0:
        r = 0.0
    elif r > 255.0:
        r = 255.0
    if g < 0.0:
        g = 0.0
    elif g > 255.0:
        g = 255.0
    if b < 0.0:
        b = 0.0
    elif b > 255.0:
        b = 255.0
    return r, g, b


@njit(inline="always")
def _hue_rgb(u):
    """u in [0,1) -> saturated rainbow color (r, g, b) in 0..255."""
    f = u * 6.0
    i = int(f)
    if i > 5:
        i = 5
    t = f - i
    a = 255.0 * (1.0 - t)
    b = 255.0 * t
    if i == 0:
        return 255.0, b, 0.0
    elif i == 1:
        return a, 255.0, 0.0
    elif i == 2:
        return 0.0, 255.0, b
    elif i == 3:
        return 0.0, a, 255.0
    elif i == 4:
        return b, 0.0, 255.0
    return 255.0, 0.0, a


@njit(inline="always")
def _cell_color(cx, cy, fi, es, salt, boost):
    """Color for one 4x4 sub-cell of a corrupt block.

    Sub-cell granularity (not per pixel) is what makes a bad DCT decode
    read as digital block garbage instead of analog-style static.
    """
    ha = hash3_01(cx, cy, fi, es ^ salt)
    hb = hash3_01(cx, cy, fi, es ^ (salt + 0x1111))
    return _boost(ha * 255.0, hb * 255.0,
                  ((ha + hb) - np.floor(ha + hb)) * 255.0, boost)


@njit(cache=True)
def update_hz_state(
    hz,
    duty, dur_lo, dur_hi, gap_lo, gap_hi,
    w_rain, w_lines, w_black,
    band_speed, band_frac, env_rate, mp_speed,
    loss_p, loss_dur_lo, loss_dur_hi,
    seed, fi,
):
    """Advance multipath envelope/drift, the interference event machine,
    and the signal-loss latch."""
    es = seed ^ (fi * 0x9E3779B9)

    # multipath cluster field: slow drift + burst envelope
    hz[HZ_DRIFT] += mp_speed * (0.6 + 0.8 * hash2_01(fi, 9, es ^ 0x4D50))
    hz[HZ_ENV] = value_noise1f(fi * env_rate, seed ^ 0xE9E9)

    # signal-loss latch: runs of "all signal lost" (no-signal screen).
    # The run tracks its elapsed time (patch-coverage ramp) and a per-run
    # id so patch positions stay stable while a run accumulates coverage.
    if hz[HZ_LOSS] > 0.0:
        hz[HZ_LOSS] -= 1.0
        hz[HZ_LOSS_T] += 1.0
    elif loss_p > 0.0 and hash2_01(fi, 43, es ^ 0x1055) < loss_p:
        hz[HZ_LOSS] = loss_dur_lo + hash2_01(fi, 47, es ^ 0x1055) * (
            loss_dur_hi - loss_dur_lo)
        hz[HZ_LOSS_T] = 1.0
        hz[HZ_LOSS_SEED] = float(fi & 0xFFFFFF)

    if hz[HZ_EVT_ACTIVE] > 0.5:
        hz[HZ_EVT_TIMER] -= 1.0
        if hz[HZ_EVT_TYPE] < 0.5:
            # band sweeps top->bottom, snapping back when it clears the frame
            y0 = hz[HZ_EVT_Y0] + band_speed
            lim = 1.0 - hz[HZ_EVT_H]
            if lim <= 0.0:
                lim = 0.0001
            if y0 > lim:
                y0 -= lim
            hz[HZ_EVT_Y0] = y0
        if hz[HZ_EVT_TIMER] <= 0.0:
            hz[HZ_EVT_ACTIVE] = 0.0
            hz[HZ_EVT_CD] = gap_lo + hash2_01(fi, 13, es ^ 0x1F2E) * (gap_hi - gap_lo)
    elif hz[HZ_EVT_CD] > 0.0:
        hz[HZ_EVT_CD] -= 1.0
    elif hash2_01(fi, 7, es ^ 0x1F2E) < duty:
        hz[HZ_EVT_ACTIVE] = 1.0
        u = hash2_01(fi, 11, es ^ 0x1F2E)
        if u < w_rain:
            hz[HZ_EVT_TYPE] = EV_RAINBOW
        elif u < w_rain + w_lines:
            hz[HZ_EVT_TYPE] = EV_LINES
        elif u < w_rain + w_lines + w_black:
            hz[HZ_EVT_TYPE] = EV_BLACK
        else:
            hz[HZ_EVT_TYPE] = EV_BAND
        hz[HZ_EVT_TIMER] = dur_lo + hash2_01(fi, 17, es ^ 0x1F2E) * (dur_hi - dur_lo)
        hz[HZ_EVT_Y0] = hash2_01(fi, 19, es ^ 0x1F2E)
        hz[HZ_EVT_H] = band_frac * (0.7 + 0.6 * hash2_01(fi, 23, es ^ 0x1F2E))
        if hz[HZ_EVT_H] > 0.9:
            hz[HZ_EVT_H] = 0.9


@njit(parallel=True, fastmath=True, cache=True)
def compute_bmap(bmap, block_err, cl_cov, cl_boost, white_blk, env, drift, seed, fi):
    """Per-block corruption type for this frame.

    Signal: scattered (epoch-stable + per-frame mixture).
    Multipath: blob-field clusters that drift and pulse with env; inside a
    cluster the type mix skews to DCT-noise/solid (punch-out look) and can
    override the signal decision.
    """
    bh, bw = bmap.shape
    es = seed ^ (fi * 0x9E3779B9)
    epoch = fi >> 2
    thr = 1.0 - cl_cov * (0.35 + 0.65 * env)
    for by in prange(bh):
        for bx in range(bw):
            t = T_CLEAN
            if block_err > 0.0:
                w = (0.62 * hash3_01(bx, by, epoch, es ^ 0x51A1)
                     + 0.38 * hash3_01(bx, by, fi, es ^ 0x51A1))
                if w < block_err:
                    u = hash3_01(bx, by, fi, es ^ 0x5717)
                    if u < white_blk:
                        t = T_WHITE
                    elif u < white_blk + 0.38:
                        t = T_CONFETTI
                    elif u < white_blk + 0.60:
                        t = T_SOLID
                    elif u < white_blk + 0.80:
                        t = T_DCT
                    else:
                        t = T_SHIFTED
            if cl_cov > 0.0:
                nv = value_noise2f(bx * 0.09 + drift, by * 0.09, es ^ 0x4D51)
                if nv > thr:
                    wc = (0.55 * hash3_01(bx, by, epoch, es ^ 0x4D52)
                          + 0.45 * hash3_01(bx, by, fi, es ^ 0x4D52))
                    if wc < cl_boost:
                        u = hash3_01(bx, by, fi, es ^ 0x4D53)
                        if u < 0.42:
                            t = T_DCT
                        elif u < 0.70:
                            t = T_SOLID
                        elif u < 0.90:
                            t = T_SHIFTED
                        else:
                            t = T_CONFETTI
            bmap[by, bx] = t


@njit(parallel=True, fastmath=True, cache=True)
def fill_blocks(out, src, bmap, seed, fi):
    """Paint corrupt blocks according to the per-block type map."""
    h, w = out.shape[0], out.shape[1]
    bh, bw = bmap.shape
    es = seed ^ (fi * 0x9E3779B9)
    for by in prange(bh):
        y0 = by * BS
        y1 = y0 + BS
        if y1 > h:
            y1 = h
        for bx in range(bw):
            t = bmap[by, bx]
            if t == T_CLEAN:
                continue
            x0 = bx * BS
            x1 = x0 + BS
            if x1 > w:
                x1 = w
            if t == T_CONFETTI or t == T_DCT:
                # 4x4 sub-cells: saturated (confetti) vs muted (bad DCT)
                if t == T_CONFETTI:
                    salt = 0x1111
                    boost = 2.4
                else:
                    salt = 0x3333
                    boost = 1.4
                for sy in range(2):
                    for sx in range(2):
                        cr, cg, cb = _cell_color(
                            bx * 4 + sx, by * 4 + sy, fi, es, salt, boost)
                        yy0 = y0 + sy * 4
                        yy1 = yy0 + 4
                        if yy1 > y1:
                            yy1 = y1
                        xx0 = x0 + sx * 4
                        xx1 = xx0 + 4
                        if xx1 > x1:
                            xx1 = x1
                        for y in range(yy0, yy1):
                            for x in range(xx0, xx1):
                                out[y, x, 0] = cr
                                out[y, x, 1] = cg
                                out[y, x, 2] = cb
            elif t == T_SOLID:
                u = hash3_01(bx, by, fi, es ^ 0x2222)
                idx = int(u * 8.0)
                if idx > 7:
                    idx = 7
                if idx == 0:
                    cr, cg, cb = 30.0, 235.0, 70.0
                elif idx == 1:
                    cr, cg, cb = 235.0, 40.0, 225.0
                elif idx == 2:
                    cr, cg, cb = 10.0, 10.0, 12.0
                elif idx == 3:
                    cr, cg, cb = 250.0, 250.0, 250.0
                elif idx == 4:
                    cr, cg, cb = 128.0, 132.0, 140.0
                elif idx == 5:
                    cr, cg, cb = 40.0, 60.0, 235.0
                elif idx == 6:
                    cr, cg, cb = 240.0, 140.0, 30.0
                else:
                    ha = hash3_01(bx, by, fi, es ^ 0x3333)
                    hb = hash3_01(bx, by, fi, es ^ 0x4444)
                    cr, cg, cb = _boost(ha * 255.0, hb * 255.0,
                                        ((ha + hb) - np.floor(ha + hb)) * 255.0, 2.2)
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        out[y, x, 0] = cr
                        out[y, x, 1] = cg
                        out[y, x, 2] = cb
            elif t == T_SHIFTED:
                h1 = hash3_01(bx, by, fi, es ^ 0x5555)
                h2 = hash3_01(bx, by, fi, es ^ 0x6666)
                dx = int(h1 * 23.0) - 11
                dy = int(h2 * 15.0) - 7
                for y in range(y0, y1):
                    sy = y + dy
                    if sy < 0:
                        sy = 0
                    elif sy >= h:
                        sy = h - 1
                    for x in range(x0, x1):
                        sx = x + dx
                        if sx < 0:
                            sx = 0
                        elif sx >= w:
                            sx = w - 1
                        out[y, x, 0] = src[sy, sx, 0] * 0.92
                        out[y, x, 1] = src[sy, sx, 1] * 0.92
                        out[y, x, 2] = src[sy, sx, 2] * 0.92
            else:  # T_WHITE
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        out[y, x, 0] = 255.0
                        out[y, x, 1] = 255.0
                        out[y, x, 2] = 255.0


@njit(parallel=True, fastmath=True, cache=True)
def chroma_quant(img, q):
    """Quantize color deviations around luma (rate-drop codec banding)."""
    if q <= 0.0:
        return
    h, w = img.shape[0], img.shape[1]
    for j in prange(h):
        for i in range(w):
            r = img[j, i, 0]
            g = img[j, i, 1]
            b = img[j, i, 2]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            dr = (r - y) / q
            dg = (g - y) / q
            db = (b - y) / q
            dr = np.floor(dr + 0.5) * q
            dg = np.floor(dg + 0.5) * q
            db = np.floor(db + 0.5) * q
            rr = y + dr
            gg = y + dg
            bb = y + db
            # round on cast: bare float->uint8 truncation would shift an
            # unchanged 128.0 (y +/- eps) down to 127 and dirty clean pixels
            rr = np.rint(rr)
            gg = np.rint(gg)
            bb = np.rint(bb)
            if rr < 0.0:
                rr = 0.0
            elif rr > 255.0:
                rr = 255.0
            if gg < 0.0:
                gg = 0.0
            elif gg > 255.0:
                gg = 255.0
            if bb < 0.0:
                bb = 0.0
            elif bb > 255.0:
                bb = 255.0
            img[j, i, 0] = rr
            img[j, i, 1] = gg
            img[j, i, 2] = bb


@njit(inline="always")
def black_hit(p, seed, fi):
    """True if this frame's hash lands inside the blackout probability."""
    es = seed ^ (fi * 0x9E3779B9)
    return hash2_01(fi, 31, es ^ 0xB1AC) < p


@njit(parallel=True, fastmath=True, cache=True)
def apply_speckle(img, p, seed, fi):
    """Sparse pure-white single-pixel hits (bit errors)."""
    if p <= 0.0:
        return
    h, w = img.shape[0], img.shape[1]
    es = seed ^ (fi * 0x9E3779B9)
    for j in prange(h):
        for i in range(w):
            if hash3_01(i, j, fi, es ^ 0x534B) < p:
                img[j, i, 0] = 255.0
                img[j, i, 1] = 255.0
                img[j, i, 2] = 255.0


@njit(parallel=True, fastmath=True, cache=True)
def fill_black(img, enabled):
    if enabled <= 0.0:
        return
    h, w = img.shape[0], img.shape[1]
    for j in prange(h):
        for i in range(w):
            img[j, i, 0] = 0.0
            img[j, i, 1] = 0.0
            img[j, i, 2] = 0.0


@njit(parallel=True, fastmath=True, cache=True)
def fill_loss(img, on, seed, fi):
    """Total-signal-loss screen: vertical rainbow column garbage.

    The HDZero "rainbow screen of death": 4 px columns, each one saturated
    hue held down the full column height (color holds ~4 frames; ~10% of
    16 px segments hue-flip), with coherent 2 px weave/dash texture that
    re-randomizes every frame and a few silver/white columns. Replaces the
    picture entirely while the loss latch is active.
    """
    if on <= 0.0:
        return
    h, w = img.shape[0], img.shape[1]
    es = seed ^ (fi * 0x9E3779B9)
    epoch = fi >> 2
    hs = seed ^ (epoch * 0x9E3779B9)
    for j in prange(h):
        y = j
        seg = y >> 4
        yr = y >> 1
        yrep = yr & 7
        for i in range(w):
            x = i
            col = x >> 3
            xin = x & 7

            # hue: one color per column down its full height, held ~4
            # frames; occasional segment flips (color break mid-column)
            h1 = hash3_01(col, 0, 0, hs ^ 0x10A5)
            hu = h1 * 3.0
            hu -= np.floor(hu)
            if hash3_01(col, seg, 7, hs ^ 0x10A6) < 0.08:
                hu += 0.5
                if hu >= 1.0:
                    hu -= 1.0
            silver = hash3_01(col, 3, epoch, hs ^ 0x10A7) < 0.10

            # texture: repeating vertical glyph pattern (period 16 px) with
            # a segment-level shift, re-randomized per frame; coherent
            # across the column width
            t1 = hash3_01(col, yrep, 0, es ^ 0x10A8)
            t2 = hash3_01(col, seg, 9, hs ^ 0x10AB)
            t = 0.60 * t1 + 0.40 * t2
            lv = int(t * 4.0)
            if lv > 3:
                lv = 3
            v = 75.0 + lv * 60.0

            # weave style per column+segment (stable within the run)
            sty = int(hash3_01(col, seg, 13, hs ^ 0x10A9) * 4.0)
            if sty > 3:
                sty = 3
            if sty == 0:
                # checkerboard weave
                if ((x + y) & 1) == 0:
                    v *= 0.50
            elif sty == 1:
                # bright right edge + sparse dark ticks
                if xin >= 6:
                    v = 255.0
                elif hash3_01(col, yr, 5, es ^ 0x10AA) < 0.25:
                    v *= 0.50
            elif sty == 2:
                # dark seam on the column's left edge
                if xin == 0:
                    v *= 0.50
            else:
                # split column: right half shifts hue slightly
                if xin >= 4:
                    hu += 0.07
                    if hu >= 1.0:
                        hu -= 1.0

            if silver:
                img[j, i, 0] = v
                img[j, i, 1] = v
                img[j, i, 2] = v
            else:
                cr, cg, cb = _hue_rgb(hu)
                s = v * (1.0 / 255.0)
                img[j, i, 0] = cr * s
                img[j, i, 1] = cg * s
                img[j, i, 2] = cb * s


@njit(parallel=True, fastmath=True, cache=True)
def fill_loss_patches(img, loss, cov, run_seed, seed, fi):
    """Growing partial-loss screen: random 8x8 cells of the real no-signal image.

    While a signal-loss run is still ramping up (cov < 1) the picture stays
    visible except for cells whose stable hash lands below cov — each painted
    cell shows a random crop of the full no-signal screen (offsets stable per
    run + cell, so coverage only accumulates as cov grows). At cov >= 1 the
    engine blits the full screen instead. fi is unused: layout must not
    flicker while a run builds.
    """
    if cov <= 0.0:
        return
    h, w = img.shape[0], img.shape[1]
    lh, lw = loss.shape[0], loss.shape[1]
    bh = (h + BS - 1) // BS
    bw = (w + BS - 1) // BS
    ks = seed ^ ((int(run_seed) & 0xFFFFFF) * 0x9E3779B9)
    for by in prange(bh):
        y0 = by * BS
        ch = BS
        if y0 + ch > h:
            ch = h - y0
        for bx in range(bw):
            if hash3_01(bx, by, 11, ks ^ 0x10C0) >= cov:
                continue
            x0 = bx * BS
            cw = BS
            if x0 + cw > w:
                cw = w - x0
            # random crop window inside the no-signal screen (stable per cell)
            sx = int(hash3_01(bx, by, 23, ks ^ 0x10C1) * (lw - cw + 1))
            sy = int(hash3_01(bx, by, 29, ks ^ 0x10C2) * (lh - ch + 1))
            if sx > lw - cw:
                sx = lw - cw
            if sy > lh - ch:
                sy = lh - ch
            if sx < 0:
                sx = 0
            if sy < 0:
                sy = 0
            for y in range(ch):
                lyy = sy + y
                for x in range(cw):
                    lxx = sx + x
                    img[y0 + y, x0 + x, 0] = loss[lyy, lxx, 0]
                    img[y0 + y, x0 + x, 1] = loss[lyy, lxx, 1]
                    img[y0 + y, x0 + x, 2] = loss[lyy, lxx, 2]


@njit(parallel=True, fastmath=True, cache=True)
def render_event(out, src, etype, active, y0f, hf, density, seed, fi):
    """Takeover events. active gates the work; all branches compile together."""
    if active <= 0.0:
        return
    h, w = out.shape[0], out.shape[1]
    es = seed ^ (fi * 0x9E3779B9)
    bh, bw = (h + BS - 1) // BS, (w + BS - 1) // BS

    if etype < 0.5:
        # band: rows of block garbage + horizontal tear; reads/writes same row
        # only (single writer per row -> deterministic)
        y0 = int(y0f * h)
        y1 = y0 + int(hf * h)
        if y1 > h:
            y1 = h
        if y0 < 0:
            y0 = 0
        for y in prange(y0, y1):
            off = int((hash2_01(y, fi, es ^ 0xBA9) - 0.5) * 88.0)
            by = y >> 3
            for bx in range(bw):
                x0 = bx * BS
                x1 = x0 + BS
                if x1 > w:
                    x1 = w
                eat = hash3_01(bx, by, fi, es ^ 0xBE6) < 0.72
                if eat:
                    u = hash3_01(bx, by, fi, es ^ 0xE711)
                    if u < 0.5:
                        cy = y >> 2
                        ccx = -1
                        cr = 0.0
                        cg = 0.0
                        cb = 0.0
                        for x in range(x0, x1):
                            cx = x >> 2
                            if cx != ccx:
                                ccx = cx
                                cr, cg, cb = _cell_color(
                                    cx, cy, fi, es, 0x1111, 2.4)
                            out[y, x, 0] = cr
                            out[y, x, 1] = cg
                            out[y, x, 2] = cb
                    elif u < 0.8:
                        ha = hash3_01(bx, y, fi, es ^ 0x3333)
                        cr = 40.0 + ha * 215.0
                        cg = 40.0 + ((ha * 7.13) % 1.0) * 215.0
                        cb = 40.0 + ((ha * 3.71) % 1.0) * 215.0
                        for x in range(x0, x1):
                            out[y, x, 0] = cr
                            out[y, x, 1] = cg
                            out[y, x, 2] = cb
                    else:
                        for x in range(x0, x1):
                            out[y, x, 0] = 12.0
                            out[y, x, 1] = 12.0
                            out[y, x, 2] = 14.0
                else:
                    # tear: shift this row's content horizontally
                    if off > 0:
                        for x in range(w - 1, -1, -1):
                            sx = x - off
                            if sx < 0:
                                sx = 0
                            out[y, x, 0] = out[y, sx, 0]
                            out[y, x, 1] = out[y, sx, 1]
                            out[y, x, 2] = out[y, sx, 2]
                    elif off < 0:
                        for x in range(w):
                            sx = x - off
                            if sx >= w:
                                sx = w - 1
                            out[y, x, 0] = out[y, sx, 0]
                            out[y, x, 1] = out[y, sx, 1]
                            out[y, x, 2] = out[y, sx, 2]
        return

    if etype < 1.5:
        # rainbow screen: every block becomes confetti/solid/DCT garbage
        for by in prange(bh):
            y0 = by * BS
            y1 = y0 + BS
            if y1 > h:
                y1 = h
            for bx in range(bw):
                u = hash3_01(bx, by, fi, es ^ 0xE711)
                if u < 0.55:
                    for sy in range(2):
                        for sx in range(2):
                            cr, cg, cb = _cell_color(
                                bx * 4 + sx, by * 4 + sy, fi, es, 0x1111, 2.4)
                            yy0 = y0 + sy * 4
                            yy1 = min(yy0 + 4, y1)
                            xx0 = bx * BS + sx * 4
                            xx1 = min(xx0 + 4, bx * BS + BS, w)
                            for y in range(yy0, yy1):
                                for x in range(xx0, xx1):
                                    out[y, x, 0] = cr
                                    out[y, x, 1] = cg
                                    out[y, x, 2] = cb
                elif u < 0.85:
                    ha = hash3_01(bx, by, fi, es ^ 0x3333)
                    hb = hash3_01(bx, by, fi, es ^ 0x4444)
                    cr, cg, cb = _boost(ha * 255.0, hb * 255.0,
                                        ((ha + hb) - np.floor(ha + hb)) * 255.0, 2.2)
                    for y in range(y0, y1):
                        for x in range(bx * BS, min(bx * BS + BS, w)):
                            out[y, x, 0] = cr
                            out[y, x, 1] = cg
                            out[y, x, 2] = cb
                else:
                    for sy in range(2):
                        for sx in range(2):
                            cr, cg, cb = _cell_color(
                                bx * 4 + sx, by * 4 + sy, fi, es, 0x3333, 1.4)
                            yy0 = y0 + sy * 4
                            yy1 = min(yy0 + 4, y1)
                            xx0 = bx * BS + sx * 4
                            xx1 = min(xx0 + 4, bx * BS + BS, w)
                            for y in range(yy0, yy1):
                                for x in range(xx0, xx1):
                                    out[y, x, 0] = cr
                                    out[y, x, 1] = cg
                                    out[y, x, 2] = cb
        return

    if etype < 2.5:
        # colored-line gibberish: saturated horizontal lines over the image
        for y in prange(h):
            if hash2_01(y, fi, es ^ 0x4C4F) >= density:
                continue
            v = hash2_01(y, 3, es ^ 0x4C50)
            sl = 30 + int(hash2_01(y, 5, es ^ 0x4C51) * 90.0)
            cr = 255.0
            cg = 255.0
            cb = 255.0
            for x in range(w):
                if x % sl == 0:
                    if v < 0.12:
                        cr, cg, cb = 0.0, 0.0, 0.0
                    elif v < 0.22:
                        cr, cg, cb = 255.0, 255.0, 255.0
                    else:
                        ha = hash3_01(x, y, fi, es ^ 0x4C52)
                        hb = hash3_01(x, y, fi, es ^ 0x4C53)
                        cr, cg, cb = _boost(
                            ha * 255.0, hb * 255.0,
                            ((ha + hb) - np.floor(ha + hb)) * 255.0, 2.6)
                    v = hash3_01(x, y, fi, es ^ 0x4C54)
                out[y, x, 0] = cr
                out[y, x, 1] = cg
                out[y, x, 2] = cb
        return

    # black screen
    for j in prange(h):
        for i in range(w):
            out[j, i, 0] = 0.0
            out[j, i, 1] = 0.0
            out[j, i, 2] = 0.0


__all__ = [
    "T_CLEAN", "T_CONFETTI", "T_SOLID", "T_DCT", "T_SHIFTED", "T_WHITE",
    "EV_BAND", "EV_RAINBOW", "EV_LINES", "EV_BLACK",
    "update_hz_state", "compute_bmap", "fill_blocks",
    "chroma_quant", "apply_speckle", "fill_black", "fill_loss",
    "fill_loss_patches", "render_event", "black_hit",
]
