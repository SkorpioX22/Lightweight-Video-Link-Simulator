#version 330 core
// fpv_breakup.frag — single-pass GLSL port of AnalogVideoBreakupEngine v1.0
// Deterministic: all animation is a pure function of uFrame/uSeed (no feedback).
// Pipeline order matches engine/renderer/engine.py:
//   inverse geometry (roll + row_shift + seam)
//     -> multipath (echo taps on clean samples, smear, blocks, ripple, null/flash)
//     -> weak signal (snow, chroma blotch, sat/killer, hue, sparklies, tear)
//     -> interference (burst bands, slices, fringe, impulses, desense, false-sync)
//     -> RGB
// All severity curves replicate engine/core/params.py exactly.

#ifndef SHADERTOY
uniform sampler2D uSrc;         // clean RGB frame (simulator's rendered output)
uniform vec2  uResolution;      // pixels
uniform float uTime;            // seconds (reserved; animation uses uFrame)
uniform uint  uFrame;           // frame counter — drives ALL animation
uniform uint  uSeed;
uniform float uSignalStrength;  // 0..99 (0 = no effect, 99 = extreme)
uniform float uMultipath;       // 0..99
uniform float uRFInterference;  // 0..99
#else
// Shadertoy provides iChannel0/iResolution/iTime/iFrame; params are plain
// globals written by mainImage before calling fpvEffect (uniforms are
// read-only, and GLSL has no `static` qualifier).
uniform sampler2D iChannel0;
uniform vec3  iResolution;
uniform float iTime;
uniform int   iFrame;
#define uSrc iChannel0
#define uResolution iResolution.xy
#define uTime iTime
uint  uSeed;
uint  uFrame;
float uSignalStrength;
float uMultipath;
float uRFInterference;
#endif

#ifndef SHADERTOY
out vec4 fragColor;
#endif

// ---------------------------------------------------------------------------
// hashing / noise (mirrors engine/core/rng.py)
// ---------------------------------------------------------------------------
uint hash_u32(uint x, uint seed) {
    uint v = x ^ seed;
    v ^= v >> 16u;
    v *= 0x7FEB352Du;
    v ^= v >> 15u;
    v *= 0x846CA68Bu;
    v ^= v >> 16u;
    return v;
}
uint hash2_u32(uint x, uint y, uint seed) {
    uint xm = x * 0x9E3779B1u;
    return hash_u32(xm ^ y, seed);
}
uint hash3_u32(uint x, uint y, uint z, uint seed) {
    uint xm = x * 0x9E3779B1u;
    uint ym = y * 0x85EBCA77u;
    return hash_u32(xm ^ ym ^ z, seed);
}
float hash01(uint x, uint seed)  { return float(hash_u32(x, seed)) * (1.0 / 4294967296.0); }
float hash2_01(uint x, uint y, uint seed) { return float(hash2_u32(x, y, seed)) * (1.0 / 4294967296.0); }
float hash3_01(uint x, uint y, uint z, uint seed) { return float(hash3_u32(x, y, z, seed)) * (1.0 / 4294967296.0); }
float hash_c1(uint x, uint seed) { return hash01(x, seed) * 2.0 - 1.0; }

float smf(float t) { return t * t * (3.0 - 2.0 * t); }

// GLSL uint(negative int) is implementation-defined; Python's np.uint32 wraps
// two's-complement, so emulate the wrap explicitly for negative lattice coords.
uint wrap_int(int x) {
    return x >= 0 ? uint(x) : uint(~x + 1);  // two's-complement bit pattern
}

float value_noise1f(float x, uint seed) {
    int xi = int(floor(x));
    float xf = x - float(xi);
    float a = hash2_01(wrap_int(xi), 0u, seed);
    float b = hash2_01(wrap_int(xi + 1), 0u, seed);
    return mix(a, b, smf(xf));
}
float value_noise1c(float x, uint seed) { return value_noise1f(x, seed) * 2.0 - 1.0; }
float value_noise2f(float x, float y, uint seed) {
    int xi = int(floor(x));
    int yi = int(floor(y));
    float xf = x - float(xi);
    float yf = y - float(yi);
    float a = hash2_01(wrap_int(xi), wrap_int(yi), seed);
    float b = hash2_01(wrap_int(xi + 1), wrap_int(yi), seed);
    float c = hash2_01(wrap_int(xi), wrap_int(yi + 1), seed);
    float d = hash2_01(wrap_int(xi + 1), wrap_int(yi + 1), seed);
    float t = smf(xf);
    float u = smf(yf);
    return mix(mix(a, b, t), mix(c, d, t), u);
}

// ---------------------------------------------------------------------------
// severity curves (mirrors engine/core/params.py)
// ---------------------------------------------------------------------------
float ss(float e0, float e1, float x) {
    float t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}
float ease_in_pow(float x, float p) { return pow(clamp(x, 0.0, 1.0), p); }

struct SigSt {
    float luma_noise, chroma_noise, sat_loss, hue_wander;
    float sparklie, h_jitter, tear, roll, static_kill;
};
struct MpSt {
    float comb_k, smear_px, ghost1_d, ghost1_a, ghost2_d, ghost2_a, ghost3_a;
    float block_shift, band_ripple, edge_spike, null_rate, flash_rate;
    float null_snow, desat_bands;
};
struct IntSt {
    float duty, band_amp, band_thr, scroll, band_count, tint, fringe;
    float slice, impulse, desense, long_event, sync_false;
};

SigSt signal_stages(float r) {
    SigSt s;
    s.luma_noise  = 42.0 * ease_in_pow(ss(0.015, 1.0, r), 1.22) / 255.0;
    s.chroma_noise = ss(0.04, 0.40, r) * 0.11;
    s.sat_loss    = ss(0.18, 0.50, r);
    s.hue_wander  = ss(0.12, 0.45, r) * 0.55;
    s.sparklie    = ease_in_pow(ss(0.42, 0.88, r), 2.0) * 0.5;
    s.h_jitter    = ss(0.52, 0.80, r) * 7.0;
    s.tear        = ss(0.62, 0.92, r);
    s.roll        = ss(0.72, 0.97, r);
    s.static_kill = ss(0.88, 1.0, r);
    return s;
}
MpSt multipath_stages(float r) {
    MpSt m;
    m.comb_k       = ss(0.04, 0.55, r) * 0.32;
    m.smear_px     = ss(0.18, 0.92, r) * 3.5;
    m.ghost1_d     = 4.0 + ss(0.12, 1.0, r) * 30.0;
    m.ghost1_a     = ss(0.12, 0.75, r) * 0.85;
    m.ghost2_d     = 16.0 + ss(0.40, 1.0, r) * 75.0;
    m.ghost2_a     = ss(0.38, 0.85, r) * 0.50;
    m.ghost3_a     = ss(0.30, 0.90, r) * 0.40;
    m.block_shift  = ss(0.45, 0.95, r) * 42.0;
    m.band_ripple  = ss(0.28, 0.85, r) * 0.14;
    m.edge_spike   = ss(0.35, 0.9, r) * 0.55;
    m.null_rate    = ease_in_pow(ss(0.55, 1.0, r), 1.5) * 0.05;
    m.flash_rate   = ss(0.42, 0.98, r) * 0.05;
    m.null_snow    = ss(0.5, 1.0, r) * 0.50;
    m.desat_bands  = ss(0.55, 0.95, r) * 0.8;
    return m;
}
IntSt interference_stages(float r) {
    IntSt t;
    t.duty       = ss(0.06, 0.97, r) * 0.60;
    t.band_amp   = ss(0.10, 0.95, r);
    t.band_thr   = 0.72 - 0.14 * r;
    t.scroll     = 0.4 + ss(0.1, 1.0, r) * 5.0;
    t.band_count = 3.0 + r * 12.0;
    t.tint       = ss(0.25, 0.95, r);
    t.fringe     = ss(0.28, 0.85, r) * 0.15;
    t.slice      = ss(0.48, 0.99, r) * 0.55;
    t.impulse    = ss(0.35, 0.95, r) * 0.45;
    t.desense    = ss(0.3, 0.95, r) * 0.35;
    t.long_event = ss(0.65, 1.0, r) * 0.03;
    t.sync_false = ss(0.7, 1.0, r) * 18.0;
    return t;
}

// ---------------------------------------------------------------------------
// BT.601 full-range color (mirrors engine/core/color.py)
// ---------------------------------------------------------------------------
vec3 rgb_to_yuv(vec3 c) {
    float y = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
    float u = -0.168736 * c.r - 0.331264 * c.g + 0.5 * c.b;
    float v = 0.5 * c.r - 0.418688 * c.g - 0.081312 * c.b;
    return vec3(y, u, v);
}
vec3 yuv_to_rgb(vec3 c) {
    float y = c.x, u = c.y, v = c.z;
    float r = y + 1.402 * v;
    float g = y - 0.344136 * u - 0.714136 * v;
    float b = y + 1.772 * u;
    return clamp(vec3(r, g, b), 0.0, 1.0);
}

// sample clean source as YUV at pixel coords (i, j)
vec3 src_yuv(sampler2D tex, vec2 res, float i, float j) {
    vec2 uv = (vec2(i, j) + 0.5) / res;
    uv = clamp(uv, vec2(0.0), vec2(1.0));
    return rgb_to_yuv(texture(tex, uv).rgb);
}

// ---------------------------------------------------------------------------
// temporal state machines (stateless approximations of Python's machines)
// All are pure functions of (uFrame, uSeed) — deterministic per frame.
// ---------------------------------------------------------------------------

// --- multipath ghost amplitudes/delays (exact port of update_state math) ---
struct MpDyn { float a1, a2, a3, d1, d2, d3, band_ph; bool null_on, flash_on; };
MpDyn mp_dynamic(MpSt st, uint fi, uint es) {
    MpDyn m;
    float env1 = 0.65 + 0.35 * value_noise1f(float(fi) * 0.9, es ^ 0xE1u);
    float env2 = 0.65 + 0.35 * value_noise1f(float(fi) * 1.3 + 31.0, es ^ 0xE2u);
    float env3 = 0.65 + 0.35 * value_noise1f(float(fi) * 1.7 + 61.0, es ^ 0xE3u);
    float sg1 = hash2_01(fi, 101u, es ^ 0xF1u) < 0.78 ? 1.0 : -1.0;
    float sg2 = hash2_01(fi, 103u, es ^ 0xF2u) < 0.72 ? 1.0 : -1.0;
    float sg3 = hash2_01(fi, 107u, es ^ 0xF3u) < 0.65 ? 1.0 : -1.0;
    float fl1 = 0.78 + 0.22 * (value_noise1f(float(fi) * 2.3, es ^ 0xA1u) * 2.0 - 1.0);
    float fl2 = 0.78 + 0.22 * (value_noise1f(float(fi) * 3.1 + 50.0, es ^ 0xA2u) * 2.0 - 1.0);
    float fl3 = 0.78 + 0.22 * (value_noise1f(float(fi) * 3.9 + 90.0, es ^ 0xA3u) * 2.0 - 1.0);
    m.a1 = st.ghost1_a * env1 * fl1 * sg1;
    m.a2 = st.ghost2_a * env2 * fl2 * sg2;
    m.a3 = st.ghost3_a * env3 * fl3 * sg3;
    float drift1 = 1.0 + 0.22 * (value_noise1f(float(fi) * 0.035, es ^ 0xD1u) * 2.0 - 1.0);
    float drift2 = 1.0 + 0.18 * (value_noise1f(float(fi) * 0.028 + 17.0, es ^ 0xD2u) * 2.0 - 1.0);
    float drift3 = 1.0 + 0.15 * (value_noise1f(float(fi) * 0.022 + 41.0, es ^ 0xD3u) * 2.0 - 1.0);
    m.d1 = st.ghost1_d * drift1;
    m.d2 = st.ghost2_d * drift2;
    m.d3 = st.ghost1_d * 0.45 * drift3;
    m.band_ph = float(fi) * (0.05 + 0.11 * value_noise1f(float(fi) * 0.05, es ^ 0xBDu));

    // null machine: rare, 1-4 frame windows with cooldown (block-hashed)
    // null blocks of 64 frames; onset only in first half so cooldown approx holds
    uint nblk = fi / 64u;
    float nt = float(fi % 64u);
    float nh = hash2_01(nblk, 41u, es ^ 0xF2u);
    float n_start = hash2_01(nblk, 43u, es) * 30.0;
    float n_len = 1.0 + hash2_01(nblk, 44u, es) * 2.0;   // 1-3 frames, like Python
    m.null_on = (st.null_rate > 0.0) && (nh * 64.0 < st.null_rate * 64.0 * 8.0)
                && nt >= n_start && nt < n_start + n_len;

    // flash machine: single-frame events, ~flash_rate per frame, min gap 4
    uint fblk = fi / 8u;
    float ft = float(fi % 8u);
    float fh = hash2_01(fblk, 29u, es ^ 0xF1u);
    float f_at = hash2_01(fblk, 30u, es) * 4.0;
    m.flash_on = (st.flash_rate > 0.0) && (fh * 8.0 < st.flash_rate * 8.0 * 6.0)
                 && abs(ft - f_at) < 1.0;
    return m;
}

// --- signal roll event (stateless: position derived per frame) ---
// active window per block; roll_pos = dir * speed * elapsed within window.
// Block onset prob tuned so duty ≈ 28% at roll=1 (matches Python machine).
float signal_roll_pos(SigSt st, uint fi, uint es) {
    if (st.roll <= 0.0) return 0.0;
    uint blk = fi / 48u;
    float t = float(fi % 48u);
    float hh = hash2_01(blk, 13u, es ^ 0x51A1u);
    if (hh > st.roll * 0.65) return 0.0;
    float start = hash2_01(blk, 14u, es) * 10.0;
    float len = 6.0 + hash2_01(blk, 17u, es) * 30.0;
    if (t < start || t >= start + len) return 0.0;
    float dir = hash2_01(blk, 19u, es) < 0.5 ? 1.0 : -1.0;
    float speed = 12.0 + hash2_01(fi, 7u, es) * 26.0;
    return dir * speed * (t - start);
}

// --- interference burst (stateless: thresholded value noise + hangover scan) ---
struct IntDyn { bool burst, long_on; float hang, scroll, fringe_ph; };
IntDyn int_dynamic(IntSt st, uint fi, uint es) {
    IntDyn d;
    // long takeover: blocks of 256 frames, onset prob long_event*256*4
    uint lblk = fi / 256u;
    float lt = float(fi % 256u);
    float lh = hash2_01(lblk, 67u, es ^ 0xEEu);
    float l_start = hash2_01(lblk, 68u, es) * 150.0;
    float l_len = 9.0 + hash2_01(lblk, 71u, es) * 66.0;
    d.long_on = (st.long_event > 0.0) && (lh < st.long_event * 4.0)
                && lt >= l_start && lt < l_start + l_len;

    // packet bursts: P(burst) ≈ eff_duty per frame, ~3-frame correlation
    float cluster = value_noise1f(float(fi) * 0.07, es ^ 0xC1u);
    float eff_duty = clamp(st.duty * (0.35 + 1.3 * cluster), 0.0, 0.9);
    float n = value_noise1f(float(fi) * 0.45, es ^ 0x88u);
    d.burst = d.long_on || (n < eff_duty);

    // desense hangover: scan back for burst end, decay 0.78/frame
    // (guard fi >= k: uint underflow at frame 0 would wrap to huge values)
    d.hang = 0.0;
    if (!d.burst && !d.long_on && st.desense > 0.0 && fi >= 1u) {
        for (int k = 1; k <= 8; k++) {
            if (fi < uint(k)) break;
            float nk = value_noise1f(float(fi - uint(k)) * 0.45, es ^ 0x88u);
            if (nk < eff_duty) {           // frame (fi-k) was a burst
                d.hang = st.desense * pow(0.78, float(k - 1));
                break;
            }
        }
    }

    d.scroll = float(fi) * st.scroll;
    d.fringe_ph = float(fi) * (0.7 + 1.4 * value_noise1f(float(fi) * 0.09, es ^ 0x2Bu));
    return d;
}

// ---------------------------------------------------------------------------
// geometry helpers: row_shift accumulation + seam (mirrors engine.py)
// ---------------------------------------------------------------------------
float row_shift_of(int j, SigSt ss_, MpSt ms_, IntSt isf_, IntDyn idyn,
                   uint fi, uint es_sig, uint es_mp, uint es_int,
                   uint tear_block, uint band_seed, float band_h) {
    float shift = 0.0;
    // multipath block displacement (24-row plateaus)
    if (ms_.block_shift > 0.0) {
        int blk = j / 24;
        float r = hash2_01(uint(blk), 0xB1u, es_mp ^ 0xD1u);
        if (r < 0.35) {
            float nfield = value_noise2f(float(blk) * 0.8, float(fi) * 0.02, es_mp ^ 0xB2u) * 2.0 - 1.0;
            shift += nfield * ms_.block_shift;
        }
    }
    // signal jitter + tear
    uint js = uint(j) * 7919u;
    float jx = hash_c1(js ^ (fi << 3), es_sig ^ 0x1111u) * ss_.h_jitter;
    if (ss_.h_jitter > 0.0) {
        jx += value_noise1c(float(j) * 0.013 + float(fi) * 0.05, es_sig ^ 0x2222u) * ss_.h_jitter * 0.55;
    }
    if (ss_.tear > 0.0) {
        uint bid = uint(j) / uint(max(band_h, 1.0));
        float r = hash3_01(bid, tear_block, 0x7Eu, band_seed);
        if (r < 0.16 * ss_.tear) {
            float amp = ss_.tear * (10.0 + 70.0 * hash2_01(bid, tear_block, es_sig ^ 0x3333u));
            jx += (hash2_01(bid, tear_block, es_sig ^ 0x4444u) < 0.5) ? amp : -amp;
        }
    }
    shift += jx;
    // interference false-sync during bursts
    if ((idyn.burst || idyn.long_on) && isf_.sync_false > 0.0) {
        uint bid = uint(j) / 5u;
        float p = 0.03 * isf_.sync_false / 18.0 + 0.015 * (isf_.sync_false / 18.0);
        if (hash3_01(bid, fi, 0x5Fu, es_int) < p) {
            shift += hash_c1(bid ^ fi, es_int ^ 0x5Au) * isf_.sync_false;
        }
    }
    return shift;
}

// ---------------------------------------------------------------------------
// main: effect body as a pure function of fragment coord (testable entry)
// ---------------------------------------------------------------------------
vec4 fpvEffect(vec2 fc) {
    float i = fc.x - 0.5;
    float j = floor(fc.y - 0.5);   // output row
    int oi = int(i);
    int oj = int(j);
    float H = uResolution.y;
    float W = uResolution.x;

    float rs = clamp(uSignalStrength, 0.0, 99.0) / 99.0;
    float rm = clamp(uMultipath, 0.0, 99.0) / 99.0;
    float ri = clamp(uRFInterference, 0.0, 99.0) / 99.0;

    // all-zero fast path: passthrough
    if (rs == 0.0 && rm == 0.0 && ri == 0.0) {
        return texture(uSrc, fc / uResolution);
    }

    SigSt ss_ = signal_stages(rs);
    MpSt  ms_ = multipath_stages(rm);
    IntSt isf_ = interference_stages(ri);

    uint fi = uFrame;
    uint es      = uSeed ^ (fi * 0x9E3779B9u);
    uint es_sig  = es ^ 0x51A1u;
    uint es_mp   = es ^ 0x3A77u;
    uint es_int  = es ^ 0x1F4Eu;

    MpDyn mpd = mp_dynamic(ms_, fi, es_mp);
    IntDyn idyn = int_dynamic(isf_, fi, es_int);

    float roll_pos = signal_roll_pos(ss_, fi, es_sig);
    float roll = float(int(roll_pos));
    bool seam_on = abs(roll_pos) >= 1.0 && roll_pos != roll;
    // seam row matches engine: (H - int(roll)) % H when rolling
    int seam_row = (roll != 0.0) ? int(mod(H - roll, H)) : -10;

    uint tear_block = fi >> 2;
    uint band_seed = es_sig ^ (tear_block * 0x2545u);
    float band_h = 4.0 + hash2_01(tear_block, 3u, band_seed) * 36.0;

    // -------- inverse geometry: recover source row --------
    int sj = oj + int(roll);
    sj = int(mod(float(sj), H));
    float shift = row_shift_of(sj, ss_, ms_, isf_, idyn, fi,
                               es_sig, es_mp, es_int, tear_block, band_seed, band_h);

    int isi = int(floor(shift));
    float frac = shift - float(isi);
    int xa = oi - isi;
    int xb = xa - 1;
    xa = clamp(xa, 0, int(W) - 1);
    xb = clamp(xb, 0, int(W) - 1);

    // -------- sample clean source at both subpixel taps --------
    vec3 s0 = src_yuv(uSrc, uResolution, float(xa), float(sj));
    vec3 s1 = src_yuv(uSrc, uResolution, float(xb), float(sj));

    vec3 chroma[2];
    for (int tap = 0; tap < 2; tap++) {
        int ci = (tap == 0) ? xa : xb;
        vec3 samp = (tap == 0) ? s0 : s1;
        float y0 = samp.x;
        float u0 = samp.y;
        float v0 = samp.z;
        float base_y = y0;

        // =================== MULTIPATH ===================
        float val = y0;
        float uu = u0;
        float vv = v0;

        // echo taps (mix-out form: val = base + a*(echo - base))
        if (ms_.comb_k > 1e-4) {
            int i1 = int(round(mpd.d1));
            if (i1 < 1) i1 = 1;
            float frac1 = mpd.d1 - float(i1);
            int xaa = clamp(ci - i1, 0, int(W) - 1);
            int xbb = clamp(xaa - 1, 0, int(W) - 1);
            float e0 = src_yuv(uSrc, uResolution, float(xaa), float(sj)).x;
            float e1 = src_yuv(uSrc, uResolution, float(xbb), float(sj)).x;
            float echo = e0 * (1.0 - frac1) + e1 * frac1;
            val += ms_.comb_k * (echo - base_y);
        }
        if (abs(mpd.a1) > 1e-4) {
            int xaa = clamp(ci - int(round(mpd.d1)), 0, int(W) - 1);
            float e = src_yuv(uSrc, uResolution, float(xaa), float(sj)).x;
            val += mpd.a1 * (e - base_y);
        }
        if (abs(mpd.a2) > 1e-4) {
            int xaa = clamp(ci - int(round(mpd.d2)), 0, int(W) - 1);
            float e = src_yuv(uSrc, uResolution, float(xaa), float(sj)).x;
            val += mpd.a2 * (e - base_y);
        }
        if (abs(mpd.a3) > 1e-4) {
            int xaa = clamp(ci - int(round(mpd.d3)), 0, int(W) - 1);
            float e = src_yuv(uSrc, uResolution, float(xaa), float(sj)).x;
            val += mpd.a3 * (e - base_y);
        }

        // horizontal smear (blended, weight capped at 0.65 so echoes survive)
        int radius = int(round(ms_.smear_px));
        if (radius > 0) {
            float acc = 0.0;
            float cnt = 0.0;
            for (int k = -6; k <= 6; k++) {
                if (abs(k) > radius) continue;
                int xaa = clamp(ci + k, 0, int(W) - 1);
                acc += src_yuv(uSrc, uResolution, float(xaa), float(sj)).x;
                cnt += 1.0;
            }
            float blur = acc / max(cnt, 1.0);
            float smear_w = 0.30 + 0.35 * min(1.0, ms_.smear_px / 3.5);
            val = val * (1.0 - smear_w) + blur * smear_w;
        }

        // FM edge spikes
        if (ms_.edge_spike > 0.0) {
            int sp = 3 + int(hash2_01(fi, 55u, es_mp) * 5.0);
            int xaa = ci - sp;
            if (xaa >= 1 && xaa < int(W) - 1) {
                float ga = src_yuv(uSrc, uResolution, float(xaa), float(sj)).x;
                float gb = src_yuv(uSrc, uResolution, float(float(xaa) - 1.0), float(sj)).x;
                float g = ga - gb;
                if (g > 0.28)      val += ms_.edge_spike * min(g, 1.0) * 0.55;
                else if (g < -0.28) val -= ms_.edge_spike * min(-g, 1.0) * 0.55;
            }
        }

        // white ripple band
        if (ms_.band_ripple > 0.0) {
            float bn = value_noise1f(float(sj) * 0.045 + mpd.band_ph, es_mp ^ 0xBD2u);
            if (bn > 0.62) val += ms_.band_ripple * (bn - 0.62) * 2.8;
        }

        // monochrome block (desat bands)
        bool mono = false;
        if (ms_.desat_bands > 0.0) {
            int blk = sj / 24;
            float r2 = hash2_01(uint(blk), 0xC1u, es_mp ^ 0xD2u);
            if (r2 < ms_.desat_bands * 0.30) mono = true;
        }

        // deep null: short monochrome static window
        if (mpd.null_on) {
            uint js = uint(sj) * 7919u;
            float nn1 = hash2_01(uint(ci), js ^ fi, es_mp ^ 0xE1u);
            float nn2 = hash2_01(uint(ci) + 1u, js ^ fi, es_mp ^ 0xE1u);
            float noise_field = (nn1 + nn2 - 1.0) * 1.9;
            float staticv = 0.08 + abs(noise_field) * ms_.null_snow;
            float keep = 0.10;
            val = val * keep + staticv * (1.0 - keep);
            val += noise_field * ms_.null_snow * 0.30;
            uu = 0.0;
            vv = 0.0;
        }
        // white AGC flash
        if (mpd.flash_on) {
            val = val * 1.06 + 0.28;
            uu *= 0.2;
            vv *= 0.2;
        }
        if (mono) { uu = 0.0; vv = 0.0; }

        // =================== WEAK SIGNAL ===================
        uint js = uint(sj) * 7919u;
        float env = 0.72 + 0.56 * value_noise2f(float(fi) * 0.13, 3.7, es_sig);
        float sigma = ss_.luma_noise * env;
        float n_gain = sigma * 2.45;
        float c_gain = ss_.chroma_noise * env;
        float img_keep = 1.0 - ss_.static_kill * 0.93;

        // color killer (stateless threshold on sat_loss; Python uses hysteresis
        // but sat_loss is monotonic in param so threshold matches for constant params)
        float sat_loss = ss_.sat_loss;
        bool killer = sat_loss >= 0.50;
        float sat_gain = killer ? 0.0 : max(0.0, 1.0 - sat_loss);

        float yy = val * img_keep;
        float flow_x = float(ci) * 0.055 + float(fi) * 1.9;
        float flow_y = float(sj) * 0.055 + float(fi) * 0.7;
        float flow = value_noise2f(flow_x, flow_y, es_sig ^ 0x333u) * 2.0 - 1.0;
        float g1 = hash2_01(uint(ci), js, es_sig ^ 0x77u);
        float g2 = hash2_01(uint(ci) + 1u, js, es_sig ^ 0x77u);
        float grain = g1 + g2 - 1.0;
        float n = n_gain * (0.50 * flow + 0.50 * grain * 1.9);

        if (ss_.static_kill > 0.0) {
            float static_field = n_gain * 2.6 * (0.5 * flow + 0.5 * grain * 2.2) + 0.03;
            yy = yy * (1.0 - ss_.static_kill) + static_field * ss_.static_kill;
        }
        // occasional static line runs
        if (ss_.tear > 0.0) {
            uint seg = uint(sj) / 3u;
            if (hash3_01(seg, fi, 0x55u, es_sig) < 0.004 * ss_.tear) {
                yy = 0.04 + abs(grain) * sigma * 4.0;
            }
        }
        yy += n;
        yy = clamp(yy, 0.0, 1.0);

        // chroma path
        float hue_ph = float(fi) * (0.028 + 0.09 * hash_c1(fi, es_sig ^ 0xABCu));
        float line_ang = ss_.hue_wander * (
            sin(hue_ph + float(sj) * 0.11)
            + 0.5 * hash_c1(js ^ fi, es_sig ^ 0xBEEFu)
        );
        float ca = cos(line_ang);
        float sa = sin(line_ang);
        float ru = uu * ca - vv * sa;
        float rv = uu * sa + vv * ca;
        uu = ru; vv = rv;

        if (!killer && c_gain > 0.0) {
            float cx = float(ci) * 0.12;
            float cy = float(sj) * 0.12 + float(fi) * 0.9;
            float cn1 = value_noise2f(cx, cy, es_sig ^ 0x901u) * 2.0 - 1.0;
            float cn2 = value_noise2f(cx + 37.0, cy + 11.0, es_sig ^ 0x902u) * 2.0 - 1.0;
            float dark = 0.35 + 0.65 * (1.0 - yy);
            uu += cn1 * c_gain * dark;
            vv += cn2 * c_gain * dark;
        }
        if (sat_gain < 1.0) { uu *= sat_gain; vv *= sat_gain; }
        if (killer) { uu = 0.0; vv = 0.0; }

        // sparklies (override luma)
        if (ss_.sparklie > 0.0) {
            int dash_len = 2 + int(hash2_01(uint(sj), fi, es_sig ^ 0xD0u) * 10.0);
            int gx = ci / max(dash_len, 1);
            float sp = hash3_01(uint(gx), uint(sj), fi, es_sig ^ 0xD1u);
            if (sp < ss_.sparklie * 0.055) {
                yy = (hash2_01(uint(gx), uint(sj), es_sig ^ 0xD2u) < 0.5) ? 1.0 : 0.0;
            }
        }

        // =================== RF INTERFERENCE ===================
        float fringe_ph = idyn.fringe_ph;
        if (isf_.fringe > 1e-4) {
            float f_k = 0.055 + hash2_01(fi, 97u, es_int ^ 0x33u) * 0.11;
            float f_k2 = f_k * 1.13;
            float s = sin(float(sj) * f_k + fringe_ph) + sin(float(sj) * f_k2 - fringe_ph * 0.83);
            yy += isf_.fringe * 0.5 * s;
        }

        // scrolling noise bands
        float band_gate = 0.0;
        int sid = sj / max(6 + int(hash2_01(fi, 91u, es_int ^ (fi * 0x5BD1u)) * 30.0), 1);
        bool slice_on = false;
        if ((idyn.burst || idyn.long_on) && isf_.band_amp > 0.0) {
            float yb = (float(sj) - idyn.scroll) * isf_.band_count / max(H, 1.0);
            float bn = value_noise1f(yb, es_int ^ 0xBBu);
            if (bn > isf_.band_thr) {
                band_gate = min((bn - isf_.band_thr) * 3.0, 1.0) * isf_.band_amp;
            }
            if (idyn.long_on) band_gate = max(band_gate, isf_.band_amp * 0.25);
        }
        if ((idyn.burst || idyn.long_on) && isf_.slice > 0.0) {
            uint slice_seed = es_int ^ (fi * 0x5BD1u);
            if (hash2_01(uint(sid), fi, slice_seed ^ 0x99u) < isf_.slice * 0.22) {
                slice_on = true;
            }
        }
        float tint_sign = hash2_01(slice_on ? uint(sid) : uint(sj), 5u, es_int ^ 0x71u) < 0.5 ? -1.0 : 1.0;

        if (band_gate > 0.0) {
            uint ib = uint(ci) >> 3;
            float n1 = hash2_01(ib, js ^ fi, es_int ^ 0xB1u);
            float n2 = hash2_01(ib + 1u, js ^ fi, es_int ^ 0xB1u);
            float streak = (n1 + n2 - 1.0) * 1.9;
            float replacement = 0.5 + streak * 0.55;
            yy = yy * (1.0 - band_gate) + replacement * band_gate;
            if (isf_.tint > 0.0) {
                // replace scene chroma in the bar so the tint coin decides polarity
                uu *= (1.0 - band_gate);
                vv *= (1.0 - band_gate);
                float tt = band_gate * isf_.tint * 0.30;
                if (tint_sign > 0.0) { uu += tt; vv += tt * 0.7; }
                else                  { uu -= tt; vv -= tt * 0.6; }
            }
            if (isf_.impulse > 0.0) {
                if (hash2_01(uint(ci), js ^ (fi << 2), es_int ^ 0xB2u) < isf_.impulse * 0.07) {
                    yy = (hash2_01(uint(ci), js, es_int ^ 0xB3u) < 0.5) ? 1.0 : 0.0;
                }
            }
        }
        if (slice_on) {
            float sn  = hash2_01(uint(ci) >> 1, js ^ (fi * 3u), es_int ^ 0x51u);
            float sn2 = hash2_01((uint(ci) >> 1) + 1u, js ^ (fi * 3u), es_int ^ 0x51u);
            float staticv = 0.5 + (sn + sn2 - 1.0) * 1.4;
            float mode = hash2_01(uint(sid), fi, es_int ^ 0x61u);
            yy = (mode < 0.25) ? (1.0 - yy) : staticv;
            if (isf_.tint > 0.2) {
                uu += isf_.tint * 0.10 * tint_sign;
                vv -= isf_.tint * 0.08 * tint_sign;
            }
        }
        if (idyn.hang > 0.001) {
            float hn  = hash2_01(uint(ci), js ^ (fi * 7u), es_int ^ 0x41u);
            float hn2 = hash2_01(uint(ci) + 1u, js ^ (fi * 7u), es_int ^ 0x41u);
            yy += (hn + hn2 - 1.0) * isf_.desense * idyn.hang * 2.4;
        }

        yy = clamp(yy, 0.0, 1.0);
        ymix[tap] = yy;
        chroma[tap] = vec3(yy, uu, vv);
    }

    // subpixel lerp between the two taps (matches apply_geometry)
    vec3 outy = mix(chroma[0], chroma[1], frac);

    // roll seam: blanking-interval noise
    if (seam_on && abs(oj - seam_row) < 3) {
        uint js = uint(sj) * 7919u;
        float nn = hash2_01(uint(oi), js ^ fi, es ^ 0x5EAu);
        outy.x = 0.03 + nn * nn * 0.5;
        outy.y *= 0.1;
        outy.z *= 0.1;
    }

    vec3 rgb = yuv_to_rgb(outy);
    // ordered dither to kill 8-bit banding after corruption
    float dth = (hash2_01(uint(oi), uint(oj), uSeed ^ 0xD17hu) - 0.5) / 255.0;
    rgb += dth;
    return vec4(clamp(rgb, 0.0, 1.0), 1.0);
}

#ifndef SHADERTOY
void main() {
    fragColor = fpvEffect(gl_FragCoord.xy);
}
#endif

// ---------------------------------------------------------------------------
// Shadertoy wrapper: delete the #version line, compile with -DSHADERTOY,
// bind the clean frame to iChannel0. Params via the three defines.
// ---------------------------------------------------------------------------
#ifdef SHADERTOY
#define FPV_SIGNAL_DEFAULT 75.0
#define FPV_MULTIPATH_DEFAULT 40.0
#define FPV_INTERFERENCE_DEFAULT 40.0
void mainImage(out vec4 o, in vec2 fc) {
    uSeed = 12345u;
    uFrame = uint(iFrame);
    uSignalStrength = FPV_SIGNAL_DEFAULT;
    uMultipath = FPV_MULTIPATH_DEFAULT;
    uRFInterference = FPV_INTERFERENCE_DEFAULT;
    o = fpvEffect(fc);
}
#endif
