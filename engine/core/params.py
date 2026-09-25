"""Parameter normalization and per-effect severity curves.

Public inputs are integers 0..99 (clamped):
  0  = no effect (clean link / no multipath / no interference)
  99 = extreme effect

Raw severity r = clamp(v,0,99)/99 in [0,1]. Each effect module maps r
through its own nonlinear response shaped by the physical research
(docs/research): stages appear in the observed real-world order and are
deliberately NOT linear in r.
"""
import numpy as np

PARAM_MIN = 0
PARAM_MAX = 99


def clamp_param(v):
    try:
        iv = int(round(float(v)))
    except (TypeError, ValueError):
        iv = 0
    return max(PARAM_MIN, min(PARAM_MAX, iv))


def raw(v):
    """0..99 -> 0..1"""
    return clamp_param(v) / float(PARAM_MAX)


def smoothstep(e0, e1, x):
    if e1 == e0:
        t = 1.0 if x >= e1 else 0.0
    else:
        t = (x - e0) / (e1 - e0)
        t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def ease_in_pow(x, p):
    return max(0.0, min(1.0, x)) ** p


def signal_stages(r):
    """Weak-signal severity -> stage intensities.

    Ordering matches research (ImmersionRC-style progression):
    fine grain -> snow -> hue/sat instability -> color kill -> heavy snow +
    sparklies -> jitter/tear -> roll events -> near-total static.
    """
    return {
        "luma_noise": 42.0 * ease_in_pow(smoothstep(0.015, 1.0, r), 1.22) / 255.0,
        # chroma blotches: subtle, dark-weighted in module (color dies first, but
        # early specks must not become rainbow confetti)
        "chroma_noise": smoothstep(0.04, 0.40, r) * 0.11,
        "sat_loss": smoothstep(0.18, 0.50, r),          # saturation collapse before killer
        "hue_wander": smoothstep(0.12, 0.45, r) * 0.55,  # radians-ish amplitude
        # color-killer thresholds on RAW (hysteresis applied in module)
        "killer_on_th": 0.50,
        "killer_off_th": 0.42,
        "sparklie": ease_in_pow(smoothstep(0.42, 0.88, r), 2.0) * 0.5,
        "h_jitter": smoothstep(0.52, 0.80, r) * 7.0,      # px
        "tear": smoothstep(0.62, 0.92, r),                # probability scale
        "roll": smoothstep(0.72, 0.97, r),                # roll event probability
        "static_kill": smoothstep(0.88, 1.0, r),          # image -> pure static
        "black_lift": smoothstep(0.05, 0.7, r) * 0.04,    # snow lifts black level
    }


def multipath_stages(r):
    """Multipath severity -> stage intensities.

    Structural-first: halo/comb -> single ghost -> multi-ghost + smear ->
    block displacement + bands -> null dropouts / white flashes.
    Snow stays near zero (except during null bursts handled by module).
    """
    return {
        "comb_k": smoothstep(0.04, 0.55, r) * 0.32,       # sub-pixel echo / HF comb
        "smear_px": smoothstep(0.18, 0.92, r) * 3.5,      # horizontal box blur radius
        "ghost1_d": 4.0 + smoothstep(0.12, 1.0, r) * 30.0,
        # amps pushed so typical |a1| > 0.5: visible double image, not just blend
        "ghost1_a": smoothstep(0.12, 0.75, r) * 0.85,
        "ghost2_d": 16.0 + smoothstep(0.40, 1.0, r) * 75.0,
        "ghost2_a": smoothstep(0.38, 0.85, r) * 0.50,
        "ghost3_a": smoothstep(0.30, 0.90, r) * 0.40,     # short-delay ghost (~0.45*d1)
        "block_shift": smoothstep(0.45, 0.95, r) * 42.0,  # px plateaus on some row-blocks
        "band_ripple": smoothstep(0.28, 0.85, r) * 0.14,  # white horizontal bands
        "edge_spike": smoothstep(0.35, 0.9, r) * 0.55,    # FM edge spike stripes
        # nulls: real but RARE + SHORT (pocket dropouts, not half-second static)
        "null_rate": ease_in_pow(smoothstep(0.55, 1.0, r), 1.5) * 0.05,
        "flash_rate": smoothstep(0.42, 0.98, r) * 0.05,
        "null_snow": smoothstep(0.5, 1.0, r) * 0.50,      # click-noise during nulls
        "desat_bands": smoothstep(0.55, 0.95, r) * 0.8,   # B&W row-blocks (burst echo)
    }


def interference_stages(r):
    """RF interference severity -> stage intensities.

    Bursty + banded: duty-cycle scrolling noise bars -> fringe -> slice
    corruption -> impulse hash during bursts -> rare long "power-up" events.
    """
    return {
        # curves stretched to ~0.95 so 75 -> 99 is clearly heavier
        "duty": smoothstep(0.06, 0.97, r) * 0.60,         # fraction of time in burst
        "band_amp": smoothstep(0.10, 0.95, r),
        "band_thr": 0.72 - 0.14 * r,                      # lower => denser bars at high I
        "scroll": 0.4 + smoothstep(0.1, 1.0, r) * 5.0,    # lines per frame
        "band_count": 3.0 + r * 12.0,                     # discrete bars across height
        "tint": smoothstep(0.25, 0.95, r),                # green/purple band tint (scaled in module)
        "fringe": smoothstep(0.28, 0.85, r) * 0.15,       # heterodyne luminance sine
        "slice": smoothstep(0.48, 0.99, r) * 0.55,        # slice replace probability
        "impulse": smoothstep(0.35, 0.95, r) * 0.45,      # salt in bursts
        "desense": smoothstep(0.3, 0.95, r) * 0.35,       # post-burst snow hangover
        "long_event": smoothstep(0.65, 1.0, r) * 0.03,    # rare multi-frame takeovers
        "sync_false": smoothstep(0.7, 1.0, r) * 18.0,     # false-sync row shifts (px)
    }


DEFAULT_PARAMS = {"signalStrength": 0, "multipath": 0, "rfInterference": 0}

PRESETS = {
    "CLEAN": {"signalStrength": 0, "multipath": 0, "rfInterference": 0},
    "LONG RANGE": {"signalStrength": 45, "multipath": 8, "rfInterference": 0},
    "WEAK SIGNAL": {"signalStrength": 72, "multipath": 10, "rfInterference": 5},
    "HEAVY MULTIPATH": {"signalStrength": 12, "multipath": 85, "rfInterference": 3},
    "WIFI INTERFERENCE": {"signalStrength": 8, "multipath": 15, "rfInterference": 72},
    "SEVERE INTERFERENCE": {"signalStrength": 30, "multipath": 22, "rfInterference": 95},
    "NEAR VIDEO LOSS": {"signalStrength": 88, "multipath": 55, "rfInterference": 70},
}

PARAM_KEYS = ("signalStrength", "multipath", "rfInterference")


def normalize_params(params=None):
    """Merge + clamp a params dict -> ints 0..99 for the three keys."""
    out = dict(DEFAULT_PARAMS)
    if params:
        for k in PARAM_KEYS:
            if k in params and params[k] is not None:
                out[k] = clamp_param(params[k])
    return out
