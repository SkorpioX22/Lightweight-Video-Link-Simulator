"""HDZero severity curves + presets.

Same public contract as the analog core: three 0..99 keys, 0 = clean,
99 = extreme. Physical mapping (docs/research/hdzero.md):

  signalStrength   - bit error rate      -> scattered corrupt 8x8 DCT blocks,
                     white speckle, breakup-to-black at the extreme
  multipath        - burst bit errors    -> clustered punch-out patches of
                     corrupt/stale (held) blocks
  rfInterference   - burst takeovers     -> blocky band scrambles, rainbow
                     screen, colored-line gibberish, black screen events
"""
from ..core.params import PARAM_MAX, PARAM_MIN, PRESETS, clamp_param, raw, smoothstep

HDZ_PRESETS = dict(PRESETS)


def hdz_signal_stages(r):
    """Weak digital link -> block-error field parameters."""
    return {
        # per-block corrupt probability (mixture of epoch-stable + per-frame)
        "block_err": smoothstep(0.04, 0.97, r) * 0.62,
        # per-pixel white speckle probability (single-bit errors)
        "speckle": smoothstep(0.12, 0.92, r) * 0.0035,
        # full-frame blackout probability (digital breakup-to-black)
        "black_frame": smoothstep(0.80, 1.0, r) * 0.15,
        # extra white-block style share inside corrupted blocks
        "white_blk": smoothstep(0.30, 0.90, r) * 0.05,
        # codec chroma quant step (always-on codec look; engine gates on
        # signal > 0 so 0-signal links never get it)
        "chroma_q": 6.0 + smoothstep(0.0, 1.0, r) * 18.0,
        # signal-loss latch: entry chance per frame + run length (frames).
        # The no-signal screen starts a little before blackout and sustains
        # in runs, so high severity spends most of its time on the loss
        # screen instead of flickering per frame.
        "loss_p": smoothstep(0.74, 1.0, r) * 0.10,
        "loss_dur_lo": 4.0 + r * 14.0,
        "loss_dur_hi": 10.0 + r * 40.0,
        # frames from loss-run start to full no-signal coverage: random
        # 8x8 patches of the real no-signal screen accumulate until the
        # whole frame is the screen (higher severity ramps faster)
        "loss_ramp": 8.0 + (1.0 - r) * 8.0,
    }


def hdz_multipath_stages(r):
    """Digital multipath -> spatially clustered error bursts."""
    return {
        "cl_cov": smoothstep(0.08, 1.0, r) * 0.42,       # cluster coverage
        "cl_boost": smoothstep(0.25, 1.0, r) * 0.80,     # corrupt prob in cluster
        "env_rate": 0.045 + r * 0.10,                    # burst envelope speed
        "mp_speed": 0.002 + r * 0.018,                   # cluster drift
    }


def hdz_interference_stages(r):
    """Digital interference -> full-frame/band takeover event machine."""
    return {
        "duty": smoothstep(0.05, 0.97, r) * 0.72,        # time inside events
        # at max severity an event lasts ~0.2-0.4 s and gaps shrink to a few
        # frames -> the link is taken over most of the time (abrupt, busy);
        # low severity gets short rare events instead
        "dur_lo": 2.0 + r * 4.0,                         # event length (frames)
        "dur_hi": 4.0 + r * 8.0,
        "gap_lo": 3.0,
        "gap_hi": 8.0 + (1.0 - r) * 60.0,                # high severity: short gaps
        "band_speed": 0.004 + r * 0.02,                  # scroll, frame-normalized
        "band_frac": 0.05 + r * 0.28,                    # band height / frame
        # cumulative event-type weights (else -> blocky band scramble)
        "w_rain": smoothstep(0.45, 1.0, r) * 0.30,       # rainbow screen
        "w_lines": smoothstep(0.35, 0.95, r) * 0.30,     # colored-line gibberish
        "w_black": smoothstep(0.70, 1.0, r) * 0.25,      # black screen
        "density": smoothstep(0.20, 0.95, r) * 0.75,     # line-event row density
    }


__all__ = [
    "HDZ_PRESETS",
    "hdz_signal_stages",
    "hdz_multipath_stages",
    "hdz_interference_stages",
    "PARAM_MIN",
    "PARAM_MAX",
    "clamp_param",
    "raw",
]
