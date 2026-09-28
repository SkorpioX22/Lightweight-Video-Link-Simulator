"""HDZero digital link simulation for LVLS.

from engine.hdzero import HDZeroEngine, PRESETS
eng = HDZeroEngine(seed=42)
out = eng.process(frame, {"signalStrength": 40, "multipath": 70, "rfInterference": 10})
"""
from .engine import HDZeroEngine
from .params import HDZ_PRESETS, PARAM_MAX, PARAM_MIN, clamp_param, raw

PRESETS = HDZ_PRESETS

__all__ = [
    "HDZeroEngine",
    "HDZ_PRESETS",
    "PRESETS",
    "PARAM_MIN",
    "PARAM_MAX",
    "clamp_param",
    "raw",
]
