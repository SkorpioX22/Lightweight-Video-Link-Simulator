"""LVLS — Lightweight Video Link Simulation.

RF-to-video degradation model for analog FPV links.

Public API:
    from engine import AnalogVideoBreakupEngine, PRESETS
    eng = AnalogVideoBreakupEngine(seed=42)
    out = eng.process(frame, {"signalStrength": 40, "multipath": 70, "rfInterference": 10})
"""
from .core.params import PRESETS, PARAM_MIN, PARAM_MAX, clamp_param, normalize_params
from .renderer.engine import AnalogVideoBreakupEngine

__all__ = [
    "AnalogVideoBreakupEngine",
    "PRESETS",
    "PARAM_MIN",
    "PARAM_MAX",
    "clamp_param",
    "normalize_params",
    "__version__",
]

__version__ = "1.0.0"
