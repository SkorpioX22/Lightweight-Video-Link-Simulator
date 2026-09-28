"""LVLS — Lightweight Video Link Simulation.

Video-link degradation models for FPV simulators:
  - AnalogVideoBreakupEngine: analog RF link (snow, ghosts, tear, roll...)
  - HDZeroEngine: digital HDZero link (block errors, punch-outs, takeovers)

Public API:
    from engine import AnalogVideoBreakupEngine, HDZeroEngine, PRESETS
    eng = AnalogVideoBreakupEngine(seed=42)
    out = eng.process(frame, {"signalStrength": 40, "multipath": 70, "rfInterference": 10})
"""
from .core.params import PRESETS, PARAM_MIN, PARAM_MAX, clamp_param, normalize_params
from .hdzero.engine import HDZeroEngine
from .hdzero.params import HDZ_PRESETS
from .renderer.engine import AnalogVideoBreakupEngine

__all__ = [
    "AnalogVideoBreakupEngine",
    "HDZeroEngine",
    "PRESETS",
    "HDZ_PRESETS",
    "PARAM_MIN",
    "PARAM_MAX",
    "clamp_param",
    "normalize_params",
    "__version__",
]

__version__ = "1.0.0"
