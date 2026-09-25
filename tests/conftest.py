"""Pytest setup: project root on sys.path + session-wide Numba JIT warmup."""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import AnalogVideoBreakupEngine  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def numba_warmup():
    """Compile all Numba kernels once per session so timing tests measure
    steady-state performance, not first-call JIT compilation."""
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    frame[:, :40, 0] = 200
    frame[:, 40:, 2] = 180
    eng = AnalogVideoBreakupEngine(seed=12345)
    eng.process(frame, {"signalStrength": 60, "multipath": 60, "rfInterference": 60})
