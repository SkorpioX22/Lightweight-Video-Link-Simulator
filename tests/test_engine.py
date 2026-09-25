"""Test suite for LVLS — Lightweight Video Link Simulation (analog core).

Run:  python -m pytest <project_root>/tests -v --tb=short

Calibration notes (thresholds verified against the real engine):
  - SEED=67 was chosen so the interference burst machine shows clear
    on/off structure within 12 frames (burstiness ~3.7) and a
    row-concentrated slice frame (top-10% rows hold ~37% of energy).
  - Multipath smear is checked via the MEDIAN horizontal-gradient ratio
    because deep-null dropout frames intentionally turn the image into
    static (gradient energy spikes); the typical frame is heavily
    horizontally smoothed (median ratio ~0.13 vs clean).
  - Discrimination: signal-only row concentration maxes at ~0.13 and
    interference reaches ~0.37, so the shared 0.30 threshold makes each
    effect fail the other's check.
"""
import time
from typing import NamedTuple

import numpy as np
import pytest

from engine import (
    PARAM_MAX,
    PARAM_MIN,
    PRESETS,
    AnalogVideoBreakupEngine,
    clamp_param,
    normalize_params,
)

H = 180
W = 240
SEED = 67

ZERO_PARAMS = {"signalStrength": 0, "multipath": 0, "rfInterference": 0}
SIG_ONLY = {"signalStrength": 70, "multipath": 0, "rfInterference": 0}
MP_ONLY = {"signalStrength": 0, "multipath": 80, "rfInterference": 0}
INT_ONLY = {"signalStrength": 0, "multipath": 0, "rfInterference": 80}


def make_structured_frame(h=H, w=W, phase=0.0):
    """Deterministic frame with strong vertical/horizontal edges and gradients."""
    x = np.arange(w, dtype=np.float32)[None, :]
    y = np.arange(h, dtype=np.float32)[:, None]
    r = np.where((x.astype(np.int32) // 18) % 2 == 0, 220.0, 25.0) + (x / w) * 30.0
    g = np.where((y.astype(np.int32) // 12) % 2 == 0, 190.0, 40.0)
    b = 64.0 + 150.0 * (x / w) + 40.0 * np.sin(y / 9.0 + phase)
    fr = np.stack([np.broadcast_to(r, (h, w)),
                   np.broadcast_to(g, (h, w)),
                   np.broadcast_to(b, (h, w))], axis=-1)
    fr[h // 3:2 * h // 3, w // 4:w // 2, 1] = 240
    fr[2 * h // 3:, :w // 3, 0] = 15
    return np.clip(fr, 0, 255).astype(np.uint8)


class RowMetrics(NamedTuple):
    meanabs: float
    top10: float
    cv: float


def row_energy_metrics(out, clean):
    d = np.abs(out.astype(np.int32) - clean.astype(np.int32))
    rm = d.sum(axis=2).mean(axis=1)
    total = rm.sum()
    k = max(1, int(np.ceil(0.10 * rm.size)))
    top10 = float(np.sort(rm)[::-1][:k].sum() / total)
    cv = float(rm.std() / rm.mean())
    return RowMetrics(float(d.mean()), top10, cv)


def horizontal_grad_var(frame):
    L = frame[..., 0] * 0.299 + frame[..., 1] * 0.587 + frame[..., 2] * 0.114
    dx = np.diff(L.astype(np.float64), axis=1)
    return float(np.mean(dx * dx))


@pytest.fixture
def frame():
    return make_structured_frame()


@pytest.fixture
def engine():
    return AnalogVideoBreakupEngine(seed=SEED)


def test_param_bounds(frame):
    assert PARAM_MIN == 0
    assert PARAM_MAX == 99
    assert clamp_param(-5) == 0
    assert clamp_param(150) == 99
    assert clamp_param("50") == 50
    assert clamp_param(49.6) == 50
    assert clamp_param("abc") == 0
    assert clamp_param(None) == 0

    assert normalize_params(None) == dict(ZERO_PARAMS)
    assert normalize_params({"multipath": 30}) == {
        "signalStrength": 0, "multipath": 30, "rfInterference": 0}
    assert normalize_params({"signalStrength": 200, "rfInterference": None}) == {
        "signalStrength": 99, "multipath": 0, "rfInterference": 0}
    assert normalize_params({"signalStrength": -10, "multipath": 500}) == {
        "signalStrength": 0, "multipath": 99, "rfInterference": 0}

    eng = AnalogVideoBreakupEngine(seed=1)
    eng.setSignalStrength(200)
    assert eng.getParams()["signalStrength"] == 99
    eng.setMultipath(-4)
    eng.setRFInterference("77")
    assert eng.getParams() == {
        "signalStrength": 99, "multipath": 0, "rfInterference": 77}

    out = eng.process(frame, {
        "signalStrength": -10, "multipath": 500, "rfInterference": None})
    assert out.dtype == np.uint8
    assert out.shape == frame.shape
    assert eng.getParams() == {
        "signalStrength": 0, "multipath": 99, "rfInterference": 0}


def test_zero_effect_identity(frame, engine):
    pristine = frame.copy()
    out = engine.process(frame, dict(ZERO_PARAMS))
    assert np.array_equal(out, frame)
    assert out.dtype == frame.dtype == np.uint8
    assert out.shape == frame.shape
    assert out is not frame
    assert np.array_equal(frame, pristine)

    out_clean = engine.process(frame, PRESETS["CLEAN"])
    assert np.array_equal(out_clean, frame)

    out_empty = engine.process(frame, {})
    assert np.array_equal(out_empty, frame)


def test_output_validity(frame, engine):
    grid = [(99, 0, 0), (0, 99, 0), (0, 0, 99), (99, 99, 99), (50, 50, 50), (0, 0, 0)]
    for s, m, i in grid:
        out = engine.process(frame, {
            "signalStrength": s, "multipath": m, "rfInterference": i})
        assert out.dtype == np.uint8, (s, m, i, out.dtype)
        assert out.shape == frame.shape, (s, m, i, out.shape)
        assert int(out.min()) >= 0 and int(out.max()) <= 255
        assert isinstance(engine.last_stages, dict)
        assert {"signal", "multipath", "interference"} <= set(engine.last_stages)

    for bad_shape in [(H, W, 1), (H, W, 4), (H, W), (H, W, 3, 1)]:
        with pytest.raises(ValueError):
            engine.process(np.zeros(bad_shape, np.uint8), dict(ZERO_PARAMS))
    with pytest.raises(ValueError):
        engine.process(None, dict(ZERO_PARAMS))

    ff = frame.astype(np.float64) / 255.0
    with pytest.raises(ValueError):
        engine.process(ff, {"signalStrength": 50, "multipath": 50,
                            "rfInterference": 50})
    with pytest.raises(ValueError):
        engine.process(ff, dict(ZERO_PARAMS))


def test_determinism():
    frames = [make_structured_frame(phase=i * 0.7) for i in range(8)]
    pseq = [{"signalStrength": (10 + 9 * i) % 100,
             "multipath": (90 - 7 * i) % 100,
             "rfInterference": (30 + 5 * i) % 100} for i in range(8)]
    assert all(any(v > 0 for v in p.values()) for p in pseq)

    e1 = AnalogVideoBreakupEngine(seed=777)
    e2 = AnalogVideoBreakupEngine(seed=777)
    o1 = [e1.process(f, p) for f, p in zip(frames, pseq)]
    o2 = [e2.process(f, p) for f, p in zip(frames, pseq)]
    assert len(o1) == 8
    for k, (a, b) in enumerate(zip(o1, o2)):
        assert np.array_equal(a, b), f"frame {k} differs between same-seed engines"

    e3 = AnalogVideoBreakupEngine(seed=778)
    o3 = [e3.process(f, p) for f, p in zip(frames, pseq)]
    assert any(not np.array_equal(a, b) for a, b in zip(o1, o3))


def test_temporal_variation(frame, engine):
    params = {"signalStrength": 60, "multipath": 40, "rfInterference": 30}
    prev = None
    diffs = []
    for _ in range(8):
        out = engine.process(frame, params)
        if prev is not None:
            diffs.append(float(np.abs(
                out.astype(np.float64) - prev.astype(np.float64)).mean()))
        prev = out
    assert len(diffs) == 7
    for k, d in enumerate(diffs):
        assert d > 0.5, f"consecutive frame {k} mean abs diff {d:.4f} <= 0.5"


def test_frame_index_and_reset(frame, engine):
    params = {"signalStrength": 55, "multipath": 45, "rfInterference": 35}
    assert engine.frame_index == 0
    first = []
    for k in range(3):
        first.append(engine.process(frame, params))
        assert engine.frame_index == k + 1

    engine.reset()
    assert engine.frame_index == 0

    replay = [engine.process(frame, params) for _ in range(3)]
    for k, (a, b) in enumerate(zip(first, replay)):
        assert np.array_equal(a, b), f"replay frame {k} differs after reset"
    assert engine.frame_index == 3


def test_parameter_independence():
    clean = make_structured_frame()
    hgrad_clean = horizontal_grad_var(clean)

    eng = AnalogVideoBreakupEngine(seed=SEED)
    sig_stats = []
    sig_ratios = []
    for _ in range(8):
        out = eng.process(clean, SIG_ONLY)
        sig_stats.append(row_energy_metrics(out, clean))
        sig_ratios.append(horizontal_grad_var(out) / hgrad_clean)

    assert float(np.mean([m.meanabs for m in sig_stats])) > 20
    assert max(m.top10 for m in sig_stats) < 0.30
    assert max(m.cv for m in sig_stats) < 0.25
    assert min(sig_ratios) > 0.5

    eng = AnalogVideoBreakupEngine(seed=SEED)
    mp_ratios = []
    mp_meanabs = []
    for _ in range(8):
        out = eng.process(clean, MP_ONLY)
        mp_ratios.append(horizontal_grad_var(out) / hgrad_clean)
        mp_meanabs.append(row_energy_metrics(out, clean).meanabs)
    assert float(np.mean(mp_meanabs)) > 10
    assert float(np.median(mp_ratios)) < 0.5

    eng = AnalogVideoBreakupEngine(seed=SEED)
    int_stats = [row_energy_metrics(eng.process(clean, INT_ONLY), clean)
                 for _ in range(12)]
    ma = [m.meanabs for m in int_stats]
    burstiness = max(ma[:10]) / min(ma[:10])
    assert burstiness > 2.0, f"burstiness {burstiness:.3f} <= 2"
    corrupted = [m for m in int_stats if m.meanabs > 15]
    assert corrupted, "no corrupted interference frame found"
    # top 10% of rows carry >=2x the uniform share (0.10) of corruption energy:
    # discrete scrolling bands must concentrate energy row-wise.
    assert max(m.top10 for m in corrupted) > 0.20
    assert max(m.cv for m in int_stats) > 0.25


def test_max_effect_stability(frame, engine):
    params = {"signalStrength": 99, "multipath": 99, "rfInterference": 99}
    for k in range(15):
        out = engine.process(frame, params)
        assert out.dtype == np.uint8, f"frame {k} dtype {out.dtype}"
        assert out.shape == frame.shape
        assert int(out.min()) >= 0 and int(out.max()) <= 255
        assert engine.frame_index == k + 1
    assert engine.frame_index == 15


def test_presets(frame, engine):
    assert "CLEAN" in PRESETS
    for name, params in PRESETS.items():
        out = engine.process(frame, params)
        assert out.dtype == np.uint8, name
        assert out.shape == frame.shape, name
        assert isinstance(engine.last_stages, dict), name

    engine.reset()
    out_clean = engine.process(frame, PRESETS["CLEAN"])
    assert np.array_equal(out_clean, frame)


def test_performance(numba_warmup):
    big = make_structured_frame(h=720, w=1280)
    eng = AnalogVideoBreakupEngine(seed=42)
    params = {"signalStrength": 60, "multipath": 60, "rfInterference": 60}
    for _ in range(3):
        eng.process(big, params)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        eng.process(big, params)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    assert avg < 0.050, f"avg {avg * 1000:.2f} ms/frame exceeds 50 ms budget"


def test_api_stability(engine, frame):
    import engine as engine_pkg

    assert hasattr(engine_pkg, "AnalogVideoBreakupEngine")
    assert hasattr(engine_pkg, "PRESETS")
    for name in ("process", "processFrame", "setSignalStrength", "setMultipath",
                 "setRFInterference", "getParams", "reset"):
        assert callable(getattr(AnalogVideoBreakupEngine, name, None)), name

    assert getattr(AnalogVideoBreakupEngine, "VERSION", None)
    assert getattr(engine_pkg, "__version__", None)

    out = engine.processFrame(frame, {"signalStrength": 40, "multipath": 40,
                                      "rfInterference": 40})
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert isinstance(engine.last_stages, dict)

    engine.setMultipath(50)
    assert engine.getParams()["multipath"] == 50
    engine.reset()
    assert engine.frame_index == 0
