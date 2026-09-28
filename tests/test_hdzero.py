"""Test suite for LVLS — HDZero digital link engine.

Run:  python -m pytest <project_root>/tests -v --tb=short

Calibration notes (thresholds verified against the real engine):
  - Digital distinctness: signal corruption is scattered 8x8 blocks ->
    coarse-cell energy CV stays low (~0.4-0.6); multipath punch-outs are
    blob-clustered (~88 px noise cells) -> CV is clearly higher (>0.9).
  - Interference is an event machine: frames are either clean or taken
    over, so burstiness (max - min of per-frame mean abs diff) is large
    while idle frames stay near zero.
  - Speckle check: isolated pure-white pixels inside otherwise all-source
    gray 8x8 blocks can only come from the speckle kernel (fill_blocks
    always paints whole blocks).
  - Clean-block survival: a digital link never touches valid pixels, so
    even at signal=70 roughly half the blocks stay bit-identical on a
    flat gray source.
"""
import time
from typing import NamedTuple

import numpy as np
import pytest

from engine import (
    PARAM_MAX,
    PARAM_MIN,
    PRESETS,
    HDZeroEngine,
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


def gray_frame(h=H, w=W, v=128):
    return np.full((h, w, 3), v, dtype=np.uint8)


class CellMetrics(NamedTuple):
    cv: float
    top20: float


def cell_energy_metrics(out, clean, cell=20):
    """Spatial clustering of corruption energy on a coarse cell grid."""
    d = np.abs(out.astype(np.int32) - clean.astype(np.int32)).sum(axis=2)
    h, w = d.shape
    eh, ew = h // cell, w // cell
    d = d[:eh * cell, :ew * cell]
    cells = d.reshape(eh, cell, ew, cell).mean(axis=(1, 3))
    total = cells.sum()
    if total <= 1e-9:
        return CellMetrics(0.0, 0.0)
    k = max(1, int(np.ceil(0.20 * cells.size)))
    top20 = float(np.sort(cells.ravel())[::-1][:k].sum() / total)
    return CellMetrics(float(cells.std() / cells.mean()), top20)


def speckle_hits(out, source):
    """Count pure-white pixels in otherwise-untouched 8x8 blocks.

    fill_blocks only ever paints complete blocks, so a partial white block
    whose non-white pixels still equal the source is a speckle hit.
    """
    white = np.all(out == 255, axis=2)
    h, w = white.shape
    bh, bw = h // 8, w // 8
    count = 0
    for by in range(bh):
        for bx in range(bw):
            wblk = white[by * 8:by * 8 + 8, bx * 8:bx * 8 + 8]
            n = int(wblk.sum())
            if n == 0 or n == 64:
                continue
            obl = out[by * 8:by * 8 + 8, bx * 8:bx * 8 + 8]
            src = source[by * 8:by * 8 + 8, bx * 8:bx * 8 + 8]
            nonwhite = obl[~wblk]
            srcnw = src[~wblk]
            if np.all(nonwhite == srcnw) and np.all(srcnw == 128):
                count += n
    return count


def clean_block_fraction(out, source):
    """Fraction of 8x8 blocks left bit-identical to the source."""
    same = np.all(out == source, axis=2)
    h, w = same.shape
    bh, bw = h // 8, w // 8
    blocks = same[:bh * 8, :bw * 8].reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)
    return float(np.mean(np.all(blocks, axis=(2, 3))))


@pytest.fixture
def frame():
    return make_structured_frame()


@pytest.fixture
def engine():
    return HDZeroEngine(seed=SEED)


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

    eng = HDZeroEngine(seed=1)
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

    # fast path also fills a caller-provided buffer bit-identically
    buf = np.empty_like(frame)
    out_buf = engine.process(frame, dict(ZERO_PARAMS), out=buf)
    assert out_buf is buf
    assert np.array_equal(buf, frame)


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

    e1 = HDZeroEngine(seed=777)
    e2 = HDZeroEngine(seed=777)
    o1 = [e1.process(f, p) for f, p in zip(frames, pseq)]
    o2 = [e2.process(f, p) for f, p in zip(frames, pseq)]
    assert len(o1) == 8
    for k, (a, b) in enumerate(zip(o1, o2)):
        assert np.array_equal(a, b), f"frame {k} differs between same-seed engines"

    e3 = HDZeroEngine(seed=778)
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
    """Signal scatters, multipath clusters, interference bursts.

    A flat gray source isolates the corruption field: chroma quant is a
    no-op on neutral pixels, so cell metrics reflect block errors only.
    """
    gray = gray_frame()

    eng = HDZeroEngine(seed=SEED)
    sig_cells = []
    sig_clean = []
    for _ in range(8):
        out = eng.process(gray, SIG_ONLY)
        sig_cells.append(cell_energy_metrics(out, gray))
        sig_clean.append(clean_block_fraction(out, gray))
    sig_cv = float(np.mean([m.cv for m in sig_cells]))
    assert float(np.mean(sig_clean)) > 0.25, \
        f"clean-block survival {np.mean(sig_clean):.3f} <= 0.25 (digital must not touch valid pixels)"

    eng = HDZeroEngine(seed=SEED)
    mp_cells = []
    for _ in range(8):
        out = eng.process(gray, MP_ONLY)
        mp_cells.append(cell_energy_metrics(out, gray))
    mp_cv = float(np.mean([m.cv for m in mp_cells]))
    assert sig_cv < 0.75, f"signal cell CV {sig_cv:.3f} >= 0.75 (should be scattered)"
    assert mp_cv > 0.95, f"multipath cell CV {mp_cv:.3f} <= 0.95 (should be clustered)"
    assert mp_cv > sig_cv + 0.2, f"mp CV {mp_cv:.3f} not clearly above sig CV {sig_cv:.3f}"

    eng = HDZeroEngine(seed=SEED)
    int_ma = []
    for _ in range(16):
        out = eng.process(gray, INT_ONLY)
        int_ma.append(float(np.abs(
            out.astype(np.int32) - gray.astype(np.int32)).mean()))
    burst = max(int_ma) - min(int_ma)
    assert max(int_ma) > 8.0, f"max interference meanabs {max(int_ma):.2f} <= 8"
    assert min(int_ma) < 2.0, f"min interference meanabs {min(int_ma):.2f} >= 2 (idle frames must stay clean)"
    assert burst > 6.0, f"burstiness {burst:.2f} <= 6"


def test_speckle_and_blackout():
    gray = gray_frame()
    # speckle on a visible picture: signal=70 (loss runs start at 0.74 raw,
    # so this severity can never latch; signal=99 is the dead link and
    # shows the no-signal screen only)
    eng = HDZeroEngine(seed=SEED)
    hits = []
    for _ in range(4):
        out = eng.process(gray, {"signalStrength": 70, "multipath": 0,
                                 "rfInterference": 0})
        hits.append(speckle_hits(out, gray))
    assert sum(hits) > 10, f"speckle hits {hits} total <= 10"
    assert all(h >= 0 for h in hits)

    # blackouts live just below the dead link (signal 99 shows the no-signal
    # screen only, black included) -> test at 98
    eng = HDZeroEngine(seed=SEED)
    black = 0
    for _ in range(30):
        out = eng.process(gray, {"signalStrength": 98, "multipath": 0,
                                 "rfInterference": 0})
        if int(out.max()) == 0:
            black += 1
    assert black >= 1, "no breakup-to-black frame within 30 frames at signal=98"

    # mid signal must never black out (black_frame starts at raw 0.80)
    eng = HDZeroEngine(seed=SEED)
    for _ in range(20):
        out = eng.process(gray, {"signalStrength": 70, "multipath": 0,
                                 "rfInterference": 0})
        assert int(out.max()) > 0, "unexpected blackout at signal=70"


def test_signal_loss_screen():
    """signal 99 = the dead link: the full no-signal screen every frame.
    97 ramps patch runs in; mid severity never latches."""
    gray = gray_frame()

    # signal 99: full screen on every frame, no partials, no blackouts
    eng = HDZeroEngine(seed=SEED)
    loss = full = partial = 0
    for _ in range(60):
        out = eng.process(gray, {"signalStrength": 99, "multipath": 0,
                                 "rfInterference": 0})
        if eng.in_loss:
            loss += 1
            if np.array_equal(out, eng._loss_full):
                full += 1            # exact full screen
            elif int(out.max()) != 0:
                partial += 1         # ramping: picture partly visible
    assert eng._loss_full is not None, "no-signal asset failed to load"
    assert loss >= 55, f"loss screen only {loss}/60 frames at signal=99"
    assert full >= 55, f"only {full} full no-signal frames at signal=99"
    assert partial == 0, \
        f"{partial} partial frames at signal=99 (dead link = full screen only)"

    # signal 97: loss runs ramp — patch frames first, then the full screen
    eng = HDZeroEngine(seed=SEED)
    loss = full = partial = 0
    for _ in range(60):
        out = eng.process(gray, {"signalStrength": 97, "multipath": 0,
                                 "rfInterference": 0})
        if eng.in_loss:
            loss += 1
            if np.array_equal(out, eng._loss_full):
                full += 1
            elif int(out.max()) != 0:
                partial += 1
    assert loss >= 30, f"loss screen only {loss}/60 frames at signal=97"
    assert full >= 3, f"only {full} full no-signal frames at signal=97"
    assert partial >= 3, f"only {partial} partial patch frames at signal=97"

    # loss onset is raw 0.74 -> signal=70 must never latch
    eng = HDZeroEngine(seed=SEED)
    for _ in range(40):
        eng.process(gray, {"signalStrength": 70, "multipath": 0,
                           "rfInterference": 0})
        assert not eng.in_loss, "loss screen at signal=70"

    # zero params leave the machine clear
    eng = HDZeroEngine(seed=SEED)
    eng.process(gray, dict(ZERO_PARAMS))
    assert not eng.in_loss


def test_loss_patch_ramp():
    """Patch coverage follows cov: stable per-run layout, accumulating."""
    from engine.hdzero.kernels import fill_loss_patches

    w, h = 64, 64
    loss = make_structured_frame(h, w)  # all pixels nonzero -> painted == cell
    imgs = []
    for cov in (0.3, 0.7):
        img = np.zeros((h, w, 3), np.uint8)
        fill_loss_patches(img, loss, cov, 7.0, 99, 0)
        imgs.append(img)
    m1 = imgs[0].sum(axis=2) > 0
    m2 = imgs[1].sum(axis=2) > 0
    c1, c2 = float(m1.mean()), float(m2.mean())
    assert 0.15 < c1 < 0.45, f"cov=0.3 painted {c1:.3f} (want ~0.30)"
    assert 0.55 < c2 < 0.85, f"cov=0.7 painted {c2:.3f} (want ~0.70)"
    assert c1 <= c2, "coverage must grow with cov"
    assert np.all(m2[m1]), "layout must be stable (high cov keeps low cells)"
    # painted cells are real crops of the no-signal image, cell-aligned
    ys, xs = np.nonzero(m1)
    if ys.size:
        by, bx = ys[0] // 8 * 8, xs[0] // 8 * 8
        cell = imgs[0][by:by + 8, bx:bx + 8]
        found = any(np.array_equal(
                cell, loss[sy:sy + 8, sx:sx + 8])
            for sy in range(h - 7) for sx in range(w - 7))
        assert found, "patch cell is not a crop of the no-signal image"


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
    eng = HDZeroEngine(seed=42)
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

    assert hasattr(engine_pkg, "HDZeroEngine")
    assert hasattr(engine_pkg, "PRESETS")
    for name in ("process", "processFrame", "setSignalStrength", "setMultipath",
                 "setRFInterference", "getParams", "reset"):
        assert callable(getattr(HDZeroEngine, name, None)), name

    assert getattr(HDZeroEngine, "VERSION", None)
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
