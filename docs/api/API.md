# API Reference — LVLS (Lightweight Video Link Simulation)

`AnalogVideoBreakupEngine` converts clean RGB frames into analog-FPV-style
degraded video, driven by three severity parameters.

- Version: `1.0.0` (`AnalogVideoBreakupEngine.VERSION`, `engine.__version__`)
- Pure CPU (Numba-compiled kernels); no GPU required
- Deterministic: all randomness derives from `(seed, frame_index, coordinates)`

## 1. Requirements and install

```
Python 3.11+
pip install numpy numba          # engine runtime
pip install opencv-python        # only for demos / QA scripts
```

No packaging metadata ships with the tree — run from the project root so
`engine/` is importable (or add the project root to `sys.path`).

## 2. Quick start

```python
import numpy as np
from engine import AnalogVideoBreakupEngine, PRESETS

eng = AnalogVideoBreakupEngine(seed=42)

clean = np.zeros((720, 1280, 3), dtype=np.uint8)   # your RGB frame (H,W,3)

out = eng.process(clean, {
    "signalStrength": 40,   # 0..99
    "multipath": 70,        # 0..99
    "rfInterference": 10,   # 0..99
})
# out: new uint8 (720,1280,3) array — clean frame is never modified
```

With OpenCV (note: OpenCV reads BGR — convert first):

```python
import cv2
frame_bgr = cv2.imread("demo/sample_media/sample_frame.png")
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
out_rgb = eng.process(frame_rgb, PRESETS["WIFI INTERFERENCE"])
cv2.imwrite("out.png", cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
```

## 3. Constructor

```python
AnalogVideoBreakupEngine(seed=12345, width=None, height=None)
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `seed` | int | `12345` | Master seed; masked to 32 bits (`& 0xFFFFFFFF`). Same seed + same call sequence => identical output |
| `width`, `height` | int or None | `None` | Optional preallocation of internal buffers for a known frame size; buffers also allocate lazily from the first frame |

Initial state: `frame_index = 0`, temporal state cleared, params all `0`,
`last_stages = None`.

## 4. Frame contract — `process()`

```python
out = eng.process(frame, params=None, advance=True)
out = eng.processFrame(frame, params=None)   # alias; advance defaults True
```

| | |
|---|---|
| `frame` | uint8 RGB array, shape `(H, W, 3)` |
| `params` | optional dict `{signalStrength, multipath, rfInterference}`, each 0..99; see Section 5 for omission semantics |
| `advance` | if `True` (default) `frame_index` increments after processing; pass `False` for idempotent/repeated processing (tests, reference rendering) |
| returns | a **new** uint8 `(H, W, 3)` array; the input array is never modified |

Pipeline applied per frame (compositing order):

```
clean RGB -> planar YUV (full-range BT.601, float32)
          -> multipath    (echoes / comb / smear / blocks / nulls)
          -> weak signal  (snow / color kill / sparklies; + jitter, tear)
          -> interference (burst bands / slices / fringe; + false-sync shifts)
          -> geometry     (row_shift + vertical roll + roll-seam noise)
          -> RGB uint8 (clipped)
```

### Zero-effect identity guarantee

When all three raw parameters are exactly `0`, `process` takes a fast path
and returns `arr.copy()` — **bit-identical to the input** (content; the
object is still a fresh copy). `frame_index` still advances if
`advance=True`.

Any nonzero parameter runs the full YUV pipeline. Very small nonzero values
(e.g. `signalStrength=1`, below the first stage onset of raw 0.015) produce
no visible effect, but still round-trip through YUV — bit-exact identity is
guaranteed only for the `(0, 0, 0)` fast path.

## 5. Parameter semantics

- Keys: `signalStrength`, `multipath`, `rfInterference`
  (`engine.core.params.PARAM_KEYS`).
- Domain: integers `0..99` (`PARAM_MIN=0`, `PARAM_MAX=99`).
  - `0` = no effect (clean link / no multipath / no interference)
  - `99` = extreme effect
- **Clamping** (`clamp_param`): values are `round`-ed to int, then clamped
  to `[0, 99]`. Non-numeric / `None` inputs coerce to `0` (no exception).
- Internally each param becomes raw severity `r = clamp(v)/99 ∈ [0,1]`,
  then passes through that module's nonlinear stage curves (Section 9).
- **Omission semantics** (`normalize_params`): a params dict is merged over
  the **defaults (all 0)**, not over previous values. Passing a partial
  dict sets the omitted keys to `0` for that call and stores that result in
  `last_params`. Use the individual setters to change one axis without
  touching the others.
- Unknown keys in the dict are ignored.

## 6. Parameter and control methods

| Method | Behavior |
|---|---|
| `setSignalStrength(v)` | Set only `signalStrength` (clamped). Other keys unchanged |
| `setMultipath(v)` | Set only `multipath` (clamped) |
| `setRFInterference(v)` | Set only `rfInterference` (clamped) |
| `setParams(params)` | Replace all params via `normalize_params` (omitted keys -> 0); returns the normalized dict |
| `getParams()` | Return a **copy** of the current params dict |
| `setSeed(seed)` | Change seed (32-bit mask). Does **not** reset `frame_index` or temporal state |
| `reset()` | Clear temporal state (roll events, killer latch, burst machine, ghost flicker, row shifts), set `frame_index = 0`. Params are kept |
| `eng.frame_index` | Public int: current frame counter (advances per `process` when `advance=True`) |
| `eng.last_params` | Most recently resolved normalized params dict |
| `eng.last_stages` | Most recently resolved stage-intensity dicts (`{"signal": {...}, "multipath": {...}, "interference": {...}, "raw": {...}}`) — useful for debugging/tuning |

`reset()` vs `frame_index`: `reset()` is the full rewind (state +
counter); setting `frame_index` manually is not supported — control replay
identity via `reset()` + identical call sequence, or via `advance=False`
for single-frame idempotence.

## 7. Package exports

```python
from engine import (
    AnalogVideoBreakupEngine,   # main class
    PRESETS,                    # named parameter sets
    PARAM_MIN, PARAM_MAX,       # 0, 99
    clamp_param, normalize_params,
    __version__,                # "1.0.0"
)
```

## 8. PRESETS

```python
from engine import PRESETS
out = eng.process(frame, PRESETS["HEAVY MULTIPATH"])
```

| Preset | signalStrength | multipath | rfInterference | Scenario |
|---|---:|---:|---:|---|
| `CLEAN` | 0 | 0 | 0 | No effect (identity fast path) |
| `LONG RANGE` | 45 | 8 | 0 | Mid snow, mild echo |
| `WEAK SIGNAL` | 72 | 10 | 5 | Heavy snow, color dead, sync stress |
| `HEAVY MULTIPATH` | 12 | 85 | 3 | Structural ghosts/bands, low snow |
| `WIFI INTERFERENCE` | 8 | 15 | 72 | Bursty bands/fringe dominates |
| `SEVERE INTERFERENCE` | 30 | 22 | 95 | Slices, false sync, takeovers |
| `NEAR VIDEO LOSS` | 88 | 55 | 70 | All axes near collapse |

## 9. Stage mapping tables (source of truth: `engine/core/params.py`)

Stage curves convert raw severity `r ∈ [0,1]` into per-effect intensities.
`smoothstep(e0,e1,x)` is the standard cubic Hermite clamped to `[0,1]`;
`ease_in(x,p) = clamp(x,0,1)^p`.

### 9.1 `signal_stages(r)` — weak signal

| Key | Formula | Onset r (param value) | Full r (param value) |
|---|---|---|---|
| `luma_noise` | `42 * ease_in(smoothstep(0.015,1.0,r), 1.22) / 255` | 0.015 (~1.5) | 1.0 (99) |
| `chroma_noise` | `smoothstep(0.04,0.40,r) * 0.11` (dark-weighted in module) | 0.04 (~4) | 0.40 (~40) |
| `black_lift` | `smoothstep(0.05,0.7,r) * 0.04` | 0.05 (~5) | 0.70 (~69) |
| `hue_wander` | `smoothstep(0.12,0.45,r) * 0.55` | 0.12 (~12) | 0.45 (~45) |
| `sat_loss` | `smoothstep(0.18,0.50,r)` | 0.18 (~18) | 0.50 (~50) |
| `killer_on_th` | constant `0.50` (raw space) | — | — |
| `killer_off_th` | constant `0.42` (raw space) | — | — |
| `sparklie` | `ease_in(smoothstep(0.42,0.88,r), 2.0) * 0.5` | 0.42 (~42) | 0.88 (~87) |
| `h_jitter` | `smoothstep(0.52,0.80,r) * 7.0` px | 0.52 (~51) | 0.80 (~79) |
| `tear` | `smoothstep(0.62,0.92,r)` | 0.62 (~61) | 0.92 (~91) |
| `roll` | `smoothstep(0.72,0.97,r)` | 0.72 (~71) | 0.97 (~96) |
| `static_kill` | `smoothstep(0.88,1.0,r)` | 0.88 (~87) | 1.0 (99) |

Notes: the color-killer uses hysteresis on raw severity (latch on at
`killer_on_th`, release at `killer_off_th`); the renderer converts those raw
thresholds into `sat_loss`-space internally. Roll event start probability uses
`roll * 0.12` per eligible frame. `black_lift` is applied after snow: the
receiver's black level rises as `yy += black_lift * (1 - yy)`.

### 9.2 `multipath_stages(r)` — multipath

| Key | Formula | Onset r (param value) | Full r (param value) |
|---|---|---|---|
| `comb_k` | `smoothstep(0.04,0.55,r) * 0.32` | 0.04 (~4) | 0.55 (~54) |
| `smear_px` | `smoothstep(0.18,0.92,r) * 3.5` px | 0.18 (~18) | 0.92 (~91) |
| `ghost1_d` | `4 + smoothstep(0.12,1.0,r) * 30` px | 0.12 (~12) | 1.0 (99) |
| `ghost1_a` | `smoothstep(0.12,0.75,r) * 0.85` | 0.12 (~12) | 0.75 (~74) |
| `ghost2_d` | `16 + smoothstep(0.40,1.0,r) * 75` px | 0.40 (~40) | 1.0 (99) |
| `ghost2_a` | `smoothstep(0.38,0.85,r) * 0.50` | 0.38 (~38) | 0.85 (~84) |
| `ghost3_a` | `smoothstep(0.30,0.90,r) * 0.40` | 0.30 (~30) | 0.90 (~89) |
| `block_shift` | `smoothstep(0.45,0.95,r) * 42` px | 0.45 (~45) | 0.95 (~94) |
| `band_ripple` | `smoothstep(0.28,0.85,r) * 0.14` | 0.28 (~28) | 0.85 (~84) |
| `edge_spike` | `smoothstep(0.35,0.9,r) * 0.55` | 0.35 (~35) | 0.90 (~89) |
| `null_rate` | `ease_in(smoothstep(0.55,1.0,r), 1.5) * 0.05` | 0.55 (~54) | 1.0 (99) |
| `flash_rate` | `smoothstep(0.42,0.98,r) * 0.05` | 0.42 (~42) | 0.98 (~97) |
| `null_snow` | `smoothstep(0.5,1.0,r) * 0.50` | 0.50 (~50) | 1.0 (99) |
| `desat_bands` | `smoothstep(0.55,0.95,r) * 0.8` | 0.55 (~54) | 0.95 (~94) |

Ghost-3 delay is derived by the renderer as `ghost1_d * 0.45` (short-delay
tap). Ghost amplitudes are additionally modulated per frame with a signed
flicker envelope (`env ∈ [0.65,1]`, `fl ∈ [0.56,1]`, occasional polarity
flips); delays drift slowly. Null windows last 1–3 frames with a 20–80 frame
cooldown. Echo mix is blended with the box smear using a weight capped at
0.65 so ghost taps stay correlatable.

### 9.3 `interference_stages(r)` — RF interference

| Key | Formula | Onset r (param value) | Full r (param value) |
|---|---|---|---|
| `duty` | `smoothstep(0.06,0.97,r) * 0.60` | 0.06 (~6) | 0.97 (~96) |
| `band_amp` | `smoothstep(0.10,0.95,r)` | 0.10 (~10) | 0.95 (~94) |
| `band_thr` | `0.72 - 0.14 * r` (lower = denser bars) | — | — |
| `scroll` | `0.4 + smoothstep(0.1,1.0,r) * 5.0` lines/frame | 0.10 (~10) | 1.0 (99) |
| `band_count` | `3.0 + r * 12.0` (linear) | 0.00 (0) | 1.0 (99) |
| `tint` | `smoothstep(0.25,0.95,r)` (×0.30 in module) | 0.25 (~25) | 0.95 (~94) |
| `fringe` | `smoothstep(0.28,0.85,r) * 0.15` | 0.28 (~28) | 0.85 (~84) |
| `slice` | `smoothstep(0.48,0.99,r) * 0.55` | 0.48 (~48) | 0.99 (~98) |
| `impulse` | `smoothstep(0.35,0.95,r) * 0.45` | 0.35 (~35) | 0.95 (~94) |
| `desense` | `smoothstep(0.3,0.95,r) * 0.35` | 0.30 (~30) | 0.95 (~94) |
| `long_event` | `smoothstep(0.65,1.0,r) * 0.03` | 0.65 (~64) | 1.0 (99) |
| `sync_false` | `smoothstep(0.7,1.0,r) * 18.0` px | 0.70 (~69) | 1.0 (99) |

Effective burst duty inside the machine is further modulated by a traffic
cluster envelope, capped at `0.75`; burst chain probability uses
`eff_duty * 0.35`. Bar tint polarity is a per-band-run coin (green vs
purple) and replaces the scene's chroma inside the bar.

## 10. Determinism contract

- All randomness comes from counter-based integer hashes
  (`engine/core/rng.py`) over `(seed, frame_index, spatial coordinates)`.
  There is no hidden PRNG state beyond the engine's explicit temporal state
  arrays.
- **Guarantee:** for a given engine instance, the same `seed`, the same
  params sequence, the same `frame_index`, and the same temporal state
  produce identical output bytes. Replaying `reset()` + the same call
  sequence reproduces the video bit-for-bit.
- The engine is **pure CPU** (Numba/LLVM); there is no GPU code path, so
  GPU-vs-CPU variance is not a factor. Output can vary across Numba/NumPy
  versions or platforms if floating-point codegen changes — determinism is
  promised per environment, not across environments.

## 11. Performance

| Topic | Behavior |
|---|---|
| First call (cold) | Numba JIT-compiles each kernel on first use: **~6-11 s per effect family, one-time** |
| Disk cache | All kernels use `cache=True` — compiles persist in `engine/**/__pycache__/*.nbc` across process launches; subsequent launches pay **~0 s** |
| Warmup | `eng.warmup()` precompiles every kernel on a tiny throwaway frame (~19 s cold, ~instant warm). The GUI demo calls an equivalent staged warmup at startup so slider drags never stall |
| Steady state | **~3-8 ms/frame at 720p** on a modern desktop CPU (~12 ms incl. RGB/YUV conversion + copy) |
| Parallelism | Kernel loops parallelize over image rows (`prange`); threads are Numba's internal pool |
| Buffers | Reused across frames (`_buf`); resolution change reallocates once |
| Advice | Call `warmup()` once after construction if you need to exclude compile time from interactive paths; pre-size via `width`/`height` to avoid first-frame allocation |

## 12. Thread safety

- **Not thread-safe to share**: one `AnalogVideoBreakupEngine` instance per
  thread. Internal buffers and temporal state are mutated in place during
  `process()`.
- Distinct instances on distinct threads are fine (each carries its own
  state); the shared Numba thread pool is managed by Numba.
- Do not call `process()` on the same instance concurrently from multiple
  threads.

## 13. Error handling

| Condition | Behavior |
|---|---|
| `frame is None` | raises `ValueError("frame is None")` |
| shape not `(H, W, 3)` | raises `ValueError(f"expected (H,W,3) RGB frame, got shape ...")` |
| non-numeric / out-of-range params | never raises; coerced to `0` / clamped to `0..99` |
| wrong dtype (e.g. float frame) | raises `ValueError(f"expected uint8 frame, got dtype ...")` — pass uint8 as contracted |
| resolution change mid-stream | supported; buffers reallocate automatically |

## 14. Related entry points

- GUI demo: `python demo/gui/main.py` (uses
  `demo/gui/demo_video1.mp4`, falling back to the synthetic
  `demo/sample_media/sample.mp4`; regenerate the latter with
  `python demo/sample_media/generate_sample.py`)
- QA render matrix: `python tests/render_matrix.py` (writes
  `tests/qa/matrix/*.png` and short clips under `tests/qa/videos/`)
