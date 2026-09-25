# LVLS — Lightweight Video Link Simulation

**LVLS** (Lightweight Video Link Simulation) is a standalone video-link
degradation model for FPV (first-person view) applications. It takes clean RGB
frames — typically a simulator's rendered output — and produces plausibly
authentic **5.8 GHz analog FPV** breakup, driven by three independent severity
parameters:

| Parameter | Range | 0 | 99 |
|---|---|---|---|
| `signalStrength` | 0..99 | clean link | near-total video loss (static) |
| `multipath` | 0..99 | no reflections | heavy ghosting, dropouts, roll |
| `rfInterference` | 0..99 | no interferers | sustained noisy bars / slices |

Each parameter is independent; setting one to 99 while the others are 0 must
produce a phenomenon that is visually distinct from the other two. That
requirement is the core acceptance criterion of the project and is verified by
the QA harness in `tests/`.

**Parameter semantics note:** the public getter/setter keeps the name
`signalStrength` for spec compatibility, but the value acts as **weak-signal
severity** — 0 = no effect, 99 = extreme (a worse link).

## Features

- Three independent 0..99 parameters with physically-researched stage curves
  (grain → snow → color kill → tear/roll → static; echo → smear → displacement
  → nulls; bars → fringe → slices → false sync)
- Deterministic: all randomness derives from `(seed, frame_index)` counter
  hashes — same inputs, same outputs, on any machine
- Bit-identical passthrough when all parameters are 0
- Fast path for repeated calls (`advance=False`), plus temporal state machines
  for roll events, color-killer hysteresis, multipath null/flash and
  interference bursts
- Pure CPU (Numba-compiled kernels); no GPU required —
  `pip install numpy numba`
- Optional GLSL fragment-shader port for real-time / in-simulator use
- PySide6 GUI demo, sample media, pytest suite, visual QA harness

## Install

```bash
pip install numpy numba            # engine runtime
pip install opencv-python pytest   # demos / tests (optional)
pip install PySide6                # GUI demo (optional)
```

Run from the project root so `engine/` is importable (or add the root to
`sys.path`). There is no packaging metadata by design — the tree is
self-contained.

## Quick start

```python
import numpy as np
from engine import AnalogVideoBreakupEngine, PRESETS

eng = AnalogVideoBreakupEngine(seed=42)

clean = np.zeros((720, 1280, 3), dtype=np.uint8)  # your RGB frame (H,W,3)

out = eng.process(clean, {
    "signalStrength": 40,   # 0..99
    "multipath": 70,        # 0..99
    "rfInterference": 10,   # 0..99
})
# out: new uint8 (720,1280,3) array — the input is never modified
```

With OpenCV (note: OpenCV reads BGR — convert first):

```python
import cv2
from engine import AnalogVideoBreakupEngine, PRESETS

frame_bgr = cv2.imread("demo/sample_media/sample_frame.png")
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

eng = AnalogVideoBreakupEngine(seed=42)
out_rgb = eng.process(frame_rgb, PRESETS["WIFI INTERFERENCE"])
cv2.imwrite("out.png", cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
```

Full API: [`docs/api/API.md`](docs/api/API.md).

## Presets

Defined in `engine/core/params.py` and exported as `PRESETS`:

| Preset | signal | multipath | interference |
|---|---|---|---|
| `CLEAN` | 0 | 0 | 0 |
| `LONG RANGE` | 45 | 8 | 0 |
| `WEAK SIGNAL` | 72 | 10 | 5 |
| `HEAVY MULTIPATH` | 12 | 85 | 3 |
| `WIFI INTERFERENCE` | 8 | 15 | 72 |
| `SEVERE INTERFERENCE` | 30 | 22 | 95 |
| `NEAR VIDEO LOSS` | 88 | 55 | 70 |

## Architecture

```
clean RGB
  -> planar YUV (full-range BT.601, float32)
  -> multipath     echoes / comb / smear / block displacement / ripple bands /
                   short mono nulls / white AGC flash        (src -> dst)
  -> weak signal   snow / chroma blotch / sat collapse / color killer /
                   hue wander / sparklies / jitter + tear    (in place)
  -> RF interference  burst noise bars / slices / fringe / impulses /
                   desense / false-sync shifts               (in place)
  -> geometry      per-row horizontal shift + vertical roll + seam noise
  -> RGB uint8
```

- **Multipath runs first on the clean source** so echo taps sample pristine
  pixels; it accumulates row-block displacement into `row_shift`.
- **Weak signal** owns the snow envelope, color-killer hysteresis and the
  vertical-roll event machine; jitter/tear accumulate into `row_shift`.
- **Interference** owns the packet-burst machine and adds false-sync shifts.
- **Geometry is applied once at the end** from a double buffer.

### Parameter curves

Each 0..99 input is normalized to `r ∈ [0,1]` and pushed through per-stage
`smoothstep`/power curves in `engine/core/params.py`. Curves are deliberately
non-linear and ordered to match field research (see `docs/research/`): color
dies before sync lock, ghosting precedes nulls, interference duty stretches so
75 → 99 is visibly heavier.

### Determinism

Every random value is `hash(seed, frame_index, coordinate[, effect_id])` —
32-bit integer hashes with explicit uint64 masking (Numba promotes uint32
multiplies to int64; the mask is load-bearing). There is no hidden PRNG state;
`process(frame, params, advance=False)` is idempotent.

## Using the demo

The GUI demo plays a sample clip and lets you scrub the three severity
parameters live, with a side-by-side original/processed view.

```bash
# 1. install the demo dependency
pip install PySide6 opencv-python

# 2. launch the GUI (from the project root)
python demo/gui/main.py
```

**Controls**

| Control | Action |
|---|---|
| **Play / Pause** | playback runs at the clip's native FPS |
| **Restart** | seek back to frame 0 |
| **Step Frame** | advance one frame while paused |
| **Signal / Multipath / Interference sliders** | live 0..99 severity; the processed pane re-renders after a short debounce |
| **Preset dropdown** | load a preset combination (CLEAN, LONG RANGE, …) |
| **Reset** | set all sliders back to CLEAN |

**Demo video:** the GUI looks for `demo/gui/demo_video1.mp4` first. If it is
absent (it is not shipped in the repository — see
[Repository contents](#repository-contents)), it automatically falls back to
the bundled synthetic clip `demo/sample_media/sample.mp4`. You can also drop
in your own clip:

```bash
# regenerate the bundled synthetic sample clip (buildings/trees/poles/road)
python demo/sample_media/generate_sample.py

# or use any of your own footage
cp path/to/your_clip.mp4 demo/gui/demo_video1.mp4
```

The status bar shows the measured playback FPS (locked to the clip's native
rate), processing time per frame, the current frame number, and the engine's
deterministic `frame_index`.

### Other demos and tools

```bash
# Render the visual test matrix + QA clips
python tests/render_matrix.py

# Test suite (11 tests)
python -m pytest tests/

# Runnable simulator-loop integration example
python examples/integration_example.py
```

## GLSL port

`engine/glsl/fpv_breakup.frag` is a single-pass, stateless GLSL 330 fragment
shader with the same three uniforms (`uSignalStrength`, `uMultipath`,
`uRFInterference`), driven by `uFrame`/`uSeed`. Severity curves are copied
verbatim from `params.py`. Temporal machines are approximated with block
hashes so the output is a pure function of uniforms — no feedback FBO.
See [`engine/glsl/README.md`](engine/glsl/README.md) for the interface,
Shadertoy notes, verification status and deliberate differences.

## Documentation

| Doc | Contents |
|---|---|
| [`docs/api/API.md`](docs/api/API.md) | full API reference |
| [`docs/integration/INTEGRATION.md`](docs/integration/INTEGRATION.md) | wiring into a simulator loop |
| [`docs/research/`](docs/research/README.md) | per-phenomenon field research (weak signal, multipath, interference, DVR vs RF) |
| [`engine/glsl/README.md`](engine/glsl/README.md) | shader port interface & notes |
| [`examples/integration_example.py`](examples/integration_example.py) | runnable simulator-loop example |

## Repository contents

| Path | Shipped in repo? | Notes |
|---|---|---|
| `engine/` | yes | simulation core (Python/Numba + GLSL) |
| `demo/gui/main.py` | yes | PySide6 GUI demo |
| `demo/sample_media/` | yes | synthetic sample clip + generator |
| `demo/gui/demo_video1.mp4` | **no** | large footage file — add your own clip (see demo instructions) |
| `tests/`, `docs/`, `examples/` | yes | test suite, QA harness, docs |

## Testing & QA

- `tests/test_engine.py` — 11 pytest tests: identity fast path, determinism,
  uint8 contract, parameter independence (each pillar must visibly change the
  frame), reset/seed behavior, presets.
- `tests/render_matrix.py` — renders labeled stills (`tests/qa/matrix/`) and
  75-frame clips (`tests/qa/videos/`) across the severity grid.
- Visual QA protocol: extract frames, check three-phenomenon distinctness,
  ghost correlability (LS-fit echo coefficient), null-window budgets,
  interference burstiness/tint, and pilot plausibility. Two full review rounds
  were run; all first-round defects were verified fixed in round two.

## Known limitations

- CPU-bound: Numba kernels are fast (a 720p frame processes in a few ms after
  JIT warm-up) but this is not a realtime GPU path — use the GLSL port for that.
- The GLSL port has been statically checked but **not** GPU-compiled on this
  machine (no GL context available); compile it on your target before shipping.
- `signalStrength` is named per spec but behaves as weak-signal severity (see
  the semantics note at the top).

## Future additions

- **Performance upgrades** — further kernel optimizations and a realtime
  GPU path so the simulation can run at full frame rate inside a live
  simulator render loop.
- **Digital video link simulations** — models for digital FPV systems such as
  **HDZero**, **DJI OcuSync (O4, O3, and related generations)**, and other
  digital transmission links, each with its own characteristic failure modes
  (compression artifacts, block corruption, breakup-to-black behavior) alongside
  the existing analog model.
