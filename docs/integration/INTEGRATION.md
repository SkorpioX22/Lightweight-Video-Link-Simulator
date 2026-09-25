# Simulator Integration Guide

How to drive `AnalogVideoBreakupEngine` from a flight simulator, game, or
video tool. The engine only corrupts frames — **your simulator decides how
bad the link is** each frame and feeds three 0..99 parameters.

## 1. Pipeline overview

```
+-----------------+     +--------------------------+     +------------------+
| sim renders     |     | RF condition model       |     | engine           |
| clean camera    | --> | (link budget, reflectors,| --> | AnalogVideo-     |
| frame (RGB      |     |  interferers from world) |     | BreakupEngine    |
| uint8 HxWx3)    |     |   -> 3 params 0..99      |     | .process(...)    |
+-----------------+     +--------------------------+     +--------+---------+
                                                                 |
       +-------------------+     +-------------------+            v
       | DVR / recorder    | <-- | display / goggles | <--- degraded frame
       | (out of scope)    |     | (what pilot sees) |
       +-------------------+     +-------------------+
```

```
sim world model          engine params                visual result
---------------          ------------                -------------
distance / obstacles --> signalStrength (0..99) --> snow, color kill, roll
reflections / indoor --> multipath (0..99)       --> ghosts, smear, flashes
Wi-Fi / other VTXs   --> rfInterference (0..99)  --> burst bands, slices
```

The engine deliberately contains **no RF physics**: it never infers range,
reflectors, or interferers. All world-to-param mapping lives in the sim.

## 2. Deriving the parameters

### 2.1 `signalStrength` — from the link budget

Compute received power:

```
P_rx (dBm) = P_tx + G_tx + G_rx - L_path(distance, freq)
                          - L_blockage(Fresnel/obstacles)
                          - L_polarization(mismatch)
```

Anchor on the receiver's observed sensitivity window (first snow
~-94 dBm, sync loss ~-106 dBm, ~12 dB span) and map to 0..99:

```
signalStrength =
    0                                if P_rx >= -94 dBm      # below first snow
    round((-94 - P_rx) / 12 * 99)    if -106 < P_rx < -94    # clamp 0..99
    99                               if P_rx <= -106 dBm     # sync lost
```

Concrete piecewise example (with a margin band so antennas/orientation
wiggle does not chatter at the edges):

```
margin = P_rx - (-94)          # dB above first-snow threshold

if   margin >=  12:  v = 0     # comfortable: e.g. P_rx >= -82 dBm
elif margin <= -12:  v = 99    # hopeless:    e.g. P_rx <= -106 dBm
else:                v = round((12 - margin) / 24 * 99)
```

Tips: include antenna gain/orientation and polarization loss in
`L_polarization`; Fresnel-zone obstruction belongs in `L_blockage`; slow
altitude/terrain fades produce the seconds-to-minutes envelope the research
calls "range fade".

### 2.2 `multipath` — from reflector proximity

The engine needs a single severity score summarizing early reflected energy:

1. Raycast against world geometry (buildings, ground, metal objects) from
   RX to TX: collect echo paths with delay `tau_k` and reflection power
   `P_echo_k` (use material reflectivity; ground bounce ~ -3..-6 dB per hit
   for concrete/metal at grazing incidence is a reasonable starting point).
2. Score the strongest *early* reflection (delay < ~3 us) relative to the
   direct path, in dB:

```
S_db = 10*log10( max_k P_echo_k / P_direct )      # e.g. -40 (weak) .. 0 (equal)
```

Concrete piecewise example:

```
indoor = 1 if bando/warehouse/indoors else 0       # geometry flag
pol    = 0.0 matched circular polarization
         1.0 linear-crossed / wrong-hand CP / linear vs CP

if   S_db >=  -6:  base = 1.0                     # strong echo
elif S_db <= -30:  base = 0.0                     # negligible echo
else:              base = (-6 - S_db) / 24        # linear across [-30,-6]

score = clamp01(base + 0.25 * indoor + 0.20 * pol)
multipath = round(99 * score)
```

Notes:
- Echo *delays* determine ghost offsets in the physical world (13.5 px/us);
  the engine maps severity to its own delay curves — you only send strength.
- Linear polarization outdoors over ground is worse than CP; CP rejects a
  single bounce by ~20-30 dB (reduce `base` accordingly when CP is in use).
- Multipath is position-locked, not distance-monotonic: fixed bad spots are
  correct behavior — do not smooth them away with distance heuristics.
- Higher TX power does not fix multipath; do not couple `P_tx` into this
  axis the way you do for `signalStrength`.

### 2.3 `rfInterference` — from nearby interferers

Enumerate interferers (Wi-Fi APs on overlapping UNII channels, other
pilots' VTXs on R1-R6, CW sources):

```
overlap  : 0..1   channel spectral overlap (0 = clear, e.g. R8 vs Wi-Fi;
                  1 = co-channel)
I_db     : interferer power at RX input relative to wanted signal
           (dB; negative = interferer weaker)
duty     : 0..1   traffic activity (Wi-Fi data load / beacon-only ~0.05,
                  saturated ~0.9)
```

Concrete piecewise example:

```
if   overlap == 0 or I_db <= -20:  base = 0.0     # no interaction
elif I_db >=   0:                  base = 1.0     # interferer captures RX
else:                              base = (I_db + 20) / 20

score = clamp01(base * (0.3 + 0.7 * duty) * (0.4 + 0.6 * overlap))
rfInterference = round(99 * score)
```

Notes: the engine's internal burst machines already model packet ON/OFF
timing — feed it the *session average* (duty/strength), not per-packet
frames. A VTX power-up event in your sim can briefly force `rfInterference`
near 99 for 0.1-0.3 s.

## 3. Per-frame usage pattern (Python)

```python
import numpy as np
from engine import AnalogVideoBreakupEngine

eng = AnalogVideoBreakupEngine(seed=42)          # one instance, main thread

def clamp01(x): return max(0.0, min(1.0, x))

# raw physics -> 0..99 (examples from Section 2)
def compute_raw_params(world):
    p_rx = link_budget(world)                    # dBm
    if   p_rx >= -94: s = 0
    elif p_rx <= -106: s = 99
    else: s = int(round((-94 - p_rx) / 12.0 * 99))

    s_db = strongest_early_echo_db(world)        # dB, or -99 if LOS clear
    if   s_db >= -6:  m_base = 1.0
    elif s_db <= -30: m_base = 0.0
    else:             m_base = (-6 - s_db) / 24.0
    m = int(round(99 * clamp01(m_base + 0.25 * world.indoor + 0.2 * world.pol)))

    ov, i_db, duty = wifi_state(world)
    if ov == 0 or i_db <= -20: i_base = 0.0
    elif i_db >= 0:            i_base = 1.0
    else:                      i_base = (i_db + 20) / 20.0
    i = int(round(99 * clamp01(i_base * (0.3 + 0.7*duty) * (0.4 + 0.6*ov))))

    return {"signalStrength": s, "multipath": m, "rfInterference": i}

# temporal smoothing: physics can be noisy; feed plausible envelopes
smoothed = {"signalStrength": 0, "multipath": 0, "rfInterference": 0}
ALPHA = 0.2                                          # EMA per frame @30fps

def smooth(raw):
    for k in smoothed:
        smoothed[k] = int(round((1 - ALPHA) * smoothed[k] + ALPHA * raw[k]))
    return dict(smoothed)

while sim.running:
    frame_rgb = sim.render_camera()                 # uint8 (H,W,3) RGB
    params = smooth(compute_raw_params(sim.world))
    degraded = eng.process(frame_rgb, params)       # new uint8 array
    display.show(degraded)                          # goggles / monitor / capture
```

Consecutive `process()` calls share temporal state (roll events, burst
machines, ghost flicker) — call it exactly once per displayed frame, in
order, on one instance.

### Engine init / reset

- Construct once: `AnalogVideoBreakupEngine(seed=...)`.
- Call `eng.reset()` on sim restart/replay to clear temporal state and
  `frame_index` (keeps params); or create a new instance per recording
  session for a clean state machine.
- Keep `seed` fixed for reproducible QA; vary it only to sample different
  event realizations.

## 4. Generic pseudocode (Unity / Unreal / Godot)

```
// once
engine = AnalogVideoBreakupEngine(seed: 42)

// per rendered camera frame (any engine; render camera to RGB8 buffer)
world      = get_world_state()
p_rx       = tx_power + gains - path_loss(distance) - blockage_db - pol_db
signal     = piecewise(p_rx, -94, -106)              // 0..99
echo_db    = max_raycast_reflection_power_db(world)  // vs direct path
multipath  = piecewise_echo(echo_db, indoor_flag, pol_mismatch)
interf     = piecewise_wifi(overlap, intf_db_at_rx, traffic_duty)

// optional envelope smoothing (EMA, alpha ~0.1..0.3)
signal, multipath, interf = ema_smooth(prev, {signal, multipath, interf}, 0.2)

rgb8       = camera_target.ReadPixels()              // H x W x 3, uint8, RGB order
out_rgb8   = engine.Process(rgb8, {signal, multipath, interf})
blit_to_screen(out_rgb8)                             // do NOT blit the clean frame
```

Engine binding notes for game engines:

- The Python engine runs out-of-process (or via embedded Python / IPC);
  for in-process GPU integration a GLSL port is planned (Section 6).
- Feed **RGB8**, not sRGB-compressed or BGR textures — swizzle first.
- One engine instance per rendering thread/sim instance.
- Render the camera at the sim's native resolution; the engine handles
  arbitrary `(H,W)` and reallocates on resolution change.

## 5. Temporal smoothing advice

- Physics-derived params (raycasts, RSSI estimators) can jitter every
  frame. The engine already handles micro-fluctuation *inside* its visual
  language (snow envelope flicker, burst machines, ghost amplitude flicker)
  — do not ask it to average raw sensor noise.
- Feed **plausible envelopes**:
  - EMA with alpha ~0.1-0.3 per frame (30 fps): smooth enough to look like
    a real link, fast enough to track fast fades (0.1-2 s).
  - Or rate-limit: max ~5-10 param units change per frame.
- Keep *slow* structure: range fades over seconds-minutes, fixed multipath
  bad spots, Wi-Fi traffic sessions. Over-smoothing removes the non-
  monotonic character that distinguishes multipath from distance.
- Step changes are legitimate (entering a bando, VTX power-up) — the
  engine's event machines (roll, nulls, takeovers) handle abrupt severity
  jumps by design.

## 6. GLSL port (planned)

A GLSL companion port under `engine/glsl/` is **planned** (not yet present
in this tree) for direct GPU integration — render pass or compute-shader
equivalent of the per-frame pipeline, keeping the same three-parameter
contract and stage curves. Until it lands, use the Python engine (CPU,
out-of-process or embedded) as the reference implementation; treat any
future GLSL port as behaviorally matched to the tables in
`docs/api/API.md` Section 9.

## 7. Demo and QA

```
# GUI demo (video: demo/gui/demo_video1.mp4)
python demo/gui/main.py

# regenerate the sample clip / still
python demo/sample_media/generate_sample.py

# render the visual QA matrix + short clips (tests/qa/)
python tests/render_matrix.py
```

Sample media: the GUI demo plays `demo/gui/demo_video1.mp4` (real-flight
clip, 1280x720 @30fps). `demo/sample_media/sample.mp4` is a synthetic
FPV-style regenerable fallback (panorama with strong horizontal/vertical
structure — good for judging ghosts, tearing, and bands) also used by
the QA matrix.
