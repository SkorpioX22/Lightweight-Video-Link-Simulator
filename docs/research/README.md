# Research Notes — LVLS (Lightweight Video Link Simulation)

Consolidated research findings behind the engine's degradation model. The
engine emulates how 5.8 GHz analog FPV video (FM-modulated CVBS) breaks up in
the real world, driven by three independent severity axes:

| Axis (engine param)      | Physical cause                                   | Visual signature |
|--------------------------|--------------------------------------------------|------------------|
| `signalStrength` (0..99) | Low received power (range, obstacles, antennas)  | Noise-dominant: snow, color kill, tearing, roll, static |
| `multipath` (0..99)      | Reflected copies of the signal arriving late     | Structural: echoes, smear, displaced blocks, white flashes |
| `rfInterference` (0..99) | Other transmitters (Wi-Fi, other VTXs, CW)       | Bursty + banded: scrolling noise stripes, slices, fringe |

Higher param value = worse link. `0` = clean, `99` = extreme.

## Documents

| File | Contents |
|------|----------|
| `weak_signal.md` | Staged weak-signal progression, FM/CVBS causation chain, temporal scales, engine stage thresholds |
| `multipath.md`   | Echo/comb physics, FM phase-ripple effects, ghost stages, null dropouts, engine implementation |
| `interference.md`| Wi-Fi/VTX/CW signatures, burst/band structure, causation, engine burst machine |
| `dvr_vs_rf.md`   | Which artifacts live in the RF chain vs the recorder; what the engine deliberately does not emulate |

## End-to-end signal chain

```
VTX camera --> CVBS (luma + chroma @ 3.58/4.43 MHz subcarrier + sync)
            --> FM modulate @ 5.8 GHz
            --> air link: path loss + multipath + interference + noise
            --> receiver antenna/LNA
            --> FM discriminator (limiter removes AM, phase ripple survives)
            --> deemphasis (tau ~ 0.75-0.8 us)
            --> CVBS decode: sync slicers (H/V), chroma PLL, AGC
            --> display (goggles / monitor) or DVR recorder
```

Each research doc maps stages of this chain to what the viewer sees, then to
the engine module that reproduces it.

## Cross-cutting distinctions

The three axes are intentionally distinguishable at a glance:

```
weak signal   = CONTINUOUS, uniform, monotonic with distance
                ("more snow the further you go")
multipath     = STRUCTURAL, low snow, non-monotonic in position
                (echoes / displacement / smear / fixed bad spots)
interference  = BURSTY + BANDED, chroma-tinted, sharp onset/offset
                (clean frames between bursts, scrolling stripes)
```

- Weak signal and multipath are physically separable: raising VTX power
  fixes weak signal but can worsen multipath indoors.
- Interference is separable from both by its temporal sparsity (packet
  bursts) and row-wise energy concentration (bands/slices).
- CP antennas reject single-bounce multipath by roughly 20-30 dB; they do
  not help weak signal in free space or interference from a same-polarized
  source.

## Shared temporal scales

| Phenomenon | Timescale | Notes |
|---|---|---|
| Snow boil | every field (50/60 Hz) | redrawn noise, additive in signal space |
| FM sparklies | Poisson ~10-200 events/s | hard black/white dots and dashes |
| Hue wander | 0.5-5 s | slow phase drift before color kill |
| Color-killer | flicker then latch | hysteresis: one-way until signal recovers |
| V-roll events | 0.2-2 s then re-lock | bursty, with cooldown between events |
| Range fade | seconds-minutes | slow envelope under fast fading |
| Fast fades | 0.1-2 s | deep dips, especially indoors |
| Ghost flicker | ~19 Hz per m/s path change @ 5.8 GHz | phase flips every ~2.6 cm of path length |
| Wi-Fi beacons | ~102 ms | periodic burst clusters |
| Desense recovery | 30-150 ms | AGC hangover after a burst |
| Long takeovers | 0.3-2.5 s | VTX power-up / sustained co-channel |

## Sensitivity anchors (ImmersionRC-style)

The weak-signal axis is anchored on measured receiver behavior:

```
first visible snow   ~ -94 dBm
sync loss            ~ -106 dBm
                     ------------
window                  ~12 dB
```

That ~12 dB window contains the entire visible degradation ladder
(grain -> snow -> color death -> sync stress -> static), which is why stage
thresholds in `engine/core/params.py` are densely packed and deliberately
nonlinear rather than evenly spaced over 0..99.

## Engine mapping summary

| Research doc | Engine module | Stage table |
|---|---|---|
| `weak_signal.md` | `engine/signal/weak_signal.py` | `signal_stages()` in `engine/core/params.py` |
| `multipath.md` | `engine/multipath/multipath.py` | `multipath_stages()` |
| `interference.md` | `engine/interference/interference.py` | `interference_stages()` |
| `dvr_vs_rf.md` | renderer compositing order in `engine/renderer/engine.py` | (scope boundary) |

Compositing order per frame: multipath (structural, needs clean source) ->
weak signal (snow/color/sparklies, accumulates jitter+tear) -> interference
(bands/slices/fringe, accumulates false-sync shifts) -> geometry (row shifts
+ vertical roll) -> RGB.
