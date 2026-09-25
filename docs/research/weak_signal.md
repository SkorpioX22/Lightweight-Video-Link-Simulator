# Research: Weak Signal Progression

How an analog FPV video link degrades as received power falls. The
progression is staged, ordered, and anchored on ImmersionRC-style receiver
sensitivity:

```
first snow     ~ -94 dBm
sync loss      ~ -106 dBm
window         ~ 12 dB
```

Everything visible happens inside that ~12 dB window. The stages below are
ordered as observed in the real world and map onto `signalStrength` 0..99
(nonlinearly — see thresholds).

## 1. Staged progression

| Stage | Name | What the viewer sees |
|-------|------|----------------------|
| S0 | Clean | Perfect image, no artifacts |
| S1 | Subtle grain | Faint luma grain in dark areas + chroma specks + slow hue wander |
| S2 | Fine snow | Visible fine monochrome snow; additive, uniform in signal space; perceptually stronger in darks |
| S3 | Color failure | Saturation collapse -> hue flicker -> color-killer latch -> locked B&W; color dies well before monochrome detail |
| S4 | Sync stress | Per-line horizontal jitter/tearing -> isolated corrupted lines -> intermittent vertical roll events (0.2-2 s) with recovery; FM "sparklies": hard black/white dots and dashes rising steeply below threshold |
| S5 | Near-total loss | Boiling static, image crushed, sync lost |

Notes on S2: snow is additive and uniform in *signal* space, but the eye
perceives it more strongly in dark regions (Weber-like contrast). S3 is a
distinct cliff: color (chroma subcarrier) fails long before the monochrome
picture becomes unusable. S4 combines two sync systems failing at different
rates: horizontal (line) sync tears locally first, vertical (frame) sync
rolls in bursts.

## 2. Technical causation chain

```
5.8 GHz FM video
  -> receiver frontend noise (adds to signal before detection)
  -> FM discriminator
       triangular noise PSD proportional to f^2
         -> noise power grows toward high baseband freq
         -> fine grain after deemphasis (tau ~ 0.75-0.8 us)
  -> CVBS composite
       chroma rides a suppressed subcarrier far lower SNR than luma
         -> color dies first (S3 before S4/S5)
       sync tips are the largest bipolar excursion in the waveform
         -> survive the lowest CNR (sync outlasts picture)
       Rice "click" noise near threshold
         -> sparklies: hard black/white dots and dashes (S4)
       H-sync slicer fails locally on noisy lines
         -> tearing / line displacement (S4)
       V-sync integrator fails intermittently
         -> vertical roll events with recovery (S4)
  -> total C/N below sync capability
         -> boiling static, lost lock (S5)
```

Key ordering fact: chroma < luma < sync in robustness. This single
inequality produces the whole S3-before-S4-before-S5 ordering the engine
reproduces.

## 3. Temporal characteristics

| Artifact | Temporal behavior |
|---|---|
| Snow | Redrawn every field -> 50/60 Hz boil |
| Sparklies | Poisson arrivals, ~10-200/s, steep rise below threshold |
| Hue wander | Slow drift, ~0.5-5 s timescale, per-line jitter on top |
| Color killer | Flickers at onset, then latches to B&W until recovery |
| Roll events | 0.2-2 s bursts with snap re-lock and cooldown between |
| Range fade | Slow envelope over seconds-minutes, with fast fades 0.1-2 s layered on it |

The engine models the slow range fade and fast fades together as a snow
envelope flicker (multiplicative on noise amplitude) so snow "breathes"
instead of sitting at constant density.

## 4. How the engine reproduces it

Module: `engine/signal/weak_signal.py`
Stage table: `signal_stages()` in `engine/core/params.py`

### 4.1 Stage thresholds (raw severity r = signalStrength/99 in [0,1])

| Stage key | Curve (from source) | Onset r | Full r | Meaning |
|---|---|---|---|---|
| `luma_noise` | `46 * ease_in(smoothstep(0.015,1.0,r), 1.35) / 255` | 0.015 | 1.0 | grain starts almost immediately, ease-in ramp (amplitude 0..46/255) |
| `chroma_noise` | `smoothstep(0.04,0.42,r) * 0.28` | 0.04 | 0.42 | coarse chroma blotches |
| `black_lift` | `smoothstep(0.05,0.7,r) * 0.04` | 0.05 | 0.70 | snow lifts black level (stage present; see API notes) |
| `hue_wander` | `smoothstep(0.12,0.45,r) * 0.55` | 0.12 | 0.45 | hue rotation amplitude |
| `sat_loss` | `smoothstep(0.18,0.50,r)` | 0.18 | 0.50 | saturation collapse, smooth ramp into killer |
| killer hysteresis | on at raw 0.50, off at raw 0.42 | 0.42-0.50 | — | latch: color dies at S3, recovers only after signal improves |
| `sparklie` | `ease_in(smoothstep(0.42,0.88,r), 2.0) * 0.5` | 0.42 | 0.88 | quadratic ease-in: steep rise below FM threshold |
| `h_jitter` | `smoothstep(0.52,0.80,r) * 7.0` | 0.52 | 0.80 | per-line horizontal jitter, up to 7 px |
| `tear` | `smoothstep(0.62,0.92,r)` | 0.62 | 0.92 | probability scale for displaced bands / corrupted lines |
| `roll` | `smoothstep(0.72,0.97,r)` | 0.72 | 0.97 | vertical roll event probability (used as `roll * 0.12` per frame) |
| `static_kill` | `smoothstep(0.88,1.0,r)` | 0.88 | 1.0 | picture crushed into pure static |

All stage onsets are ordered to match S1 -> S5: grain (0.015) -> chroma
specks (0.04) -> hue (0.12) -> saturation (0.18) -> killer/sparklies (0.42)
-> jitter (0.52) -> tear (0.62) -> roll (0.72) -> static (0.88). The
nonlinear spacing concentrates stages inside the physical ~12 dB window.

### 4.2 Noise structure

- **Dual-scale noise**: luma grain sampled on a fine lattice (spatial scale
  ~0.055) vs chroma noise on a coarse ~1/8-resolution lattice (scale 0.12)
  -> colored blotches instead of per-pixel chroma hash.
- **60/40 split**: ~60% temporally coherent flow noise (value noise scrolling
  with frame time) + ~40% per-frame white-ish grain with 2 px horizontal
  correlation -> boiling snow that does not look like static TV snow.
- **Snow envelope flicker**: `env = 0.72 + 0.56 * value_noise(frame*0.13)`
  -> slow breathing on top of whatever envelope the simulator feeds in.
- **Black lift**: noise raises the floor of pure-black regions (S1 tell).

### 4.3 Color pipeline

- Saturation: `sat_gain = 1 - sat_loss`, applied by scaling U/V toward 0.
- Hue: per-line rotation `hue_wander * (sin(phase + y*0.11) + 0.5*hash)` —
  slow global wander + per-line/per-frame burst jitter.
- Killer: hysteresis latch in state; when latched, `U = V = 0` exactly
  (true B&W), until raw severity drops below the off-threshold.

### 4.4 Sync / geometry

- **H jitter**: per-line random ± plus a slow AFC wander (value noise along
  y and time), amplitude up to 7 px.
- **Tear bands**: rows grouped into bands 4-40 rows tall; pattern held ~4
  frames (`frame_idx >> 2`); each band displaced ±(10..80)*tear px with
  probability `0.16 * tear`. Occasionally a 3-row segment is fully replaced
  by static (probability `0.004 * tear`) -> "isolated corrupted lines".
- **Roll event machine**: with per-frame probability `roll*0.12` (after
  cooldown), start an event lasting 6-60 frames (0.2-2 s @30fps), scrolling
  12-38 rows/frame in a random direction; on expiry the position snaps to 0
  (V-sync re-lock) and a 30-120 frame cooldown begins.
- **Sparklies**: horizontal dashes 2-11 px long; per-dash probability
  `sparklie * 0.055`; each hit writes a hard bipolar value (luma exactly 1.0
  or 0.0) -> the classic FM-threshold black/white dots and dashes.
- **Static kill**: `img_keep = 1 - 0.93 * static_kill` crushes the picture
  while a full-field static mix rises to 1.0 at r = 1.
