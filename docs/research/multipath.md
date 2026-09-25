# Research: Multipath

Reflected copies of the transmitted signal arriving with delay and phase,
mixing with the direct path at the receiver. Unlike weak signal, multipath
is *structural*: it distorts image geometry and adds echoes with little
background snow (except during deep nulls).

## 1. Physical model

```
r(t) = s(t) + SUM_k  a_k * e^(j*phi_k) * s(t - tau_k)
```

- **Delays**: nearby reflectors 10-100 ns (sub-pixel on screen) up to
  0.3-3 us for distant reflectors. On-screen ghost offset is approximately
  **13.5 px per us** of delay.
- **Phase**: phi_k flips every ~2.6 cm of path-length change -> fast
  flicker; a 1 m/s platform speed at 5.8 GHz gives a Doppler-like phase
  rate of ~19 Hz per m/s.
- **FM specifics**:
  - Amplitude notches are removed by the receiver limiter, but the
    **phase ripple survives** into the discriminator.
  - Discriminator error scales with baseband frequency -> HF (edges)
    distort most: comb-filter smear + ghost echoes + FM edge spikes
    (hard bright/dark stripes at a fixed offset to the right of strong
    vertical edges).
  - Deep cancellation (a ~ 1, phi ~ 180 degrees) -> white AGC flash or
    momentary dropout (click noise) even at close range / high power —
    multipath is distance-independent and position-locked.

## 2. Visual stages

| Stage | What the viewer sees |
|-------|----------------------|
| S0 | Sub-pixel halo / edge ringing (no snow) |
| S1 | Single right-offset ghost at 15-40% brightness + horizontal smear + white horizontal ripple bands |
| S2 | Echo train: 2-3 ghosts with +/- polarity, row/block horizontal displacement plateaus ("blocks of monochrome / shifted picture" as reported on rcgroups), wavy per-line displacement |
| S3 | Deep nulls: sharp position-locked white flashes / dropout bursts (2-14 frames), mixed snow bursts only during nulls, sync disturbance |

## 3. Temporal behavior

- **Ghost OFFSET** is smooth in time (it is geometry: reflector positions
  change slowly relative to the frame rate).
- **Ghost BRIGHTNESS / POLARITY** flickers fast, including full sign flips
  (phase walking at ~19 Hz per m/s of path change).
- The channel is **line-coherent within a frame** (constant over ~16 ms of
  a 60 Hz field / 33 ms of a 30 fps frame) — all lines in one frame see the
  same tap gains.
- Flicker events correlate with platform motion; during a static hover the
  ghost is quasi-static.
- Nulls and dropouts are sudden, position-locked, and distance-independent.

## 4. Distinction from weak signal

| | Multipath | Weak signal |
|---|---|---|
| Character | Structural (echoes, displacement, smear, bands) | Noise-dominant (snow) |
| Snow level | Low (except inside nulls) | Rising monotonically |
| With distance | Non-monotonic; fixed bad spots | Monotonic degradation |
| With VTX power | Can worsen indoors (stronger reflections) | Improves |
| Antennas | CP rejects single bounce ~20-30 dB | Needs gain/directivity |

Rule of thumb: if the picture is *shifted, doubled, or banded but clean*,
it is multipath; if it is *snowy but aligned*, it is weak signal.

## 5. How the engine reproduces it

Module: `engine/multipath/multipath.py`
Stage table: `multipath_stages()` in `engine/core/params.py`

### 5.1 Stage thresholds (raw severity r = multipath/99 in [0,1])

| Stage key | Curve (from source) | Onset r | Full r | Meaning |
|---|---|---|---|---|
| `comb_k` | `smoothstep(0.04,0.45,r) * 0.30` | 0.04 | 0.45 | sub-pixel comb tap (S0 halo/ringing) |
| `ghost1_d` | `4 + smoothstep(0.12,1.0,r) * 34` px | 0.12 | 1.0 | first ghost delay (smooth drift on top) |
| `ghost1_a` | `smoothstep(0.14,0.55,r) * 0.42` | 0.14 | 0.55 | first ghost amplitude envelope |
| `smear_px` | `smoothstep(0.18,0.70,r) * 3.5` px | 0.18 | 0.70 | horizontal box-smear radius |
| `band_ripple` | `smoothstep(0.30,0.75,r) * 0.16` | 0.30 | 0.75 | white horizontal ripple bands |
| `edge_spike` | `smoothstep(0.35,0.8,r) * 0.55` | 0.35 | 0.80 | FM edge spike stripes right of vertical edges |
| `ghost2_d` | `14 + smoothstep(0.40,1.0,r) * 70` px | 0.40 | 1.0 | second ghost delay |
| `ghost2_a` | `smoothstep(0.42,0.80,r) * 0.30` | 0.42 | 0.80 | second ghost amplitude |
| `flash_rate` | `smoothstep(0.45,0.95,r) * 0.05` | 0.45 | 0.95 | per-frame white AGC flash probability |
| `block_shift` | `smoothstep(0.48,0.90,r) * 34` px | 0.48 | 0.90 | row-block displacement plateaus |
| `null_rate` | `ease_in(smoothstep(0.50,1.0,r), 1.6) * 0.10` | 0.50 | 1.0 | deep-null dropout probability/frame |
| `null_snow` | `smoothstep(0.5,1.0,r) * 0.55` | 0.50 | 1.0 | click-noise snow during nulls only |
| `desat_bands` | `smoothstep(0.55,0.9,r) * 0.8` | 0.55 | 0.90 | B&W (monochrome) row-blocks via burst echo |
| `ghost3_a` | `smoothstep(0.65,1.0,r) * 0.18` | 0.65 | 1.0 | third ghost amplitude (delay = ghost2_d * 1.7, set by renderer) |

### 5.2 Echo mixing

Mix-out form with DC gain 1:

```
out = base + SUM_k a_k * (echo_k - base)
```

Equivalent to `base*(1 - sum a) + sum a*echo_k` — aligned taps pass the
image unchanged, so ghosts add without changing average brightness.

- **2-3 ghost taps**: integer-pixel delays with slow drift
  (`drift = 1 + 0.15..0.22 * value_noise(frame * 0.022..0.035)`), so ghost
  offset glides smoothly (geometry), while amplitudes are signed and
  flicker fast (`value_noise` at rates 1.9 / 2.7 / 3.3 per frame) including
  polarity flips — the Doppler sign inversion.
- **Sub-pixel comb tap**: ~1 px echo with fractional lerp (uses the
  fractional part of ghost-1 delay) -> S0 halo/edge ringing with no visible
  ghost yet.

### 5.3 Structural artifacts

- **Horizontal smear**: box blur radius `round(smear_px)`; below
  `smear_px <= 0.8` the blur is blended 45/55 with the echo mix, above it
  fully replaces — proxy for frequency-selective fade comb smear.
- **FM edge spikes**: vertical gradient detection (`|g| > 0.28` sampled
  3-7 px to the left of the current pixel) adds/subtracts a stripe ->
  bright/dark spikes at fixed offset right of strong vertical edges.
- **White ripple bands**: 1D value noise along y with slowly advancing
  phase; thresholded at 0.62 -> discrete white horizontal bands.
- **Block displacement**: 24-row blocks; ~35% of blocks get a smooth
  drifting horizontal shift up to `block_shift` px -> "shifted picture
  plateaus". Accumulates into `row_shift`, applied once by the geometry
  pass.
- **Monochrome blocks**: each 24-row block goes B&W with probability
  `desat_bands * 0.30` (color-burst echo cancellation on those bands).

### 5.4 Null / flash event machines

- **White AGC flash**: 1-frame event, probability `flash_rate`, cooldown
  4-18 frames. Luma `val*1.15 + 0.45`, chroma scaled to 0.6.
- **Deep null**: 2-14 frame dropout, cooldown 10-50 frames, probability
  `null_rate`. Image collapses toward click-noise static: picture kept at
  10%, static field at `null_snow` strength, chroma scaled to 0.15 — snow
  appears *only* during nulls, keeping baseline multipath snow-free.
