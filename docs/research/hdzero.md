# Research: HDZero Digital Link

HDZero is a digital 5.8 GHz FPV system: the camera feed is compressed with a
block-based (DCT) codec at a constrained bitrate, FEC-protected, and sent as a
digital bitstream. The failure signature is fundamentally different from
analog FM: **there is no snow**. The picture stays pixel-clean until the bit
error rate outruns the FEC, then degrades in discrete, ugly steps — corrupt
blocks, line gibberish, stutter/freeze, and abrupt **breakup to black** — and
recovers as abruptly as it failed ("cliff effect").

> **Copyright notice:** "HDZero" is a trademark of its respective owner.
> This document is independent field research — not affiliated with,
> sponsored, or endorsed by HDZero. The bundled no-signal screen capture
> (`engine/hdzero/no_signal.png`) is © its respective owner and is used
> here only as a reference for the simulated loss screen.

## 1. Failure modes and signatures

| Mode | Signature |
|------|-----------|
| **Bit errors near range limit** | Isolated 8x8 DCT blocks replaced by confetti/solid garbage; sparse pure-white single-pixel speckle; blocks flicker frame to frame |
| **Burst errors (multipath/interference)** | Spatially **clustered** patches of corrupt blocks — punch-out blobs that drift slowly, not uniform scattering |
| **Codec at constrained bitrate** | Always-on chroma quantization/banding; color blocks get coarse and flat as rate drops |
| **Interference takeover** | Abrupt full-frame or banded disruption: horizontal tear bands of block garbage, rainbow full-screen scramble, colored horizontal line gibberish |
| **Loss of lock** | The whole screen becomes the **real HDZero no-signal screen** (captured rainbow-column pattern) in sustained runs: random 8x8 patches of it accumulate while the picture is still partially visible, then the full screen takes over — punctuated by occasional full black frames — and the picture snaps back when sync returns. At exactly signal 99 (the dead link) the full screen holds on **every** frame: no ramp, no black punch-through, multipath/interference ignored |

## 2. Visual characteristics

- **Block-quantized**: all corruption lives on the 8x8 grid (or is masked by
  it). Corrupt blocks show 4x4 sub-cell garbage — saturated "rainbow" or
  muted "bad decode" tones — never per-pixel analog static.
- **Valid pixels stay clean**: undamaged areas are bit-exact to the source
  (plus codec chroma quant). There is no global desaturation, no snow, no
  hue wander — color only dies when black frames arrive.
- **White speckle**, not snow: sparse single white pixels (bit flips), not a
  full-field boiling noise layer.
- **Abrupt edges**: events switch on and off in one frame; corruption is
  either present or absent — no amplitude fade.
- **Line gibberish** during strong interference: full-width colored-line
  noise bands, denser than anything analog produces (rows of corrupted
  entropy-decoded data).
- **Loss screen**: on total signal loss the receiver paints its no-signal
  screen. The engine ships the real captured screen (`engine/hdzero/
  no_signal.png`, center-cropped to the frame aspect, nearest-scaled once
  per resolution) instead of synthesizing it. While a loss run is still
  ramping up, random **8x8 block-cell patches** of that screen — crops
  stable per run, layout randomized per run/seed — accumulate over
  `loss_ramp` frames until coverage reaches 1 and the full screen takes
  over. The picture stays visible between patches until then, and the
  procedural vertical-rainbow-column generator remains as a fallback when
  no asset/OpenCV is available.

## 3. Temporal structure

| Phenomenon | Timescale | Notes |
|---|---|---|
| Block flicker | per frame / 4-frame epochs | mixture of epoch-stable (same blocks bad for ~4 frames) and per-frame randomness |
| Cluster drift/pulse | ~0.5-2 s envelope | burst-error blobs wander and breathe |
| Takeover event | ~0.2-0.4 s at high severity (2-12 frames) | one burst = one visible "hit" |
| Gap between events | few frames at severity 99 → many seconds at severity 25 | duty cycle climbs with severity (measured occupancy: ~0% → ~11% → ~36% → ~59% at 5/25/80/99) |
| Signal-loss runs | ~0.07-0.8 s per run (4-50 frames) | latch: entry chance per frame at raw ≥ 0.74; at severity 99 the dead link holds the full screen on 100% of frames (~95% occupancy just below, 0 at ≤ 70); first ~8 frames of a natural run ramp: growing 8x8 no-signal patches, then full screen |
| Blackout frames | 1-2 frames each, ~15% of frames at severity ~99 | punch through the loss screen; instant recovery; suppressed at the 99 dead link (full screen only) |

## 4. Causation

- **FEC cliff**: below the decoder threshold errors are corrected and the
  picture is perfect; above it, decoded bits are wrong. That maps to
  `block_err` probability crossing from ~0 to substantial — visible onset is
  early (r≈0.04) but weight is heavily biased to high severity.
- **DCT block decode**: one wrong coefficient set corrupts the whole 8x8
  block — hence block-quantized damage with sub-cell structure, not grain.
- **De-interleaver burst errors**: a short RF fade lands on a contiguous
  span of the bitstream, which de-interleaves into a *spatial cluster* of
  bad blocks — multipath axis drives a drifting blob field.
- **Rate drop**: bandwidth pressure lowers effective bitrate → chroma is
  quantized harder first (chroma subsampling/quant is cheapest to cut).
- **Loss of lock**: below the sync threshold the receiver decodes a
  garbage bitstream and paints its error pattern — the captured
  rainbow-column screen — until sync is re-acquired. The model latches
  into loss runs (entry chance per frame + run-length timer) rather than
  flickering per frame, matching the observed "screen goes rainbow for a
  moment, then snaps back" behavior, and ramps coverage from scattered
  8x8 patches of the screen to the full screen over the run's first
  frames (the picture is still partially visible at first — "signal
  coming and going" — before the screen fully takes over).

## 5. Distinction from the analog axes

| | Analog weak signal | Analog interference | **HDZero (digital)** |
|---|---|---|---|
| Noise floor | full-field snow grows continuously | banded, inside bursts | **none** — pixels stay clean |
| Corruption unit | pixel | scanline/slice | **8x8 block** |
| Color behavior | desaturates → color killer → hue wander | green/purple tint in bands | flat quantized blocks, rainbow garbage; clean elsewhere |
| Failure edge | gradual | bursty but signal persists | **cliff**: clean ↔ rainbow/black, 1-frame transitions |
| Recovery | gradual (AGC, hysteresis) | hangover decay | instant snap back |

## 6. How the engine reproduces it

Module: `engine/hdzero/` — `params.py` (stage curves), `kernels.py`
(Numba kernels), `state.py` (temporal state), `engine.py` (`HDZeroEngine`),
plus the shipped no-signal asset `no_signal.png`. The public contract is
identical to `AnalogVideoBreakupEngine` (same three 0..99 keys, same
`process()` semantics).

The asset loads lazily on first frame: `no_signal.npy` if present, else
`no_signal.png` via OpenCV (the engine itself only requires numpy + numba;
without cv2 or the file it falls back to the procedural loss screen). It is
fitted once per resolution (center-crop to frame aspect + nearest scale).

### 6.1 Pipeline (per frame, after the all-zero fast path)

```
clean RGB
  -> chroma_quant      codec color banding (gated: signal must be > 0)
  -> compute_bmap      per-8x8-block type: clean / confetti / solid /
                       dct / shifted / white  (signal scatter + mp clusters)
  -> fill_blocks       paint corrupt blocks (4x4 sub-cell garbage,
                       palette solids, displaced copies, white)
  -> apply_speckle     sparse white single-pixel bit errors
  -> render_event      takeover events: band / rainbow / lines / black
  -> loss screen       latched runs: growing random 8x8 patches of the
                       real no-signal screen (fill_loss_patches) ramping
                       to the full screen blit; procedural rainbow-column
                       fallback when the asset is unavailable
  -> fill_black        extreme-signal breakup-to-black (runs last,
                       overrides the loss screen)
```

Every kernel is called unconditionally on every non-zero frame with
flag/level arguments, so one Numba call compiles all of its branches (no
first-use compile stall mid-interaction); `HDZeroEngine.warmup()` additionally
forces each takeover event type, a partial-coverage loss frame (compiles
`fill_loss_patches` or the procedural fallback) and the blackout path — the
full-screen state itself is a plain array copy.

### 6.2 Signal stages (r = signalStrength/99 ∈ [0,1])

| Stage key | Curve | Onset r | Full r | Meaning |
|---|---|---|---|---|
| `block_err` | `smoothstep(0.04,0.97,r) * 0.62` | 0.04 | 0.97 | per-block corrupt probability (0.62 epoch-stable + 0.38 per-frame mixture) |
| `speckle` | `smoothstep(0.12,0.92,r) * 0.0035` | 0.12 | 0.92 | per-pixel white hit probability (~58 hits/frame at 99 on 720p) |
| `white_blk` | `smoothstep(0.30,0.90,r) * 0.05` | 0.30 | 0.90 | share of fully-white blocks inside the corrupt set |
| `black_frame` | `smoothstep(0.80,1.0,r) * 0.15` | 0.80 | 1.0 | per-frame probability of a full blackout (measured: 0 at 70, ~15% of frames at 98; black frames punch through the loss screen below 99 — suppressed at the dead link) |
| `loss_p` | `smoothstep(0.74,1.0,r) * 0.10` | 0.74 | 1.0 | chance per frame to latch into a signal-loss run (at exactly 99 the dead link forces the run continuously) |
| `loss_dur_lo/hi` | `4 + r*14` … `10 + r*40` | — | — | loss-run length in frames (~0.07-0.8 s); 100% full-screen at r=1 (dead link), ~95% occupancy at 98, 0 at r≤0.707 |
| `loss_ramp` | `8.0 + (1-r) * 8.0` | — | — | frames from run start to full no-signal coverage (8 at r=1, 16 at r=0; patches accumulate until coverage ≥ 1, then the full screen replaces the picture) |
| `chroma_q` | `6.0 + smoothstep(0.0,1.0,r) * 18.0` | 0.00 | 1.0 | chroma quant step — always-on codec look, but the engine only applies it when signal > 0 so a pure 0,0,x link stays bit-clean outside events |

Note: effects that render *before* the loss blit (block corruption,
speckle, white blocks, blackouts) are invisible at the 99 dead link — the
full no-signal screen replaces the whole frame after them.

Corrupt-block type mix (from `compute_bmap`): confetti 38%, solid 22%,
DCT-noise 20%, shifted ~20% minus `white_blk`, plus `white_blk` itself —
weighted so confetti (rainbow-block) dominates the visible look.

### 6.3 Multipath stages (r = multipath/99)

| Stage key | Curve | Meaning |
|---|---|---|
| `cl_cov` | `smoothstep(0.08,1.0,r) * 0.42` | blob-field coverage of the cluster mask |
| `cl_boost` | `smoothstep(0.25,1.0,r) * 0.80` | corrupt probability inside a cluster (can override the signal decision) |
| `env_rate` | `0.045 + r * 0.10` | burst-envelope speed (value noise over frames) |
| `mp_speed` | `0.002 + r * 0.018` | cluster-field drift phase per frame |

Inside clusters the type mix skews to DCT-noise/solid/shifted (42/28/20%)
— the "punch-out" look: coherent patches of damaged blocks, not scatter.
The cluster threshold is modulated by the envelope
(`thr = 1 - cl_cov * (0.35 + 0.65 * env)`), so clusters pulse as well as
drift.

### 6.4 Interference stages (r = rfInterference/99)

| Stage key | Curve | Meaning |
|---|---|---|
| `duty` | `smoothstep(0.05,0.97,r) * 0.72` | probability per eligible frame of starting an event |
| `dur_lo/hi` | `2 + r*4` … `4 + r*8` | event length in frames (~0.03-0.3 s) |
| `gap_lo/hi` | `3` … `8 + (1-r)*60` | cooldown frames; short at high severity, many seconds at low |
| `band_speed` | `0.004 + r*0.02` | top→bottom sweep rate of band events (frame-normalized) |
| `band_frac` | `0.05 + r*0.28` | band height as fraction of frame |
| `w_rain` | `smoothstep(0.45,1.0,r) * 0.30` | cumulative weight: rainbow-screen events |
| `w_lines` | `smoothstep(0.35,0.95,r) * 0.30` | cumulative weight: colored-line gibberish |
| `w_black` | `smoothstep(0.70,1.0,r) * 0.25` | cumulative weight: black-screen events |
| `density` | `smoothstep(0.20,0.95,r) * 0.75` | row density of the line-gibberish event |

Event machine (`update_hz_state`): idle frames roll `hash(seed,fi) < duty`;
when an event starts, a second hash picks the type from the cumulative
weights, `dur_lo..dur_hi` sets the timer, and a cooldown of
`gap_lo..gap_hi` frames follows. Bands sweep downward and wrap. With
`rfInterference = 0` any running event is force-killed and no new event can
start. Measured occupancy over 120 frames (seed 42, 720p source):
**r=5 → 0%, r=25 → 11%, r=80 → 36%, r=99 → 59%** — the severity ladder is
the duty/occupancy curve.

### 6.5 Event renderers (`render_event`)

| Type | Render |
|---|---|
| `EV_BAND` (0, default) | horizontal band of block garbage sweeping the frame + tear rows of line noise at its edges |
| `EV_RAINBOW` (1) | full-frame 4x4 sub-cell saturated garbage (rainbow scramble) |
| `EV_LINES` (2) | full-width colored horizontal line gibberish at `density` row probability |
| `EV_BLACK` (3) | full-frame blackout for the event duration |

### 6.6 Determinism and contract

- Same contract as the analog engine: all randomness is
  `hash(seed, frame_index, coordinates)`; `process(frame, {}, advance=False)`
  is idempotent; `process(frame, None)` uses persistent `last_params`
  (mirrors analog semantics).
- `(0,0,0)` fast path returns a bit-identical copy; `reset()` clears the
  event machine + cluster field + loss latch and zeroes `frame_index`.
- `engine.in_loss` reports whether the last processed frame was inside a
  signal-loss run (useful for driving UI state like a "SIGNAL LOST"
  indicator).
- QA artifacts: `tests/render_matrix_hdzero.py` renders
  `tests/qa/hdzero/*.png` (peak-effect frame selection — digital effects are
  bursty, idle frames are clean), a forced-run progression strip
  (`loss_ramp_strip.png`, patch coverage 0.13 → 1.0 → full screen) plus
  60-frame clips under `tests/qa/hdzero_clips/`.
