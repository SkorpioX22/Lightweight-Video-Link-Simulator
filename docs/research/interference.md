# Research: RF Interference

Artifacts caused by other RF sources landing in or near the 5.8 GHz FPV
band: Wi-Fi, other pilots' VTXs, broadband noise, CW carriers. The signature
that separates interference from weak signal and multipath is
**bursty + banded**: structured rows of energy that switch on and off with
sharp edges, with clean frames between bursts.

## 1. Interference types and signatures

| Type | Signature |
|------|-----------|
| **Wi-Fi (5 GHz UNII)** | UNII bands overlap FPV R1-R6 (R8 is clear). Narrow horizontal noise lines that travel vertically (beat against carrier offset). Bursty, following packet traffic: 802.11 frames last microseconds-ms, beacons ~102 ms, duty cycle can exceed 50% under load. |
| **Other VTXs (co-channel)** | Capture-effect flicker; on power-up a full-band glitch lasting ~100-300 ms then sustained; adjacent-channel "frosting" + faint ghost. |
| **Other VTXs (nearby freq)** | Co-channel beating -> diagonal herringbone / fringe crawl (crawl rate equals the frequency difference) or low-frequency horizontal bars. |
| **Broadband noise** | Continuous, snow-like — but range-independent (does not worsen with distance). |
| **Narrowband / CW** | Steady or slowly crawling stripes. |
| **Microwave / oven-like** | Gated bursts at ~50 Hz. |

## 2. Visual characteristics

- Banded / striped / line-structured, often **tinted** (purple/green casts
  from corruption of the chroma region of the spectrum).
- Bands **scroll vertically** at a rate set by the carrier offset.
- **Bursty**: bursts separated by clean frames; sharp onset and offset.
- **Slice corruption** with hard line boundaries: burst duration maps to a
  number of scanlines at the 15.7 kHz line rate — **1 ms ~ 16 lines**.
- More chroma-biased than weak signal (the interferer hits the chroma
  subcarrier region preferentially).

## 3. Temporal structure

| Layer | Timescale |
|---|---|
| Single burst ON | 0.5-5 ms (~1-6 frames at 30 fps presentation) |
| OFF gap between bursts | shorter than ON at high load |
| Traffic cluster envelope | 50-100 Hz-ish sessions lasting 0.3-5 s |
| Rare long takeover | 0.3-2.5 s (VTX power-up, sustained co-channel) |
| Desense hangover | AGC recovery 30-150 ms after a burst ends |

## 4. Causation

- **Bursty OOBE/blocking** is well modeled as gated noise
  (per ITU-R BT.2382 studies of wireless microphone / video interference).
- **Capture effect**: during a burst the stronger interferer captures the
  FM demodulator; when it ends, the picture snaps back -> hard edges.
- **Beat tones**: a nearby carrier beats with the video carrier at the
  difference frequency -> a video-rate tone -> horizontal bars / diagonal
  fringe whose crawl rate equals the beat frequency.
- **False sync**: strong bursts push the slicer into false triggers ->
  tearing and row displacement.

## 5. Distinction from the other axes

| | Interference | Weak signal | Multipath |
|---|---|---|---|
| Temporal | Bursty, sparse, sharp on/off | Continuous | Continuous (flicker) |
| Spatial | Banded / row-structured, scrolling | Uniform full-field snow | Echoes / displacement / smear |
| Chroma | Often tinted (purple/green) | Desaturates then kills color | Occasional mono bands |
| With range | Independent of distance | Monotonic with distance | Position-locked spots |

## 6. How the engine reproduces it

Module: `engine/interference/interference.py`
Stage table: `interference_stages()` in `engine/core/params.py`

### 6.1 Stage thresholds (raw severity r = rfInterference/99 in [0,1])

| Stage key | Curve (from source) | Onset r | Full r | Meaning |
|---|---|---|---|---|
| `duty` | `smoothstep(0.06,0.85,r) * 0.65` | 0.06 | 0.85 | fraction of time inside a burst (before traffic clustering) |
| `band_amp` | `smoothstep(0.10,0.70,r)` | 0.10 | 0.70 | scrolling noise-band amplitude |
| `scroll` | `0.4 + smoothstep(0.1,1.0,r) * 5.0` lines/frame | 0.10 | 1.0 | vertical band scroll rate |
| `band_count` | `3.0 + r * 14.0` (linear) | 0.00 | 1.0 | number of bands across the frame |
| `impulse` | `smoothstep(0.35,0.9,r) * 0.45` | 0.35 | 0.90 | hard black/white impulse "salt" inside bursts |
| `desense` | `smoothstep(0.3,0.95,r) * 0.35` | 0.30 | 0.95 | post-burst snow hangover amplitude |
| `tint` | `smoothstep(0.25,0.8,r)` | 0.25 | 0.80 | green/purple band tint |
| `fringe` | `smoothstep(0.28,0.75,r) * 0.22` | 0.28 | 0.75 | heterodyne luminance sine (beating carriers) |
| `slice` | `smoothstep(0.48,0.92,r) * 0.55` | 0.48 | 0.92 | slice-replace probability |
| `long_event` | `smoothstep(0.65,1.0,r) * 0.03` | 0.65 | 1.0 | rare multi-frame takeover probability/frame |
| `sync_false` | `smoothstep(0.7,1.0,r) * 18.0` px | 0.70 | 1.0 | false-sync row shifts |

Ordering matches the research: bands appear first (0.06-0.10), then tint
and fringe, then impulses, then slices, then false-sync geometry damage —
burst structure exists from near-zero severity, structural damage only when
severe.

### 6.2 Packet burst machine (state)

- Alternating ON/OFF counters: ON = 1-6 frames, extended by +2 frames with
  probability `eff_duty * 0.5` during heavy traffic; OFF gap = 1-9 frames
  (`1 + g^2 * 8`, biased short).
- **Traffic cluster envelope**: `eff_duty = duty * (0.35 + 1.3 *
  value_noise(frame*0.07))`, capped at **0.9** — sessions of higher/lower
  burst density.
- **Long takeover**: probability `long_event` per frame -> ON for 9-75
  frames (0.3-2.5 s), forces burst ON and raises the band floor; cooldown
  60-180 frames after.
- **Desense hangover**: when a burst ends, hangover set to `desense` then
  decays `*= 0.78` per frame (reaches ~0 in 30-150 ms at 30 fps) — extra
  fine snow while the receiver AGC recovers.

### 6.3 Visual layers (applied inside `apply_interference`)

- **Scrolling noise bands**: 1D value noise sampled along
  `(y - scroll) * band_count/height * 6`, thresholded at 0.58 -> discrete
  bars with hard-ish edges that translate vertically each frame. Inside a
  band: streaky horizontally-correlated noise (2 px x-correlation)
  replacement, plus optional per-slice **tint** (U +/- / V -/+ opposing
  deltas -> green or purple). Long takeovers raise a floor so nearly all
  rows are implicated.
- **Slice corruption**: rows grouped into 6-36-row slices; each slice is
  replaced (probability `slice * 0.30`) by either static (~75%) or an
  inverted-video strip (~25%), with sharp line boundaries — maps 1:1 to the
  ms-to-scanlines burst math above.
- **Heterodyne fringe**: two beating sines on luma
  (`sin(y*f_k + ph) + sin(y*f_k*1.13 - ph*0.83)`), `f_k` in 0.055-0.165
  rad/row — runs **continuously even between bursts** (models a narrowband
  CW-like interferer), producing diagonal/horizontal fringe crawl.
- **Impulses inside bursts**: probability `impulse * 0.10` per pixel ->
  hard black/white salt.
- **False-sync row shifts**: during bursts, 5-row blocks shift by up to
  `sync_false` px with probability proportional to `sync_false/18` —
  accumulated into `row_shift` and applied by the geometry pass (tearing /
  displacement from false slicer triggers).
