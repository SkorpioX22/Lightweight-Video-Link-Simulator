# GLSL port — `fpv_breakup.frag`

Single-pass GLSL 330 core fragment shader port of `AnalogVideoBreakupEngine` v1.0.

## Interface (desktop GL / three.js / raw OpenGL)

| Uniform | Type | Meaning |
|---|---|---|
| `uSrc` | `sampler2D` | clean RGB frame (the simulator's rendered output) |
| `uResolution` | `vec2` | output size in pixels |
| `uTime` | `float` | reserved (all animation is frame-driven) |
| `uFrame` | `uint` | frame counter — drives every temporal machine |
| `uSeed` | `uint` | deterministic seed |
| `uSignalStrength` | `float` | 0..99 (0 = no effect) |
| `uMultipath` | `float` | 0..99 |
| `uRFInterference` | `float` | 0..99 |

Output: `fragColor` (RGBA). All-zero params → bit-identical passthrough of `uSrc`.

## Pipeline (identical order to the Python engine)

1. inverse geometry (vertical roll + per-row horizontal shift + seam noise)
2. multipath — echo taps sampled from **clean** source, smear blend (weight ≤ 0.65 so echoes survive), edge spikes, ripple bands, short mono null windows, white flash
3. weak signal — mono-correct snow (2 px x-correlated, flow+grain 50/50), dark-weighted chroma blotches, sat collapse → color killer, hue wander, dash sparklies, tear/roll accumulation
4. interference — burst-gated threshold bands with 8 px streaks + green/purple tint, slice corruption, beating fringe (continuous between bursts), impulses, desense hangover, false-sync shifts
5. YUV → RGB + TPDF dither

All severity curves are copied verbatim from `engine/core/params.py`.

## Determinism / stateless machines

Output is a pure function of `(uSrc, uFrame, uSeed, params)` — no feedback FBO.
Python's temporal state machines are approximated as block-hashed windows:

| Python state machine | GLSL approximation |
|---|---|
| null/flash cooldown timers | 64/8-frame blocks, hashed onset+length |
| roll event (6–60 f + cooldown) | 48-frame blocks, hashed window, duty ≈ Python's |
| burst/gap packet machine | threshold on value noise at ~3-frame correlation, duty-scaled by traffic cluster |
| desense hangover decay | 8-frame scan-back after burst end, ×0.78/frame |
| fringe/scroll phase accumulators | closed-form `frame × rate` |

The color-killer uses a plain threshold instead of hysteresis; `sat_loss` is
monotonic in the parameter, so for constant params the visible behavior matches.

## Verification status

No GL context or `glslangValidator` is available on this machine, so the shader
was verified by a static checker: balanced delimiters, no `texture2D`/
`gl_FragColor`, every uniform used, every called function defined before use,
no bare-int literals in float contexts, `#version` first, `#ifdef`/`#endif`
balanced, all `params.py` stage constants present. **Compile it on your target
GPU before shipping** — and if a driver rejects something, the fixes belong here.

Known portability notes:
- `wrap_int()` emulates two's-complement `uint()` wrap for negative noise
  lattice coordinates (GLSL leaves `uint(negative)` implementation-defined;
  Python's `np.uint32` wraps).
- uint underflow guarded in the hangover scan-back (frame 0).

## Shadertoy

1. Delete the `#version 330 core` line (Shadertoy supplies its preamble).
2. Compile with `-DSHADERTOY` (or paste the body and adapt).
3. Bind the clean frame to `iChannel0`.
4. Set the three `FPV_*_DEFAULT` defines (or write the globals in `mainImage`).

## Differences vs Python (deliberate)

- Stateless temporal machines (table above) — statistically equivalent, not
  sample-identical frame-by-frame.
- Killer hysteresis → threshold (identical for constant params).
- One-pass geometry: echo taps and row shifts are applied in the same shader
  invocation; Python runs multipath before geometry in separate buffers.
- Added per-pixel dither (Python relies on the float pipeline + uint8 rounding).
