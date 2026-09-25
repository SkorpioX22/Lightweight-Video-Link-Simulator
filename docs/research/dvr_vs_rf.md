# Research: DVR vs RF — What the Engine Emulates (and What It Does Not)

An artifact seen in an FPV recording can come from the **RF chain**
(transmitter -> air -> receiver -> analog decode) or from the **recorder**
(DVR / goggles DVR / capture device). The engine models the RF chain only.
This document records the classification rule and the scope decision.

## 1. DVR-independent RF artifacts

These live in the demodulated composite signal itself. They are visible
**live** and in **recordings on every device**, because any recorder fed
the same CVBS will show them:

- snow (additive noise)
- ghosting (multipath echoes)
- tearing (H-sync slicer failures)
- displacement / shifted rows (false sync / AFC loss)
- vertical roll (V-sync integrator failure)
- hue shift and color dropout / color-killer (chroma SNR collapse)
- dot crawl (chroma-luma crosstalk)

**Rule of thumb:** additive speckle + geometric tear/roll + color-kill =
RF chain -> the engine reproduces these.

## 2. Receiver-profile layer (partially modeled)

Two different things get called "sync":

- **Signal-level sync integrity** — whether the composite waveform still
  contains usable H/V tips (what the RF chain degrades). This is what the
  engine models.
- **Display PLL behavior** — how a particular receiver reacts when sync
  fails. Modern sync-reconstruction firmware (RapidFire, Fusion,
  SteadyView) converts roll into hold / scroll / split-screen artifacts
  instead of a free-running roll; diversity receivers can tear at the
  moment of antenna switching.

The engine's default profile is a **plain-CRT-like response**: on vertical
sync loss the image free-rolls (with snap re-lock on recovery). A
sync-reconstruction profile is a possible future option, not present in v1.

## 3. DVR-specific artifacts (deliberately NOT emulated)

Recorder-chain effects — they depend on encoder firmware, not on the RF
link, and are out of scope as a primary visual language:

| Artifact | Source |
|---|---|
| Encoder blockiness on noise (MJPEG vs H.264 differ markedly) | recorder codec |
| Frame drops / freezes | recorder throughput |
| PAL/NTSC re-lock black flashes (e.g. Fatshark firmware behavior) | recorder re-sync |
| B&W-recording bugs (records monochrome while live view is color) | recorder bug |
| Blue screens / "VIDEO LOSS" banners | recorder input detection |
| OSD chrome (overlay graphics burned in by the goggle/DVR) | recorder/goggle UI |
| Deinterlace ghosting in the saved file | recorder post-processing |

**Rule of thumb:** blue screens / freezes / macroblocks = recorder chain ->
not the engine.

## 4. Scope summary

```
                 RF / analog chain                    Recorder chain
             +--------------------------+       +------------------------+
air/link --> | snow, ghosts, tearing,   | -->   | blockiness, frame      |
noise        | displacement, roll,      |       | drops, blue screens,   |
multipath    | hue shift, color kill,   |       | OSD, B&W bugs,         |
interf.      | dot crawl                |       | deinterlace ghosting   |
             +--------------------------+       +------------------------+
                  ^ engine lives here               NOT emulated
```

Engine architecture = a single RF-to-analog corruption model applied to the
clean rendered frame. A "DVR skin" (codec artifacts, freezes, banners)
could be added later as a thin optional post-process layer without touching
the RF model.
