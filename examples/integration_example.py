"""Simulator-loop integration example for LVLS — Lightweight Video Link Simulation.

Shows the three ways an FPV simulator typically drives the engine:

  1. constant link quality (pilot flying a fixed-distance course)
  2. distance-driven link (quality degrades as the quad flies away)
  3. scenario keyframes (scripted race: clean start, multipath in a
     warehouse gap, interference over the WiFi zone)

Run from the project root:

    python examples/integration_example.py

Outputs a few example frames to examples/output/ (requires opencv-python).
The engine itself only needs numpy + numba.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from engine import PRESETS, AnalogVideoBreakupEngine  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def make_test_frame(w=1280, h=720):
    """Synthetic clean frame (sky / horizon / ground) so the example runs
    without opencv. Replace with your simulator's rendered RGB buffer."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    frame = np.empty((h, w, 3), dtype=np.uint8)
    horizon = h * 0.55
    horizon_i = int(horizon)
    sky = yy < horizon
    frame[..., 0] = np.where(sky, 90 + 60 * (1 - yy / horizon), 70)
    frame[..., 1] = np.where(sky, 140 + 50 * (1 - yy / horizon), 110)
    frame[..., 2] = np.where(sky, 200, 60)
    # vertical-edge "buildings" so multipath ghosts are visible
    for x0 in range(80, w, 160):
        top = int(horizon - 120 - 40 * ((x0 // 160) % 3))
        frame[top:horizon_i, x0:x0 + 70, :] = (55, 55, 65)
        for wy in range(top + 20, horizon_i - 20, 36):
            for wx in range(x0 + 8, x0 + 62, 20):
                frame[wy:wy + 16, wx:wx + 12, :] = (220, 200, 120)
    # horizon line / road
    frame[horizon_i:, :, :] = (80, 120, 70)
    frame[horizon_i + 4:horizon_i + 8, :, :] = (160, 160, 160)
    return frame


def example_constant_quality(eng, frame):
    """1. Fixed link quality — pass the same params every frame."""
    print("[1] constant quality:", PRESETS["WIFI INTERFERENCE"])
    for _ in range(3):
        out = eng.process(frame, PRESETS["WIFI INTERFERENCE"])
    return out


def example_distance_driven(eng, frame, start_m=20, end_m=450, frames=60):
    """2. Distance-driven link: map range to the three parameters.

    A typical 25 mW VTX on 5.8 GHz: close = clean, far = snow creeping in,
    and a reflective structure mid-range adding multipath.
    """
    print(f"[2] distance-driven: {start_m}m -> {end_m}m over {frames} frames")
    out = None
    for k in range(frames):
        t = k / max(frames - 1, 1)
        dist = start_m + (end_m - start_m) * t
        # link budget heuristic: SNR falls roughly with distance^2
        signal = int(np.clip((dist - 40) / 4.2, 0, 99))          # 0 at 40m
        multipath = int(np.clip(70 - abs(dist - 180) / 3.0, 0, 70))  # peak near 180m
        interference = 6 if dist < 300 else 18                    # urban noise floor
        out = eng.process(frame, {
            "signalStrength": signal,
            "multipath": multipath,
            "rfInterference": interference,
        })
    print(f"    final params: signal={signal} multipath={multipath} "
          f"interference={interference}")
    return out


def example_scenario_keyframes(eng, frame):
    """3. Scripted race segments via keyframe interpolation."""
    print("[3] scenario keyframes: clean -> warehouse -> WiFi zone")
    # (frame_at_segment_start, params)
    keyframes = [
        (0,  {"signalStrength": 0,  "multipath": 0,  "rfInterference": 0}),
        (20, {"signalStrength": 10, "multipath": 78, "rfInterference": 5}),
        (40, {"signalStrength": 8,  "multipath": 15, "rfInterference": 72}),
        (60, {"signalStrength": 88, "multipath": 55, "rfInterference": 70}),
    ]

    def params_at(frame_idx):
        # step hold up to the next keyframe (use lerp for smoother transitions)
        active = keyframes[0][1]
        for at, p in keyframes:
            if frame_idx >= at:
                active = p
        return active

    out = None
    for k in range(80):
        out = eng.process(frame, params_at(k))
    return out


def example_reset_and_seed():
    """Determinism: same seed + same call sequence => identical frames."""
    frame = make_test_frame(320, 180)
    a = AnalogVideoBreakupEngine(seed=7)
    b = AnalogVideoBreakupEngine(seed=7)
    params = {"signalStrength": 55, "multipath": 40, "rfInterference": 30}
    fa = a.process(frame, params)
    fb = b.process(frame, params)
    assert np.array_equal(fa, fb), "determinism broken"
    # and the all-zero fast path is bit-identical to the input
    clean = a.process(frame, {"signalStrength": 0, "multipath": 0,
                              "rfInterference": 0})
    assert np.array_equal(clean, frame), "identity fast path broken"
    print("[4] determinism + identity checks: OK")


def try_save(name, frame_bgr):
    try:
        import cv2
    except ImportError:
        print(f"    (opencv not installed — skipping save of {name})")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    cv2.imwrite(path, cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR))
    print(f"    saved {path}")


def main():
    frame = make_test_frame()
    eng = AnalogVideoBreakupEngine(seed=42)

    try_save("clean.png", frame)
    try_save("constant_wifi.png", example_constant_quality(eng, frame))
    try_save("distance_end.png", example_distance_driven(eng, frame))
    try_save("scenario_end.png", example_scenario_keyframes(eng, frame))
    example_reset_and_seed()

    print("\nIntegration tips:")
    print("  - call eng.process() once per rendered simulator frame, in order")
    print("  - keep one engine instance per session so temporal machines")
    print("    (roll events, bursts, nulls) advance naturally")
    print("  - call eng.reset() when restarting a flight")
    print("  - pass advance=False to re-process the same frame idempotently")
    print("  - see docs/integration/INTEGRATION.md for the full guide")


if __name__ == "__main__":
    main()
