"""Render the visual test matrix (docs/qa) from the sample clip.

Outputs:
  tests/qa/matrix/<label>.png     - single degraded frame per cell
  tests/qa/matrix/grid.png        - overview grid
  tests/qa/videos/<label>.mp4     - short clips for temporal QA (subset)

Usage: python tests/render_matrix.py [--frames N]
"""
import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import AnalogVideoBreakupEngine, PRESETS  # noqa: E402

SAMPLE = os.path.join(ROOT, "demo", "sample_media", "sample.mp4")
OUT_PNG = os.path.join(ROOT, "tests", "qa", "matrix")
OUT_VID = os.path.join(ROOT, "tests", "qa", "videos")

# (label, signal, multipath, interference)
MATRIX = [
    ("clean_000", 0, 0, 0),
    ("sig_75", 75, 0, 0),
    ("sig_50", 50, 0, 0),
    ("sig_25", 25, 0, 0),
    ("sig_05", 5, 0, 0),
    ("sig_99", 99, 0, 0),
    ("mp_99", 0, 99, 0),
    ("mp_75", 0, 75, 0),
    ("mp_50", 0, 50, 0),
    ("mp_25", 0, 25, 0),
    ("int_99", 0, 0, 99),
    ("int_75", 0, 0, 75),
    ("int_50", 0, 0, 50),
    ("int_25", 0, 0, 25),
    ("mp25_at_sig99", 99, 25, 0),
    ("mp50_at_sig99", 99, 50, 0),
    ("mp75_at_sig99", 99, 75, 0),
    ("mp99_at_sig99", 99, 99, 0),
    ("int25_at_sig99", 99, 0, 25),
    ("int50_at_sig99", 99, 0, 50),
    ("int75_at_sig99", 99, 0, 75),
    ("int99_at_sig99", 99, 0, 99),
    ("mix_50_50_50", 50, 50, 50),
    ("mix_25_75_50", 25, 75, 50),
    ("mix_20_20_90", 20, 20, 90),
    ("all_max_99", 99, 99, 99),
    ("preset_long_range", *(_ := [PRESETS["LONG RANGE"][k] for k in ("signalStrength", "multipath", "rfInterference")])),
    ("preset_heavy_mp", *[PRESETS["HEAVY MULTIPATH"][k] for k in ("signalStrength", "multipath", "rfInterference")]),
    ("preset_wifi", *[PRESETS["WIFI INTERFERENCE"][k] for k in ("signalStrength", "multipath", "rfInterference")]),
    ("preset_near_loss", *[PRESETS["NEAR VIDEO LOSS"][k] for k in ("signalStrength", "multipath", "rfInterference")]),
]

# subset rendered as temporal clips
CLIPS = ["clean_000", "sig_75", "sig_99", "mp_75", "mp_99", "int_75", "int_99",
         "mix_50_50_50", "all_max_99", "preset_wifi"]


def read_frame(cap, n=0):
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, fr = cap.read()
    if not ok:
        raise RuntimeError("cannot read sample frame")
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=60, help="sample frame used for stills")
    ap.add_argument("--clip-frames", type=int, default=75, help="frames per QA clip")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-clips", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_PNG, exist_ok=True)
    cap = cv2.VideoCapture(SAMPLE)
    if not cap.isOpened():
        raise SystemExit("sample.mp4 missing — run demo/sample_media/generate_sample.py")
    frame0 = read_frame(cap, args.frame)

    os.makedirs(OUT_PNG, exist_ok=True)
    results = []
    for label, s, m, i in MATRIX:
        eng = AnalogVideoBreakupEngine(seed=42)
        # warm temporal state so events (roll/burst/null) can occur:
        # feed the same frame until the target frame index
        for _ in range(min(args.frame, 90)):
            out = eng.process(frame0, {"signalStrength": s, "multipath": m, "rfInterference": i})
        # composite: original on top, processed below, with labels
        comp = np.zeros((frame0.shape[0] * 2 + 24, frame0.shape[1], 3), np.uint8)
        comp[: frame0.shape[0]] = frame0
        comp[frame0.shape[0] + 24: frame0.shape[0] * 2 + 24] = out
        cv2.putText(comp, f"ORIGINAL  |  {label}  S={s} M={m} I={i}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(comp, "PROCESSED", (8, frame0.shape[0] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        path = os.path.join(OUT_PNG, f"{label}.png")
        cv2.imwrite(path, comp)
        results.append((label, path, out))
        print(f"  {label}: mean={out.mean():.1f} std={out.std():.1f}")

    # overview grid: processed halves only, 5 cols
    thumbs = []
    tw, th = 256, 144
    for label, _, out in results:
        t = cv2.resize(out, (tw, th))
        cv2.putText(t, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(t)
    cols = 5
    rows = (len(thumbs) + cols - 1) // cols
    grid = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for idx, t in enumerate(thumbs):
        r, c = divmod(idx, cols)
        grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    cv2.imwrite(os.path.join(OUT_PNG, "grid.png"), grid)
    print(f"grid -> {os.path.join(OUT_PNG, 'grid.png')}")

    if not args.no_clips:
        os.makedirs(OUT_VID, exist_ok=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames = []
        for _ in range(args.clip_frames):
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        for label, s, m, i in [(x[0], x[1], x[2], x[3]) for x in MATRIX if x[0] in CLIPS]:
            eng = AnalogVideoBreakupEngine(seed=42)
            vw = cv2.VideoWriter(os.path.join(OUT_VID, f"{label}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                 (frames[0].shape[1], frames[0].shape[0]))
            for fr in frames:
                out = eng.process(fr, {"signalStrength": s, "multipath": m, "rfInterference": i})
                vw.write(out)
            vw.release()
            print(f"  clip {label}.mp4")
    cap.release()
    print("done")


if __name__ == "__main__":
    main()
