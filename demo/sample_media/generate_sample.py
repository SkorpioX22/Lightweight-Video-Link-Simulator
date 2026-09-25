"""Generate a synthetic FPV-style sample clip (original work, no copyright issues).

Renders a large procedural 'world' texture (sky, ground, buildings, trees,
poles, fences — strong horizontal/vertical structure, detailed textures)
and extracts a moving FPV-like camera crop with forward push + attitude
wobble. Output: demo/sample_media/sample.mp4 (mp4v) + sample_frame.png.

Usage:
    python demo/sample_media/generate_sample.py [--width 1280] [--height 720]
        [--seconds 8] [--fps 30]
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def build_world(w_world=3840, h_world=2160, seed=7):
    """Procedural outdoor FPV-ish panorama."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h_world, w_world, 3), np.float32)
    horizon = int(h_world * 0.42)

    # --- sky gradient + clouds ---
    sky_t = np.linspace(0, 1, horizon, dtype=np.float32)[:, None]
    img[:horizon, :, 0] = 0.35 + 0.25 * sky_t
    img[:horizon, :, 1] = 0.55 + 0.25 * sky_t
    img[:horizon, :, 2] = 0.85 + 0.10 * sky_t
    cloud = rng.random((horizon // 24 + 2, w_world // 48 + 2)).astype(np.float32)
    cloud = cv2.resize(cloud, (w_world, horizon), interpolation=cv2.INTER_LINEAR)
    cloud = cv2.GaussianBlur(cloud, (0, 0), 9)
    cl = np.clip((cloud - 0.55) * 3.0, 0, 1) * 0.55
    img[:horizon] += cl[..., None] * np.array([0.5, 0.5, 0.45], np.float32)

    # --- sun ---
    sx, sy = int(w_world * 0.78), int(horizon * 0.30)
    yy, xx = np.mgrid[0:horizon, 0:w_world]
    sun = np.exp(-(((xx - sx) ** 2 + (yy - sy) ** 2) / (110.0 ** 2)))
    img[:horizon] += sun[..., None] * np.array([0.7, 0.6, 0.3], np.float32)

    # --- ground: textured field with perspective-ish rows ---
    gh = h_world - horizon
    gt = np.linspace(0, 1, gh, dtype=np.float32)[:, None]
    gnoise = rng.random((gh // 8 + 2, w_world // 8 + 2)).astype(np.float32)
    gnoise = cv2.resize(gnoise, (w_world, gh), interpolation=cv2.INTER_LINEAR)
    gnoise = cv2.GaussianBlur(gnoise, (0, 0), 3)
    dirt = np.array([0.36, 0.34, 0.20], np.float32)
    grass = np.array([0.22, 0.42, 0.16], np.float32)
    base = grass * (1 - gt) + dirt * gt
    img[horizon:] = base[..., None, :].repeat(w_world, axis=1) if base.shape[1] == 1 else 0
    # fix: broadcast properly
    img[horizon:] = np.broadcast_to(base[:, None, :], (gh, w_world, 3)).copy()
    img[horizon:] *= (0.75 + 0.5 * gnoise)[..., None]
    # plow rows (horizontal structure)
    rows = (np.sin(np.arange(gh, dtype=np.float32) * (0.08 + gt[:, 0] * 0.5)[:, None]).mean(1))
    # simpler row striping
    row_stripe = 0.08 * np.sin(np.arange(gh, dtype=np.float32) * 0.15)[:, None]
    img[horizon:] += row_stripe[..., None]

    # road strip (strong horizontal + texture)
    rh = int(gh * 0.34)
    ry0 = horizon + int(gh * 0.55)
    road = np.array([0.30, 0.30, 0.32], np.float32)
    img[ry0:ry0 + rh, :] = road
    img[ry0:ry0 + rh, :] += (rng.random((rh, w_world, 3)).astype(np.float32) - 0.5) * 0.06
    # center line dashes (strong vertical-edges along x, high contrast)
    dash_w = 26
    for x in range(0, w_world, 140):
        img[ry0 + rh // 3: ry0 + 2 * rh // 3, x:x + dash_w, :] = np.array([0.85, 0.8, 0.2], np.float32)

    # --- buildings skyline above horizon (vertical edges, windows) ---
    x = 0
    while x < w_world:
        bw = int(rng.integers(160, 420))
        bh = int(rng.integers(160, 520))
        y1 = horizon
        y0 = max(0, y1 - bh)
        tone = rng.uniform(0.30, 0.65)
        bcol = np.array([tone, tone * 0.95, tone * 0.9], np.float32)
        img[y0:y1, x:min(w_world, x + bw)] = bcol
        # windows grid
        for wy in range(y0 + 18, y1 - 10, 36):
            for wx in range(x + 14, x + bw - 14, 30):
                if rng.random() < 0.75:
                    lit = rng.random() < 0.3
                    wcol = np.array([0.95, 0.9, 0.6], np.float32) if lit else np.array([0.12, 0.16, 0.24], np.float32)
                    img[wy:wy + 18, wx:wx + 16] = wcol
        # outline for crisp edges
        img[y0:y1, x:x + 2] *= 0.4
        img[y0:y1, min(w_world - 1, x + bw - 2):min(w_world, x + bw)] *= 0.4
        x += bw + int(rng.integers(10, 80))

    # --- trees (trunks = vertical edges, canopies = texture) ---
    for _ in range(40):
        tx = int(rng.integers(0, w_world - 8))
        th = int(rng.integers(80, 260))
        ty1 = horizon + int(rng.integers(0, 40))
        ty0 = ty1 - th
        trunk_w = int(rng.integers(6, 14))
        img[ty0 + th // 2:ty1, tx:tx + trunk_w, :] = np.array([0.25, 0.17, 0.10], np.float32)
        cr = int(th * 0.55)
        cy, cx = ty0 + th // 3, tx + trunk_w // 2
        yy, xx = np.ogrid[max(0, cy - cr):min(h_world, cy + cr), max(0, cx - cr):min(w_world, cx + cr)]
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= cr * cr
        region = img[max(0, cy - cr):max(0, cy - cr) + mask.shape[0], max(0, cx - cr):max(0, cx - cr) + mask.shape[1]]
        n = mask.shape
        tex = rng.random(n).astype(np.float32)
        leaf = np.array([0.10, 0.34, 0.10], np.float32) * (0.6 + 0.9 * tex)[..., None]
        region[mask] = leaf[mask]

    # --- power poles + wires (excellent for ghosting/tearing visibility) ---
    for px in range(200, w_world, 700):
        top = horizon - int(rng.integers(120, 220))
        img[top:horizon + 60, px:px + 10, :] = np.array([0.18, 0.14, 0.11], np.float32)
        arm = 70
        img[top + 20:top + 30, px - arm // 2:px + arm, :] = np.array([0.18, 0.14, 0.11], np.float32)
        # wires
        for wy, sag in [(top + 25, 40), (top + 55, 55)]:
            for x2 in range(px, min(w_world, px + 700)):
                t = (x2 - px) / 700.0
                yy = int(wy + sag * 4 * t * (1 - t))
                if 0 <= yy < h_world:
                    img[yy:yy + 2, x2] = np.array([0.1, 0.1, 0.1], np.float32)

    # --- chain-link fence band near bottom (fine texture / moire-friendly) ---
    fy0 = h_world - int(gh * 0.22)
    fence = img[fy0:fy0 + 90, :].copy()
    diag1 = (np.arange(w_world)[None, :] + np.arange(90)[:, None] * 2) % 22 < 2
    diag2 = (np.arange(w_world)[None, :] - np.arange(90)[:, None] * 2) % 22 < 2
    m = diag1 | diag2
    fence[m] = 0.75
    img[fy0:fy0 + 90, :][m] = 0.75

    # global fine detail noise (helps snow be visible against texture)
    img += (rng.random(img.shape[:2]).astype(np.float32) - 0.5)[..., None] * 0.03

    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def render_clip(out_path, width=1280, height=720, seconds=8.0, fps=30, seed=7):
    world = build_world(seed=seed)
    H, W = world.shape[:2]
    n_frames = int(seconds * fps)

    # crop path: slow yaw + forward push (scale in) + attitude wobble
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vw = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not vw.isOpened():
        raise RuntimeError("VideoWriter failed to open")

    base_crop_w = int(W * 0.42)
    frames = []
    for f in range(n_frames):
        t = f / fps
        # forward push: crop shrinks ~8% over clip
        scale = 1.0 - 0.08 * (f / max(1, n_frames - 1))
        cw = int(base_crop_w * scale)
        ch = int(cw * height / width)
        # yaw drift left-right
        cx = int(W * 0.5 + np.sin(t * 0.45) * W * 0.16 + np.sin(t * 0.11) * W * 0.05)
        # altitude bob
        cy = int(H * 0.45 + np.sin(t * 0.7) * H * 0.05 - (1 - scale) * H * 0.1)
        x0 = np.clip(cx - cw // 2, 0, W - cw)
        y0 = np.clip(cy - ch // 2, 0, H - ch)
        crop = world[y0:y0 + ch, x0:x0 + cw]
        frame = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
        # slight FPV attitude roll wobble
        ang = np.sin(t * 1.3) * 1.6 + np.sin(t * 0.4) * 1.0
        M = cv2.getRotationMatrix2D((width / 2, height / 2), ang, 1.0)
        frame = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        # mild motion blur along direction of travel (keeps sim feel)
        frame = cv2.GaussianBlur(frame, (3, 3), 0.6)
        vw.write(frame)
        if f == 0:
            frames.append(frame.copy())
    vw.release()

    # also write a PNG still
    png_path = os.path.join(HERE, "sample_frame.png")
    cv2.imwrite(png_path, frames[0])
    return out_path, png_path, n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(HERE, "sample.mp4"))
    args = ap.parse_args()
    out, png, n = render_clip(args.out, args.width, args.height, args.seconds, args.fps)
    print(f"wrote {out} ({n} frames) and {png}")


if __name__ == "__main__":
    main()
