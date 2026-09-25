"""RGB <-> planar YUV (BT.601 full-range) conversions, Numba kernels.

Input side reads uint8 RGB directly (scale 1/255 fused in the kernel).
Output side writes uint8 RGB directly (clamp + *255 fused in the kernel),
which avoids two full-frame float temporaries (~6 ms/frame at 720p).

Y in [0,1], U/V in [-0.5, 0.5] roughly (chroma centered at 0).
Full-range (JPEG-style) matrix keeps 0-effect round-trip near-lossless
when combined with float32 pipeline.
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def rgb_to_yuv_planar(rgb, y, u, v):
    h, w = y.shape
    s = 1.0 / 255.0
    for j in prange(h):
        for i in range(w):
            r = rgb[j, i, 0] * s
            g = rgb[j, i, 1] * s
            b = rgb[j, i, 2] * s
            yy = 0.299 * r + 0.587 * g + 0.114 * b
            uu = -0.168736 * r - 0.331264 * g + 0.5 * b
            vv = 0.5 * r - 0.418688 * g - 0.081312 * b
            y[j, i] = yy
            u[j, i] = uu
            v[j, i] = vv


@njit(parallel=True, fastmath=True, cache=True)
def yuv_planar_to_rgb(y, u, v, rgb):
    """Write clamped uint8 RGB (H,W,3) directly from planar float YUV."""
    h, w = y.shape
    for j in prange(h):
        for i in range(w):
            yy = y[j, i]
            uu = u[j, i]
            vv = v[j, i]
            r = (yy + 1.402 * vv) * 255.0
            g = (yy - 0.344136 * uu - 0.714136 * vv) * 255.0
            b = (yy + 1.772 * uu) * 255.0
            if r < 0.0:
                r = 0.0
            elif r > 255.0:
                r = 255.0
            if g < 0.0:
                g = 0.0
            elif g > 255.0:
                g = 255.0
            if b < 0.0:
                b = 0.0
            elif b > 255.0:
                b = 255.0
            rgb[j, i, 0] = r
            rgb[j, i, 1] = g
            rgb[j, i, 2] = b
