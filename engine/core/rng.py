"""Counter-based deterministic hashing / value noise (Numba-compatible).

All randomness in the engine derives from (seed, frame_index, coordinates)
so that process(frame, params) is reproducible without hidden PRNG state.
Integer hashes only -> stable across platforms.
"""
import numpy as np
from numba import njit

U32 = np.uint32
U64 = np.uint64


@njit(inline="always")
def hash_u32(x, seed):
    # explicit 64-bit workspace: Numba may promote uint32 multiplies to
    # int64 (no wrap), so mask back to 32 bits after every product.
    v = np.uint64(np.uint32(x)) ^ np.uint64(np.uint32(seed))
    v ^= v >> np.uint64(16)
    v = (v * np.uint64(0x7FEB352D)) & np.uint64(0xFFFFFFFF)
    v ^= v >> np.uint64(15)
    v = (v * np.uint64(0x846CA68B)) & np.uint64(0xFFFFFFFF)
    v ^= v >> np.uint64(16)
    return np.uint32(v & np.uint64(0xFFFFFFFF))


@njit(inline="always")
def hash2_u32(x, y, seed):
    xm = np.uint64(np.uint32(x)) * np.uint64(0x9E3779B1)
    return hash_u32(np.uint32(xm & np.uint64(0xFFFFFFFF)) ^ np.uint32(y), seed)


@njit(inline="always")
def hash3_u32(x, y, z, seed):
    xm = np.uint64(np.uint32(x)) * np.uint64(0x9E3779B1)
    ym = np.uint64(np.uint32(y)) * np.uint64(0x85EBCA77)
    mixed = np.uint32((xm & np.uint64(0xFFFFFFFF)) ^ (ym & np.uint64(0xFFFFFFFF)) ^ np.uint32(z))
    return hash_u32(mixed, seed)


@njit(inline="always")
def hash01(x, seed):
    return hash_u32(x, seed) * np.float32(1.0 / 4294967296.0)


@njit(inline="always")
def hash2_01(x, y, seed):
    return hash2_u32(x, y, seed) * np.float32(1.0 / 4294967296.0)


@njit(inline="always")
def hash3_01(x, y, z, seed):
    return hash3_u32(x, y, z, seed) * np.float32(1.0 / 4294967296.0)


@njit(inline="always")
def hash_c1(x, seed):
    """Hash to [-1, 1)."""
    return hash01(x, seed) * np.float32(2.0) - np.float32(1.0)


@njit(inline="always")
def hash2_c1(x, y, seed):
    return hash2_01(x, y, seed) * np.float32(2.0) - np.float32(1.0)


@njit(inline="always")
def _smooth(t):
    return t * t * (np.float32(3.0) - np.float32(2.0) * t)


@njit(inline="always")
def value_noise1f(x, seed):
    """1D value noise, output ~[0,1], C1 continuous."""
    xi = np.int32(np.floor(x))
    xf = x - np.float32(xi)
    a = hash2_01(xi, 0, seed)
    b = hash2_01(xi + 1, 0, seed)
    t = _smooth(xf)
    return a + (b - a) * t


@njit(inline="always")
def value_noise1c(x, seed):
    """1D value noise, output ~[-1,1]."""
    return value_noise1f(x, seed) * np.float32(2.0) - np.float32(1.0)


@njit(inline="always")
def value_noise2f(x, y, seed):
    """2D value noise, output ~[0,1]."""
    xi = np.int32(np.floor(x))
    yi = np.int32(np.floor(y))
    xf = x - np.float32(xi)
    yf = y - np.float32(yi)
    a = hash2_01(xi, yi, seed)
    b = hash2_01(xi + 1, yi, seed)
    c = hash2_01(xi, yi + 1, seed)
    d = hash2_01(xi + 1, yi + 1, seed)
    t = _smooth(xf)
    u = _smooth(yf)
    ab = a + (b - a) * t
    cd = c + (d - c) * t
    return ab + (cd - ab) * u


@njit(inline="always")
def fbm1f(x, seed, octaves=3):
    """Cheap 1D fbm in [0,1]-ish, octave weights 1, .5, .25 (pink-ish)."""
    total = np.float32(0.0)
    amp = np.float32(1.0)
    norm = np.float32(0.0)
    fx = x
    for _ in range(octaves):
        total += amp * value_noise1f(fx, seed)
        norm += amp
        amp *= np.float32(0.5)
        fx *= np.float32(2.0)
        seed = hash_u32(seed, 0x9E3779B9)
    return total / norm


@njit(inline="always")
def gauss_approx(r1, r2):
    """Two uniform [0,1) -> approx normal via Box-Muller, return first sample scaled."""
    r1 = max(np.float32(r1), np.float32(1e-6))
    z = np.sqrt(-np.float32(2.0) * np.log(r1)) * np.cos(np.float32(6.2831853) * r2)
    return z


@njit(inline="always")
def frame_seed(seed, frame_index):
    """Master per-frame seed."""
    return hash_u32(U32(frame_index), U32(seed) ^ U32(0xA341316C))


@njit(inline="always")
def effect_seed(fseed, effect_id):
    return hash_u32(U32(effect_id), fseed)
