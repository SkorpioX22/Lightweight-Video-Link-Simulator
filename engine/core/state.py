"""Temporal engine state.

State is kept as small float32 arrays so Numba kernels can mutate them
in place. Layout documented per array below. reset_state() clears all.
"""
import numpy as np

# signal state (float32[16]) — indices mirror signal/weak_signal.py
SIG_ROLL_ACTIVE = 0     # 1 while rolling
SIG_ROLL_POS = 1        # accumulated vertical offset (rows)
SIG_ROLL_TIMER = 2      # frames remaining in roll event
SIG_ROLL_COOLDOWN = 3   # frames until next roll can start
SIG_ROLL_DIR = 4        # +1 / -1
SIG_KILLER = 5          # 1 = color killer latched on
SIG_HUE = 6             # slow hue wander phase
SIG_NOISE_ENV = 7       # slow snow envelope (fast fades)
SIG_RESERVED = 8        # unused

# multipath state (float32[16]) — indices mirror multipath/multipath.py
MP_A1 = 0               # ghost1 amplitude (signed, flickers fast)
MP_A2 = 1
MP_A3 = 2
MP_D1 = 3               # ghost1 delay px (slow drift)
MP_D2 = 4
MP_D3 = 5
MP_PHI = 6              # slow phase / ripple
MP_NULL_ACTIVE = 7      # 1 = deep null dropout
MP_NULL_TIMER = 8       # frames remaining in null
MP_FLASH = 9            # 1 = white AGC flash this frame
MP_BAND_PHI = 10        # white band ripple phase
MP_NULL_CD = 11         # frames until next null allowed
MP_FLASH_CD = 12        # frames until next flash allowed

# interference state (float32[16]) — indices mirror interference/interference.py
INT_BURST = 0           # 1 = burst ON at frame start
INT_TIMER = 1           # frames remaining in current burst/gap
INT_SCROLL = 2          # band scroll phase (lines)
INT_FRINGE = 3          # fringe phase
INT_HANG = 4            # desense hangover 1..0
INT_LONG = 5            # 1 = long takeover event active
INT_LONG_TIMER = 6      # frames remaining in takeover
INT_CLUSTER = 7         # slow traffic-session phase
INT_LONG_CD = 8         # frames until next takeover allowed


def reset_state():
    return {
        "signal": np.zeros(16, dtype=np.float32),
        "multipath": np.zeros(16, dtype=np.float32),
        "interference": np.zeros(16, dtype=np.float32),
        # multipath delays start at sane defaults so first frame is meaningful
    }


def init_multipath_defaults(mp, width):
    mp[MP_D1] = 8.0
    mp[MP_D2] = 20.0
    mp[MP_D3] = 40.0
    mp[MP_NULL_CD] = 0.0
    mp[MP_FLASH_CD] = 0.0
