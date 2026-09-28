"""HDZero temporal state (float32[16]).

Indices mirror hdzero/kernels.py. reset_hz_state() clears all.
"""
import numpy as np

# interference event machine
HZ_EVT_ACTIVE = 0     # 1 = takeover event running
HZ_EVT_TIMER = 1      # frames remaining
HZ_EVT_TYPE = 2       # 0 band / 1 rainbow / 2 lines / 3 black (see kernels)
HZ_EVT_CD = 3         # frames until next event allowed
HZ_EVT_Y0 = 4         # band start, normalized [0,1)
HZ_EVT_H = 5          # band height fraction
# multipath cluster field
HZ_DRIFT = 6          # cluster-field drift phase
HZ_ENV = 7            # cluster burst envelope (0..1)
# signal-loss latch (digital "rainbow screen of death")
HZ_LOSS = 8           # frames remaining in signal-loss state
HZ_LOSS_T = 9         # frames elapsed in the current loss run (patch ramp)
HZ_LOSS_SEED = 10     # run id -> stable random patch layout within a run


def reset_hz_state():
    return np.zeros(16, dtype=np.float32)


__all__ = [
    "HZ_EVT_ACTIVE", "HZ_EVT_TIMER", "HZ_EVT_TYPE", "HZ_EVT_CD",
    "HZ_EVT_Y0", "HZ_EVT_H", "HZ_DRIFT", "HZ_ENV", "HZ_LOSS",
    "HZ_LOSS_T", "HZ_LOSS_SEED",
    "reset_hz_state",
]
