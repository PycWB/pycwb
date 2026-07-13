"""Packet-energy parameter helpers shared by max-energy backends."""

from __future__ import annotations

import numpy as np


def _compute_packet_energy_params(M, T, pattern, edge, wavelet_rate, f_low, f_high, df):
    """Pre-compute bounds and neighbor offset table for _wdm_packet_energy_nb."""
    J = M * T
    jb = int(edge * float(wavelet_rate) / 4.0) * M
    jb = max(jb, 4 * M)
    je = J - jb
    jb = max(jb, 0)
    je = min(je, J)

    mL = int(f_low / df + 0.1)
    mH = int(f_high / df + 0.1)
    mL = max(mL, 0)
    mH = min(mH, M - 1)

    pattern = abs(int(pattern))
    if pattern in (1, 3, 4):
        mean = 3.0
        mL += 1
        mH -= 1
    elif pattern == 2:
        mean = 3.0
    elif pattern in (5, 6):
        mean = 5.0
        mL += 2
        mH -= 2
    elif pattern in (7, 8):
        mean = 5.0
        mL += 1
        mH -= 1
    elif pattern == 9:
        mean = 9.0
        mL += 1
        mH -= 1
    else:
        mean = 1.0

    p = np.zeros(9, dtype=np.int64)
    if pattern == 1:
        p[1], p[2] = 1, -1
    elif pattern == 2:
        p[1], p[2] = M, -M
    elif pattern == 3:
        p[1], p[2] = M + 1, -M - 1
    elif pattern == 4:
        p[1], p[2] = -M + 1, M - 1
    elif pattern == 5:
        p[1], p[2], p[3], p[4] = M + 1, -M - 1, 2 * M + 2, -2 * M - 2
    elif pattern == 6:
        p[1], p[2], p[3], p[4] = -M + 1, M - 1, -2 * M + 2, 2 * M - 2
    elif pattern == 7:
        p[1], p[2], p[3], p[4] = 1, -1, M, -M
    elif pattern == 8:
        p[1], p[2], p[3], p[4] = M + 1, -M + 1, M - 1, -M - 1
    elif pattern == 9:
        p[1:9] = [1, -1, M, -M, M + 1, M - 1, -M + 1, -M - 1]

    return jb, je, mL, mH, mean, p
