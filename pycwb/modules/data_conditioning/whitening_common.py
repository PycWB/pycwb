"""Shared cWB frequency-bin masking for whitening-noise maps.

Wavelet and MESA noise estimators remain separate; this helper only replaces
out-of-band rows with the requested constant, preserving cWB bin boundaries.
"""

import numpy as np

def _apply_cwb_bandpass_constant(nrms_map, f1, f2, a, df, f_low_map, f_high_map):
    """
    Mirror cWB `WSeries::bandpass(f1, f2, a)` behavior on TF rows.

    For `bandpass(16., 0., 1.)`, rows below the low edge are set to 1.
    """
    if nrms_map.ndim != 2:
        raise ValueError("Expected 2D RMS map")

    out = np.array(nrms_map, copy=True)
    n_freq = out.shape[0]
    if n_freq == 0:
        return out

    dF = float(df)
    fl = abs(float(f1)) if abs(float(f1)) > 0.0 else float(f_low_map)
    fh = abs(float(f2)) if abs(float(f2)) > 0.0 else float(f_high_map)

    n = int((fl + dF / 2.0) / dF + 0.1)
    m = int((fh + dF / 2.0) / dF + 0.1) - 1

    if n > m:
        return out

    n = max(0, min(n, n_freq - 1))
    m = max(0, min(m, n_freq - 1))

    indices = np.arange(n_freq)

    keep = np.zeros(n_freq, dtype=bool)
    if f1 >= 0 and f2 >= 0:
        keep = (indices > n) & (indices <= m)
    elif f1 < 0 and f2 < 0:
        keep = (indices < n) | (indices > m)
    elif f1 < 0 and f2 >= 0:
        keep = indices < n
    elif f1 >= 0 and f2 < 0:
        keep = indices >= m

    out[~keep, :] = float(a)
    return out
