"""Shared native-order host sampling and finalization for CPU and CUDA chirp scoring."""

from __future__ import annotations

import math

import numpy as np
from numba import njit

TRIALS = 1000
"""Number of bootstrap trials; fixed by the release algorithm."""

PICKS_PER_TRIAL = 6
"""Micropixels drawn per trial by the release algorithm."""


@njit(cache=True)
def prepare_bootstrap(
    x: np.ndarray, f: np.ndarray, energy: np.ndarray, mindt: float, uniforms: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
    """Preserve native sampling order; emit independent trial slopes and times.

    Parameters
    ----------
    x, f, energy : numpy.ndarray
        Float64 micropixel times, frequencies and energies, equal length ``n``.
    mindt : float
        Minimum time resolution; unused here, kept for signature symmetry
        with the native bootstrap.
    uniforms : numpy.ndarray
        Float64 uniform stream consumed sequentially in native order.

    Returns
    -------
    tuple
        ``(y, weights, slopes, mergers, valid, ellipticity, cursor, ready)``:
        ``y`` and ``weights`` are float64 arrays of length ``n``; ``slopes``
        and ``mergers`` are float64 arrays of length :data:`TRIALS`; ``valid``
        is a boolean array of length :data:`TRIALS` marking trials with a
        well-defined fit; ``cursor`` is the number of uniforms consumed and
        ``ready`` is ``False`` when the stream ran out before all trials were
        sampled (the caller then returns NaN and retries with a longer stream).
    """
    n = len(x)
    y = np.empty(n)
    weights = np.empty(n)
    cdf = np.zeros(n + 1)
    sx = sy = 0.0
    for i in range(n):
        y[i] = 1.0 / (f[i] / 128.0) ** (8.0 / 3.0)
        # The oracle takes sqrt(likelihood), squares it in double, then
        # stores the histogram sampling weight in float32.
        weights[i] = math.sqrt(energy[i]) ** 2
        cdf[i + 1] = cdf[i] + np.float32(weights[i])
        sx += x[i]
        sy += y[i]
    cdf /= cdf[n]
    mx, my = sx / n, sy / n
    xx = yy = xy = 0.0
    for i in range(n):
        xx += (x[i] - mx) ** 2
        yy += (y[i] - my) ** 2
        xy += (x[i] - mx) * (y[i] - my)
    delta = math.sqrt((xx - yy) ** 2 + 4 * xy * xy)
    a, b = (
        math.sqrt(max(0.0, (xx + yy + delta) / 2)),
        math.sqrt(max(0.0, (xx + yy - delta) / 2)),
    )
    ellipticity = abs((a - b) / (a + b)) if a + b else 0.0
    used = np.zeros(n, dtype=np.bool_)
    cursor = 0
    slopes = np.zeros(TRIALS)
    mergers = np.zeros(TRIALS)
    valid = np.zeros(TRIALS, dtype=np.bool_)
    for trial in range(TRIALS):
        used[:] = False
        sx = sy = sx2 = sxy = 0.0
        for pick in range(PICKS_PER_TRIAL):
            while True:
                if cursor >= len(uniforms):
                    return (
                        y,
                        weights,
                        slopes,
                        mergers,
                        valid,
                        ellipticity,
                        cursor,
                        False,
                    )
                r = uniforms[cursor]
                cursor += 1
                cell = np.searchsorted(cdf, r, side="right") - 1
                # TH1F spans [0, n-1] with n bins. Preserve its unusual
                # interpolation and integer truncation before duplicate checks.
                width = (n - 1.0) / n
                sample = cell * width
                if r > cdf[cell]:
                    sample += width * (r - cdf[cell]) / (cdf[cell + 1] - cdf[cell])
                k = int(sample)
                if not used[k]:
                    break
            used[k] = True
            sx += x[k]
            sy += y[k]
            sx2 += x[k] * x[k]
            sxy += x[k] * y[k]
        numerator, denominator = sy * sx - sxy * 6, sx2 * 6 - sx * sx
        if numerator == 0 or denominator == 0:
            continue
        sl = numerator / denominator
        tm = (sy + sl * sx) / 6 / sl
        sl = -sl
        slopes[trial] = sl
        mergers[trial] = tm
        valid[trial] = True
    return y, weights, slopes, mergers, valid, ellipticity, cursor, True


@njit(cache=True)
def finish_bootstrap(
    x: np.ndarray,
    f: np.ndarray,
    weights: np.ndarray,
    mindt: float,
    slope: float,
    merger: float,
    selected: int,
    ellipticity: float,
    symmetry: float,
    cursor: int,
) -> tuple[np.ndarray, int]:
    """Compute the native chirp metadata for the winning trial.

    Parameters
    ----------
    x, f, weights : numpy.ndarray
        Float64 micropixel times, frequencies and sampling weights.
    mindt : float
        Minimum time resolution used for the energy-fraction window.
    slope, merger : float
        Winning trial's chirp slope and merger time.
    selected : int
        Number of micropixels the winning trial classified as on-track.
    ellipticity, symmetry : float
        Metadata computed earlier for the winning trial.
    cursor : int
        Number of uniforms consumed, returned unchanged.

    Returns
    -------
    tuple[numpy.ndarray, int]
        ``([mass, merger, ellipticity, energy_fraction, symmetry], cursor)``;
        five zeros when fewer than four micropixels were selected.
    """
    n = len(x)
    if selected < 4:
        return np.zeros(5), cursor
    total = signal = 0.0
    for i in range(n):
        total += weights[i]
        offset = 2 * mindt if slope < 0 else -2 * mindt
        t = x[i] - merger - offset
        if t * slope <= 0:
            continue
        df = f[i] - 128.0 * (slope * t) ** (-3.0 / 8.0)
        dt = t - (f[i] / 128.0) ** (-8.0 / 3.0) / slope
        if df > 0:
            if abs(dt) > abs(offset) and abs(df * (dt + offset)) > 1:
                continue
        elif abs(df * dt) > 1:
            continue
        signal += weights[i]
    c = 299792458.0
    mgc = 256.0 * math.pi / 5.0 * (6.67259e-11 * 1.98892e30 * math.pi / c / c / c) ** (5.0 / 3.0) * 128.0 ** (8.0 / 3.0)
    mass = abs(slope / mgc) ** 0.6 * (1 if slope < 0 else -1)
    return np.array([mass, merger, ellipticity, signal / total, symmetry]), cursor
