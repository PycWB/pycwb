"""Pixel-derived event metadata with explicit time and rounding conventions."""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def pixel_moments(time, frequency, layers, rate, likelihood, analysis_rate):
    """Release ``netcluster::get(..., 'L', ..., core=false)`` moments.

    All cluster pixels define the sub-bin resolution, including zero-weight
    halo pixels. Positive halo likelihood also contributes to the moments.
    Keep accumulation in pixel order and the release's variance fallback.
    """
    if len(rate) == 0:
        return -1., -1., -1.
    min_rate = max_rate = int(rate[0] + .1)
    for r in rate:
        min_rate = min(min_rate, int(r + .1))
        max_rate = max(max_rate, int(r + .1))
    if min_rate <= 0:
        raise ValueError('Pixel moments require positive rates')
    at = bt = dt2 = af = bf = df2 = 0.
    for i in range(len(rate)):
        dt = 1. / rate[i]
        offset = 0. if layers[i] == int(analysis_rate / rate[i] + .1) else .5
        nt = int(max_rate * dt)
        step = 1. / max_rate
        t = (time[i] // layers[i] - offset) * dt + step / 2.
        weight = max(0., likelihood[i]) / (nt * nt)
        for j in range(nt):
            at += t * weight
            bt += weight
            dt2 += t * t * weight
            t += step
        nf = int(1. / (min_rate * dt))
        step = min_rate / 2.
        f = (frequency[i] - offset) / dt / 2. + step / 2.
        weight = max(0., likelihood[i]) / (nf * nf)
        for j in range(nf):
            af += f * weight
            bf += weight
            df2 += f * f * weight
            f += step
    duration = bandwidth = central_frequency = -1.
    if bt > 0:
        variance = (dt2 - at * at / bt) / bt
        duration = (math.sqrt(variance) * bt if variance > 0 else bt / max_rate) / bt
    if bf > 0:
        variance = (df2 - af * af / bf) / bf
        bandwidth = (math.sqrt(variance) * bf if variance > 0 else bf * min_rate / 2.) / bf
        central_frequency = af / bf
    return central_frequency, duration, bandwidth


def noise_amplitude(rms, *, inverse=False):
    """cWB's float32 logarithmic noise export, including zero-RMS counts."""
    total = 0.
    for value in rms:
        value = float(value)
        if value > 0:
            total += 1./value/value if inverse else value*value
    if len(rms):
        total /= len(rms)
    if inverse and total > 0:
        total = 1./total
    logarithm = float(np.float32(math.log(total)/2./math.log(10.))) if total > 0 else 0.
    return 10.**logarithm


def aligned_pixel_bounds(pixels, n_ifo, edge, duration, lag_shifts=None):
    """Core-pixel supports in the catalog's unshifted analysis coordinates.

    Detector pixel indices address circularly shifted data. Undo each lag in
    integer bin coordinates before taking extrema, so wrap-edge events remain
    compact. No time-of-flight correction belongs in these pixel supports.
    """
    mask = pixels.core & (pixels.rate > 0) & (pixels.layers > 0)
    if not np.any(mask):
        return None
    rate = pixels.rate[mask].astype(np.float64)
    layers = pixels.layers[mask].astype(np.int64)
    shifts = np.zeros(n_ifo) if lag_shifts is None else np.asarray(lag_shifts)
    if shifts.shape != (n_ifo,):
        raise ValueError('One lag shift per detector is required')
    starts, stops = [], []
    for i in range(n_ifo):
        bins = pixels.pixel_index[i, mask].astype(np.int64) // layers
        first = (edge*rate+.001).astype(np.int64)
        count = np.rint(duration*rate).astype(np.int64)
        if np.any(count <= 0):
            raise ValueError('Nonpositive analyzed duration in pixel bins')
        shift = ((float(shifts[i])-float(np.min(shifts)))*rate+.001).astype(np.int64)
        in_window = (bins >= first) & (bins < first+count)
        bins = np.where(in_window, first + (bins-first-shift) % count, bins)
        starts.append(float(np.min(bins/rate)))
        stops.append(float(np.max((bins+1)/rate)))
    return starts, stops
