"""Reference cWB ``detector::setsim`` energy for target-SNR injections."""

import numpy as np
from scipy.fft import irfft, rfft


def cwb_snr_energy(series, gps, half_window, f_low, f_high):
    """Reproduce the reference centroid, 99.9% crop and packed-FFT cut.

    ``series`` must already be whitened using the pre-conditioning noise RMS.
    The frequency-cut loop intentionally preserves the original packed-FFT
    convention, including its upper loop bound and joint DC/Nyquist handling.
    """
    rate = float(series.sample_rate)
    values = np.asarray(series.data)
    size = len(values)
    origin = float(series.t0)
    center_time = float(gps) - origin
    for _ in range(2):
        lo = max(0, int((center_time - half_window) * rate))
        hi = min(size, int((center_time + half_window) * rate))
        if hi <= lo:
            return 0.0
        energy = values[lo:hi] ** 2
        total_energy = float(energy.sum())
        if total_energy <= 0:
            return 0.0
        center_time = float(np.dot(np.arange(lo, hi) / rate, energy) / total_energy)

    # Preserve the C++ absolute-GPS round trip before selecting the waveform.
    time = center_time + origin
    window_size = int(2 * half_window * rate)
    offset = int((time - half_window - origin) * rate)
    indices = offset + np.arange(window_size)
    valid = (indices >= 0) & (indices < size)
    window_energy = float(np.sum(values[indices[valid]] ** 2))
    if window_energy <= 0:
        return 0.0
    center = offset + window_size // 2
    accumulated = float(values[center] ** 2) if 0 <= center < size else 0.0
    for radius in range(1, window_size // 2):
        if 0 <= center - radius < size:
            accumulated += float(values[center - radius] ** 2)
        if 0 <= center + radius < size:
            accumulated += float(values[center + radius] ** 2)
        if accumulated / window_energy > 0.999:
            break
    else:
        radius = window_size // 2

    midpoint = int((time - origin) * rate)
    lo = max(0, midpoint - radius)
    hi = min(size, midpoint + radius)
    waveform = values[lo:hi].copy()
    if not len(waveform):
        return 0.0
    spectrum = rfft(waveform)
    # CWB visits packed indices j < N/2, step 2; use integer division even
    # when a segment-edge crop produces an odd-length waveform.
    bins = np.arange((len(waveform) // 2 + 1) // 2)
    frequency = bins * rate / len(waveform)
    cut = bins[(frequency < f_low) | (frequency > f_high)]
    spectrum[cut] = 0
    if 0 in cut:
        spectrum[-1] = 0
    waveform = irfft(spectrum, n=len(waveform))
    return float(waveform @ waveform)
