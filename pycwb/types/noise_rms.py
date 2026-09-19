"""Whitening-noise anchors and cWB-compatible pixel noise lookup."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from wdm_wavelet.types.time_frequency_map import TimeFrequencyMap


@dataclass
class NoiseRMSMap(TimeFrequencyMap):
    """Noise anchors, with timing separate from the original WDM transform.

    The inherited dt/t0 describe the transform used to estimate the noise.
    noise_start/noise_rate describe the much more sparsely sampled anchors.
    segment_start is the origin of detector pixel indices, including lags.
    """

    noise_start: float
    noise_rate: float
    segment_start: float


def make_noise_rms_map(tf_map, anchors, edge_length):
    """Preserve the anchor lattice used by WSeries::white / wavearray::white."""
    data = np.asarray(anchors, dtype=np.float64)
    n_time = tf_map.data.shape[1]
    intervals = data.shape[1] - 1
    if intervals < 1:
        raise ValueError("Pixel RMS requires at least two whitening-noise anchors")
    tf_rate = 1.0 / float(tf_map.dt)
    offset = int(float(edge_length) * tf_rate + 0.5)
    offset -= offset & 1
    step = (n_time - 2 * offset) // intervals
    step -= step & 1
    if step < 2:
        raise ValueError("Segment is too short for the whitening-noise anchor lattice")
    first = (n_time - step * intervals) // 2
    return NoiseRMSMap(
        data=data,
        df=float(tf_map.df),
        dt=float(tf_map.dt),
        t0=float(tf_map.t0),
        len_timeseries=int(tf_map.len_timeseries),
        wdm_params=dict(tf_map.wdm_params),
        noise_start=float(tf_map.t0) + first / tf_rate,
        noise_rate=tf_rate / step,
        segment_start=float(tf_map.t0),
    )


@njit(cache=True)
def _pixel_noise_rms(frequencies, indices, layers, rates, start, noise, noise_start, noise_rate, df):
    """Port of detector::setrms for WDM pixels without an nVAR map."""
    result = np.empty(len(frequencies), dtype=np.float64)
    nf, nt = noise.shape
    for i in range(len(frequencies)):
        rate = rates[i]
        if rate <= 0 or layers[i] <= 1 or frequencies[i] <= 0:
            raise ValueError("Invalid WDM pixel for noise lookup")
        low = int((frequencies[i] - 0.5) * rate / 2 / df + 0.6)
        high = min(int((frequencies[i] + 0.5) * rate / 2 / df + 0.6), nf)
        # Keep the C++ order of operations, including GPS addition and truncation.
        t = indices[i] / rate / layers[i] + start
        k = int((t - noise_start) * noise_rate)
        if k >= nt:
            k -= 1
        if k < 0 or k >= nt or low < 0 or low >= high:
            raise ValueError("Pixel lies outside the whitening-noise anchor map")
        inv_variance = 0.0
        for j in range(low, high):
            value = noise[j, k]
            if value <= 0 or not np.isfinite(value):
                raise ValueError("Whitening-noise RMS must be finite and positive")
            inv_variance += 1.0 / value / value
        result[i] = np.sqrt((high - low) / inv_variance)
    return result


def lookup_pixel_noise_rms(frequencies, detector_indices, layers, rates, noise_maps):
    """Return float64 (n_pixels, n_detectors) RMS without a dense TF-time cache."""
    frequencies = np.asarray(frequencies, dtype=np.int64)
    indices = np.asarray(detector_indices, dtype=np.int64)
    if indices.shape != (len(frequencies), len(noise_maps)):
        raise ValueError("Detector pixel indices and noise maps must agree")
    layers = np.broadcast_to(np.asarray(layers, dtype=np.int64), frequencies.shape)
    rates = np.broadcast_to(np.asarray(rates, dtype=np.float64), frequencies.shape)
    result = np.empty(indices.shape, dtype=np.float64)
    for d, noise in enumerate(noise_maps):
        if not isinstance(noise, NoiseRMSMap):
            raise ValueError("NoiseRMSMap anchor timing is required for pixel RMS lookup")
        if noise.noise_rate <= 0 or noise.df <= 0:
            raise ValueError("Noise anchor rate and frequency spacing must be positive")
        result[:, d] = _pixel_noise_rms(
            frequencies,
            indices[:, d],
            layers,
            rates,
            noise.segment_start,
            np.asarray(noise.data, dtype=np.float64),
            noise.noise_start,
            noise.noise_rate,
            noise.df,
        )
    return result
