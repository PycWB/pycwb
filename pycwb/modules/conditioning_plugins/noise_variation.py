"""Apply plugin-produced noise variation to lagged pixel RMS values.

The ordinary TimeFrequencyMap holds one float32 correction row, its time
lattice, and the affected frequency band. This implements cWB detector::setrms
and WSeries<float>::get, including float rounding when averaging the window.
It is shared plugin support, not a directly configurable conditioning hook.
"""

import numpy as np
from numba import njit

from pycwb.types.time_frequency_map import TimeFrequencyMap


@njit(cache=True)
def _apply_noise_variation(rms, frequencies, indices, layers, rates, start,
                           values, var_start, var_rate, low, high):
    """detector::setrms band overlap and wavearray::get RMS averaging."""
    for i in range(len(rms)):
        f = (frequencies[i] - .5) * rates[i] / 2.
        upper = f + rates[i] / 2.
        overlap = max(0., min(upper, high) - max(f, low)) * 2. / rates[i]
        t = indices[i] / rates[i] / layers[i] + start - var_start
        center = int(t * var_rate)
        if center < 0 or center >= len(values):
            raise ValueError('Pixel outside noise-variation timeline')
        half = .5 / rates[i] + .51 / var_rate
        a = max(0, min(len(values)-1, int((t-half)*var_rate)))
        b = max(0, min(len(values)-1, int((t+half)*var_rate)))
        square = values[center] ** 2
        if a < b:
            square = 0.
            for j in range(a, b+1):
                # nVAR is a WSeries<float>: products and returned RMS round to float.
                value = np.float32(values[j])
                square += np.float32(value * value)
            value = np.float64(np.float32(np.sqrt(square / (b-a+1))))
            square = value * value
        rms[i] /= np.sqrt(1-overlap+overlap*square)
    return rms


def apply_noise_variation(rms, frequencies, indices, layers, rates, segment_start, variation):
    """Update one detector's RMS array in place from a single-band correction.

    Pixel arrays are one-dimensional, with one entry per RMS value. Their
    indices include detector lags and are relative to segment_start.
    """
    if not isinstance(variation, TimeFrequencyMap):
        raise ValueError('Noise variation requires a TimeFrequencyMap')
    values = np.asarray(variation.data)
    if (values.ndim != 2 or values.shape[0] != 1 or not values.shape[1]
            or variation.f_low is None or variation.f_high is None
            or not np.isfinite([variation.dt, variation.start,
                               variation.f_low, variation.f_high]).all()
            or variation.dt <= 0 or variation.f_high <= variation.f_low):
        raise ValueError('Invalid noise-variation map')
    if np.iscomplexobj(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Noise variation must be finite and positive')
    return _apply_noise_variation(
        rms, frequencies, indices, layers, rates, segment_start,
        np.asarray(values[0], dtype=np.float64), variation.start,
        1.0 / variation.dt, variation.f_low, variation.f_high,
    )
