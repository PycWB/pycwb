"""Reference cWB SNR-mode injection resampling, separate from noise/SNR estimation."""
import numpy as np
from scipy.signal import oaconvolve
from pycwb.types.time_series import TimeSeries
from .meyer_coefficients import MEYER_HALF


def meyer_downsample(values, levels):
    """Return cWB Meyer(1024) low-pass layers with periodic boundaries.

    Each layer preserves wavelet energy normalization (sqrt(2) for DC).
    ForwardFWT's default odd parity starts the 1024 taps at sample -511.
    """
    if int(levels) != levels or levels < 0:
        raise ValueError('Meyer levels must be a nonnegative integer')
    result = np.asarray(values, dtype=np.float64)
    half = np.asarray(MEYER_HALF)
    filt = np.concatenate(([0.0], half[:0:-1], half))
    for _ in range(int(levels)):
        if len(result) < 2048 or len(result) % 2:
            raise ValueError('Meyer input must be even and at least 2048 samples per level')
        padded = np.pad(result, (511, 512), mode='wrap')
        result = oaconvolve(padded, filt[::-1], mode='valid')[::2].copy()
    return result


def uses_cwb_snr_resampling(config, injections):
    """Return whether the configured SNR path needs reference resampling."""
    if getattr(config, 'injection_resampling', 'fft') != 'cwb':
        return False
    targeted = [float(p.get('target_snr', p.get('targeted_snr', 0))) > 0 for p in injections]
    if any(targeted) and not all(targeted):
        raise ValueError('cWB resampling requires separate trials for target-SNR and fixed-hrss injections')
    return any(targeted)


def resample_snr_injection(strain, config):
    """Meyer-resample final SNR-scaled strain; do not use for estimating its SNR."""
    strain = TimeSeries.from_input(strain)
    if float(strain.sample_rate) != float(config.inRate):
        raise ValueError('Injection rate does not match inRate')
    if config.fResample > 0:
        strain = strain.cwb_resampling(float(config.fResample))
    rate = float(strain.sample_rate) / (1 << config.levelR)
    return TimeSeries(meyer_downsample(strain.data, config.levelR), dt=1/rate, t0=strain.t0)
