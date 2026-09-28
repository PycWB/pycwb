"""Data container for whitening-noise anchors and optional variability."""

from dataclasses import dataclass

from wdm_wavelet.types.time_frequency_map import TimeFrequencyMap as WaveletTimeFrequencyMap

from .time_frequency_map import TimeFrequencyMap


@dataclass
class NoiseRMSMap(WaveletTimeFrequencyMap):
    """Noise anchors, with timing separate from the original WDM transform.

    The inherited dt/t0 describe the transform used to estimate the noise.
    noise_start/noise_rate describe the much more sparsely sampled anchors.
    segment_start is the origin of detector pixel indices, including lags.
    variation is a single-band correction map, not wavelet coefficients.
    """

    noise_start: float
    noise_rate: float
    segment_start: float
    variation: TimeFrequencyMap | None = None
