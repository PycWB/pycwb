"""Shared helpers for time-delay max-energy backends."""

from __future__ import annotations

import numpy as np


def validate_time_delay_inputs(
    tf_map, dt, downsample, *, require_wavelet_api: bool
) -> None:
    """Validate backend-independent max-energy inputs."""
    if require_wavelet_api and (
        not hasattr(tf_map.wavelet, "t2w") or not hasattr(tf_map.wavelet, "w2t")
    ):
        raise ValueError(
            "time_delay_max_energy requires a WDM wavelet with t2w/w2t APIs"
        )
    if downsample <= 0:
        raise ValueError("downsample must be >= 1")
    if not np.isfinite(dt):
        raise ValueError("dt must be finite")


def time_series_length(tf_map) -> int:
    """Return the original time-series length represented by a TF map."""
    if tf_map.len_timeseries is not None:
        return int(tf_map.len_timeseries)
    return max(1, int(round((tf_map.stop - tf_map.start) / tf_map.dt)))


def sample_rate_from_tf_map(tf_map, n_freq: int) -> float:
    """Return the sample rate implied by WDM frequency spacing."""
    return float(2.0 * float(tf_map.df) * (int(n_freq) - 1))


def frequency_bounds(tf_map, n_freq: int) -> tuple[float, float]:
    """Return low/high frequency bounds for packet-energy kernels."""
    f_low = 0.0 if tf_map.f_low is None else float(tf_map.f_low)
    f_high = (
        (float(tf_map.df) * (int(n_freq) - 1))
        if tf_map.f_high is None
        else float(tf_map.f_high)
    )
    return f_low, f_high


__all__ = [
    "validate_time_delay_inputs",
    "time_series_length",
    "sample_rate_from_tf_map",
    "frequency_bounds",
]
