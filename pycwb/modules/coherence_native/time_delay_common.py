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


__all__ = [
    "validate_time_delay_inputs",
    "time_series_length",
    "sample_rate_from_tf_map",
    "frequency_bounds",
]
