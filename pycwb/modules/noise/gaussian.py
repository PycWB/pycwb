"""
Coloured Gaussian noise generation.

Replaces ``pycbc.noise.noise_from_psd`` using ``lalsimulation.SimNoise``
directly (the same engine PyCBC uses).
"""

import logging

import lal
import lalsimulation
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["gaussian_noise_from_psd", "generate_noise"]


def gaussian_noise_from_psd(
    n_samples: int,
    delta_t: float,
    psd: np.ndarray,
    delta_f: float,
    seed: int | None = None,
    start_time: float = 0.0,
):
    """Generate coloured Gaussian noise from a PSD array.

    Uses ``lalsimulation.SimNoise`` with an overlap-save scheme identical
    to the one inside PyCBC, so noise statistics are the same for a given
    seed.

    Parameters
    ----------
    n_samples : int
        Number of time-domain samples to generate.
    delta_t : float
        Sampling interval (seconds).
    psd : numpy.ndarray
        One-sided PSD array (strain²/Hz) on a uniform grid with spacing
        *delta_f*.  Length must be at least ``N//2 + 1`` where
        ``N = int(1 / delta_t / delta_f)``.
    delta_f : float
        Frequency resolution of the PSD (Hz).
    seed : int or None
        RNG seed for reproducibility.  If ``None`` a random seed is chosen.
    start_time : float
        GPS start time assigned to the output time series.

    Returns
    -------
    pycwb.types.time_series.TimeSeries
    """
    from pycwb.types.time_series import TimeSeries

    N = int(1.0 / delta_t / delta_f)
    n = N // 2 + 1
    stride = N // 2

    if n > len(psd):
        raise ValueError(
            f"PSD length ({len(psd)}) too short for the requested delta_t "
            f"({delta_t}) and delta_f ({delta_f}): need at least {n} bins"
        )

    # Build a LAL REAL8FrequencySeries from the numpy PSD
    lal_psd = lal.CreateREAL8FrequencySeries(
        "psd", lal.LIGOTimeGPS(0), 0.0, delta_f,
        lal.DimensionlessUnit, n,
    )
    lal_psd.data.data[:] = psd[:n]
    lal_psd.data.data[0] = 0.0
    lal_psd.data.data[n - 1] = 0.0

    # RNG
    if seed is None:
        seed = np.random.randint(0, 2**31)
    rng = lal.gsl_rng("ranlux", int(seed))

    # Output buffer
    out = np.zeros(n_samples, dtype=np.float64)

    # LAL segment for SimNoise
    segment = lal.CreateREAL8TimeSeries(
        "noise", lal.LIGOTimeGPS(0), 0.0, delta_t,
        lal.DimensionlessUnit, N,
    )

    # First call — full segment initialisation
    lalsimulation.SimNoise(segment, 0, lal_psd, rng)

    generated = 0
    while generated < n_samples:
        chunk = min(stride, n_samples - generated)
        out[generated : generated + chunk] = segment.data.data[:chunk]
        generated += chunk
        if generated < n_samples:
            lalsimulation.SimNoise(segment, stride, lal_psd, rng)

    return TimeSeries(data=out, dt=delta_t, t0=start_time)


def generate_noise(
    psd: str | np.ndarray | None = None,
    f_low: float = 30.0,
    delta_f: float = 0.25,
    duration: float = 32.0,
    sample_rate: float = 16384.0,
    seed: int | None = None,
    start_time: float = 0.0,
    noise_type: str = "gaussian",
    psd_model: str = "aLIGOZeroDetHighPower",
):
    """High-level noise generator — drop-in replacement for mdc.generate_noise.

    Parameters
    ----------
    psd : str, numpy.ndarray, or None
        * ``str`` — path to a two-column (freq, ASD) text file.
        * ``numpy.ndarray`` — pre-computed PSD array on the target grid.
        * ``None`` — use the analytic model given by *psd_model*.
    f_low : float
        Low-frequency cutoff (Hz).
    delta_f : float
        Frequency resolution (Hz).
    duration : float
        Duration of the output time series (seconds).
    sample_rate : float
        Sampling rate (Hz).
    seed : int or None
        RNG seed.
    start_time : float
        GPS epoch of the output time series.
    noise_type : str
        ``"gaussian"`` (default).  Other types raise ``NotImplementedError``.
    psd_model : str
        Analytic PSD model name when *psd* is ``None``.

    Returns
    -------
    pycwb.types.time_series.TimeSeries
    """
    if noise_type != "gaussian":
        raise NotImplementedError(
            f"Noise type '{noise_type}' is not implemented yet. "
            "Only 'gaussian' is supported."
        )

    from .psd import analytic_psd, load_psd

    flen = int(sample_rate / delta_f) + 1
    delta_t = 1.0 / sample_rate

    if isinstance(psd, str):
        logger.info(
            "Using psd file %s with f_low %s, delta_f %s, flen %s",
            psd, f_low, delta_f, flen,
        )
        psd_arr = load_psd(psd, flen, delta_f, f_low)
    elif isinstance(psd, np.ndarray):
        psd_arr = psd
    else:
        logger.info(
            "Using %s psd with f_low %s, delta_f %s, flen %s",
            psd_model, f_low, delta_f, flen,
        )
        psd_arr = analytic_psd(psd_model, flen, delta_f, f_low)

    # Ensure PSD covers the full bandwidth
    desired_length = int(1.0 / delta_t / delta_f) // 2 + 1
    if len(psd_arr) < desired_length:
        logger.warning(
            "PSD length %d is less than desired length %d, zero-padding PSD",
            len(psd_arr), desired_length,
        )
        padded = np.zeros(desired_length, dtype=np.float64)
        padded[: len(psd_arr)] = psd_arr
        psd_arr = padded

    n_samples = int(duration / delta_t)

    return gaussian_noise_from_psd(
        n_samples=n_samples,
        delta_t=delta_t,
        psd=psd_arr,
        delta_f=delta_f,
        seed=seed,
        start_time=start_time,
    )
