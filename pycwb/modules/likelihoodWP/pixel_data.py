"""Pixel data loading: extract time-delay data from pixels.

Provides :func:`extract_pixel_time_delay_data` (fast SoA path),
:func:`_extract_pixel_array_time_delay_data` (internal fast path), and
:func:`build_sky_delay_and_antenna_patterns`.

Legacy aliases ``load_data_from_pixels``, ``_load_data_from_pixel_arrays``,
and ``load_data_from_ifo`` remain available.
"""

from __future__ import annotations

import numpy as np
from pycwb.config.config import Config
from pycwb.types.network_pixel import Pixel
from pycwb.types.time_series import TimeSeries
from pycwb.types.detector import compute_sky_delay_and_patterns
from .pixel_batch_ops import load_data_from_pixels_vectorized

def extract_pixel_time_delay_data(
    pixels: list[Pixel],
    nifo: int,
    pixel_arrays=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract pixel time-delay quadrature arrays for numba / JAX processing.

    Fast path
    ---------
    When ``pixel_arrays`` (a :class:`~pycwb.types.pixel_arrays.PixelArrays`)
    is provided and its ``td_amp`` is populated, the function reads directly
    from the pre-computed SoA arrays — zero per-pixel Python iteration.

    Fallback
    --------
    Otherwise delegates to the vectorised implementation in
    ``pixel_batch_ops`` which still avoids the worst of the per-pixel loops.

    Returns
    -------
    noise_weights : (nifo, n_pix) float32
        Normalised inverse-RMS weights.
    td_phase0 : (nifo, n_pix, tsize2) float32
        Time-delay samples in the 0-degree WDM phase.
    td_phase90 : (nifo, n_pix, tsize2) float32
        Time-delay samples in the 90-degree WDM phase.
    td_energy : (nifo, n_pix, tsize2) float32
        ``td_phase0 ** 2 + td_phase90 ** 2``.
    """
    n_detectors = nifo
    if pixel_arrays is not None and pixel_arrays.has_td_amp():
        return _extract_pixel_array_time_delay_data(pixel_arrays)
    return load_data_from_pixels_vectorized(pixels, n_detectors)


def _extract_pixel_array_time_delay_data(
    pixel_arrays,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast path: extract noise weights and quadratures from ``PixelArrays``."""
    # noise_rms: (n_ifo, n_pix) float32
    inverse_noise_rms = 1.0 / pixel_arrays.noise_rms.astype(np.float64)
    pixel_rms_norm = 1.0 / np.sqrt(np.sum(inverse_noise_rms ** 2, axis=0))
    noise_weights = (inverse_noise_rms * pixel_rms_norm[np.newaxis, :]).astype(np.float32)

    # td_amp_dense: (n_pix, n_ifo, tsize) → split into 00/90 halves
    time_delay_amplitudes = pixel_arrays.td_amp_dense()
    phase_size = time_delay_amplitudes.shape[2] // 2
    td_phase0 = time_delay_amplitudes[:, :, :phase_size].transpose(1, 0, 2)
    td_phase90 = time_delay_amplitudes[:, :, phase_size:].transpose(1, 0, 2)
    td_energy = td_phase0 ** 2 + td_phase90 ** 2

    return noise_weights, td_phase0, td_phase90, td_energy


def build_sky_delay_and_antenna_patterns(
    nIFO: int,
    strains: list[TimeSeries] | None = None,
    config: Config | None = None,
    ml: np.ndarray | None = None,
    FP: np.ndarray | None = None,
    FX: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sky-delay and antenna-pattern arrays for numba processing.
    Parameters
    ----------
    nIFO : int
        Number of interferometers.
    strains : list[TimeSeries] or None, optional
        Whitened strain time series; required when ``ml``/``FP``/``FX`` are not provided.
    config : Config or None, optional
        Analysis configuration; required when ``ml``/``FP``/``FX`` are not provided.
    ml : np.ndarray or None, optional
        Pre-computed sky-delay index array (nIFO, n_sky). When provided together
        with ``FP`` and ``FX``, the sky-pattern computation is skipped.
    FP : np.ndarray or None, optional
        Pre-computed f+ antenna patterns (nIFO, n_sky).
    FX : np.ndarray or None, optional
        Pre-computed fx antenna patterns (nIFO, n_sky).

    Returns
    -------
    ml_arr : np.ndarray
        Array of time-delay indices for each sky location, shape (nIFO, n_sky).
    fp_arr : np.ndarray
        f+ polarization data for each interferometer, shape (nIFO, n_sky).
    fx_arr : np.ndarray
        fx polarization data for each interferometer, shape (nIFO, n_sky).
    """
    if ml is not None and FP is not None and FX is not None:
        return np.asarray(ml), np.asarray(FP), np.asarray(FX)

    if strains is None or config is None:
        raise ValueError("strains and config are required when ml/FP/FX are not provided")

    strains = [TimeSeries.from_input(s) for s in strains]
    gps_time = float(strains[0].t0)
    _upTDF_lh = int(getattr(config, 'upTDF', 1))
    _TDRate_lh = int(getattr(config, 'TDRate', int(getattr(config, 'rateANA')) * _upTDF_lh))
    sky_delay_samples, plus_antenna_patterns, cross_antenna_patterns = compute_sky_delay_and_patterns(
        ifos=getattr(config, "ifo"),
        ref_ifo=getattr(config, "refIFO"),
        sample_rate=float(_TDRate_lh),
        td_size=max(int(getattr(config, "TDSize")) * _upTDF_lh,
                    int(getattr(config, "max_delay", 0.0) * float(_TDRate_lh)) + 1),
        gps_time=gps_time,
        healpix_order=int(getattr(config, "healpix", 0)) if hasattr(config, "healpix") else None,
        n_sky=None,
    )
    return sky_delay_samples, plus_antenna_patterns, cross_antenna_patterns


# Legacy aliases
load_data_from_pixels = extract_pixel_time_delay_data
_load_data_from_pixel_arrays = _extract_pixel_array_time_delay_data
load_data_from_ifo = build_sky_delay_and_antenna_patterns

__all__ = [
    "load_data_from_pixels", "load_data_from_ifo",
    "extract_pixel_time_delay_data",
    "build_sky_delay_and_antenna_patterns",
]
