"""Extract time-delay quadratures and normalized noise weights from pixels.

extract_pixel_time_delay_data uses PixelArrays when available, with a legacy
Pixel-list fallback. build_sky_delay_and_antenna_patterns bridges detector
geometry to the scan array layout.
"""

from __future__ import annotations

import numpy as np
from pycwb.config.config import Config
from pycwb.types.network_pixel import Pixel
from pycwb.types.time_series import TimeSeries
from pycwb.types.detector import compute_sky_delay_and_patterns


def _extract_legacy_pixel_time_delay_data(pixels, nifo):
    """
    Vectorized replacement for the per-pixel Python loop in ``extract_pixel_time_delay_data``.

    Extracts noise RMS and time-delay amplitudes from a list of Pixel objects
    into pre-allocated numpy arrays using bulk array operations instead of
    element-wise Python loops.

    Parameters
    ----------
    pixels : list[Pixel]
    nifo   : int  — number of interferometers

    Returns
    -------
    rms       : np.ndarray, shape (nifo, n_pix)         float32
    td00      : np.ndarray, shape (nifo, n_pix, tsize2)  float32
    td90      : np.ndarray, shape (nifo, n_pix, tsize2)  float32
    td_energy : np.ndarray, shape (nifo, n_pix, tsize2)  float32
    """
    n_pix = len(pixels)
    tsize = int(np.asarray(pixels[0].td_amp[0]).shape[0])
    tsize2 = tsize // 2

    # ---- Extract noise_rms: shape (nifo, n_pix) ----
    inv_rms_arr = np.empty((nifo, n_pix), dtype=np.float64)
    for i in range(nifo):
        inv_rms_arr[i] = [1.0 / pix.data[i].noise_rms for pix in pixels]

    # Per-pixel network RMS: 1 / sqrt(sum(inv_rms^2))
    rms_pix = 1.0 / np.sqrt(np.sum(inv_rms_arr ** 2, axis=0))  # (n_pix,)

    # Normalised inverse rms: shape (nifo, n_pix)
    rms = (inv_rms_arr * rms_pix[np.newaxis, :]).astype(np.float32)

    # ---- Extract td_amp: shape (nifo, n_pix, tsize) ----
    # pixel.td_amp[i] can be a numpy array or a list of floats
    td_amp_arr = np.empty((nifo, n_pix, tsize), dtype=np.float32)
    for i in range(nifo):
        for pid, pix in enumerate(pixels):
            td_amp_arr[i, pid] = pix.td_amp[i]

    # Split into 00 and 90 quadrature halves
    td00 = td_amp_arr[:, :, :tsize2]
    td90 = td_amp_arr[:, :, tsize2:tsize]
    td_energy = td00 ** 2 + td90 ** 2

    return rms, td00, td90, td_energy


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
    Otherwise delegates to :func:`_extract_legacy_pixel_time_delay_data`, which
    extracts arrays from legacy Pixel objects.

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
    return _extract_legacy_pixel_time_delay_data(pixels, n_detectors)


def _extract_pixel_array_time_delay_data(
    pixel_arrays,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast path: extract noise weights and quadratures from ``PixelArrays``."""
    # noise_rms: (n_ifo, n_pix) float32
    inverse_noise_rms = 1.0 / pixel_arrays.noise_rms.astype(np.float64)
    pixel_rms_norm = 1.0 / np.sqrt(np.sum(inverse_noise_rms**2, axis=0))
    noise_weights = (inverse_noise_rms * pixel_rms_norm[np.newaxis, :]).astype(np.float32)

    # td_amp_dense: (n_pix, n_ifo, tsize) → split into 00/90 halves
    time_delay_amplitudes = pixel_arrays.td_amp_dense()
    phase_size = time_delay_amplitudes.shape[2] // 2
    td_phase0 = time_delay_amplitudes[:, :, :phase_size].transpose(1, 0, 2)
    td_phase90 = time_delay_amplitudes[:, :, phase_size:].transpose(1, 0, 2)
    td_energy = td_phase0**2 + td_phase90**2

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
    _upTDF_lh = int(getattr(config, "upTDF", 1))
    _TDRate_lh = int(getattr(config, "TDRate", int(getattr(config, "rateANA")) * _upTDF_lh))
    sky_delay_samples, plus_antenna_patterns, cross_antenna_patterns = compute_sky_delay_and_patterns(
        detectors=config.detectors,
        ref_ifo=getattr(config, "refIFO"),
        sample_rate=float(_TDRate_lh),
        td_size=max(
            int(getattr(config, "TDSize")) * _upTDF_lh, int(getattr(config, "max_delay", 0.0) * float(_TDRate_lh)) + 1
        ),
        gps_time=gps_time,
        healpix_order=int(getattr(config, "healpix", 0)) if hasattr(config, "healpix") else None,
        n_sky=None,
    )
    return sky_delay_samples, plus_antenna_patterns, cross_antenna_patterns

__all__ = [
    "extract_pixel_time_delay_data",
    "build_sky_delay_and_antenna_patterns",
]
