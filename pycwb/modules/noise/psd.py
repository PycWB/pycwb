"""
PSD loading and generation utilities.

Replaces ``pycbc.psd.from_txt`` and ``pycbc.psd.aLIGOZeroDetHighPower``.
"""

import logging

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

__all__ = ["load_psd", "analytic_psd"]


def load_psd(path: str, flen: int, delta_f: float, f_low: float) -> np.ndarray:
    """Load a two-column ASD text file and return a PSD array on a uniform grid.

    The text file is expected to have two whitespace-separated columns:

    * Column 1 — frequency (Hz)
    * Column 2 — amplitude spectral density (strain / √Hz)

    The ASD values are **squared** to produce a one-sided power spectral
    density, then interpolated onto the uniform frequency grid
    ``[0, delta_f, 2*delta_f, …, (flen-1)*delta_f]``.

    Frequencies below *f_low* or outside the file's frequency range are
    set to zero.

    Parameters
    ----------
    path : str
        Path to the two-column text file.
    flen : int
        Number of frequency bins in the output (typically ``int(sample_rate / delta_f) // 2 + 1``).
    delta_f : float
        Frequency resolution (Hz).
    f_low : float
        Low-frequency cutoff (Hz); bins below this are zeroed.

    Returns
    -------
    numpy.ndarray
        1-D array of length *flen* containing the one-sided PSD (strain²/Hz).
    """
    raw = np.loadtxt(path)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"PSD file {path} must have at least two columns")

    freq_file = raw[:, 0]
    asd_file = raw[:, 1]

    # Square ASD → PSD
    psd_file = asd_file ** 2

    # Build the target frequency grid
    freqs = np.arange(flen) * delta_f

    # Determine the usable range
    f_max_file = freq_file[-1]
    if freqs[-1] > f_max_file:
        logger.warning(
            "Requested number of samples exceeds the highest available "
            "frequency in the input data, will use max available frequency "
            "instead. (requested %f Hz, available %f Hz)",
            freqs[-1], f_max_file,
        )

    # Interpolate onto the uniform grid
    interp_fn = interp1d(
        freq_file, psd_file,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    psd_out = interp_fn(freqs)

    # Zero below f_low
    psd_out[freqs < f_low] = 0.0

    return psd_out


def analytic_psd(model: str, flen: int, delta_f: float, f_low: float) -> np.ndarray:
    """Evaluate a named lalsimulation noise model on a uniform frequency grid.

    Parameters
    ----------
    model : str
        Name of the noise model.  Must correspond to a
        ``lalsimulation.SimNoisePSD<model>`` function, e.g.
        ``"aLIGOZeroDetHighPower"``, ``"aLIGOaLIGODesignSensitivityT1800044"``,
        ``"AdVDesignSensitivityP1200087"``.
    flen : int
        Number of frequency bins.
    delta_f : float
        Frequency resolution (Hz).
    f_low : float
        Low-frequency cutoff (Hz); bins below this are zeroed.

    Returns
    -------
    numpy.ndarray
        1-D PSD array of length *flen* (strain²/Hz).
    """
    import lalsimulation

    func_name = f"SimNoisePSD{model}"
    psd_func = getattr(lalsimulation, func_name, None)
    if psd_func is None:
        raise ValueError(
            f"Unknown PSD model '{model}': lalsimulation.{func_name} not found"
        )

    freqs = np.arange(flen) * delta_f
    psd_out = np.zeros(flen, dtype=np.float64)

    for i, f in enumerate(freqs):
        if f >= f_low:
            psd_out[i] = psd_func(f)

    return psd_out
