"""
Compatibility module for pure-Python data conditioning.

Implementations have been split into:
- regression_py.py
- whitening_py.py

Existing imports from data_conditioning_python are preserved.
"""

import time
import logging

from .regression_py import (
    regression_python,
)
from .whitening_py import (
    whitening_python,
    _estimate_noise_rms,
    _estimate_noise_rms_cwb,
    _bandpass_rms,
    _whiten_coefficients,
    _apply_wiener_filter,
    _average_phases,
)
from .whitening_mesa_py import whitening_mesa_python

logger = logging.getLogger(__name__)


def data_conditioning(config, strains, nproc=1):
    """
    Perform pure-Python data conditioning (regression + whitening).

    Parameters
    ----------
    config : Config
        Configuration object.
    strains : list
        Input strain time series per detector.
    nproc : int
        Unused. Kept for API compatibility.

    Returns
    -------
    tuple
        `(conditioned_strains, nRMS_list)` where conditioned strains are
        time-domain series and `nRMS_list` are 2D TF maps.
    """
    timer_start = time.perf_counter()

    white_method = getattr(config, "whiteMethod", "wavelet")
    if white_method not in {"wavelet", "python", "mesa"}:
        logger.error(f"Method {white_method} is not a valid pure-Python whitening method")
        raise ValueError(f"Method {white_method} is not a valid pure-Python whitening method")

    data_regressions = [regression_python(config, h) for h in strains]
    if white_method == "mesa":
        res = [whitening_mesa_python(config, h) for h in data_regressions]
    else:
        res = [whitening_python(config, h) for h in data_regressions]

    conditioned_strains, nRMS_list = zip(*res)

    timer_end = time.perf_counter()
    logger.info("-------------------------------------------------------")
    logger.info(f"Pure-Python Data Conditioning Time: {timer_end - timer_start:.2f} seconds")
    logger.info("-------------------------------------------------------")

    return conditioned_strains, nRMS_list


def data_conditioning_single(config, strain):
    """
    Perform pure-Python data conditioning for a single detector strain.

    Parameters
    ----------
    config : Config
        Configuration object.
    strain : object
        Input strain time series.

    Returns
    -------
    tuple
        `(conditioned_strain, nRMS)`.
    """
    white_method = getattr(config, "whiteMethod", "wavelet")
    if white_method not in {"wavelet", "python", "mesa"}: 
        logger.error(f"Method {white_method} is not a valid pure-Python whitening method")
        raise ValueError(f"Method {white_method} is not a valid pure-Python whitening method")

    data_regression = regression_python(config, strain)
    if white_method == 'mesa':
        conditioned_strain, nRMS = whitening_mesa_python(config, data_regression)
    else:
        conditioned_strain, nRMS = whitening_python(config, data_regression)

    return conditioned_strain, nRMS

__all__ = [
    "data_conditioning",
    "data_conditioning_single",
    "regression_python",
    "whitening_python",
    "whitening_mesa_python",
    "_cwb_percentile_mean",
    "_cwb_rotated_products",
    "_build_correlation_matrix",
    "_build_crosscorr_vector",
    "_estimate_noise_rms",
    "_estimate_noise_rms_cwb",
    "_bandpass_rms",
    "_whiten_coefficients",
    "_apply_wiener_filter",
    "_average_phases",
]
