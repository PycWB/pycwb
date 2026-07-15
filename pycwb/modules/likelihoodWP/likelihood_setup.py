"""Likelihood setup: segment-level precomputation.

Provides :func:`prepare_likelihood_inputs` which computes all job-segment-level
(lag/cluster-independent) inputs for the likelihood pipeline.
"""

from __future__ import annotations

import logging
import numpy as np
from pycwb.config.config import Config
from pycwb.types.time_series import TimeSeries
from pycwb.types.detector import _build_sky_directions
from .pixel_data import build_sky_delay_and_antenna_patterns
from .sky_mask import compute_sky_valid_indices
from .backends import normalize_likelihood_backend

logger = logging.getLogger(__name__)


def prepare_likelihood_inputs(
    config: Config,
    strains: list[TimeSeries],
    nIFO: int,
    ml: np.ndarray | None = None,
    FP: np.ndarray | None = None,
    FX: np.ndarray | None = None,
    ml_big: np.ndarray | None = None,
    FP_big: np.ndarray | None = None,
    FX_big: np.ndarray | None = None,
    big_cluster_healpix_order: int | None = None,
) -> dict:
    """
    Pre-compute all job-segment-level inputs for likelihood evaluation.

    Call this once per job segment, then pass the returned dict as ``setup=`` to
    every :func:`likelihood` call.  This avoids repeating:

    - Runtime parameter resolution from config
    - Sky delay / antenna pattern computation
    - ``_build_sky_directions`` healpix grid construction
    - FP / FX transpose + float32 cast

    Parameters
    ----------
    config : Config
        Analysis configuration.
    strains : list[TimeSeries]
        Whitened strain data (one per IFO); used only to determine GPS time and
        sample rate for sky-delay computation when ``ml``/``FP``/``FX`` are not
        provided.
    nIFO : int
        Number of interferometers.
    ml : np.ndarray, optional
        Pre-computed sky-delay index array (nIFO, n_sky) from ``setup_supercluster``.
        When provided, ``compute_sky_delay_and_patterns`` is skipped entirely.
    FP : np.ndarray, optional
        Pre-computed f+ antenna patterns (nIFO, n_sky) from ``setup_supercluster``.
    FX : np.ndarray, optional
        Pre-computed fx antenna patterns (nIFO, n_sky) from ``setup_supercluster``.

    Returns
    -------
    dict
        Keys: ``network_energy_threshold``, ``gamma_regulator``,
        ``delta_regulator``, ``net_rho_threshold``, ``netEC_threshold``, ``netCC``, ``ml``, ``FP``,
        ``FX``, ``FP_t``, ``FX_t``, ``n_sky``, ``healpix_order``, ``ra_arr``,
        ``dec_arr``, ``likelihood_backend``, and ``likelihood_extension_plan``.
    """
    n_detectors = nIFO
    if config is None:
        raise ValueError("config is required for pure-Python likelihood")
    # Resolve names once per segment setup rather than repeating registry
    # validation for every one of potentially millions of clusters.
    from .extensions import resolve_extension_plan
    likelihood_extension_plan = resolve_extension_plan(config)
    likelihood_backend = normalize_likelihood_backend(
        getattr(config, "likelihood_backend", "numba")
    )
    acor = float(getattr(config, "Acore"))
    gamma = float(getattr(config, "gamma", 0.0))
    delta = float(getattr(config, "delta", 0.0))
    net_rho = float(getattr(config, "netRHO", 0.0))
    netCC = float(getattr(config, "netCC", 0.0))
    xgb_rho_mode = bool(getattr(config, "xgb_rho_mode", False))

    network_energy_threshold = 2 * acor * acor * n_detectors
    gamma_regulator = gamma * gamma * 2 / 3
    # Mirror C++ constraint(): delta==0 is stored as 0.00001 to avoid a degenerate regulator
    if delta == 0.0:
        delta = 0.00001
    delta_regulator = abs(delta) if abs(delta) < 1 else 1
    net_rho_threshold = abs(net_rho)
    netEC_threshold = abs(net_rho) * abs(net_rho) * 2

    if ml is not None and FP is not None and FX is not None:
        # Reuse pre-computed arrays from setup_supercluster to avoid a duplicate
        # compute_sky_delay_and_patterns call (~same GPS time, same config).
        sky_delay_samples, plus_antenna_patterns, cross_antenna_patterns = (
            np.asarray(ml), np.asarray(FP), np.asarray(FX)
        )
    else:
        sky_delay_samples, plus_antenna_patterns, cross_antenna_patterns = (
            build_sky_delay_and_antenna_patterns(n_detectors, strains, config)
        )
    n_sky = int(sky_delay_samples.shape[1])

    # Pre-transpose and cast to float32 so per-cluster calls skip that work
    plus_antenna_patterns_t = plus_antenna_patterns.T.astype(np.float32)
    cross_antenna_patterns_t = cross_antenna_patterns.T.astype(np.float32)

    # Big-cluster coarse sky arrays (for bBB handling in network::likelihoodWP)
    if ml_big is not None and FP_big is not None and FX_big is not None:
        sky_delay_samples_big = np.asarray(ml_big)
        plus_antenna_patterns_big_t = np.asarray(FP_big).T.astype(np.float32)
        cross_antenna_patterns_big_t = np.asarray(FX_big).T.astype(np.float32)
        n_sky_big  = int(sky_delay_samples_big.shape[1])
    else:
        sky_delay_samples_big = None
        plus_antenna_patterns_big_t = None
        cross_antenna_patterns_big_t = None
        n_sky_big  = None

    healpix_order = int(getattr(config, 'healpix', 0)) if hasattr(config, 'healpix') else None
    # _build_sky_directions returns the cWB Earth-fixed grid.  Keep legacy
    # ra_arr/dec_arr aliases below, but use frame-explicit names internally.
    phi_geo_arr, latitude_arr = _build_sky_directions(n_sky, healpix_order)

    # Sky mask: restrict the sky scan to a user-defined region (mirrors C++ skyMask).
    # Parsed once per job segment and stored as a sorted int64 index array.
    _sky_mask_config = getattr(config, 'sky_mask', None)
    t_ref = (
        float(strains[0].t0)
        if strains is not None and len(strains) > 0
        else None
    )
    sky_valid_indices = compute_sky_valid_indices(
        phi_geo_arr, latitude_arr, _sky_mask_config, t_ref=t_ref
    )

    # Separate valid-index array for the coarse (big-cluster) sky grid
    if n_sky_big is not None:
        phi_geo_arr_big, latitude_arr_big = _build_sky_directions(
            n_sky_big, big_cluster_healpix_order
        )
        sky_valid_indices_big = compute_sky_valid_indices(
            phi_geo_arr_big, latitude_arr_big, _sky_mask_config, t_ref=t_ref
        )
    else:
        phi_geo_arr_big = None
        latitude_arr_big = None
        sky_valid_indices_big = None

    return {
        "network_energy_threshold": network_energy_threshold,
        "xgb_rho_mode": xgb_rho_mode,
        "gamma_regulator": gamma_regulator,
        "delta_regulator": delta_regulator,
        "net_rho_threshold": net_rho_threshold,
        "netEC_threshold": netEC_threshold,
        "netCC": netCC,
        "ml": sky_delay_samples,      # legacy key: (nIFO, n_sky)
        "FP": plus_antenna_patterns,  # legacy key: (nIFO, n_sky)
        "FX": cross_antenna_patterns, # legacy key: (nIFO, n_sky)
        "FP_t": plus_antenna_patterns_t,
        "FX_t": cross_antenna_patterns_t,
        "n_sky": n_sky,
        "healpix_order": healpix_order,
        "phi_geo_arr": phi_geo_arr,
        "latitude_arr": latitude_arr,
        # Deprecated internal aliases retained for callers outside this module.
        "ra_arr": phi_geo_arr,
        "dec_arr": latitude_arr,
        "sky_mask_config": _sky_mask_config,
        "segment_start_gps": t_ref,
        "likelihood_extension_plan": likelihood_extension_plan,
        "likelihood_backend": likelihood_backend,
        "sky_valid_indices": sky_valid_indices,
        "ml_big_cluster": sky_delay_samples_big,
        "FP_big_cluster_t": plus_antenna_patterns_big_t,
        "FX_big_cluster_t": cross_antenna_patterns_big_t,
        "n_sky_big_cluster": n_sky_big,
        "big_cluster_healpix_order": big_cluster_healpix_order,
        "phi_geo_arr_big_cluster": phi_geo_arr_big,
        "latitude_arr_big_cluster": latitude_arr_big,
        "sky_valid_indices_big": sky_valid_indices_big,
    }



# Stable public alias.
setup_likelihood = prepare_likelihood_inputs

__all__ = [
    "setup_likelihood",
    "prepare_likelihood_inputs",
]
