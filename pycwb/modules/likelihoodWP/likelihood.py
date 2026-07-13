"""Thin facade module — public entry points for likelihoodWP.

Import the public API from here:
    from pycwb.modules.likelihoodWP.likelihood import (
        setup_likelihood, likelihood, likelihood_wrapper,
        prepare_likelihood_inputs, evaluate_cluster_likelihood,
        evaluate_fragment_clusters,
    )

All helper functions have been extracted to phase submodules:
    - ``likelihood_setup.py``   — prepare_likelihood_inputs
    - ``pixel_data.py``         — extract_pixel_time_delay_data, ...
    - ``sky_scan.py``           — scan_sky_for_best_fit (@njit)
    - ``sky_statistics.py``     — compute_statistics_at_sky_position
    - ``detection_statistics.py`` — get_likelihood_rejection_reason,
                                    populate_detection_statistics,
                                    update_chirp_mass_statistics,
                                    compute_sky_error_region, ...
    - ``packet_ops.py``         — avx_noise_ps, avx_packet_ps, packet_norm_numpy, ...
"""

from __future__ import annotations

import logging
import time
import numpy as np

from pycwb.types.network_cluster import Cluster, FragmentCluster
from pycwb.types.time_series import TimeSeries
from pycwb.types.time_frequency_map import TimeFrequencyMap
from pycwb.modules.xtalk.type import XTalk

# Phase submodule imports
from .likelihood_setup import (
    prepare_likelihood_inputs,
    populate_pixel_noise_from_maps,
)
from .pixel_data import extract_pixel_time_delay_data as _extract_pixel_time_delay_data
from .sky_scan import scan_sky_for_best_fit as _scan_sky_for_best_fit
from .sky_statistics import compute_statistics_at_sky_position as _compute_statistics_at_sky_position
from .detection_statistics import (
    get_likelihood_rejection_reason as _get_likelihood_rejection_reason,
    populate_detection_statistics as _populate_detection_statistics,
    update_chirp_mass_statistics as _update_chirp_mass_statistics,
    compute_sky_error_region as _compute_sky_error_region,
)
from .dpf import calculate_dpf as _calculate_dpf
from .typing import SkyStatistics, SkyMapStatistics

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pycwb.config.config import Config

logger = logging.getLogger(__name__)


def evaluate_fragment_clusters(
    config: Config,
    fragment_clusters: list[FragmentCluster],
    strains: list[TimeSeries],
    MRAcatalog: str,
    nRMS: list[TimeFrequencyMap] | None = None,
    xtalk: XTalk | None = None,
) -> list[list[tuple[Cluster, SkyMapStatistics]]]:
    """
    Convenience wrapper for interactive / legacy use.

    Internally calls :func:`prepare_likelihood_inputs` once and then calls
    :func:`evaluate_cluster_likelihood` for every surviving cluster across all lags, avoiding
    repeated sky-pattern computation and runtime-parameter resolution.

    Parameters
    ----------
    config : Config
        Analysis configuration.
    fragment_clusters : list[FragmentCluster]
        One :class:`~pycwb.types.network_cluster.FragmentCluster` per lag —
        the direct output of
        :func:`~pycwb.modules.super_cluster_native.super_cluster.supercluster_wrapper`.
        Clusters with ``cluster_status != 0`` are skipped automatically.
    strains : list
        Whitened strain time series (one per IFO); used for sky-pattern
        computation inside :func:`prepare_likelihood_inputs`.
    MRAcatalog : str
        Path to the MRA catalog (cross-talk coefficients).
    nRMS : list | None
        Per-IFO TF noise maps from data conditioning.  When provided, each
        pixel's ``noise_rms`` is populated so physical-unit quantities (hrss,
        noise) are correct.
    xtalk : XTalk | None
        Pre-loaded cross-talk catalog.  When *None* it is loaded from
        ``MRAcatalog``.

    Returns
    -------
    list[list[tuple[Cluster, SkyMapStatistics]]]
        ``results[lag]`` is a list of ``(result_cluster, sky_stats)`` tuples
        for every cluster that passed the likelihood veto in that lag.
        Empty inner lists indicate no accepted clusters for that lag.
    """
    timer_start = time.perf_counter()

    strains = [TimeSeries.from_input(s) for s in strains]

    if xtalk is None:
        xtalk = XTalk.load(MRAcatalog, dump=True)

    likelihood_setup = prepare_likelihood_inputs(config, strains, config.nIFO)

    results = []
    for fragment_cluster in fragment_clusters:
        lag_results = []
        for k, selected_cluster in enumerate(fragment_cluster.clusters):
            if selected_cluster.cluster_status > 0:
                continue
            selected_cluster.cluster_id = k + 1
            result_cluster, sky_stats = evaluate_cluster_likelihood(
                config.nIFO, selected_cluster, config,
                cluster_id=k + 1, nRMS=nRMS, setup=likelihood_setup, xtalk=xtalk,
            )
            if result_cluster is None or result_cluster.cluster_status != -1:
                logger.info("likelihood rejected cluster %d (%d pixels)",
                            k + 1, len(selected_cluster.pixel_arrays))
                continue
            logger.info("likelihood accepted cluster %d (%d pixels)",
                        k + 1, len(result_cluster.pixel_arrays))
            lag_results.append((result_cluster, sky_stats))
        results.append(lag_results)

    total_accepted = sum(len(r) for r in results)
    logger.info("Likelihood wrapper done: %d accepted cluster(s) across %d lag(s)", total_accepted, len(fragment_clusters))
    logger.info("Likelihood wrapper time: %.2f s", time.perf_counter() - timer_start)

    return results


def evaluate_cluster_likelihood(
    nIFO: int,
    cluster: Cluster,
    config: Config,
    MRAcatalog: str | None = None,
    strains: list[TimeSeries] | None = None,
    cluster_id: int | None = None,
    nRMS: list[TimeFrequencyMap] | None = None,
    setup: dict | None = None,
    xtalk: XTalk | None = None,
    supercluster_setup: dict | None = None,
) -> tuple[Cluster | None, SkyMapStatistics | None]:
    """
    Evaluate the likelihood for a single cluster.

    When ``setup`` and ``xtalk`` are pre-computed (the normal multi-lag workflow),
    they are used directly.  For one-off standalone use, pass ``MRAcatalog`` and
    ``strains`` (or ``supercluster_setup``) and they are built automatically.
    For multi-cluster / multi-lag processing, prefer :func:`evaluate_fragment_clusters`.

    Parameters
    ----------
    nIFO : int
        Number of interferometers.
    cluster : Cluster
        Cluster with ``td_amp`` already set on every pixel
        (guaranteed by :func:`~pycwb.modules.super_cluster_native.super_cluster.supercluster_single_lag`).
    config : Config
        Analysis configuration.
    MRAcatalog : str or None, optional
        Path to the MRA catalog; used to load ``xtalk`` when ``xtalk`` is *None*.
    strains : list or None, optional
        Whitened strain time series; used to build ``setup`` when ``setup`` is *None*
        and ``supercluster_setup`` is also *None*.
    cluster_id : int or None, optional
        Opaque cluster identifier for logging.
    nRMS : list or None, optional
        Per-IFO TF noise maps from data conditioning.  When provided each pixel's
        ``noise_rms`` is populated so that physical-unit quantities (hrss, noise) are correct.
    setup : dict or None, optional
        Pre-computed segment-level inputs from :func:`prepare_likelihood_inputs`.  Built
        automatically when *None* if ``strains`` or ``supercluster_setup`` is provided.
    xtalk : XTalk or None, optional
        Pre-loaded cross-talk catalog.  Loaded from ``MRAcatalog`` automatically when *None*.
    supercluster_setup : dict or None, optional
        If provided and ``setup`` is *None*, its sky-pattern arrays are reused to avoid
        duplicate computation.

    Returns
    -------
    tuple[Cluster or None, SkyMapStatistics or None]
        The updated cluster and full skymap statistics, or ``(None, None)`` if the
        cluster is rejected.
    """
    if xtalk is None:
        if MRAcatalog is None:
            raise ValueError(
                "likelihood(): xtalk or MRAcatalog must be provided. "
                "For multi-cluster / multi-lag use, call evaluate_fragment_clusters() instead."
            )
        xtalk = XTalk.load(MRAcatalog, dump=True)
    if setup is None:
        input_sky_delay_samples = None
        input_plus_antenna_patterns = None
        input_cross_antenna_patterns = None
        if supercluster_setup is not None:
            input_sky_delay_samples = supercluster_setup.get("ml_likelihood", supercluster_setup.get("ml"))
            input_plus_antenna_patterns = supercluster_setup.get("FP_likelihood", supercluster_setup.get("FP"))
            input_cross_antenna_patterns = supercluster_setup.get("FX_likelihood", supercluster_setup.get("FX"))
        if strains is None and input_sky_delay_samples is None:
            raise ValueError(
                "likelihood(): setup, strains, or supercluster_setup must be provided. "
                "For multi-cluster / multi-lag use, call evaluate_fragment_clusters() instead."
            )
        setup = prepare_likelihood_inputs(
            config, strains, nIFO,
            ml=input_sky_delay_samples,
            FP=input_plus_antenna_patterns,
            FX=input_cross_antenna_patterns,
            ml_big=supercluster_setup.get("ml_big_cluster") if supercluster_setup else None,
            FP_big=supercluster_setup.get("FP_big_cluster") if supercluster_setup else None,
            FX_big=supercluster_setup.get("FX_big_cluster") if supercluster_setup else None,
            big_cluster_healpix_order=supercluster_setup.get("big_cluster_healpix_order") if supercluster_setup else None,
        )
    if config is None:
        raise ValueError(
            "likelihood(): config must be provided. Without it, hrss/strain are zero and "
            "gps_time/central_freq fall back to coarser supercluster estimates."
        )
    timer_start = time.perf_counter()
    stage_timings: dict[str, float] = {}
    logger.info("-------------------------------------------------------")
    logger.info("-> Processing cluster-id=%d|pixels=%d", int(cluster_id) if cluster_id is not None else -1, len(cluster.pixel_arrays))
    logger.info("   ----------------------------------------------------")

    # Populate pixel noise_rms from the nRMS TF maps so downstream physical-unit quantities
    # (hrss, noise) are correct.  Each pixel stores the noise floor at its (freq_bin, time_bin).
    if nRMS is not None and len(nRMS) == nIFO:
        cluster.pixel_arrays.populate_noise_rms(nRMS)

    network_energy_threshold = setup["network_energy_threshold"]
    xgb_rho_mode             = setup["xgb_rho_mode"]
    gamma_regulator          = setup["gamma_regulator"]
    delta_regulator          = setup["delta_regulator"]
    net_rho_threshold        = setup["net_rho_threshold"]
    netEC_threshold          = setup["netEC_threshold"]
    netCC                    = setup["netCC"]
    sky_delay_samples        = setup["ml"]    # legacy key: (nIFO, n_sky)
    plus_antenna_patterns    = setup["FP_t"]  # (n_sky, nIFO) float32 — already transposed
    cross_antenna_patterns   = setup["FX_t"]  # (n_sky, nIFO) float32 — already transposed
    n_sky                    = setup["n_sky"]
    sky_valid_indices        = setup.get("sky_valid_indices", np.arange(n_sky, dtype=np.int64))

    # regularization[0] = delta * sqrt(2): amplitude regulator; regularization[1] filled below by DPF scan
    regularization = np.array([delta_regulator * np.sqrt(2), 0., 0.], dtype=np.float32)
    n_pixels = len(cluster.pixel_arrays)

    # --- Big-cluster sky thinning (mirrors C++ network::likelihoodWP bBB logic) ---
    # C++: bBB = (V > wdmMRA.nRes * csize) → use coarser healpix sky grid in the sky loop.
    # C++ does NOT truncate pixels — it keeps all pixels and reduces the sky resolution.
    _precision = int(abs(getattr(config, 'precision', 0) or 0))
    _csize = _precision % 65536
    _nres  = int(getattr(config, 'nRES', 1) or 1)
    _bBB = (_csize > 0 and n_pixels > _nres * _csize
            and setup.get("ml_big_cluster") is not None)
    if _bBB:
        sky_delay_samples = setup["ml_big_cluster"]
        plus_antenna_patterns = setup["FP_big_cluster_t"]
        cross_antenna_patterns = setup["FX_big_cluster_t"]
        n_sky = setup["n_sky_big_cluster"]
        sky_valid_indices = setup.get("sky_valid_indices_big", np.arange(n_sky, dtype=np.int64))
        logger.info(
            "Cluster-id=%s is big (%d px > csize_threshold=%d): "
            "using coarse sky grid (%d directions, healpix order=%s)",
            cluster_id, n_pixels, _nres * _csize, n_sky,
            setup.get("big_cluster_healpix_order"),
        )

    # --- Prepare per-cluster inputs ---
    _t0 = time.perf_counter()
    cluster_xtalk_lookup, cluster_xtalk = xtalk.get_xtalk_pixels(cluster.pixel_arrays, True)
    noise_weights, td_phase0, td_phase90, td_energy = _extract_pixel_time_delay_data(
        None, nIFO, pixel_arrays=cluster.pixel_arrays
    )
    # Reshape to (ndelay, nifo, npix) and cast to float32 for numba; FP/FX already prepared in setup
    td_phase0 = np.transpose(td_phase0.astype(np.float32), (2, 0, 1))
    td_phase90 = np.transpose(td_phase90.astype(np.float32), (2, 0, 1))
    noise_weights = noise_weights.T.astype(np.float32)
    stage_timings["data_prep"] = time.perf_counter() - _t0

    # regularization[1]: DPF-based energy regulator (gamma-corrected, sky-scan average)
    _t0 = time.perf_counter()
    regularization[1] = _calculate_dpf(
        plus_antenna_patterns, cross_antenna_patterns, noise_weights, n_sky, nIFO,
        gamma_regulator, network_energy_threshold, sky_valid_indices,
    )
    stage_timings["dpf_regulator"] = time.perf_counter() - _t0

    # --- Sky scan: find the optimal sky direction (l_max) ---
    # Returns a tuple; numba cannot return dataclasses directly
    _t0 = time.perf_counter()
    skymap_statistics = _scan_sky_for_best_fit(
        nIFO, n_pixels, n_sky,
        plus_antenna_patterns, cross_antenna_patterns, noise_weights,
        td_phase0, td_phase90, sky_delay_samples, regularization, netCC,
        delta_regulator, network_energy_threshold, sky_valid_indices,
    )
    skymap_statistics = SkyMapStatistics.from_tuple(skymap_statistics)
    stage_timings["sky_scan"] = time.perf_counter() - _t0

    if skymap_statistics.sky_stat_max <= 0.0:
        logger.info("Cluster rejected: non-positive sky_stat_max=%.3f", skymap_statistics.sky_stat_max)
        stage_timings["total"] = time.perf_counter() - timer_start
        logger.info("-------------------------------------------------------")
        logger.info("Total events: %d", 0)
        logger.info("Total time: %.2f s", stage_timings["total"])
        logger.info("-------------------------------------------------------")
        return None, None

    # --- Compute normalised sky probability map (softmax over nSkyStat) ---
    _t0 = time.perf_counter()
    _sky_stat_f64 = skymap_statistics.nSkyStat.astype(np.float64)
    _sky_stat_shifted = _sky_stat_f64 - _sky_stat_f64.max()
    _exp_stat = np.exp(_sky_stat_shifted)
    skymap_statistics.nProbability = (_exp_stat / _exp_stat.sum()).astype(np.float32)
    stage_timings["sky_probability"] = time.perf_counter() - _t0

    # --- Convert l_max index to (theta, phi) sky angles ---
    _t0 = time.perf_counter()
    _healpix_order = setup["healpix_order"]
    _ra_arr        = setup["ra_arr"]
    _dec_arr       = setup["dec_arr"]
    _l_max = int(skymap_statistics.l_max)
    # cWB theta: co-latitude in degrees [0, 180]; phi: longitude in degrees [0, 360)
    _theta_rad = float(np.pi / 2.0 - _dec_arr[_l_max])  # co-latitude from declination
    _phi_rad = float(_ra_arr[_l_max])
    _theta_deg = float(np.degrees(_theta_rad)) % 180.0
    _phi_deg = float(np.degrees(_phi_rad)) % 360.0
    stage_timings["sky_coords"] = time.perf_counter() - _t0

    # calculate sky statistics for the cluster at the optimal sky location l_max,
    # dozens of parameters will be returned in SkyStatistics dataclass
    _t0 = time.perf_counter()
    sky_statistics: SkyStatistics = _compute_statistics_at_sky_position(
        skymap_statistics.l_max, nIFO, n_pixels,
        plus_antenna_patterns, cross_antenna_patterns, noise_weights,
        td_phase0, td_phase90, sky_delay_samples, regularization,
        network_energy_threshold,
        cluster_xtalk, cluster_xtalk_lookup,
        xgb_rho_mode=xgb_rho_mode,
    )
    stage_timings["sky_statistics_at_lmax"] = time.perf_counter() - _t0

    # --- Threshold cuts — reject cluster if any condition fails ---
    _t0 = time.perf_counter()
    selected_core_pixels = int(np.count_nonzero(np.asarray(sky_statistics.pixel_mask) > 0))
    logger.info("Selected core pixels: %d / %d", selected_core_pixels, n_pixels)

    rejected = _get_likelihood_rejection_reason(
        sky_statistics,
        network_energy_threshold,
        netEC_threshold,
        net_rho_threshold=net_rho_threshold,
        xgb_rho_mode=xgb_rho_mode,
    )
    stage_timings["threshold_cut"] = time.perf_counter() - _t0
    if rejected:
        logger.debug("Cluster rejected due to threshold cuts: %s", rejected)
        logger.info("   cluster-id|pixels: %5d|%d", int(cluster_id) if cluster_id is not None else -1, len(cluster.pixel_arrays))
        logger.info("\t <- rejected    ")
        stage_timings["total"] = time.perf_counter() - timer_start
        logger.info("-------------------------------------------------------")
        logger.info("Total events: %d", 0)
        logger.info("Total time: %.2f s", stage_timings["total"])
        logger.info("-------------------------------------------------------")
        return None, None

    # --- Fill detection statistics (rho, netCC, waveform energies, per-pixel data) ---
    _t0 = time.perf_counter()
    # Build wdm_list once per cluster here and pass it into populate_detection_statistics
    # to avoid _create_wdm_set_python being called again inside (~1 s saving).
    if config is not None:
        from pycwb.modules.reconstruction.getMRAwaveform import _create_wdm_set_python
        _wdm_list = _create_wdm_set_python(config)
    else:
        _wdm_list = None
    _populate_detection_statistics(
        sky_statistics, skymap_statistics, cluster=cluster,
        n_ifo=nIFO, xtalk=xtalk,
        network_energy_threshold=network_energy_threshold,
        xgb_rho_mode=xgb_rho_mode,
        config=config,
        cluster_xtalk=cluster_xtalk,
        cluster_xtalk_lookup=cluster_xtalk_lookup,
        wdm_list=_wdm_list,
    )
    stage_timings["populate_detection_statistics"] = time.perf_counter() - _t0

    # --- Post-processing: chirp mass and error region ---
    _t0 = time.perf_counter()
    pat0 = (getattr(config, 'pattern', 10) == 0) if config is not None else False
    _update_chirp_mass_statistics(cluster, xgb_rho_mode=xgb_rho_mode, pat0=pat0)
    stage_timings["update_chirp_mass_statistics"] = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    _compute_sky_error_region(cluster)
    stage_timings["compute_sky_error_region"] = time.perf_counter() - _t0

    # --- Store sky localisation metadata ---
    _t0 = time.perf_counter()
    cluster.cluster_meta.l_max = _l_max
    cluster.cluster_meta.theta = _theta_deg
    cluster.cluster_meta.phi = _phi_deg
    # Fall back to supercluster estimates if populate_detection_statistics did not set these
    if cluster.cluster_meta.c_time == 0.0:
        cluster.cluster_meta.c_time = cluster.cluster_time
    if cluster.cluster_meta.c_freq == 0.0:
        cluster.cluster_meta.c_freq = cluster.cluster_freq
    # Time-of-flight delays at l_max per IFO — used by getMRAwaveform for ToF correction
    cluster.sky_time_delay = [float(sky_delay_samples[i, _l_max]) for i in range(nIFO)]
    stage_timings["sky_metadata"] = time.perf_counter() - _t0

    # Mark accepted (mirrors C++ sCuts[id-1] = -1)
    cluster.cluster_status = -1

    detected = cluster.cluster_status == -1
    logger.info("   cluster-id|pixels: %5d|%d", int(cluster_id) if cluster_id is not None else -1, len(cluster.pixel_arrays))
    if detected:
        logger.info("\t -> SELECTED !!!")
    else:
        logger.info("\t <- rejected    ")

    stage_timings["total"] = time.perf_counter() - timer_start
    logger.info("-------------------------------------------------------")
    logger.info("Total events: %d", 1 if detected else 0)
    logger.info("Total time: %.2f s", stage_timings["total"])
    logger.info("Stage timings (CPU):")
    for _stage, _t in stage_timings.items():
        if _stage != "total":
            logger.info("  %-30s %.4f s  (%5.1f%%)", _stage, _t,
                        100.0 * _t / stage_timings["total"] if stage_timings["total"] > 0 else 0)
    logger.info("-------------------------------------------------------")

    # Attach stage timings to skymap_statistics for benchmark collection
    skymap_statistics.stage_timings = stage_timings

    return cluster, skymap_statistics


# ---------------------------------------------------------------------------
# Friendly aliases for researcher readability
# ---------------------------------------------------------------------------

setup_likelihood = prepare_likelihood_inputs
likelihood = evaluate_cluster_likelihood
likelihood_wrapper = evaluate_fragment_clusters
_populate_pixel_noise_rms = populate_pixel_noise_from_maps

# Public API surface for the facade
__all__ = [
    "setup_likelihood", "likelihood", "likelihood_wrapper",
    "prepare_likelihood_inputs",
    "evaluate_cluster_likelihood", "evaluate_fragment_clusters",
]
