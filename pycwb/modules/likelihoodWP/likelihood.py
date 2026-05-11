from __future__ import annotations

from math import sqrt
import logging
import time
import numpy as np
from numba import njit, prange, float32
from pycwb.types.network_cluster import Cluster, FragmentCluster
from pycwb.types.network_pixel import Pixel
from pycwb.types.time_series import TimeSeries
from pycwb.types.time_frequency_map import TimeFrequencyMap
from pycwb.types.detector import compute_sky_delay_and_patterns, _build_sky_directions
from .dpf import calculate_dpf, dpf_np_loops_vec
from .sky_stat import avx_GW_ps, avx_ort_ps, avx_stat_ps, load_data_from_td
from .utils import avx_packet_ps, packet_norm_numpy, gw_norm_numpy, avx_noise_ps, \
        avx_setAMP_ps, avx_pol_ps, avx_loadNULL_ps, xtalk_energy_sum_numpy
from .pixel_batch_ops import load_data_from_pixels_vectorized
from pycwb.modules.xtalk.type import XTalk
from pycwb.modules.xtalk.monster import _compute_null_likelihood_numba
from .typing import SkyStatistics, SkyMapStatistics
from pycwb.modules.reconstruction.getMRAwaveform import (
    _create_wdm_set_python, get_MRA_wave, _pa_to_tuple, _build_wdm_njit_data,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pycwb.config.config import Config


logger = logging.getLogger(__name__)


def _populate_pixel_noise_rms(pixels: list[Pixel], nRMS: list[TimeFrequencyMap]) -> None:
    """
    Populate each ``pixel.data[i].noise_rms`` from the per-IFO TF noise maps.

    The nRMS maps come from the highest-resolution whitening step.  For pixels at
    other resolutions the frequency bin is scaled proportionally to the nRMS grid.

    Parameters
    ----------
    pixels : list[Pixel]
        Cluster pixels.
    nRMS : list[TimeFrequencyMap]
        One TF noise map per IFO from whitening_python.  ``data`` shape is
        ``(n_freq_bins, n_time_bins)`` where n_freq_bins covers [0, fNyq].
    """
    n_ifo = len(nRMS)
    # Precompute nRMS data arrays once
    nrms_data = []
    nrms_shapes = []
    for i in range(n_ifo):
        arr = np.asarray(nRMS[i].data, dtype=np.float64)
        nrms_data.append(arr)
        nrms_shapes.append(arr.shape)  # (n_freq, n_time)

    for pixel in pixels:
        freq_bin = int(pixel.frequency)
        n_freq_pix = int(pixel.layers)  # number of frequency bins at this resolution
        # Derive time bin from composite pixel.time = time_idx * n_freq + freq_bin
        if n_freq_pix > 0:
            time_bin_pix = int(pixel.time) // n_freq_pix
        else:
            time_bin_pix = 0

        for i in range(n_ifo):
            try:
                nf, nt = nrms_shapes[i]
                # Map pixel freq_bin (at resolution n_freq_pix) to nRMS freq bin
                if n_freq_pix > 0 and nf > 0:
                    fb = int(round(freq_bin * nf / n_freq_pix))
                    fb = min(max(fb, 0), nf - 1)
                else:
                    fb = 0
                # Map time bin 
                tb = min(time_bin_pix, nt - 1) if nt > 0 else 0
                val = float(np.abs(nrms_data[i][fb, tb]))
                if val > 0.0:
                    pixel.data[i].noise_rms = val
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Failed to populate noise_rms for pixel at freq_bin=%d, ifo=%d",
                    freq_bin, i, exc_info=True
                )


def likelihood_wrapper(
    config: Config,
    fragment_clusters: list[FragmentCluster],
    strains: list[TimeSeries],
    MRAcatalog: str,
    nRMS: list[TimeFrequencyMap] | None = None,
    xtalk: XTalk | None = None,
) -> list[list[tuple[Cluster, SkyMapStatistics]]]:
    """
    Convenience wrapper for interactive / legacy use.

    Internally calls :func:`setup_likelihood` once and then calls
    :func:`likelihood` for every surviving cluster across all lags, avoiding
    repeated sky-pattern computation and runtime-parameter resolution.

    Parameters
    ----------
    config : Config
        Analysis configuration.
    fragment_clusters : list[FragmentCluster]
        One :class:`~pycwb.types.network_cluster.FragmentCluster` per lag —
        the direct output of
        :func:`~pycwb.modules.super_cluster.super_cluster.supercluster_wrapper`.
        Clusters with ``cluster_status != 0`` are skipped automatically.
    strains : list
        Whitened strain time series (one per IFO); used for sky-pattern
        computation inside :func:`setup_likelihood`.
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

    likelihood_setup = setup_likelihood(config, strains, config.nIFO)

    results = []
    for fragment_cluster in fragment_clusters:
        lag_results = []
        for k, selected_cluster in enumerate(fragment_cluster.clusters):
            if selected_cluster.cluster_status > 0:
                continue
            selected_cluster.cluster_id = k + 1
            result_cluster, sky_stats = likelihood(
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


def setup_likelihood(
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
    Pre-compute all job-segment-level (lag/cluster-independent) inputs for likelihood.

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
        ``dec_arr``.
    """

    if config is None:
        raise ValueError("config is required for pure-Python likelihood")
    acor = float(getattr(config, "Acore"))
    gamma = float(getattr(config, "gamma", 0.0))
    delta = float(getattr(config, "delta", 0.0))
    net_rho = float(getattr(config, "netRHO", 0.0))
    netCC = float(getattr(config, "netCC", 0.0))
    xgb_rho_mode = bool(getattr(config, "xgb_rho_mode", False))

    network_energy_threshold = 2 * acor * acor * nIFO
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
        ml_raw, FP_raw, FX_raw = np.asarray(ml), np.asarray(FP), np.asarray(FX)
    else:
        ml_raw, FP_raw, FX_raw = load_data_from_ifo(nIFO, strains, config)
    n_sky = int(ml_raw.shape[1])

    # Pre-transpose and cast to float32 so per-cluster calls skip that work
    FP_t = FP_raw.T.astype(np.float32)  # (n_sky, nIFO)
    FX_t = FX_raw.T.astype(np.float32)  # (n_sky, nIFO)

    # Big-cluster coarse sky arrays (for bBB handling in network::likelihoodWP)
    if ml_big is not None and FP_big is not None and FX_big is not None:
        ml_big_raw = np.asarray(ml_big)
        FP_big_t   = np.asarray(FP_big).T.astype(np.float32)
        FX_big_t   = np.asarray(FX_big).T.astype(np.float32)
        n_sky_big  = int(ml_big_raw.shape[1])
    else:
        ml_big_raw = None
        FP_big_t   = None
        FX_big_t   = None
        n_sky_big  = None

    healpix_order = int(getattr(config, 'healpix', 0)) if hasattr(config, 'healpix') else None
    ra_arr, dec_arr = _build_sky_directions(n_sky, healpix_order)

    return {
        "network_energy_threshold": network_energy_threshold,
        "xgb_rho_mode": xgb_rho_mode,
        "gamma_regulator": gamma_regulator,
        "delta_regulator": delta_regulator,
        "net_rho_threshold": net_rho_threshold,
        "netEC_threshold": netEC_threshold,
        "netCC": netCC,
        "ml": ml_raw,          # (nIFO, n_sky) — integer time-delay indices
        "FP": FP_raw,          # (nIFO, n_sky) — raw, pre-transpose
        "FX": FX_raw,          # (nIFO, n_sky) — raw, pre-transpose
        "FP_t": FP_t,          # (n_sky, nIFO) float32 — ready for numba
        "FX_t": FX_t,          # (n_sky, nIFO) float32 — ready for numba
        "n_sky": n_sky,
        "healpix_order": healpix_order,
        "ra_arr": ra_arr,
        "dec_arr": dec_arr,
        "ml_big_cluster": ml_big_raw,
        "FP_big_cluster_t": FP_big_t,
        "FX_big_cluster_t": FX_big_t,
        "n_sky_big_cluster": n_sky_big,
        "big_cluster_healpix_order": big_cluster_healpix_order,
    }


def likelihood(
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
    Calculate the likelihood for a single cluster.

    When ``setup`` and ``xtalk`` are pre-computed (the normal multi-lag workflow),
    they are used directly.  For one-off standalone use, pass ``MRAcatalog`` and
    ``strains`` (or ``supercluster_setup``) and they are built automatically.
    For multi-cluster / multi-lag processing, prefer :func:`likelihood_wrapper`.

    Parameters
    ----------
    nIFO : int
        Number of interferometers.
    cluster : Cluster
        Cluster with ``td_amp`` already set on every pixel
        (guaranteed by :func:`~pycwb.modules.super_cluster.super_cluster.supercluster_single_lag`).
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
        Pre-computed segment-level inputs from :func:`setup_likelihood`.  Built
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
                "For multi-cluster / multi-lag use, call likelihood_wrapper() instead."
            )
        xtalk = XTalk.load(MRAcatalog, dump=True)
    if setup is None:
        ml, FP, FX = None, None, None
        if supercluster_setup is not None:
            ml = supercluster_setup.get("ml_likelihood", supercluster_setup.get("ml"))
            FP = supercluster_setup.get("FP_likelihood", supercluster_setup.get("FP"))
            FX = supercluster_setup.get("FX_likelihood", supercluster_setup.get("FX"))
        if strains is None and ml is None:
            raise ValueError(
                "likelihood(): setup, strains, or supercluster_setup must be provided. "
                "For multi-cluster / multi-lag use, call likelihood_wrapper() instead."
            )
        setup = setup_likelihood(config, strains, nIFO, ml=ml, FP=FP, FX=FX,
                                 ml_big=supercluster_setup.get("ml_big_cluster") if supercluster_setup else None,
                                 FP_big=supercluster_setup.get("FP_big_cluster") if supercluster_setup else None,
                                 FX_big=supercluster_setup.get("FX_big_cluster") if supercluster_setup else None,
                                 big_cluster_healpix_order=supercluster_setup.get("big_cluster_healpix_order") if supercluster_setup else None)
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
    ml                       = setup["ml"]    # (nIFO, n_sky)
    FP                       = setup["FP_t"]  # (n_sky, nIFO) float32 — already transposed
    FX                       = setup["FX_t"]  # (n_sky, nIFO) float32 — already transposed
    n_sky                    = setup["n_sky"]

    # REG[0] = delta * sqrt(2): amplitude regulator; REG[1] filled below by DPF scan
    REG = np.array([delta_regulator * np.sqrt(2), 0., 0.], dtype=np.float32)
    n_pix = len(cluster.pixel_arrays)

    # --- Big-cluster sky thinning (mirrors C++ network::likelihoodWP bBB logic) ---
    # C++: bBB = (V > wdmMRA.nRes * csize) → use coarser healpix sky grid in the sky loop.
    # C++ does NOT truncate pixels — it keeps all pixels and reduces the sky resolution.
    _precision = int(abs(getattr(config, 'precision', 0) or 0))
    _csize = _precision % 65536
    _nres  = int(getattr(config, 'nRES', 1) or 1)
    _bBB = (_csize > 0 and n_pix > _nres * _csize
            and setup.get("ml_big_cluster") is not None)
    if _bBB:
        ml    = setup["ml_big_cluster"]
        FP    = setup["FP_big_cluster_t"]
        FX    = setup["FX_big_cluster_t"]
        n_sky = setup["n_sky_big_cluster"]
        logger.info(
            "Cluster-id=%s is big (%d px > csize_threshold=%d): "
            "using coarse sky grid (%d directions, healpix order=%s)",
            cluster_id, n_pix, _nres * _csize, n_sky,
            setup.get("big_cluster_healpix_order"),
        )

    # --- Prepare per-cluster inputs ---
    _t0 = time.perf_counter()
    cluster_xtalk_lookup, cluster_xtalk = xtalk.get_xtalk_pixels(cluster.pixel_arrays, True)
    rms, td00, td90, td_energy = load_data_from_pixels(
        None, nIFO, pixel_arrays=cluster.pixel_arrays
    )
    # Reshape to (ndelay, nifo, npix) and cast to float32 for numba; FP/FX already prepared in setup
    td00 = np.transpose(td00.astype(np.float32), (2, 0, 1))
    td90 = np.transpose(td90.astype(np.float32), (2, 0, 1))
    rms = rms.T.astype(np.float32)
    stage_timings["data_prep"] = time.perf_counter() - _t0

    # REG[1]: DPF-based energy regulator (gamma-corrected, sky-scan average)
    _t0 = time.perf_counter()
    REG[1] = calculate_dpf(FP, FX, rms, n_sky, nIFO, gamma_regulator, network_energy_threshold)
    stage_timings["dpf_regulator"] = time.perf_counter() - _t0

    # --- Sky scan: find the optimal sky direction (l_max) ---
    # Returns a tuple; numba cannot return dataclasses directly
    _t0 = time.perf_counter()
    skymap_statistics = find_optimal_sky_localization(nIFO, n_pix, n_sky, FP, FX, rms, td00, td90, ml, REG, netCC,
                                          delta_regulator, network_energy_threshold)
    skymap_statistics = SkyMapStatistics.from_tuple(skymap_statistics)
    stage_timings["sky_scan"] = time.perf_counter() - _t0

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
    sky_statistics: SkyStatistics = calculate_sky_statistics(skymap_statistics.l_max, nIFO, n_pix, 
                                                             FP, FX, rms, td00, td90, ml, REG, 
                                                             network_energy_threshold,
                                                             cluster_xtalk, cluster_xtalk_lookup,
                                                             xgb_rho_mode=xgb_rho_mode)
    stage_timings["sky_statistics_at_lmax"] = time.perf_counter() - _t0

    # --- Threshold cuts — reject cluster if any condition fails ---
    _t0 = time.perf_counter()
    selected_core_pixels = int(np.count_nonzero(np.asarray(sky_statistics.pixel_mask) > 0))
    logger.info("Selected core pixels: %d / %d", selected_core_pixels, n_pix)

    rejected = threshold_cut(
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
    # Build wdm_list once per cluster here and pass it into fill_detection_statistic
    # to avoid _create_wdm_set_python being called again inside (~1 s saving).
    if config is not None:
        from pycwb.modules.reconstruction.getMRAwaveform import _create_wdm_set_python
        _wdm_list = _create_wdm_set_python(config)
    else:
        _wdm_list = None
    fill_detection_statistic(sky_statistics, skymap_statistics, cluster=cluster,
                             n_ifo=nIFO, xtalk=xtalk,
                             network_energy_threshold=network_energy_threshold,
                             xgb_rho_mode=xgb_rho_mode,
                             config=config,
                             cluster_xtalk=cluster_xtalk,
                             cluster_xtalk_lookup=cluster_xtalk_lookup,
                             wdm_list=_wdm_list)
    stage_timings["fill_detection_statistic"] = time.perf_counter() - _t0

    # --- Post-processing: chirp mass and error region ---
    _t0 = time.perf_counter()
    pat0 = (getattr(config, 'pattern', 10) == 0) if config is not None else False
    get_chirp_mass(cluster, xgb_rho_mode=xgb_rho_mode, pat0=pat0)
    stage_timings["get_chirp_mass"] = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    get_error_region(cluster)
    stage_timings["get_error_region"] = time.perf_counter() - _t0

    # --- Store sky localisation metadata ---
    _t0 = time.perf_counter()
    cluster.cluster_meta.l_max = _l_max
    cluster.cluster_meta.theta = _theta_deg
    cluster.cluster_meta.phi = _phi_deg
    # Fall back to supercluster estimates if fill_detection_statistic did not set these
    if cluster.cluster_meta.c_time == 0.0:
        cluster.cluster_meta.c_time = cluster.cluster_time
    if cluster.cluster_meta.c_freq == 0.0:
        cluster.cluster_meta.c_freq = cluster.cluster_freq
    # Time-of-flight delays at l_max per IFO — used by getMRAwaveform for ToF correction
    cluster.sky_time_delay = [float(ml[i, _l_max]) for i in range(nIFO)]
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


def load_data_from_pixels(
    pixels: list[Pixel],
    nifo: int,
    pixel_arrays=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load data from pixels into numpy arrays for numba / JAX processing.

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
    rms       : (nifo, n_pix) float32 — normalised inverse-RMS weights
    td00      : (nifo, n_pix, tsize2) float32
    td90      : (nifo, n_pix, tsize2) float32
    td_energy : (nifo, n_pix, tsize2) float32
    """
    if pixel_arrays is not None and pixel_arrays.has_td_amp():
        return _load_data_from_pixel_arrays(pixel_arrays)
    return load_data_from_pixels_vectorized(pixels, nifo)


def _load_data_from_pixel_arrays(
    pa,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast path: extract rms/td arrays directly from a ``PixelArrays``."""
    # noise_rms: (n_ifo, n_pix) float32
    inv_rms = 1.0 / pa.noise_rms.astype(np.float64)          # (n_ifo, n_pix)
    rms_pix = 1.0 / np.sqrt(np.sum(inv_rms ** 2, axis=0))    # (n_pix,)
    rms = (inv_rms * rms_pix[np.newaxis, :]).astype(np.float32)  # (n_ifo, n_pix)

    # td_amp_dense: (n_pix, n_ifo, tsize) → split into 00/90 halves
    td = pa.td_amp_dense()          # (n_pix, n_ifo, tsize)
    tsize2 = td.shape[2] // 2
    td00 = td[:, :, :tsize2].transpose(1, 0, 2)   # (n_ifo, n_pix, tsize2)
    td90 = td[:, :, tsize2:].transpose(1, 0, 2)
    td_energy = td00 ** 2 + td90 ** 2

    return rms, td00, td90, td_energy


def load_data_from_ifo(
    nIFO: int,
    strains: list[TimeSeries] | None = None,
    config: Config | None = None,
    ml: np.ndarray | None = None,
    FP: np.ndarray | None = None,
    FX: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the sky delay/pattern data into numpy arrays for numba processing.
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
    ml_arr, fp_arr, fx_arr = compute_sky_delay_and_patterns(
        ifos=getattr(config, "ifo"),
        ref_ifo=getattr(config, "refIFO"),
        sample_rate=float(_TDRate_lh),
        td_size=max(int(getattr(config, "TDSize")) * _upTDF_lh,
                    int(getattr(config, "max_delay", 0.0) * float(_TDRate_lh)) + 1),
        gps_time=gps_time,
        healpix_order=int(getattr(config, "healpix", 0)) if hasattr(config, "healpix") else None,
        n_sky=None,
    )
    return ml_arr, fp_arr, fx_arr


@njit(cache=True, parallel=True)
def find_optimal_sky_localization(n_ifo, n_pix, n_sky, FP, FX, rms, td00, td90, ml, REG, netCC, delta_regulator, network_energy_threshold):
    """
    Find the optimal sky localization by calculating sky statistics for each sky location.

    .. note::
        Decorated with ``@njit`` — Python type annotations are not added to the
        signature; numba infers array types at compile time.

    Parameters
    ----------
    n_ifo : int
        Number of interferometers.
    n_pix : int
        Number of pixels.
    n_sky : int
        Number of sky locations.
    FP : np.ndarray
        f+ polarization data, shape (n_sky, nIFO), float32.
    FX : np.ndarray
        fx polarization data, shape (n_sky, nIFO), float32.
    rms : np.ndarray
        Per-IFO per-pixel RMS values, shape (nIFO, n_pix), float32.
    td00 : np.ndarray
        Time-delayed in-phase amplitudes, shape (ndelay, nIFO, n_pix), float32.
    td90 : np.ndarray
        Time-delayed quadrature amplitudes, shape (ndelay, nIFO, n_pix), float32.
    ml : np.ndarray
        Sky-delay index array, shape (nIFO, n_sky), int.
    REG : np.ndarray
        Regularization parameters, shape (3,), float32.
    netCC : float
        Network correlation coefficient threshold.
    delta_regulator : float
        Delta regulator value.
    network_energy_threshold : float
        Energy threshold for the network.

    Returns
    -------
    tuple
        ``(l_max, nAntenaPrior, nAlignment, nLikelihood, nNullEnergy, nCorrEnergy,
        nCorrelation, nSkyStat, nDisbalance, nNetIndex, nEllipticity, nPolarisation)``
        where ``l_max`` is the index of the sky location with maximum cross-correlation
        statistic and all ``n*`` arrays are float32 of length n_sky.
    """
    # Arrays are pre-transposed and cast to float32 by setup_likelihood / the caller.
    REG = REG.astype(np.float32)

    # --- Allocate per-sky-location statistics arrays ---
    nAlignment = np.zeros(n_sky, dtype=float32)
    nLikelihood = np.zeros(n_sky, dtype=float32)
    nNullEnergy = np.zeros(n_sky, dtype=float32)
    nCorrEnergy = np.zeros(n_sky, dtype=float32)
    nCorrelation = np.zeros(n_sky, dtype=float32)
    nSkyStat = np.zeros(n_sky, dtype=float32)
    nProbability = np.zeros(n_sky, dtype=float32)
    nDisbalance = np.zeros(n_sky, dtype=float32)
    nNetIndex = np.zeros(n_sky, dtype=float32)
    nEllipticity = np.zeros(n_sky, dtype=float32)
    nPolarisation = np.zeros(n_sky, dtype=float32)
    nAntenaPrior = np.zeros(n_sky, dtype=float32)

    offset = int(td00.shape[0] / 2)
    # TODO: sky sky mask
    AA_array = np.zeros(n_sky, dtype=float32)
    for sky_idx in prange(n_sky):
        # --- Apply time delay and load pixel data for this sky direction ---
        v00 = np.empty((n_ifo, n_pix), dtype=float32)
        v90 = np.empty((n_ifo, n_pix), dtype=float32)
        for i in range(n_ifo):
            v00[i] = td00[ml[i, sky_idx] + offset, i]
            v90[i] = td90[ml[i, sky_idx] + offset, i]

        # --- Compute data energy and pixel mask ---
        Eo, NN, energy_total, mask = load_data_from_td(v00, v90, network_energy_threshold)

        # --- Compute DPF (dominant polarisation frame) f+/fx and their norms ---
        _, f, F, fp, fx, si, co, ni = dpf_np_loops_vec(FP[sky_idx], FX[sky_idx], rms)

        # --- Project data onto GW strain packet; select pixels above threshold ---
        Mo, ps, pS, mask, au, AU, av, AV = avx_GW_ps(v00, v90, f, F, fp, fx, ni, energy_total, mask, REG)

        # --- Orthogonalise signal amplitudes (+ and x polarisations) ---
        Lo, si, co, ee, EE = avx_ort_ps(ps, pS, mask)

        # --- Compute coherent network statistics ---
        Cr, Ec, Mp, No, coherent_energy, _, _ = avx_stat_ps(v00, v90, ps, pS, si, co, mask)

        CH = No / (n_ifo * Mo + sqrt(Mo))  # chi2 in TF domain
        cc = CH if CH > float(1.0) else 1.0  # noise correction factor in TF domain
        Co = Ec / (Ec + No * cc - Mo * (n_ifo - 1))  # network correlation coefficient in TF

        if Cr < netCC:
            continue

        # --- Sky statistics: likelihood and cross-correlation ---
        aa = Eo - No if Eo > float32(0.) else float32(0.)  # likelihood skystat
        AA = aa * Co  # x-correlation skystat
        nProbability[sky_idx] = aa if delta_regulator < 0 else AA

        # --- Antenna sensitivity: energy-weighted f+/fx norms ---
        ff, FF, ee = float32(0.), float32(0.), float32(0.)

        for j in range(n_pix):
            if mask[j] <= 0:
                continue
            ee += energy_total[j]  # total energy
            ff += fp[j] * energy_total[j]  # |f+|^2
            FF += fx[j] * energy_total[j]  # |fx|^2
        ff = ff / ee if ee > float32(0.) else float32(0.)
        FF = FF / ee if ee > float32(0.) else float32(0.)

        nAntenaPrior[sky_idx] = sqrt(ff + FF)
        nAlignment[sky_idx] = sqrt(FF / ff) if ff > float32(0.) else float32(0.)
        # --- Store all per-sky statistics ---
        nLikelihood[sky_idx] = Eo - No
        nNullEnergy[sky_idx] = No
        nCorrEnergy[sky_idx] = Ec
        nCorrelation[sky_idx] = Co
        nSkyStat[sky_idx] = AA
        nDisbalance[sky_idx] = CH
        nNetIndex[sky_idx] = cc
        nEllipticity[sky_idx] = Cr
        nPolarisation[sky_idx] = Mp

        AA_array[sky_idx] = AA
    # Mirror C++ tie-breaking: C++ uses `if (AA >= STAT)` in a forward loop,
    # so the LAST pixel with the maximum value wins on ties.
    # np.argmax returns the FIRST, so scan forward explicitly.
    STAT = np.float32(-1.e12)
    l_max = 0
    for _l in range(n_sky):
        if AA_array[_l] >= STAT:
            STAT = AA_array[_l]
            l_max = _l

    return (l_max, nAntenaPrior, nAlignment, nLikelihood, nNullEnergy, nCorrEnergy, \
              nCorrelation, nSkyStat, nDisbalance, nNetIndex, nEllipticity, nPolarisation)


def calculate_sky_statistics(
    sky_idx: int,
    n_ifo: int,
    n_pix: int,
    FP: np.ndarray,
    FX: np.ndarray,
    rms: np.ndarray,
    td00: np.ndarray,
    td90: np.ndarray,
    ml: np.ndarray,
    REG: np.ndarray,
    network_energy_threshold: float,
    cluster_xtalk: np.ndarray,
    cluster_xtalk_lookup_table: np.ndarray,
    DEBUG: bool = False,
    xgb_rho_mode: bool = False,
) -> SkyStatistics:
    """
    Calculate the sky statistics for a specific sky location.
    Parameters
    ----------
    sky_idx : int
        Index of the sky location.
    n_ifo : int
        Number of interferometers.
    n_pix : int
        Number of pixels.
    FP : np.ndarray
        f+ polarization data for each interferometer, shape (n_sky, nIFO).
    FX : np.ndarray
        fx polarization data for each interferometer, shape (n_sky, nIFO).
    rms : np.ndarray
        RMS values, shape (nIFO, n_pix).
    td00 : np.ndarray
        Time-delayed in-phase data, shape (ndelay, nIFO, n_pix).
    td90 : np.ndarray
        Time-delayed quadrature data, shape (ndelay, nIFO, n_pix).
    ml : np.ndarray
        Sky-delay index array, shape (nIFO, n_sky).
    REG : np.ndarray
        Regularization parameters, shape (3,).
    network_energy_threshold : float
        Energy threshold for the network.
    cluster_xtalk : object
        Cluster XTalk object containing cross-talk coefficients.
    cluster_xtalk_lookup_table : object
        Lookup table for cross-talk.
    DEBUG : bool, optional
        If True, emit extra debug output. Default is False.

    Returns
    -------
    SkyStatistics
        Dataclass containing the sky statistics for the specified sky location.
    """
    v00 = np.empty((n_ifo, n_pix), dtype=np.float32)
    v90 = np.empty((n_ifo, n_pix), dtype=np.float32)
    td_energy = np.zeros((n_ifo, n_pix), dtype=np.float32)

    offset = int(td00.shape[0] / 2)

    # --- Apply time delay for this sky direction ---
    for i in range(n_ifo):
        v00[i] = td00[ml[i, sky_idx] + offset, i]
        v90[i] = td90[ml[i, sky_idx] + offset, i]

    # Per-pixel TF energy (v00² + v90²)
    for i in range(n_ifo):
        for j in range(n_pix):
            td_energy[i, j] = v00[i, j] * v00[i, j] + v90[i, j] * v90[i, j]

    # --- Compute total energy, pixel activity mask ---
    Eo, NN, energy_total, mask = load_data_from_td(v00, v90, network_energy_threshold)

    # --- Dominant polarisation frame: f+/fx projections and norms ---
    _, f, F, fp, fx, si, co, ni = dpf_np_loops_vec(FP[sky_idx], FX[sky_idx], rms)

    # --- Project onto GW strain packet; select pixels above threshold ---
    Mo, ps, pS, mask, au, AU, av, AV = avx_GW_ps(v00, v90, f, F, fp, fx, ni, energy_total, mask, REG)

    # --- Orthogonalise signal amplitudes (+ and x polarisations) ---
    Lo, si, co, ee, EE = avx_ort_ps(ps, pS, mask)

    # --- Coherent network statistics ---
    _, _, _, _, coherent_energy, gn, rn = avx_stat_ps(v00, v90, ps, pS, si, co, mask)

    # --- Build data and signal packets; compute xtalk-corrected SNRs ---
    Eo, pd, pD, pD_E, pD_si, pD_co, pD_a, pD_A = avx_packet_ps(v00, v90, mask)  # data packet (Ep)
    Lo, ps, pS, pS_E, pS_si, pS_co, pS_a, pS_A = avx_packet_ps(ps, pS, mask)    # signal packet (Lp)

    detector_snr, pD_E, rn, pD_norm = packet_norm_numpy(pd, pD, cluster_xtalk, cluster_xtalk_lookup_table, mask, pD_E)  # data packet energy snr
    D_snr = np.sum(detector_snr)
    S_snr, signal_snr, pS_E, pS_norm = gw_norm_numpy(pD_norm, pD_E, pS_E, coherent_energy)  # signal norms and signal SNR
    if DEBUG:
        print(S_snr, signal_snr)
        print("Eo = ", Eo, ", Lo = ", Lo, ", Ep = ", D_snr, ", Lp = ", S_snr)

    # --- Gaussian-noise correction and coherent energy decomposition ---
    # Returns: Gn (Gaussian noise), Ec (core coherent energy), Dc (signal-core coherent energy),
    #          Rc (EC normalisation), Eh (satellite/halo energy), Es, NC, NS
    # TODO: one more pixel selected, need to be fixed
    Gn, Ec, Dc, Rc, Eh, Es, NC, NS = avx_noise_ps(pS_norm, pD_norm, energy_total, mask, coherent_energy, gn, rn)

    if DEBUG:
        print("Gn = ", Gn, ", Ec = ", Ec, ", Dc = ", Dc, ", Rc = ", Rc, ", Eh = ", Eh, ", Es = ", Es, ", NC = ", NC, ", NS = ", NS)

    # --- Set packet amplitudes and compute time-domain null / energy ---
    N, pd, pD = avx_setAMP_ps(pd, pD, pD_norm, pD_si, pD_co, pD_a, pD_A, mask)  # data amplitudes
    N = N - 1  # effective pixel count
    _, ps, pS = avx_setAMP_ps(ps, pS, pS_norm, pS_si, pS_co, pS_a, pS_A, mask)  # signal amplitudes
    pn, pN = avx_loadNULL_ps(pd, pD, ps, pS)                                     # noise = data - signal

    # Raw xtalk sums (no clamping, mirrors C++ _avx_norm_ps(-V4))
    _, pD_E, rn, _ = packet_norm_numpy(pd, pD, cluster_xtalk, cluster_xtalk_lookup_table, mask, pD_E)
    Em = xtalk_energy_sum_numpy(pd, pD, cluster_xtalk, cluster_xtalk_lookup_table, mask)  # time-domain data energy
    Np = xtalk_energy_sum_numpy(pn, pN, cluster_xtalk, cluster_xtalk_lookup_table, mask)  # time-domain null energy
    D_snr = Em  # alias for backward compat
    Lm = Em - Np - Gn   # time-domain signal energy
    norm = (Eo - Eh) / Em if Em > 0 else 1.e9
    if norm < 1:
        norm = 1
    Ec /= norm  # core coherent energy normalised to time domain
    Dc /= norm  # signal-core coherent energy normalised to time domain
    ch = (Np + Gn) / (N * n_ifo)  # chi2
    if DEBUG:
        print("Np = ", Np, ", Em = ", Em, ", Lm = ", Lm, ", norm = ", norm, ", Ec = ", Ec, ", Dc = ", Dc, ", ch = ", ch)

    # --- Detection statistic rho (mode-dependent) ---
    xrho = 0.
    penalty = 0.
    ecor = 0.
    if not xgb_rho_mode:  # original 2G
        cc = ch if ch > 1 else 1             # noise correction factor
        rho = np.sqrt(Ec * Rc / 2.) if Ec > 0 else 0  # cWB rho
        if DEBUG:
            print("cc = ", cc, ", rho = ", rho)
    else:  # XGB.rho0
        penalty = ch
        ecor = Ec
        rho = np.sqrt(ecor / (1 + penalty * (max(float(1), penalty) - 1)))
        cc = ch if ch > 1 else 1             # noise correction factor (kept for xrho)
        xrho = np.sqrt(Ec * Rc / 2.) if Ec > 0 else 0  # original 2G rho (reference only)
        if DEBUG:
            print("cc = ", cc, ", rho = ", rho, ", ecor = ", ecor, ", penalty = ", penalty, ", xrho = ", xrho)

    # --- Project residuals onto network polarisation plane (Dual Stream Transform) ---
    v00, v90, p00_POL, p90_POL = avx_pol_ps(v00, v90, mask, fp, fx, f, F)  # network-plane projection
    v00, v90, r00_POL, r90_POL = avx_pol_ps(v00, v90, mask, fp, fx, f, F)  # DSP components

    return SkyStatistics(
        Gn=np.float32(Gn),
        Ec=np.float32(Ec),
        Dc=np.float32(Dc),
        Rc=np.float32(Rc),
        Eh=np.float32(Eh),
        Es=np.float32(Es),
        Np=np.float32(Np),
        Em=np.float32(Em),
        Lm=np.float32(Lm),
        norm=np.float32(norm),
        cc=np.float32(cc),
        rho=np.float32(rho),
        xrho=np.float32(xrho),
        Lo=np.float32(Lo),
        Eo=np.float32(Eo),
        energy_array_plus=ee,
        energy_array_cross=EE,
        pixel_mask=mask,
        v00=v00,
        v90=v90,
        gaussian_noise_correction=gn,
        coherent_energy=coherent_energy,
        N_pix_effective=N,
        noise_amplitude_00=pn,
        noise_amplitude_90=pN,
        pd=pd,
        pD=pD,
        ps=ps,
        pS=pS,
        p00_POL=p00_POL,
        p90_POL=p90_POL,
        r00_POL=r00_POL,
        r90_POL=r90_POL,
        S_snr=signal_snr,
        f = f,
        F = F,
    )


def fill_detection_statistic(sky_statistics: SkyStatistics, skymap_statistics: SkyMapStatistics,
                             cluster: Cluster, n_ifo: int,
                             xtalk: XTalk,
                             network_energy_threshold: float,
                             xgb_rho_mode: bool = False,
                             config: Config = None,
                             cluster_xtalk: np.ndarray | None = None,
                             cluster_xtalk_lookup: np.ndarray | None = None,
                             wdm_list=None) -> None:
    """
    Fill the detection statistics into the cluster and pixels.
    
    Parameters
    ----------
    sky_statistics : SkyStatistics
        The sky statistics object containing the calculated statistics.
    skymap_statistics : SkyMapStatistics
        The skymap statistics object to be filled.
    cluster : Cluster
        The cluster object containing the pixels.
    n_ifo : int
        Number of interferometers.
    xtalk : XTalk
        The XTalk object for cross-talk calculations.
    network_energy_threshold : float
        Energy threshold for the network.
    xgb_rho_mode : bool, optional
        If True, use XGB.rho0 statistics (rho0 without cc division). Default False.
    config : Config
        Pipeline configuration object. Required for MRA waveform reconstruction
        (hrss, strain, accurate gps_time and central_freq). Raises ValueError if None.
    cluster_xtalk : np.ndarray or None, optional
        Pre-computed CSR xtalk coefficient array from ``xtalk.get_xtalk_pixels``.
        When provided together with ``cluster_xtalk_lookup``, the internal
        ``get_xtalk_pixels`` call is skipped (saves ~0.4 s for N=2600).
    cluster_xtalk_lookup : np.ndarray or None, optional
        Pre-computed CSR lookup array (shape (N, 2)) from ``xtalk.get_xtalk_pixels``.
    wdm_list : list or None, optional
        Pre-built WDM filter-bank list from ``_create_wdm_set_python(config)``.
        When provided, ``_create_wdm_set_python`` is not called again (saves ~1 s
        per cluster). Pass from ``likelihood()`` where it is built once.

    Returns
    -------
    None
        Modifies ``cluster`` and ``skymap_statistics`` in place.
    """
    if config is None:
        raise ValueError(
            "fill_detection_statistic(): config is required. Without it, hrss/strain "
            "are zero and gps_time/central_freq use inaccurate supercluster fallback values."
        )
    _fds_t0 = time.perf_counter()
    _fds_timings: dict[str, float] = {}

    pixel_mask = sky_statistics.pixel_mask
    energy_array_plus = sky_statistics.energy_array_plus
    energy_array_cross = sky_statistics.energy_array_cross
    pd = sky_statistics.pd
    pD = sky_statistics.pD
    ps = sky_statistics.ps
    pS = sky_statistics.pS
    gaussian_noise_correction = sky_statistics.gaussian_noise_correction
    pn = sky_statistics.noise_amplitude_00
    pN = sky_statistics.noise_amplitude_90
    coherent_energy = sky_statistics.coherent_energy
    S_snr = sky_statistics.S_snr
    Rc = sky_statistics.Rc
    Gn = sky_statistics.Gn
    Np = sky_statistics.Np
    N_pix_effective = sky_statistics.N_pix_effective

    event_size = 0 # defined as Mw in cwb
    n_coherent_pixels = 0

    # --- First pass: set core/likelihood/null flags and per-ifo data arrays ---
    n_pix = len(cluster.pixel_arrays)

    _t0 = time.perf_counter()
    # Fast path: vectorised update of pixel_arrays (avoids O(n_pix * n_ifo) Python loop).
    _pa = cluster.pixel_arrays
    _pa.set_waveform_data(
        wave         = np.asarray(pd,  dtype=np.float32),
        w_90         = np.asarray(pD,  dtype=np.float32),
        asnr         = np.asarray(ps,  dtype=np.float32),
        a_90         = np.asarray(pS,  dtype=np.float32),
        core_mask    = pixel_mask,
        energy_plus  = np.asarray(energy_array_plus,  dtype=np.float32),
            energy_cross = np.asarray(energy_array_cross, dtype=np.float32),
        )

    # Pre-convert amplitude arrays to 2-D NumPy for fast column access
    pn_arr = np.asarray(pn, dtype=np.float64)  # (n_ifo, n_pix)
    pN_arr = np.asarray(pN, dtype=np.float64)
    ps_arr = np.asarray(ps, dtype=np.float64)
    pS_arr = np.asarray(pS, dtype=np.float64)

    # Use pre-computed xtalk arrays when available (avoids redundant O(N²) numba call).
    # Fall back to computing them here only when not passed in (e.g. standalone calls).
    if cluster_xtalk is not None and cluster_xtalk_lookup is not None:
        xtalks_lookup = cluster_xtalk_lookup
        xtalks = cluster_xtalk
    else:
        xtalks_lookup, xtalks = xtalk.get_xtalk_pixels(cluster.pixel_arrays)

    # core flags from pixel_arrays — no Python iteration
    _core = _pa.core
    null_k_set = np.where(_core & (np.asarray(gaussian_noise_correction) > 0))[0].astype(np.int64)
    like_k_set = np.where(_core & (np.asarray(coherent_energy)           > 0))[0].astype(np.int64)
    _fds_timings["set_waveform_data"] = time.perf_counter() - _t0

    # --- Second pass: compute null and likelihood using the parallel numba kernel ---
    logger.debug("fill_detection_statistic: null_k_set size=%d, like_k_set size=%d, n_pix=%d",
                 len(null_k_set), len(like_k_set), n_pix)
    logger.debug("fill_detection_statistic: pn_arr shape=%s, pn range=[%g, %g]",
                 str(pn_arr.shape), float(np.min(np.abs(pn_arr))), float(np.max(np.abs(pn_arr))))
    logger.debug("fill_detection_statistic: gn range=[%g, %g], ec range=[%g, %g]",
                 float(np.min(gaussian_noise_correction)),
                 float(np.max(gaussian_noise_correction)),
                 float(np.min(coherent_energy)),
                 float(np.max(coherent_energy)))

    # null_out and like_out are written in place for the relevant pixel indices.
    # Initialise to zero so pixels not in the respective sets keep their old value
    # (matches behaviour of the previous Python loops).
    null_out = np.zeros(n_pix, dtype=np.float64)
    like_out = np.zeros(n_pix, dtype=np.float64)

    gn_arr = np.asarray(gaussian_noise_correction, dtype=np.float64)
    ec_arr = np.asarray(coherent_energy, dtype=np.float64)

    # Boolean membership masks — mirror the original inner-loop scope:
    #   original null loop:       for k in null_k_set  (core & gn > 0)
    #   original likelihood loop: for k in like_k_set  (core & ec > 0)
    null_mask = np.zeros(n_pix, dtype=np.bool_)
    null_mask[null_k_set] = True
    like_mask = np.zeros(n_pix, dtype=np.bool_)
    like_mask[like_k_set] = True

    _t0 = time.perf_counter()
    _compute_null_likelihood_numba(
        null_k_set, like_k_set,
        pn_arr, pN_arr, ps_arr, pS_arr,
        gn_arr, ec_arr,
        xtalks_lookup.astype(np.int64),
        xtalks,
        null_mask, like_mask,
        null_out, like_out,
    )
    _kernel_time = time.perf_counter() - _t0

    # Write results back into pixel_arrays
    for i in null_k_set:
        _pa.null[i] = null_out[i]
    for i in like_k_set:
        _pa.likelihood[i] = like_out[i]

    # Count statistics (sets were pre-filtered, so counts equal set sizes)
    event_size        = int(len(null_k_set))
    n_coherent_pixels = int(len(like_k_set))

    _fds_timings["null_xtalk_loop"]       = _kernel_time * len(null_k_set) / max(len(null_k_set) + len(like_k_set), 1)
    _fds_timings["likelihood_xtalk_loop"] = _kernel_time * len(like_k_set) / max(len(null_k_set) + len(like_k_set), 1)

    # --- Subnetwork statistic ---
    Nmax = 0.0
    Emax = np.max(S_snr)
    Esub = np.sum(S_snr) - Emax
    Esub = Esub * (1 + 2 * Rc * Esub / Emax)
    Nmax = Gn + Np - N_pix_effective * (n_ifo - 1)

    # --- Time-domain waveform statistics via getMRAwave reconstruction ---
    # Mirrors C++ getMRAwave('W') + getMRAwave('S') loop.
    # See docs/math/waveform_likelihood.md for the full derivation.
    #
    # Per-IFO quantities (whitened time-domain waveforms):
    #   sSNR_i = Σ_t z_signal_i(t)²             (signal energy / sSNR)  → Lw = Σ_i sSNR_i
    #   snr_i  = Σ_t z_data_i(t)²               (data   energy / snr)   → Ew_wf
    #   null_i = Σ_t (z_data - z_signal)_i(t)²  (null   energy)         → Nw_wf
    # To/Fo   = sSNR-weighted mean time / frequency over core pixels
    ps_arr_np = np.asarray(ps, dtype=np.float64)   # (n_ifo, n_pix)
    pS_arr_np = np.asarray(pS, dtype=np.float64)
    pd_arr_np = np.asarray(pd, dtype=np.float64)   # data amplitudes after avx_setAMP_ps
    pD_arr_np = np.asarray(pD, dtype=np.float64)
    # Core pixel indices — only core pixels contribute to getMRAwave
    core_indices = np.where(cluster.pixel_arrays.core)[0].tolist()

    _t0 = time.perf_counter()
    Lw = 0.0
    sSNR_ifo  = np.zeros(n_ifo, dtype=np.float64)
    snr_ifo   = np.zeros(n_ifo, dtype=np.float64)
    null_ifo  = np.zeros(n_ifo, dtype=np.float64)
    signal_energy_physical = np.zeros(n_ifo, dtype=np.float64)
    To = 0.0
    Fo = 0.0

    # if config is not None and len(core_indices) > 0:
    # --- WDM synthesis path: exact getMRAwave equivalent (pure Python, no ROOT) ---
    # Reconstructs whitened time-domain waveforms per IFO:
    #   z_i(t) = Σ_{j∈core} [ a00_ij·ψ00_j(t) + a90_ij·ψ90_j(t) ]

    # Reuse the wdm_list built in likelihood() when provided; otherwise build it
    # here (one-off / standalone calls).  Building it per-cluster was ~1 s overhead.
    if wdm_list is None:
        wdm_list = _create_wdm_set_python(config)
    rate_ana = float(config.rateANA)

    # Pre-build pixel array tuple and WDM kernel data once; shared across all
    # (ifo, a_type, whiten) combinations so get_MRA_wave skips redundant extraction.
    _pixel_arrays  = _pa_to_tuple(cluster.pixel_arrays)
    _wdm_njit_data = _build_wdm_njit_data(wdm_list)

    for ifo_i in range(n_ifo):
        z_sig_ts = get_MRA_wave(cluster, wdm_list, rate_ana, ifo_i,
                                a_type='signal', mode=0, nproc=1, whiten=True,
                                _pixel_arrays=_pixel_arrays, _wdm_njit_data=_wdm_njit_data)
        z_dat_ts = get_MRA_wave(cluster, wdm_list, rate_ana, ifo_i,
                                a_type='strain', mode=0, nproc=1, whiten=True,
                                _pixel_arrays=_pixel_arrays, _wdm_njit_data=_wdm_njit_data)
        # For hrss: get un-whitened signal energy (physical strain units)
        z_sig_physical = get_MRA_wave(cluster, wdm_list, rate_ana, ifo_i,
                                        a_type='signal', mode=0, nproc=1, whiten=False,
                                        _pixel_arrays=_pixel_arrays, _wdm_njit_data=_wdm_njit_data)
        if z_sig_ts is None or z_dat_ts is None:
            continue
        z_sig = np.asarray(z_sig_ts.data, dtype=np.float64)
        z_dat = np.asarray(z_dat_ts.data, dtype=np.float64)
        sSNR_ifo[ifo_i] = np.sum(z_sig ** 2)
        snr_ifo[ifo_i]  = np.sum(z_dat ** 2)
        null_ifo[ifo_i] = np.sum((z_dat - z_sig) ** 2)
        if z_sig_physical is not None:
            z_sig_phys = np.asarray(z_sig_physical.data, dtype=np.float64)
            signal_energy_physical[ifo_i] = np.sum(z_sig_phys ** 2)

        # getWFtime() / getWFfreq() equivalents (mirrors C++ detector::getWFtime/getWFfreq)
        # Used to compute To/Fo exactly as C++: Fo += sSNR_i * getWFfreq_i; To /= Lw
        n_fft = len(z_sig)
        rate_wf = float(z_sig_ts.sample_rate)
        e_sig = z_sig ** 2
        E_sig = float(np.sum(e_sig))
        if E_sig > 0.0:
            t_start = float(z_sig_ts.start_time)
            wf_time_ifo = t_start + float(np.dot(e_sig, np.arange(n_fft))) / (E_sig * rate_wf)
            Z_fft = np.fft.rfft(z_sig)
            power = Z_fft.real ** 2 + Z_fft.imag ** 2
            E_fft = float(np.sum(power))
            if E_fft > 0.0:
                wf_freq_ifo = float(np.dot(power, np.arange(len(power)))) * rate_wf / n_fft / E_fft
            else:
                wf_freq_ifo = 0.0
            To += sSNR_ifo[ifo_i] * wf_time_ifo
            Fo += sSNR_ifo[ifo_i] * wf_freq_ifo

    Lw    = float(np.sum(sSNR_ifo))
    Ew_wf = float(np.sum(snr_ifo))
    Nw_wf = float(np.sum(null_ifo))
    if Lw > 0.0:
        To /= Lw
        Fo /= Lw

    # else:
    #     # Fallback: xtalk-catalog double-sum (used when config is not available).
    #     # Approximate because the catalog may omit weak-overlap pixel pairs.
    #     cross_ifo = np.zeros(n_ifo, dtype=np.float64)
    #     sSNR_ifo  = np.zeros(n_ifo, dtype=np.float64)
    #     snr_ifo   = np.zeros(n_ifo, dtype=np.float64)
    #     _pa_fb    = cluster.pixel_arrays
    #     for i_idx in core_indices:
    #         for k_idx in core_indices:
    #             xt = xtalk.get_xtalk(
    #                 pix1=(_pa_fb.layers[i_idx], _pa_fb.time[i_idx]),
    #                 pix2=(_pa_fb.layers[k_idx], _pa_fb.time[k_idx]),
    #             )
    #             if xt[0] > 2:
    #                 continue
    #             ps_i = ps_arr_np[:, i_idx]
    #             pS_i = pS_arr_np[:, i_idx]
    #             ps_k = ps_arr_np[:, k_idx]
    #             pS_k = pS_arr_np[:, k_idx]
    #             pd_i = pd_arr_np[:, i_idx]
    #             pD_i = pD_arr_np[:, i_idx]
    #             pd_k = pd_arr_np[:, k_idx]
    #             pD_k = pD_arr_np[:, k_idx]
    #             sSNR_ifo += (xt[0]*ps_i*ps_k + xt[1]*ps_i*pS_k + xt[2]*pS_i*ps_k + xt[3]*pS_i*pS_k)
    #             snr_ifo  += (xt[0]*pd_i*pd_k + xt[1]*pd_i*pD_k + xt[2]*pD_i*pd_k + xt[3]*pD_i*pD_k)
    #             cross_ifo += (xt[0]*pd_i*ps_k + xt[1]*pd_i*pS_k + xt[2]*pD_i*ps_k + xt[3]*pD_i*pS_k)
    #         s_snr_pix = float(np.sum(ps_arr_np[:, i_idx] ** 2 + pS_arr_np[:, i_idx] ** 2))
    #         _r  = float(_pa_fb.rate[i_idx])
    #         _ly = float(_pa_fb.layers[i_idx])
    #         pix_time = float(_pa_fb.time[i_idx]) / (_r * _ly) if (_r > 0 and _ly > 0) else 0.0
    #         pix_freq = float(_pa_fb.frequency[i_idx]) * _r / 2.0 if _r > 0 else 0.0
    #         To += s_snr_pix * pix_time
    #         Fo += s_snr_pix * pix_freq
    #     Lw = float(np.sum(sSNR_ifo))
    #     null_ifo = snr_ifo - 2.0 * cross_ifo + sSNR_ifo
    #     Ew_wf = float(np.sum(snr_ifo))
    #     Nw_wf = float(np.sum(null_ifo))
    #     if Lw > 0.0:
    #         To /= Lw
    #         Fo /= Lw

    _fds_timings["mra_waveform_reconstruction"] = time.perf_counter() - _t0

    # xSNR per IFO: geometric mean  C++ get_XS() = sqrt(get_XX() * get_SS())
    _t0 = time.perf_counter()
    xSNR_ifo = np.sqrt(np.maximum(snr_ifo * sSNR_ifo, 0.0))

    # --- Detection statistics: netCC, norm, rho (mirrors network.cc likelihoodWP) ---
    # Energy notation:
    #   Eo    — total TF-domain data energy
    #   Eh    — satellite (halo) energy
    #   Em    — pixel-domain xtalk-corrected energy (likesky / neted[3])
    #   Ew_wf — waveform-domain data energy from getMRAwave (neted[2])
    #   Nw_wf — waveform-domain null energy from getMRAwave (neted[1] - Gn)
    # C++ formulas:
    #   ch_wf = (Nw_wf + Gn) / (N * nIFO)
    #   Cp = Ec*Rc / (Ec*Rc + (Dc+Nw_wf+Gn)       - N*(nIFO-1))   # netCC[0]
    #   Cr = Ec*Rc / (Ec*Rc + (Dc+Nw_wf+Gn)*cc_Cr - N*(nIFO-1))   # netCC[1]
    #   norm = (Eo-Eh) / Ew_wf  clamped to ≥ 1, stored as norm*2
    Dc = float(sky_statistics.Dc)
    Ec = float(sky_statistics.Ec)
    Rc_val = float(sky_statistics.Rc)
    Eo = float(sky_statistics.Eo)
    Eh = float(sky_statistics.Eh)
    Gn_val = float(sky_statistics.Gn)
    N_eff = float(N_pix_effective)
    Nw_for_stats = max(Nw_wf, 0.0)  # clamp to avoid negative chi2
    ch_td = (Nw_for_stats + Gn_val) / (N_eff * n_ifo) if (N_eff * n_ifo) > 0 else 1.0

    # cc_Cr: Cr-specific correction (NOT the simple ch used for rho)
    cc_Cr = 1.0 + (ch_td - 1.0) * 2.0 * (1.0 - Rc_val) if ch_td > 1.0 else 1.0
    denom_r = Ec * Rc_val + (Dc + Nw_for_stats + Gn_val) * cc_Cr - N_eff * (n_ifo - 1)
    denom_p = Ec * Rc_val + (Dc + Nw_for_stats + Gn_val) - N_eff * (n_ifo - 1)
    Cr_td = (Ec * Rc_val / denom_r) if denom_r > 0 else 0.0
    Cp_td = (Ec * Rc_val / denom_p) if denom_p > 0 else 0.0

    norm_td = (Eo - Eh) / Ew_wf if Ew_wf > 0 else 1.0
    if norm_td < 1.0:
        norm_td = 1.0

    # rho is divided by sqrt(cc) using Nw-based chi2 (time-domain null, matches C++ line 939)
    cc_rho_td = ch_td if ch_td > 1.0 else 1.0
    rho_reduced = float(sky_statistics.rho) / sqrt(cc_rho_td)
    _fds_timings["detection_statistics"] = time.perf_counter() - _t0

    # --- Store all fields on cluster_meta ---
    _t0 = time.perf_counter()
    cluster.cluster_meta.sky_size = event_size
    cluster.cluster_meta.sub_net = Esub / (Esub + Nmax) if (Esub + Nmax) > 0 else 0.0
    cluster.cluster_meta.sub_net2 = skymap_statistics.nCorrelation[skymap_statistics.l_max]
    cluster.cluster_meta.like_sky = float(sky_statistics.Em)          # Em (neted[3]): pixel-domain xtalk energy
    cluster.cluster_meta.energy_sky = sky_statistics.Eo               # TF-domain data energy (neted[4])
    cluster.cluster_meta.net_ecor = sky_statistics.Ec                 # packet coherent energy
    cluster.cluster_meta.norm_cor = sky_statistics.Ec * sky_statistics.Rc  # normalised coherent energy
    cluster.cluster_meta.like_net = float(Lw)                         # waveform likelihood (likenet)
    cluster.cluster_meta.energy = float(Ew_wf)                        # getMRAwave data energy (neted[2])
    cluster.cluster_meta.net_null = float(Nw_for_stats + Gn_val)      # packet null (neted[1])
    cluster.cluster_meta.net_ed = float(Nw_for_stats + Gn_val + Dc - N_eff * n_ifo)  # residual null (neted[0])
    cluster.cluster_meta.norm = float(norm_td * 2.0)                  # packet norm
    cluster.cluster_meta.net_cc = float(Cp_td)                        # network cc (netcc[0])
    cluster.cluster_meta.sky_cc = float(Cr_td)                        # reduced network cc (netcc[1])
    # c_time / c_freq from Lw-weighted centroid over core pixels
    if Lw > 0.0:
        cluster.cluster_meta.c_time = float(To)
        cluster.cluster_meta.c_freq = float(Fo)

    if not xgb_rho_mode:  # original 2G
        cluster.cluster_meta.net_rho = rho_reduced
        cluster.cluster_meta.net_rho2 = float(sky_statistics.rho)
    else:  # XGB.rho0
        # rho[0] = -netRHO = rho  (XGB rho0, no cc division — C++ netevent.cc line 979)
        cluster.cluster_meta.net_rho = float(sky_statistics.rho)
        # rho[1] = netrho = xrho/sqrt(cc)  (original 2G rho with cc — C++ netevent.cc line 980)
        cluster.cluster_meta.net_rho2 = float(sky_statistics.xrho) / sqrt(cc_rho_td)

    cluster.cluster_meta.g_net = skymap_statistics.nAntennaPrior[skymap_statistics.l_max]
    cluster.cluster_meta.a_net = skymap_statistics.nAlignment[skymap_statistics.l_max]
    cluster.cluster_meta.i_net = 0
    cluster.cluster_meta.ndof = N_pix_effective
    cluster.cluster_meta.sky_chi2 = skymap_statistics.nDisbalance[skymap_statistics.l_max]
    cluster.cluster_meta.g_noise = sky_statistics.Gn
    cluster.cluster_meta.iota = 0.0
    cluster.cluster_meta.psi = 0.0
    cluster.cluster_meta.ellipticity = 0

    # Per-IFO xtalk-corrected waveform energies (getMRAwave equivalents for snr/sSNR/xSNR)
    cluster.cluster_meta.signal_snr = sSNR_ifo.tolist()   # C++ d->sSNR = get_SS() per IFO
    cluster.cluster_meta.wave_snr   = snr_ifo.tolist()    # C++ d->enrg = get_XX() per IFO
    cluster.cluster_meta.cross_snr  = xSNR_ifo.tolist()   # C++ d->xSNR = get_XS() per IFO
    cluster.cluster_meta.signal_energy_physical = signal_energy_physical.tolist()  # physical strain energy for hrss
    cluster.cluster_meta.null_energy = null_ifo.tolist()  # null energy per IFO (C++ d->null)

    logger.debug(
        "fill_detection_statistic: sky_size=%d sub_net=%.4f net_cc=%.4f sky_cc=%.4f "
        "like_net=%.2f energy=%.2f net_null=%.4f norm=%.4f rho=%.4f "
        "Ew_wf=%.2f Nw_wf=%.4f like_sky=%.2f",
        cluster.cluster_meta.sky_size, cluster.cluster_meta.sub_net,
        cluster.cluster_meta.net_cc, cluster.cluster_meta.sky_cc,
        cluster.cluster_meta.like_net, cluster.cluster_meta.energy,
        cluster.cluster_meta.net_null, cluster.cluster_meta.norm,
        cluster.cluster_meta.net_rho,
        Ew_wf, Nw_wf, cluster.cluster_meta.like_sky,
    )

    _fds_timings["store_cluster_meta"] = time.perf_counter() - _t0
    _fds_timings["total"] = time.perf_counter() - _fds_t0
    logger.info("fill_detection_statistic stage timings:")
    for _stage, _t in _fds_timings.items():
        if _stage != "total":
            logger.info("  %-30s %.4f s  (%5.1f%%)", _stage, _t,
                        100.0 * _t / _fds_timings["total"] if _fds_timings["total"] > 0 else 0)


def threshold_cut(
    sky_statistics: SkyStatistics,
    network_energy_threshold: float,
    netEC_threshold: float,
    net_rho_threshold: float | None = None,
    xgb_rho_mode: bool = False,
) -> str:
    """
    Apply threshold cuts based on the sky statistics and network energy threshold.
    
    Parameters
    ----------
    sky_statistics : SkyStatistics
        The statistics calculated for the sky location.
    network_energy_threshold : float
        The threshold for network energy.
    netEC_threshold : float
        The threshold for net correlation energy (``netEC``).
    net_rho_threshold : float or None, optional
        Absolute ``netRHO`` threshold. In XGB mode C++ compares against
        ``fabs(netRHO)`` directly.
    xgb_rho_mode : bool, optional
        If True, apply XGB.rho0 cuts instead of the original 2G cuts.

    Returns
    -------
    str or None
        A rejection reason string if any cut fails; ``None`` if the cluster passes.
    """
    Lm = sky_statistics.Lm
    Eo = sky_statistics.Eo
    Eh = sky_statistics.Eh
    Ec = sky_statistics.Ec
    Rc = sky_statistics.Rc
    cc = sky_statistics.cc
    rho = sky_statistics.rho
    N = sky_statistics.N_pix_effective   # effective pixel count (_avx_setAMP_ps() - 1)
    if not xgb_rho_mode:
        condition_1 = Lm <= 0.
        condition_2 = (Eo - Eh) <= 0.
        condition_3 = Ec * Rc / cc < netEC_threshold
        condition_4 = N < 1   # C++: N < 1 (pixel count, not null energy)
        if condition_1 or condition_2 or condition_3 or condition_4:
            rejection_reason = ""
            if condition_1:
                rejection_reason += f"Lm > 0 but Lm = {Lm};"
            if condition_2:
                rejection_reason += f" (Eo - Eh) > 0 but (Eo - Eh) = {Eo - Eh};"
            if condition_3:
                rejection_reason += f" Ec * Rc / cc >= netEC_threshold but Ec * Rc / cc = {Ec * Rc / cc:.4f} < {netEC_threshold:.4f};"
            if condition_4:
                rejection_reason += f" N < 1 but N = {N};"
            return rejection_reason
    else:
        # For XGB.rho0 case C++ uses `rho < fabs(netRHO)` directly.
        if net_rho_threshold is None:
            net_rho_threshold = (netEC_threshold / 2.0) ** 0.5
        condition_1 = Lm <= 0.
        condition_2 = (Eo - Eh) <= 0.
        condition_3 = rho < net_rho_threshold
        condition_4 = N < 1   # C++: N < 1 (pixel count)
        if condition_1 or condition_2 or condition_3 or condition_4:
            rejection_reason = ""
            if condition_1:
                rejection_reason += f"Lm > 0 but Lm = {Lm};"
            if condition_2:
                rejection_reason += f" (Eo - Eh) > 0 but (Eo - Eh) = {Eo - Eh};"
            if condition_3:
                rejection_reason += f" rho >= |netRHO| but rho = {rho} < {net_rho_threshold};"
            if condition_4:
                rejection_reason += f" N < 1 but N = {N};"
            return rejection_reason
        
    return None  # No rejection, all conditions passed


def get_error_region(cluster: Cluster):
    # pwc->p_Ind[id - 1].push_back(Mo);
    # double T = To + pwc->start;                          // trigger time
    # std::vector<float> sArea;
    # pwc->sArea.push_back(sArea);
    # pwc->p_Map.push_back(sArea);
    #
    # double var = norm * Rc * sqrt(Mo) * (1 + fabs(1 - CH));
    #
    # // TODO: fix this
    # if (iID <= 0 || ID == id) {
    # network::getSkyArea(id, lag, T, var);       // calculate error regions
    # }
    pass


@njit(cache=True, parallel=True)
def _hough_count_overlaps_numba(x, y, xerr, yerr, kk, m_vals):
    """Phase 1 of mchirp Hough transform: compute the max interval-overlap count
    for each mass value (independent → parallelised with prange).

    For each mass m the t-f locus is a line  y = sl*x + b  in (time, F^{-8/3})
    space.  Each pixel defines an error ellipse that, projected onto the b-axis,
    gives an interval [bmin, bmax].  The maximum number of overlapping intervals
    is the Hough vote count for that mass.

    Parameters
    ----------
    x, y, xerr, yerr : 1-D float64 arrays, length n_pts
        Pixel coordinates and their uncertainties.
    kk : float
        Pre-computed chirp-mass constant.
    m_vals : 1-D float64 array, length n_mass
        Mass grid to scan.

    Returns
    -------
    nsel_arr : 1-D int64 array, length n_mass
        Maximum overlap count per mass value.
    """
    n_mass = len(m_vals)
    n_pts  = len(x)
    nsel_arr = np.zeros(n_mass, dtype=np.int64)

    for mi in prange(n_mass):
        m  = m_vals[mi]
        sl = kk * np.abs(m) ** (5.0 / 3.0)
        if m > 0.0:
            sl = -sl

        Db   = np.sqrt(2.0 * (sl * sl * xerr * xerr + yerr * yerr))
        bmin = y - sl * x - Db
        bmax = bmin + 2.0 * Db

        # Build flat endpoint list: opens (+1) followed by closes (-1)
        ep_val  = np.empty(2 * n_pts, dtype=np.float64)
        ep_type = np.empty(2 * n_pts, dtype=np.float64)
        for i in range(n_pts):
            ep_val[i]          = bmin[i]
            ep_type[i]         = 1.0
            ep_val[n_pts + i]  = bmax[i]
            ep_type[n_pts + i] = -1.0

        order = np.argsort(ep_val)

        # Walk sorted endpoints; track running overlap count
        cum    = 0
        maxcum = 0
        for i in range(2 * n_pts):
            idx = order[i]
            cum += int(ep_type[idx])
            if cum > maxcum:
                maxcum = cum

        nsel_arr[mi] = maxcum

    return nsel_arr


@njit(cache=True)
def _fine_search_numba(x, y, xerr, yerr, wgt, kk, m_vals, cand_indices, nselmax, chi2_thr):
    """Phase 2 of mchirp Hough transform: fine b-grid search among candidate masses.

    For each candidate mass (those achieving *nselmax* votes in phase 1) the
    b-axis is scanned at step 0.0025 within segments that attain the maximum
    overlap.  The (m, b) pair minimising the likelihood-weighted mean chi2 is
    returned.

    Parameters
    ----------
    x, y, xerr, yerr, wgt : 1-D float64 arrays, length n_pts
    kk : float
    m_vals : 1-D float64 array (full mass grid)
    cand_indices : 1-D int64 array — indices into m_vals with nsel == nselmax
    nselmax : int
    chi2_thr : float

    Returns
    -------
    m0, b0 : float
        Best-fit chirp-mass slope and intercept.
    """
    n_pts   = len(x)
    b_step  = 0.0025
    chi2min = 1e100
    m0 = m_vals[cand_indices[0]]
    b0 = 0.0

    for jj in range(len(cand_indices)):
        mi = cand_indices[jj]
        m  = m_vals[mi]
        sl = kk * np.abs(m) ** (5.0 / 3.0)
        if m > 0.0:
            sl = -sl

        # Per-pixel chi2 denominator
        eps  = sl * sl * xerr * xerr + yerr * yerr

        # Recompute sorted endpoints (cheap — only a few candidate masses)
        Db   = np.sqrt(2.0 * (sl * sl * xerr * xerr + yerr * yerr))
        bmin = y - sl * x - Db
        bmax = bmin + 2.0 * Db

        ep_val  = np.empty(2 * n_pts, dtype=np.float64)
        ep_type = np.empty(2 * n_pts, dtype=np.float64)
        for i in range(n_pts):
            ep_val[i]          = bmin[i]
            ep_type[i]         = 1.0
            ep_val[n_pts + i]  = bmax[i]
            ep_type[n_pts + i] = -1.0

        order = np.argsort(ep_val)
        sorted_val  = np.empty(2 * n_pts, dtype=np.float64)
        sorted_type = np.empty(2 * n_pts, dtype=np.float64)
        for i in range(2 * n_pts):
            sorted_val[i]  = ep_val[order[i]]
            sorted_type[i] = ep_type[order[i]]

        # Build cumulative-type array
        cum_types = np.empty(2 * n_pts, dtype=np.int64)
        cum = 0
        for i in range(2 * n_pts):
            cum += int(sorted_type[i])
            cum_types[i] = cum

        # Walk segments that achieve nselmax and scan b grid
        for k in range(2 * n_pts - 1):
            if cum_types[k] != nselmax:
                continue
            b_lo = sorted_val[k]
            b_hi = sorted_val[k + 1]
            if b_hi <= b_lo:
                continue

            n_b_steps = int((b_hi - b_lo) / b_step)
            if n_b_steps < 1:
                n_b_steps = 1

            for bi in range(n_b_steps + 1):
                b = b_lo + bi * b_step
                if b > b_hi:
                    b = b_hi

                chi2_sum = 0.0
                wgt_sum  = 0.0
                for i in range(n_pts):
                    res      = y[i] - sl * x[i] - b
                    chi2_val = res * res / eps[i]
                    if chi2_val <= chi2_thr:
                        chi2_sum += chi2_val * wgt[i]
                        wgt_sum  += wgt[i]

                if wgt_sum > 0.0:
                    totchi = chi2_sum / wgt_sum
                    if totchi < chi2min:
                        chi2min = totchi
                        m0 = m
                        b0 = b

    return m0, b0


def get_chirp_mass(cluster: Cluster, xgb_rho_mode: bool = False, pat0: bool = False):
    """Python implementation of C++ netcluster::mchirp().

    Computes chirpEllip and chirpEfrac via Hough-transform + PCA ellipticity
    on the cluster's pixels (which must have .likelihood already set by
    fill_detection_statistic).  Updates cluster.cluster_meta.net_rho2 with
    rho1 = rho0 * chirpEllip * sqrt(chirpEfrac), matching netevent.cc line 977:
        rho[1] = pcd->netRHO * chirp[3] * sqrt(chirp[5])   (pat0=false branch)

    net_rho2 is only updated for original 2G mode (xgb_rho_mode=False)
    with pat0=False.  In XGB mode or pat0=True the value set by
    fill_detection_statistic is preserved (mirrors netevent.cc lines 974-981).
    """
    import math

    # --- C++ watconstants (same as in netcluster::mchirp, from constants.hh) ---
    G  = 6.67259e-11        # WAT_G_SI: gravitational constant [N m^2 kg^-2]
    SM = 1.98892e30         # solar mass [kg]
    C  = 299792458.0        # speed of light [m/s]
    Pi = math.pi
    sF = 128.0              # frequency scaling (units of 128 Hz)
    chi2_thr = 2.5          # default threshold

    kk = 256.0 * Pi / 5.0 * math.pow(G * SM * Pi / (C * C * C), 5.0 / 3.0)
    kk *= math.pow(sF, 8.0 / 3.0)

    # --- Collect pixels (vectorised — no per-pixel object construction) ---
    _pa = cluster.pixel_arrays
    _valid = (_pa.likelihood > 0.0) & (_pa.frequency > 0)

    _rate_v   = _pa.rate[_valid].astype(float)
    _layers_v = _pa.layers[_valid].astype(float)
    _time_v   = _pa.time[_valid].astype(float)
    _freq_v   = _pa.frequency[_valid].astype(float)
    _lh_v     = _pa.likelihood[_valid].astype(float)

    T_v   = np.floor(_time_v / _layers_v) / _rate_v
    eT_v  = (0.5 / _rate_v) * math.sqrt(2.0)

    F_raw_v = _freq_v * _rate_v / 2.0 / sF
    _pos = F_raw_v > 0.0
    T_v, eT_v, F_raw_v, _rate_v, _lh_v = (T_v[_pos], eT_v[_pos], F_raw_v[_pos],
                                            _rate_v[_pos], _lh_v[_pos])

    eF_v = (_rate_v / 4.0 / math.sqrt(3.0)) / sF
    eF_v *= 8.0 / 3.0 / np.power(F_raw_v, 11.0 / 3.0)
    F_t_v = 1.0 / np.power(F_raw_v, 8.0 / 3.0)

    np_pts = len(T_v)
    if np_pts < 5:
        return  # insufficient pixels — leave net_rho2 unchanged

    x    = T_v
    y    = F_t_v
    xerr = eT_v
    yerr = eF_v
    wgt  = _lh_v

    # --- Hough transform: find mass(es) with maximum pixel-overlap ---
    maxM     = 100.0
    stepM    = 0.2
    m_vals   = np.arange(-maxM, maxM + 1e-9, stepM)   # 1001 values

    # Phase 1: parallel Numba scan — O(n_mass * n_pts * log n_pts) with prange
    nsel_arr = _hough_count_overlaps_numba(
        x.astype(np.float64), y.astype(np.float64),
        xerr.astype(np.float64), yerr.astype(np.float64),
        float(kk), m_vals.astype(np.float64),
    )

    nselmax      = int(np.max(nsel_arr))
    cand_indices = np.where(nsel_arr == nselmax)[0].astype(np.int64)

    # Phase 2: fine b-grid search over candidate masses — tight Numba inner loop
    m0, b0 = _fine_search_numba(
        x.astype(np.float64), y.astype(np.float64),
        xerr.astype(np.float64), yerr.astype(np.float64),
        wgt.astype(np.float64),
        float(kk), m_vals.astype(np.float64),
        cand_indices, int(nselmax), float(chi2_thr),
    )

    # --- Compute Efrac ---
    sl  = kk * math.pow(abs(m0), 5.0 / 3.0)
    if m0 > 0:
        sl = -sl

    eps = sl * sl * xerr * xerr + yerr * yerr
    residuals = y - sl * x - b0
    chi2_all  = residuals * residuals / eps
    sel_mask  = chi2_all <= chi2_thr

    totEn = float(np.sum(wgt))
    selEn = float(np.sum(wgt[sel_mask]))
    Efrac = selEn / totEn if totEn > 0.0 else 0.0

    # --- Filter to selected pixels and compute PCA ellipticity ---
    x_sel = x[sel_mask]
    y_sel = y[sel_mask]
    np_sel = len(x_sel)

    if np_sel >= 2:
        xcm = np.mean(x_sel)
        ycm = np.mean(y_sel)
        dx  = x_sel - xcm
        dy  = y_sel - ycm
        qxx = float(np.sum(dx * dx))
        qyy = float(np.sum(dy * dy))
        qxy = float(np.sum(dx * dy))

        sq_delta = math.sqrt((qxx - qyy) ** 2 + 4.0 * qxy * qxy)
        lam1 = math.sqrt((qxx + qyy + sq_delta) / 2.0)
        lam2_sq = (qxx + qyy - sq_delta) / 2.0
        lam2 = math.sqrt(max(lam2_sq, 0.0))
        denom = lam1 + lam2
        chirpEllip = abs(lam1 - lam2) / denom if denom > 0.0 else 0.0
    else:
        chirpEllip = 0.0

    # --- Update cluster metadata ---
    chrho = chirpEllip * math.sqrt(Efrac)
    rho1  = cluster.cluster_meta.net_rho * chrho

    # C++ netevent.cc lines 974-981:
    #   chrho = chirp[3] * sqrt(chirp[5])                    (always computed)
    #   if netRHO >= 0 (original 2G):
    #       rho[1] = pat0 ? netrho : netRHO * chrho           (only pat0=false uses chirp)
    #   else (XGB.rho0):
    #       rho[1] = netrho                                   (chirp result ignored)
    if not xgb_rho_mode and not pat0:
        cluster.cluster_meta.net_rho2 = rho1
    # else: net_rho2 already set correctly by fill_detection_statistic; do not overwrite
