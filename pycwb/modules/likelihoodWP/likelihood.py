"""Public likelihood entry points and flat per-cluster orchestration.

All stage outputs are ordinary local variables in
:func:`evaluate_cluster_likelihood`.  The backend boundary is explicit, and no
mutable pipeline-state object is passed through the scientific flow.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from pycwb.modules.xtalk.type import XTalk
from pycwb.modules.reconstruction.getMRAwaveform import _create_wdm_set_python
from pycwb.types.network_cluster import Cluster, FragmentCluster
from pycwb.types.time_frequency_map import TimeFrequencyMap
from pycwb.types.time_series import TimeSeries

from .backends import LikelihoodKernels, get_likelihood_backend
from .backends.base import SkyGrid
from .detection_statistics import (
    get_likelihood_rejection_reason,
    populate_detection_statistics,
    update_chirp_mass_statistics,
)
from .extensions import (
    build_legacy_sky_area,
    build_sky_probability,
    build_target_sky_indices,
    compute_target_sky_metrics,
    rank_sky_probability,
    resolve_extension_plan,
    run_likelihood_cuts,
    run_likelihood_features,
    sky_pixel_area_deg2,
    trigger_gps,
    validate_sky_temperature,
)
from .sky_mask import sky_valid_indices_for_cluster
from .likelihood_setup import prepare_likelihood_inputs
from .pixel_data import extract_pixel_time_delay_data
from .typing import SkyMapStatistics

if TYPE_CHECKING:
    from pycwb.config.config import Config

logger = logging.getLogger(__name__)


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
    backend: str | LikelihoodKernels | None = None,
) -> tuple[Cluster | None, SkyMapStatistics | None]:
    """Resolve inputs and execute one explicit, flat likelihood sequence."""

    # Resolve segment-level inputs here so the scientific stages below operate
    # only on explicit local data.  Callers processing many clusters can pass
    # precomputed setup/xtalk objects to avoid rebuilding them for every event.
    if config is None:
        raise ValueError("likelihood(): config is required")
    if xtalk is None:
        if MRAcatalog is None:
            raise ValueError(
                "likelihood(): xtalk or MRAcatalog must be provided. "
                "For multi-cluster use, call evaluate_fragment_clusters()."
            )
        xtalk = XTalk.load(MRAcatalog, dump=True)
    if setup is None:
        setup = _prepare_setup(
            nIFO, config, strains=strains, supercluster_setup=supercluster_setup
        )

    backend_choice = backend
    if backend_choice is None:
        backend_choice = setup.get(
            "likelihood_backend",
            getattr(config, "likelihood_backend", "numba"),
        )
    kernels = get_likelihood_backend(backend_choice)
    n_ifo = nIFO

    # Keep timing and event bookkeeping beside the flat workflow; this makes
    # the cost of each optional stage visible without wrapping the stage calls.
    started = time.perf_counter()
    timings: dict[str, float] = {}
    n_pixels = len(cluster.pixel_arrays)
    _log_start(kernels.name, cluster_id, n_pixels)

    # Refresh per-pixel noise weights only when whitening maps are supplied.
    if nRMS is not None and len(nRMS) == n_ifo:
        cluster.pixel_arrays.populate_noise_rms(nRMS)

    # Stage 1: select the normal or coarse sky grid, then apply static and
    # event-dependent sky masks before any sky-dependent calculation.
    stage_started = time.perf_counter()
    grid, grid_rejection = _select_sky_grid(
        config, setup, cluster, cluster_id, n_pixels
    )
    timings["sky_grid"] = time.perf_counter() - stage_started
    if grid_rejection:
        return _reject(
            grid_rejection,
            cluster_id,
            n_pixels,
            kernels.name,
            started,
            timings,
        )

    # Resolve optional features and cuts once.  An empty plan is the inexpensive
    # default for high-volume background production.
    extension_plan = setup.get("likelihood_extension_plan")
    if extension_plan is None:
        extension_plan = resolve_extension_plan(config)

    # Stage 2: convert sparse pixel/xtalk data to the dense layouts expected by
    # both numerical backends.  Noise is (pixel, detector); delayed quadratures
    # are (delay, detector, pixel).
    stage_started = time.perf_counter()
    xtalk_lookup, xtalk_coefficients = xtalk.get_xtalk_pixels(
        cluster.pixel_arrays, True
    )
    noise_weights, phase0, phase90, _ = extract_pixel_time_delay_data(
        None, n_ifo, pixel_arrays=cluster.pixel_arrays
    )
    noise_weights = noise_weights.T.astype(np.float32)
    td_phase0 = np.transpose(phase0.astype(np.float32), (2, 0, 1))
    td_phase90 = np.transpose(phase90.astype(np.float32), (2, 0, 1))
    timings["data_prep"] = time.perf_counter() - stage_started

    # Stage 3: construct the dominant-polarization-frame regulators.  The
    # second term depends on the active sky grid and is computed by the backend.
    regularization = np.array(
        [setup["delta_regulator"] * np.sqrt(2.0), 0.0, 0.0],
        dtype=np.float32,
    )
    stage_started = time.perf_counter()
    regularization[1] = kernels.calculate_dpf_regulator(
        grid.plus_patterns,
        grid.cross_patterns,
        noise_weights,
        grid.size,
        n_ifo,
        setup["gamma_regulator"],
        setup["network_energy_threshold"],
        grid.valid_indices,
    )
    timings["dpf_regulator"] = time.perf_counter() - stage_started

    # Stage 4: evaluate the coherent statistic over all valid sky directions.
    # The scan returns compact sky maps and the maximizing direction l_max.
    stage_started = time.perf_counter()
    sky_scan_result = kernels.scan_sky(
        n_ifo,
        n_pixels,
        grid.size,
        grid.plus_patterns,
        grid.cross_patterns,
        noise_weights,
        td_phase0,
        td_phase90,
        grid.delays,
        regularization,
        setup["netCC"],
        setup["delta_regulator"],
        setup["network_energy_threshold"],
        grid.valid_indices,
    )
    skymap_statistics = SkyMapStatistics.from_tuple(sky_scan_result)
    timings["sky_scan"] = time.perf_counter() - stage_started
    # A non-positive maximum means that no viable coherent sky solution exists.
    if skymap_statistics.sky_stat_max <= 0.0:
        return _reject(
            f"non-positive sky_stat_max={skymap_statistics.sky_stat_max:.3f}",
            cluster_id,
            n_pixels,
            kernels.name,
            started,
            timings,
            skymap_statistics,
        )

    # Stage 5: recompute detailed coherent and per-pixel statistics only at the
    # best-fit direction instead of retaining them for the full sky grid.
    stage_started = time.perf_counter()
    sky_statistics = kernels.statistics_at_best_fit(
        int(skymap_statistics.l_max),
        n_ifo,
        n_pixels,
        grid.plus_patterns,
        grid.cross_patterns,
        noise_weights,
        td_phase0,
        td_phase90,
        grid.delays,
        regularization,
        setup["network_energy_threshold"],
        xtalk_coefficients,
        xtalk_lookup,
        xgb_rho_mode=setup["xgb_rho_mode"],
    )
    timings["sky_statistics_at_lmax"] = time.perf_counter() - stage_started

    # Stage 6: normalize sky probability for every survivor.  The O(N log N)
    # ranking and target-region products are built only when the plan requires
    # them, then passed explicitly to features and cuts that share them.
    stage_started = time.perf_counter()
    sky_temperature = validate_sky_temperature(
        getattr(config, "likelihood_sky_temperature", 1.0)
    )
    sky_probability, evaluated_sky_indices = build_sky_probability(
        skymap_statistics.nSkyStat,
        grid.valid_indices,
        sky_temperature,
    )
    skymap_statistics.nProbability = sky_probability

    ranked_sky_indices = None
    ranked_cumulative_probability = None
    if "sky_ranking" in extension_plan.required_products:
        ranked_sky_indices, ranked_cumulative_probability = (
            rank_sky_probability(sky_probability, evaluated_sky_indices)
        )

    target_sky_indices = None
    target_metrics = None
    target_preparation_error: Exception | None = None
    target_region = getattr(config, "likelihood_target_region", None)
    needs_target_metrics = "target_metrics" in extension_plan.required_products
    if "target_indices" in extension_plan.required_products and (
        target_region or needs_target_metrics
    ):
        try:
            target_sky_indices = build_target_sky_indices(
                target_region,
                grid.phi_geo,
                grid.latitude,
                t_ref=trigger_gps(cluster, setup),
            )
            if needs_target_metrics:
                target_metrics = compute_target_sky_metrics(
                    skymap_statistics.nSkyStat,
                    sky_probability,
                    evaluated_sky_indices,
                    target_sky_indices,
                    target_level=getattr(
                        config, "likelihood_target_level", 0.9
                    ),
                    ranked_sky_indices=ranked_sky_indices,
                    ranked_cumulative_probability=(
                        ranked_cumulative_probability
                    ),
                )
        except Exception as exc:
            if (
                "target_sky_consistency"
                in extension_plan.post_sky_cut_names
                or extension_plan.feature_failure == "error"
            ):
                raise
            target_preparation_error = exc
    timings["sky_probability"] = time.perf_counter() - stage_started

    # Stage 7: apply the standard production thresholds before waveform
    # reconstruction and optional feature extraction.
    stage_started = time.perf_counter()
    selected_pixels = int(
        np.count_nonzero(np.asarray(sky_statistics.pixel_mask) > 0)
    )
    logger.info("Selected core pixels: %d / %d", selected_pixels, n_pixels)
    standard_rejection = get_likelihood_rejection_reason(
        sky_statistics,
        setup["network_energy_threshold"],
        setup["netEC_threshold"],
        net_rho_threshold=setup["net_rho_threshold"],
        xgb_rho_mode=setup["xgb_rho_mode"],
    )
    timings["threshold_cut"] = time.perf_counter() - stage_started
    if standard_rejection:
        return _reject(
            standard_rejection,
            cluster_id,
            n_pixels,
            kernels.name,
            started,
            timings,
            skymap_statistics,
        )

    # Stage 8: run optional sky-only cuts early.  A targeted search can, for
    # example, reject a cluster whose 90% HPD region misses the target region.
    stage_started = time.perf_counter()
    post_sky = run_likelihood_cuts(
        extension_plan,
        stage="post_sky",
        config=config,
        target_metrics=target_metrics,
    )
    timings["likelihood_post_sky_cuts"] = (
        time.perf_counter() - stage_started
    )
    cut_metrics = dict(post_sky.metrics)
    if not post_sky.passed:
        return _reject(
            post_sky.reason or "post-sky likelihood cut",
            cluster_id,
            n_pixels,
            kernels.name,
            started,
            timings,
            skymap_statistics,
        )

    # Stage 9: populate event statistics and reconstructed waveforms only for
    # clusters that survived all inexpensive sky-level cuts.
    stage_started = time.perf_counter()
    wdm_list = _create_wdm_set_python(config)
    populate_detection_statistics(
        sky_statistics,
        skymap_statistics,
        cluster=cluster,
        n_ifo=n_ifo,
        xtalk=xtalk,
        network_energy_threshold=setup["network_energy_threshold"],
        xgb_rho_mode=setup["xgb_rho_mode"],
        config=config,
        cluster_xtalk=xtalk_coefficients,
        cluster_xtalk_lookup=xtalk_lookup,
        wdm_list=wdm_list,
    )
    update_chirp_mass_statistics(
        cluster,
        xgb_rho_mode=setup["xgb_rho_mode"],
        pat0=getattr(config, "pattern", 10) == 0,
    )
    timings["reconstruction"] = time.perf_counter() - stage_started

    # Stage 10: reserve a clear hook for cuts that require reconstructed data.
    stage_started = time.perf_counter()
    post_reconstruction = run_likelihood_cuts(
        extension_plan,
        stage="post_reconstruction",
        config=config,
        target_metrics=target_metrics,
    )
    timings["likelihood_post_reconstruction_cuts"] = (
        time.perf_counter() - stage_started
    )
    cut_metrics.update(post_reconstruction.metrics)
    if not post_reconstruction.passed:
        return _reject(
            post_reconstruction.reason or "post-reconstruction likelihood cut",
            cluster_id,
            n_pixels,
            kernels.name,
            started,
            timings,
            skymap_statistics,
        )

    # Stage 11: compute only selected optional outputs from explicit inputs.
    stage_started = time.perf_counter()
    preparation_errors = (
        {"target_sky_metrics": target_preparation_error}
        if target_preparation_error is not None
        else None
    )
    features, feature_status = run_likelihood_features(
        extension_plan,
        config=config,
        sky_probability=sky_probability,
        evaluated_sky_indices=evaluated_sky_indices,
        phi_geo_arr=grid.phi_geo,
        latitude_arr=grid.latitude,
        healpix_order=grid.healpix_order,
        l_max=int(skymap_statistics.l_max),
        ranked_sky_indices=ranked_sky_indices,
        ranked_cumulative_probability=ranked_cumulative_probability,
        target_metrics=target_metrics,
        preparation_errors=preparation_errors,
    )
    if feature_status.get("sky_area", {}).get("ok", False):
        cluster.sky_area = build_legacy_sky_area(
            pixel_area_deg2=sky_pixel_area_deg2(len(sky_probability)),
            sky_probability=sky_probability,
            evaluated_sky_indices=evaluated_sky_indices,
            ranked_sky_indices=ranked_sky_indices,
            ranked_cumulative_probability=ranked_cumulative_probability,
            target_sky_indices=target_sky_indices,
        )
    skymap_statistics.likelihood_features = dict(features) or None
    skymap_statistics.likelihood_feature_status = dict(feature_status) or None
    skymap_statistics.likelihood_cut_metrics = dict(cut_metrics) or None
    skymap_statistics.likelihood_metadata = (
        {
            "sky_probability_temperature": sky_temperature,
            "evaluated_sky_pixel_count": int(len(evaluated_sky_indices)),
        }
        if features or feature_status or cut_metrics
        else None
    )
    timings["likelihood_features"] = time.perf_counter() - stage_started

    # Stage 12: persist the best sky position/time delays and mark acceptance.
    stage_started = time.perf_counter()
    sky_index = int(skymap_statistics.l_max)
    theta = np.pi / 2.0 - grid.latitude[sky_index]
    phi = grid.phi_geo[sky_index]
    cluster.cluster_meta.l_max = sky_index
    cluster.cluster_meta.theta = float(np.clip(np.degrees(theta), 0.0, 180.0))
    cluster.cluster_meta.phi = float(np.degrees(phi)) % 360.0
    if cluster.cluster_meta.c_time == 0.0:
        cluster.cluster_meta.c_time = cluster.cluster_time
    if cluster.cluster_meta.c_freq == 0.0:
        cluster.cluster_meta.c_freq = cluster.cluster_freq
    cluster.sky_time_delay = [
        float(grid.delays[detector, sky_index]) for detector in range(n_ifo)
    ]
    cluster.cluster_status = -1
    timings["finalize"] = time.perf_counter() - stage_started
    _log_finish(
        cluster_id,
        n_pixels,
        kernels.name,
        started,
        timings,
        skymap_statistics,
        detected=True,
    )
    return cluster, skymap_statistics


def _select_sky_grid(
    config,
    setup: dict,
    cluster: Cluster,
    cluster_id: int | None,
    n_pixels: int,
) -> tuple[SkyGrid | None, str | None]:
    # The low 16 bits of precision retain the cWB large-cluster size rule.
    # Large clusters use a coarser grid when one was prepared, limiting scan cost.
    precision = int(abs(getattr(config, "precision", 0) or 0))
    cluster_size_limit = precision % 65536
    n_resolutions = int(getattr(config, "nRES", 1) or 1)
    use_big_grid = bool(
        cluster_size_limit > 0
        and n_pixels > n_resolutions * cluster_size_limit
        and setup.get("ml_big_cluster") is not None
    )

    # Keep delays, antenna patterns, and coordinates on the same selected grid.
    suffix = "_big_cluster" if use_big_grid else ""
    delays = setup[f"ml{suffix}"]
    plus = setup[f"FP{suffix}_t"] if suffix else setup["FP_t"]
    cross = setup[f"FX{suffix}_t"] if suffix else setup["FX_t"]
    n_sky = int(delays.shape[1])

    if use_big_grid:
        valid = setup.get("sky_valid_indices_big")
        phi_geo = setup["phi_geo_arr_big_cluster"]
        latitude = setup["latitude_arr_big_cluster"]
        healpix_order = setup.get("big_cluster_healpix_order")
        logger.info(
            "Cluster-id=%s is big (%d px): using coarse sky grid (%d directions)",
            cluster_id,
            n_pixels,
            n_sky,
        )
    else:
        valid = setup.get("sky_valid_indices")
        phi_geo = setup.get("phi_geo_arr")
        latitude = setup.get("latitude_arr")
        if phi_geo is None:
            phi_geo = setup["ra_arr"]
        if latitude is None:
            latitude = setup["dec_arr"]
        healpix_order = setup.get("healpix_order")

    # The cluster-specific mask may update a time-dependent ICRS mask; when it
    # does, it replaces the setup-level list of valid directions.
    if valid is None:
        valid = np.arange(n_sky, dtype=np.int64)
    cluster_valid = sky_valid_indices_for_cluster(
        setup, cluster, use_big_grid=use_big_grid
    )
    if cluster_valid is not None:
        valid = cluster_valid
    valid = np.asarray(valid, dtype=np.int64)
    if valid.size == 0:
        return None, "sky mask selects no grid directions"

    grid = SkyGrid(
        delays=np.asarray(delays),
        plus_patterns=np.asarray(plus),
        cross_patterns=np.asarray(cross),
        valid_indices=valid,
        phi_geo=np.asarray(phi_geo),
        latitude=np.asarray(latitude),
        healpix_order=healpix_order,
    )
    return grid, None


def _log_start(
    backend_name: str,
    cluster_id: int | None,
    n_pixels: int,
) -> None:
    logger.info("-------------------------------------------------------")
    logger.info(
        "-> [%s] Processing cluster-id=%d|pixels=%d",
        backend_name,
        int(cluster_id) if cluster_id is not None else -1,
        n_pixels,
    )
    logger.info("   ----------------------------------------------------")


def _reject(
    reason: str,
    cluster_id: int | None,
    n_pixels: int,
    backend_name: str,
    started: float,
    timings: dict[str, float],
    skymap_statistics: SkyMapStatistics | None = None,
) -> tuple[None, None]:
    logger.info("Cluster-id=%s rejected: %s", cluster_id, reason)
    _log_finish(
        cluster_id,
        n_pixels,
        backend_name,
        started,
        timings,
        skymap_statistics,
        detected=False,
    )
    return None, None


def _log_finish(
    cluster_id: int | None,
    n_pixels: int,
    backend_name: str,
    started: float,
    timings: dict[str, float],
    skymap_statistics: SkyMapStatistics | None,
    *,
    detected: bool,
) -> None:
    timings["total"] = time.perf_counter() - started
    logger.info(
        "   cluster-id|pixels: %5d|%d",
        int(cluster_id) if cluster_id is not None else -1,
        n_pixels,
    )
    logger.info("\t -> SELECTED !!!" if detected else "\t <- rejected")
    logger.info("Total events: %d", int(detected))
    logger.info("Total time: %.2f s", timings["total"])
    logger.info("Stage timings (%s):", backend_name)
    for stage, elapsed in timings.items():
        if stage != "total":
            fraction = (
                100.0 * elapsed / timings["total"]
                if timings["total"] > 0
                else 0.0
            )
            logger.info("  %-34s %.4f s  (%5.1f%%)", stage, elapsed, fraction)
    logger.info("-------------------------------------------------------")

    if skymap_statistics is not None:
        skymap_statistics.stage_timings = dict(timings)
        skymap_statistics.likelihood_backend = backend_name
        if skymap_statistics.likelihood_metadata is not None:
            skymap_statistics.likelihood_metadata.setdefault(
                "likelihood_backend", backend_name
            )


def evaluate_fragment_clusters(
    config: Config,
    fragment_clusters: list[FragmentCluster],
    strains: list[TimeSeries],
    MRAcatalog: str,
    nRMS: list[TimeFrequencyMap] | None = None,
    xtalk: XTalk | None = None,
    backend: str | LikelihoodKernels | None = None,
) -> list[list[tuple[Cluster, SkyMapStatistics]]]:
    """Evaluate all surviving clusters using one setup and one backend."""

    started = time.perf_counter()
    strains = [TimeSeries.from_input(strain) for strain in strains]
    if xtalk is None:
        xtalk = XTalk.load(MRAcatalog, dump=True)

    # Setup arrays, xtalk data, and backend kernels are segment-level resources;
    # resolve them once and pass them directly into every per-cluster call.
    setup = prepare_likelihood_inputs(config, strains, config.nIFO)
    backend_choice = backend or setup["likelihood_backend"]
    kernels = get_likelihood_backend(backend_choice)

    results: list[list[tuple[Cluster, SkyMapStatistics]]] = []
    for fragment_cluster in fragment_clusters:
        lag_results: list[tuple[Cluster, SkyMapStatistics]] = []
        for index, selected_cluster in enumerate(fragment_cluster.clusters):
            if selected_cluster.cluster_status > 0:
                continue
            cluster_id = index + 1
            selected_cluster.cluster_id = cluster_id
            result_cluster, sky_stats = evaluate_cluster_likelihood(
                config.nIFO,
                selected_cluster,
                config,
                cluster_id=cluster_id,
                nRMS=nRMS,
                setup=setup,
                xtalk=xtalk,
                backend=kernels,
            )
            if result_cluster is None or result_cluster.cluster_status != -1:
                logger.info(
                    "likelihood rejected cluster %d (%d pixels)",
                    cluster_id,
                    len(selected_cluster.pixel_arrays),
                )
                continue
            logger.info(
                "likelihood accepted cluster %d (%d pixels)",
                cluster_id,
                len(result_cluster.pixel_arrays),
            )
            lag_results.append((result_cluster, sky_stats))
        results.append(lag_results)

    total_accepted = sum(len(lag) for lag in results)
    logger.info(
        "Likelihood wrapper (%s) done: %d accepted across %d lag(s) in %.2f s",
        kernels.name,
        total_accepted,
        len(fragment_clusters),
        time.perf_counter() - started,
    )
    return results


def _prepare_setup(
    n_ifo: int,
    config: Config,
    *,
    strains: list[TimeSeries] | None,
    supercluster_setup: dict | None,
) -> dict:
    # Reuse antenna/delay arrays already produced by superclustering when they
    # are available; otherwise prepare them from the strain segment.
    delays = plus = cross = None
    if supercluster_setup is not None:
        delays = supercluster_setup.get(
            "ml_likelihood", supercluster_setup.get("ml")
        )
        plus = supercluster_setup.get(
            "FP_likelihood", supercluster_setup.get("FP")
        )
        cross = supercluster_setup.get(
            "FX_likelihood", supercluster_setup.get("FX")
        )
    if strains is None and delays is None:
        raise ValueError(
            "likelihood(): setup, strains, or supercluster_setup must be provided"
        )
    return prepare_likelihood_inputs(
        config,
        strains,
        n_ifo,
        ml=delays,
        FP=plus,
        FX=cross,
        ml_big=(
            supercluster_setup.get("ml_big_cluster")
            if supercluster_setup else None
        ),
        FP_big=(
            supercluster_setup.get("FP_big_cluster")
            if supercluster_setup else None
        ),
        FX_big=(
            supercluster_setup.get("FX_big_cluster")
            if supercluster_setup else None
        ),
        big_cluster_healpix_order=(
            supercluster_setup.get("big_cluster_healpix_order")
            if supercluster_setup else None
        ),
    )


# Stable public aliases retained for existing callers.
setup_likelihood = prepare_likelihood_inputs
likelihood = evaluate_cluster_likelihood
likelihood_wrapper = evaluate_fragment_clusters

__all__ = [
    "setup_likelihood",
    "likelihood",
    "likelihood_wrapper",
    "prepare_likelihood_inputs",
    "evaluate_cluster_likelihood",
    "evaluate_fragment_clusters",
]
