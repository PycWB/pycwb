from __future__ import annotations

import logging
import time
import types as _types
from typing import TYPE_CHECKING, Any

import numpy as np

from .utils import (
    aggregate_clusters_from_links,
    apply_subnet_cut,
    calculate_statistics,
    expected_td_vec_len,
    get_cluster_links,
    get_defragment_link,
)
from ...types.detector import calculate_e2or_from_acore, compute_sky_delay_and_patterns
from ...types.network_cluster import Cluster, ClusterMeta
from ...types.time_series import TimeSeries

if TYPE_CHECKING:
    from pycwb.modules.xtalk.type import XTalk


logger = logging.getLogger(__name__)


def supercluster(
    clusters: list[Cluster],
    atype: str,
    gap: float,
    threshold: float,
    n_ifo: int,
    mini_pix: int = 3,
    core: bool = False,
    pair: bool = False,
) -> list[Cluster]:
    """
    Supercluster algorithm

    Parameters
    ----------
    cluster : List[Cluster]
        List of clusters
    atype : str
        Statistics type: 'L' for likelihood, 'E' for energy, 'P' for power
    gap : float
        Time-frequency gap used to link clusters
    threshold : float
        Threshold for discarding clusters
    n_ifo : int
        Number of interferometers
    mini_pix : int
        Minimum number of pixels in a cluster
    core : bool
        true - use only core pixels, false - use core & halo pixels
    pair : bool
        true - 2 resolutions, false - 1 resolution
    """
    # Vectorized pixel extraction for better performance with large pixel counts
    pixel_data_list = []
    for c_id, c in enumerate(clusters):
        for p in c.pixels:
            row = [p.time / p.rate / p.layers, p.frequency * p.rate, 1 / p.rate, p.rate / 2, c_id]
            row.extend([d.index / p.rate / p.layers for d in p.data])
            pixel_data_list.append(row)
    
    if not pixel_data_list:
        return clusters
        
    pixels = np.array(pixel_data_list, dtype=np.float64)

    # get the full cluster ids
    cluster_ids = np.arange(len(clusters))

    # find links between clusters
    # FIXME(#?): dF returned by get_cluster_links is the last-computed value for the final
    # pixel pair examined, so it may not represent the cluster as a whole.
    # Tracked in issue — do not use dF for per-cluster frequency normalisation until resolved.
    cluster_links, dF = get_cluster_links(pixels, gap, n_ifo)

    # remove redundant links
    # cluster_links = remove_duplicates_sorted(cluster_links[np.lexsort((cluster_links[:, 1], cluster_links[:, 0]))])

    if len(cluster_links) == 0:
        return clusters

    # aggregate clusters
    aggregated_clusters = aggregate_clusters_from_links(cluster_ids, cluster_links)

    superclusters = []
    for c_ids in aggregated_clusters:
        clusters_temp = []
        for c_id in c_ids:
            clusters_temp.append(clusters[c_id])
        # TODO: extract the per-cluster cut into a separate function so supercluster()
        # stays a pure grouping operation.
        sc = calculate_supercluster_data(clusters_temp, atype, core, pair, mini_pix, threshold, dF)
        superclusters.append(sc)

    return superclusters


def calculate_supercluster_data(
    clusters: list[Cluster],
    atype: str,
    core: bool,
    pair: bool,
    nPIX: int,
    S: float,
    dF: float,
) -> Cluster:
    """
    Merge a group of clusters into one supercluster and compute its statistics.

    Parameters
    ----------
    clusters : list[Cluster]
        Clusters to merge (already linked by :func:`supercluster`).
    atype : str
        Statistics type: ``'L'`` (likelihood), ``'E'`` (energy), or ``'P'`` (power).
    core : bool
        If ``True``, only core pixels contribute to the statistics.
    pair : bool
        If ``True``, require at least two resolution levels.
    nPIX : int
        Minimum pixel count; candidates with fewer pixels are rejected.
    S : float
        Minimum amplitude/likelihood threshold per resolution level.
    dF : float
        Frequency correction term passed to :func:`calculate_statistics`.

    Returns
    -------
    Cluster
        Merged supercluster with ``cluster_status=0`` (accepted) or
        ``cluster_status=1`` (rejected by statistics cut).
    """
    # Vectorized pixel array construction for better performance
    pixel_data_list = []
    pixel_objects = []
    
    for c_id, c in enumerate(clusters):
        for p in c.pixels:
            row = [p.core, p.time, p.frequency, p.rate, p.layers, p.likelihood]
            row.extend([d.asnr for d in p.data])
            pixel_data_list.append(row)
            pixel_objects.append(p)
    
    if not pixel_data_list:
        return Cluster(
            pixels=[],
            cluster_status=1,
            cluster_meta=ClusterMeta()
        )
    
    pixels = np.array(pixel_data_list, dtype=np.float64)
    stat = calculate_statistics(pixels, atype, core, pair, nPIX, S, dF)

    if stat is None:
        new_supercluster = Cluster(
            pixels=pixel_objects,
            cluster_status=1,
            cluster_meta=ClusterMeta()
        )
    else:
        new_supercluster = Cluster(
            pixels=pixel_objects,
            cluster_status=0,
            cluster_time=stat[0],
            cluster_freq=stat[1],
            cluster_rate=[int(d+0.01) for d in stat[2:]],
            cluster_meta=ClusterMeta(c_time=stat[0], c_freq=stat[1], like_net=stat[4], energy=stat[4])
        )
    return new_supercluster


def defragment(
    clusters: list[Cluster],
    t_gap: float,
    f_gap: float,
    n_ifo: int,
) -> list[Cluster]:
    """
    Defragmentation algorithm — merge clusters that are close in time and frequency.

    Parameters
    ----------
    clusters : list[Cluster]
        Input clusters to defragment.
    t_gap : float
        Maximum time separation (seconds) for merging two clusters.
    f_gap : float
        Maximum frequency separation (Hz) for merging two clusters.
    n_ifo : int
        Number of interferometers.

    Returns
    -------
    list[Cluster]
        Defragmented clusters; may have fewer elements than *clusters* if
        any were merged together.
    """
    # Vectorized pixel extraction
    pixel_data_list = []
    for c_id, cluster in enumerate(clusters):
        for p in cluster.pixels:
            row = [p.time / p.rate / p.layers, p.frequency * p.rate, 1 / p.rate, p.rate / 2, c_id]
            row.extend([d.index / p.rate / p.layers for d in p.data])
            pixel_data_list.append(row)
    
    if not pixel_data_list:
        return clusters
        
    pixels = np.array(pixel_data_list, dtype=np.float64)

    # get the full cluster ids
    cluster_ids = np.arange(len(clusters))

    # find links between clusters
    cluster_links = get_defragment_link(pixels, t_gap, f_gap, n_ifo)

    if len(cluster_links) == 0:
        return clusters

    # aggregate clusters
    aggregated_clusters = aggregate_clusters_from_links(cluster_ids, cluster_links)

    superclusters = []
    for c_ids in aggregated_clusters:
        # Collect all pixels from the aggregated clusters
        all_pixels = []
        for c_id in c_ids:
            all_pixels.extend(clusters[c_id].pixels)

        sc = Cluster(
            pixels=all_pixels,
            cluster_status=0,
            cluster_meta=ClusterMeta()
        )
        superclusters.append(sc)

    return superclusters


def supercluster_wrapper(
    config: Any,
    fragment_clusters: list,
    strains: list,
    xtalk_coeff: np.ndarray,
    xtalk_lookup_table: np.ndarray,
    layers: np.ndarray,
    return_td_cache: bool = False,
    job_seg: Any | None = None,
) -> list | tuple | None:
    """
    Convenience wrapper for interactive / legacy use.

    Internally calls :func:`setup_supercluster`, :func:`build_td_inputs_cache`,
    and :func:`supercluster_single_lag` for every lag, avoiding all the
    duplicated WDM/sky-pattern setup code that was previously inlined here.

    Parameters
    ----------
    config : Config
    fragment_clusters : list[list[FragmentCluster]]
        ``fragment_clusters[res][lag]`` — output of :func:`~pycwb.modules.cwb_coherence.coherence.coherence`.
    strains : list
        Whitened strain time series (one per IFO).
    xtalk_coeff : np.ndarray
    xtalk_lookup_table : np.ndarray
    layers : np.ndarray
    return_td_cache : bool
        If True, return ``(result, td_inputs_cache)`` instead of ``result``.
    job_seg : WaveSegment
        Job segment supplying lag count.

    Returns
    -------
    list[FragmentCluster] or None
        One FragmentCluster per lag, or None if any lag produces no clusters.
    """
    timer_start = time.perf_counter()

    strains = [TimeSeries.from_input(strain) for strain in strains]

    if job_seg is None:
        n_lag = len(fragment_clusters[0]) if fragment_clusters else 1
        job_seg = _types.SimpleNamespace(n_lag=n_lag)
    n_lag = int(job_seg.n_lag)
    if n_lag == 0:
        logger.info("Supercluster wrapper skipped: n_lag=0")
        return (None, {}) if return_td_cache else None

    if len(fragment_clusters) == 0 or len(fragment_clusters[0]) < n_lag:
        raise ValueError("Fragment clusters are inconsistent with lag plan")

    # Build lag-independent resources once
    from pycwb.utils.td_vector_batch import build_td_inputs_cache
    from pycwb.modules.xtalk.type import XTalk

    td_inputs_cache = build_td_inputs_cache(config, strains)
    gps_time = float(strains[0].t0)
    setup = setup_supercluster(config, gps_time)
    xtalk = XTalk(coeff=xtalk_coeff, lookup_table=xtalk_lookup_table, layers=layers, nRes=0)

    n_res = len(fragment_clusters)
    result_clusters = []

    for j in range(n_lag):
        frag_clusters_this_lag = [fragment_clusters[res][j] for res in range(n_res)]
        fragment_cluster = supercluster_single_lag(
            setup, config, frag_clusters_this_lag, j,
            xtalk=xtalk, td_inputs_cache=td_inputs_cache,
        )
        if fragment_cluster is None:
            logger.warning("No supercluster results for lag %d", j)
            return (None, td_inputs_cache) if return_td_cache else None
        result_clusters.append(fragment_cluster)

    total_clusters = sum(len(fc.clusters) for fc in result_clusters)
    total_pixels = sum(len(c.pixels) for fc in result_clusters for c in fc.clusters)
    logger.info("Supercluster wrapper done")
    logger.info("total  clusters|pixels : %6d|%d", total_clusters, total_pixels)
    logger.info("----------------------------------------")
    logger.info("Supercluster time: %.2f s", time.perf_counter() - timer_start)
    logger.info("----------------------------------------")

    return (result_clusters, td_inputs_cache) if return_td_cache else result_clusters


# ---------------------------------------------------------------------------
# Streaming-friendly API: setup once, iterate over lags
# ---------------------------------------------------------------------------

def setup_supercluster(config: Any, gps_time: float) -> dict:
    """
    Compute lag-independent supercluster resources: sky delay / antenna-pattern
    matrices and the K_td delay range.

    Call this once per job segment, then pass the returned dict together with
    a separately built ``td_inputs_cache`` to :func:`supercluster_single_lag`.

    Parameters
    ----------
    config : Config
    gps_time : float
        GPS time of the first strain sample (used for sky-delay computation).

    Returns
    -------
    dict
        Keys: ``ml``, ``FP``, ``FX``, ``n_sky``, ``ml_likelihood``,
        ``FP_likelihood``, ``FX_likelihood``, ``n_sky_likelihood``, ``K_td``.
    """
    timer_start = time.perf_counter()

    upTDF = int(getattr(config, 'upTDF', 1))
    TDRate = int(getattr(config, 'TDRate', int(config.rateANA) * upTDF))

    # ---- Sky delay / antenna-pattern matrices ----
    # Compute TWO sky-array sets that mirror ROOT/CWB behaviour:
    #   • full resolution (healpix order from config) → used by likelihood sky scan
    #   • reduced resolution (MIN_SKYRES_HEALPIX cap)  → used by apply_subnet_cut only
    if hasattr(config, "healpix") and int(config.healpix) > 0:
        healpix_order_full = int(config.healpix)
        min_skyres = int(getattr(config, "MIN_SKYRES_HEALPIX", healpix_order_full))
        healpix_order_subnet = min_skyres if healpix_order_full > min_skyres else healpix_order_full
    else:
        healpix_order_full = None
        healpix_order_subnet = None

    # K_td: delay range at TDRate resolution matching CWB's loadTDamp behaviour.
    # CWB: L = int(max_delay * TDRate) + 1 at TDRate steps.
    # Python must use the same resolution so nSkyStat discriminates directions.
    K_td = max(int(config.TDSize) * upTDF,
               int(getattr(config, 'max_delay', 0.0) * float(TDRate)) + 1)

    # Full-resolution sky arrays (for likelihood) — delays at TDRate resolution
    ml, FP, FX = compute_sky_delay_and_patterns(
        ifos=config.ifo,
        ref_ifo=config.refIFO,
        sample_rate=TDRate,
        td_size=K_td,
        gps_time=gps_time,
        healpix_order=healpix_order_full,
        n_sky=None,
    )

    # Reduced-resolution sky arrays (for subnet cut) — only recompute if different
    if healpix_order_subnet != healpix_order_full:
        ml_subnet, FP_subnet, FX_subnet = compute_sky_delay_and_patterns(
            ifos=config.ifo,
            ref_ifo=config.refIFO,
            sample_rate=TDRate,
            td_size=K_td,
            gps_time=gps_time,
            healpix_order=healpix_order_subnet,
            n_sky=None,
        )
    else:
        ml_subnet, FP_subnet, FX_subnet = ml, FP, FX

    logger.info(
        "[setup_supercluster] sky pixels: full=%d (healpix=%s), subnet=%d (healpix=%s)",
        int(ml.shape[1]), healpix_order_full,
        int(ml_subnet.shape[1]), healpix_order_subnet,
    )

    logger.info("Supercluster setup time: %.2f s", time.perf_counter() - timer_start)

    return {
        "ml": ml_subnet,              # reduced resolution for apply_subnet_cut
        "FP": FP_subnet,
        "FX": FX_subnet,
        "n_sky": int(ml_subnet.shape[1]),
        "ml_likelihood": ml,          # full resolution for likelihood sky scan
        "FP_likelihood": FP,
        "FX_likelihood": FX,
        "n_sky_likelihood": int(ml.shape[1]),
        "K_td": K_td,
    }


def supercluster_single_lag(
    setup: dict,
    config: Any,
    fragment_clusters_single_lag: list,
    lag_idx: int,
    xtalk: XTalk,
    td_inputs_cache: dict,
) -> Any | None:
    """
    Run the full supercluster pipeline for a single lag.

    Parameters
    ----------
    setup : dict
        Returned by :func:`setup_supercluster`.
    config : Config
        Configuration object — all thresholds and parameters are read
        directly from *config* rather than being pre-parsed into *setup*.
    fragment_clusters_single_lag : list[FragmentCluster]
        One ``FragmentCluster`` per resolution for *this* lag only
        (i.e. the output of :func:`~pycwb.modules.cwb_coherence.coherence.coherence_single_lag`).
    lag_idx : int
        Zero-based lag index (used for logging only).
    xtalk : XTalk
        Pre-loaded cross-talk catalog instance.
    td_inputs_cache : dict
        Pre-built TD inputs cache keyed by layer → list of per-IFO
        :class:`TDBatchInputs`.

    Returns
    -------
    FragmentCluster or None
        Processed cluster ready for likelihood, or ``None`` if all
        candidates were rejected at the subnet-cut stage.
    """
    n_ifo = config.nIFO
    K = int(setup["K_td"])

    # Merge all resolutions for this lag into one FragmentCluster.
    # No deepcopy needed: fragment_clusters_single_lag is freshly built per lag
    # by coherence_single_lag and is not referenced again after this call.
    fragment_cluster = fragment_clusters_single_lag[0]
    fragment_cluster.clusters = [c for c in fragment_cluster.clusters if c.cluster_status < 1]
    for fc in fragment_clusters_single_lag[1:]:
        fragment_cluster.clusters += [c for c in fc.clusters if c.cluster_status < 1]

    # Extract TD amplitudes from the pre-built cache
    pixel_objects = []
    for cluster in fragment_cluster.clusters:
        pixel_objects.extend(cluster.pixels)

    n_pixels = len(pixel_objects)
    if n_pixels == 0:
        logger.info("No pixels to process for lag %d", lag_idx)
        return None

    pixel_layers = np.array([int(p.layers) for p in pixel_objects], dtype=np.int32)
    pixel_indices = np.zeros((n_pixels, n_ifo), dtype=np.int32)
    for i, pixel in enumerate(pixel_objects):
        for n in range(n_ifo):
            pixel_indices[i, n] = int(pixel.data[n].index)

    unique_layers = np.unique(pixel_layers)
    pixels_by_layer = {
        int(layer): np.where(pixel_layers == layer)[0] for layer in unique_layers
    }

    td_vec_len = expected_td_vec_len(K)
    all_td_amps = np.zeros((n_pixels, n_ifo, td_vec_len), dtype=np.float32)

    for layer_key, pixel_idxs in pixels_by_layer.items():
        per_ifo_inputs = td_inputs_cache.get(layer_key)
        if per_ifo_inputs is None:
            per_ifo_inputs = (
                td_inputs_cache.get(layer_key - 1) or td_inputs_cache.get(layer_key + 1)
            )
        if per_ifo_inputs is None:
            logger.warning(
                "Missing TD input cache for layer %d, skipping %d pixels",
                layer_key, len(pixel_idxs),
            )
            continue
        layer_pixel_indices = pixel_indices[pixel_idxs]
        for ifo_idx in range(n_ifo):
            indices_np = np.asarray(layer_pixel_indices[:, ifo_idx], dtype=np.int32)
            batch_result = per_ifo_inputs[ifo_idx].extract_td_vecs(indices_np, K)
            all_td_amps[pixel_idxs, ifo_idx, :] = batch_result

    for pidx in range(n_pixels):
        pixel_objects[pidx].td_amp = [
            all_td_amps[pidx, ifo_idx, :] for ifo_idx in range(n_ifo)
        ]

    # Supercluster + subnet cut
    logger.info(
        "-> Processing lag=%d with %d clusters", lag_idx, len(fragment_cluster.clusters)
    )
    logger.info("   --------------------------------------------------")
    clusters = fragment_cluster.clusters

    super_acor = config.Acore
    super_e2or = calculate_e2or_from_acore(super_acor, n_ifo)
    subnet_acor = config.subacor if config.subacor > 0 else config.Acore
    subnet_e2or = calculate_e2or_from_acore(subnet_acor, n_ifo)
    pattern = int(getattr(config, "pattern", 0))

    superclusters = supercluster(clusters, 'L', config.TFgap, super_e2or, n_ifo)
    total_pixels = sum(len(c.pixels) for c in superclusters)
    accepted_superclusters = [sc for sc in superclusters if sc.cluster_status <= 0]
    logger.info(
        "   super clusters|pixels      : %6d|%d", len(superclusters), total_pixels
    )
    logger.info("   accepted superclusters     : %6d", len(accepted_superclusters))

    if not accepted_superclusters:
        logger.warning(
            "No accepted superclusters after supercluster stage (lag=%d)", lag_idx
        )
        return None
    
    if pattern != 0:
        accepted_superclusters = defragment(
            accepted_superclusters, config.Tgap, config.Fgap, n_ifo
        )
        logger.info("   defrag clusters|pixels     : %6d|%d", len(accepted_superclusters), sum(len(c.pixels) for c in accepted_superclusters))

    subrho = config.subrho if config.subrho > 0 else config.netRHO
    accepted_superclusters = apply_subnet_cut(
        accepted_superclusters,
        config.LOUD,
        setup["ml"],
        setup["FP"],
        setup["FX"],
        subnet_acor,
        subnet_e2or,
        n_ifo,
        setup["n_sky"],
        config.subnet,
        config.subcut,
        config.subnorm,
        subrho,
        xtalk,
    )

    if pattern == 0:
        accepted_superclusters = defragment(
            accepted_superclusters, config.Tgap, config.Fgap, n_ifo
        )

    total_pixels = sum(len(c.pixels) for c in accepted_superclusters)
    logger.info(
        "   post-cut clusters|pixels   : %6d|%d", len(accepted_superclusters), total_pixels
    )

    fragment_cluster.clusters = [c for c in accepted_superclusters if c.cluster_status <= 0]
    total_pixels = sum(len(c.pixels) for c in fragment_cluster.clusters)
    logger.info(
        "   final clusters|pixels      : %6d|%d",
        len(fragment_cluster.clusters),
        total_pixels,
    )

    # Mark all surviving pixels as core; td_amp is already populated above
    # and will be consumed directly by likelihood in the same lag iteration.
    for p in (p for c in fragment_cluster.clusters for p in c.pixels):
        p.core = 1

    return fragment_cluster


