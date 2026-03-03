import copy

import logging
import time
import numpy as np
from wdm_wavelet.wdm import WDM as WDMWavelet

from pycwb.modules.cwb_coherence.lag_plan import build_lag_plan_from_config
from .sub_net_cut import sub_net_cut
from .td_vector_batch import prepare_td_inputs, batch_extract_td_vecs
from .utils import get_cluster_links, calculate_statistics, get_defragment_link, \
    aggregate_clusters_from_links
from ...types.detector import compute_sky_delay_and_patterns, calculate_e2or_from_acore
from ...types.network_cluster import Cluster, ClusterMeta
from ...types.time_series import TimeSeries


logger = logging.getLogger(__name__)


def supercluster(clusters, atype, gap, threshold, n_ifo, mini_pix = 3, core=False, pair=False):
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
    # FIXME: understand the dF here, it must be wrong to use the last dF for all clusters
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
        # TODO: move the cut out side of the supercluster, make it additional function
        sc = calculate_supercluster_data(clusters_temp, atype, core, pair, mini_pix, threshold, dF)
        superclusters.append(sc)

    return superclusters


def calculate_supercluster_data(clusters, atype, core, pair, nPIX, S, dF):
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


def defragment(clusters, t_gap, f_gap, n_ifo):
    """
    Defragmentation algorithm

    Parameters
    ----------
    cluster : FragmentCluster
        The input cluster
    Tgap : float
        Time gap used to defragment clusters
    Fgap : float
        Frequency gap used to defragment clusters
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


def supercluster_wrapper(config, network, fragment_clusters, strains, xtalk_coeff, xtalk_lookup_table, layers,
                         return_td_cache: bool = False):
    timer_start = time.perf_counter()

    def _extract_timeseries_data(strain):
        values = np.asarray(strain.data, dtype=np.float64)
        sample_rate = float(strain.sample_rate)
        start = float(strain.t0)
        return values, sample_rate, start

    def _expected_td_vec_len(td_size):
        return 4 * int(td_size) + 2

    def _resolve_wdm_context(layer_tag, context_map):
        layer_tag = int(layer_tag)
        context = context_map.get(layer_tag)
        if context is not None:
            return context

        # cWB pixel convention stores layers as (wdm_M + 1)
        context = context_map.get(layer_tag - 1)
        if context is not None:
            return context

        raise ValueError(f"Missing WDM context for pixel layer {layer_tag}")

    def _apply_subnet_cut(superclusters, n_loudest_local, ml_local, FP_local, FX_local,
                          acor_local, e2or_local, n_ifo_local, n_sky_local,
                          subnet_local, subcut_local, subnorm_local, subrho_local,
                          xtalk_coeff_local, xtalk_lookup_table_local, layers_local):
        for i, c in enumerate(superclusters):
            c.pixels.sort(key=lambda x: x.likelihood, reverse=True)
            results = sub_net_cut(
                c.pixels[:n_loudest_local], ml_local, FP_local, FX_local,
                acor_local, e2or_local, n_ifo_local, n_sky_local,
                subnet_local, subcut_local, subnorm_local, subrho_local,
                xtalk_coeff_local, xtalk_lookup_table_local, layers_local,
            )

            if results['subnet_passed'] and results['subrho_passed'] and results['subthr_passed']:
                logger.info(
                    "Cluster %d (%d pixels) passed subnet, subrho, and subthr cut",
                    i,
                    len(c.pixels),
                )
                c.cluster_status = -1
            else:
                log_output = f"Cluster {i} ({len(c.pixels)} pixels) failed "
                if not results['subnet_passed']:
                    log_output += f"subnet cut condition: {results['subnet_condition']}, "
                if not results['subrho_passed']:
                    log_output += f"subrho cut condition: {results['subrho_condition']}, "
                if not results['subthr_passed']:
                    log_output += f"subthr cut condition: {results['subthr_condition']}, "
                logger.info(log_output)
                c.cluster_status = 1

        return [c for c in superclusters if c.cluster_status <= 0]

    strains = [TimeSeries.from_input(strain) for strain in strains]

    lag_plan = build_lag_plan_from_config(config, strains)
    n_lag = int(lag_plan.n_lag)
    if n_lag == 0:
        logger.info("Supercluster wrapper skipped: n_lag=0")
        return (None, {}) if return_td_cache else None

    if len(fragment_clusters) == 0 or len(fragment_clusters[0]) < n_lag:
        raise ValueError("Fragment clusters are inconsistent with lag plan")

    ########################
    # build wdm_wavelet contexts for each resolution and detector
    wdm_context_by_layers = {}
    for level in config.WDM_level:
        layers_at_level = 2 ** level if level > 0 else 0
        wdm_layers = max(1, int(layers_at_level))
        wdm = WDMWavelet(
            M=wdm_layers,
            K=wdm_layers,
            beta_order=config.WDM_beta_order,
            precision=config.WDM_precision,
        )
        wdm.set_td_filter(int(config.TDSize), 1)

        detector_tf_maps = []
        for n in range(config.nIFO):
            ts_data, sample_rate, t0 = _extract_timeseries_data(strains[n])
            detector_tf_maps.append(wdm.t2w(ts_data, sample_rate=sample_rate, t0=t0, MM=-1))

        context = {"wdm": wdm, "tf_maps": detector_tf_maps}
        # Accept both direct WDM layers and cWB pixel layer-tag convention (M + 1)
        wdm_context_by_layers[int(wdm_layers)] = context
        wdm_context_by_layers[int(wdm_layers) + 1] = context

    # Pre-cache JAX TD inputs (padded planes, filter tables) once per (layer, ifo).
    # These are derived from the TF maps which do not change across lags.
    td_inputs_cache = {}   # key: layer_key -> list of per-ifo dicts
    seen_keys = set()
    for layer_key, context in wdm_context_by_layers.items():
        # Both layer_key and layer_key+1 may point to the same context; deduplicate.
        ctx_id = id(context)
        if ctx_id in seen_keys:
            td_inputs_cache[layer_key] = td_inputs_cache.get(layer_key - 1) or td_inputs_cache.get(layer_key + 1)
            continue
        seen_keys.add(ctx_id)
        wdm_obj = context["wdm"]
        per_ifo = [prepare_td_inputs(context["tf_maps"][n], wdm_obj) for n in range(config.nIFO)]
        td_inputs_cache[layer_key] = per_ifo
    # For any layer_key that still has None (dedup gap), fill from its neighbour
    for layer_key in list(wdm_context_by_layers.keys()):
        if td_inputs_cache.get(layer_key) is None:
            alt = td_inputs_cache.get(layer_key - 1) or td_inputs_cache.get(layer_key + 1)
            td_inputs_cache[layer_key] = alt
    # merge cluster

    clusters_by_lag = []
    for j in range(n_lag):
        cluster = copy.deepcopy(fragment_clusters[0][j])
        cluster.clusters = [c for c in cluster.clusters if c.cluster_status < 1]
        if len(fragment_clusters) > 1:
            for fragment_cluster in fragment_clusters[1:]:
                cluster.clusters += [c for c in fragment_cluster[j].clusters if c.cluster_status < 1]
        clusters_by_lag.append(cluster)

    td_vec_default = np.zeros(_expected_td_vec_len(config.TDSize), dtype=np.float32)
    fragment_clusters = clusters_by_lag
    for lag, fragment_cluster in enumerate(fragment_clusters):
        # Batch processing: collect all pixels and pre-extract indices using numpy

        # Flatten all pixels from all clusters into a single list
        pixel_objects = []
        for cluster in fragment_cluster.clusters:
            pixel_objects.extend(cluster.pixels)
        
        n_pixels = len(pixel_objects)
        if n_pixels == 0:
            logger.info("No pixels to process for lag %d", lag)
            continue
        
        # Pre-extract pixel data into numpy arrays for vectorized operations
        # Shape: (n_pixels,)
        pixel_layers = np.array([int(p.layers) for p in pixel_objects], dtype=np.int32)
        
        # Shape: (n_pixels, n_ifo) - pre-extract all pixel indices
        pixel_indices = np.zeros((n_pixels, config.nIFO), dtype=np.int32)
        for i, pixel in enumerate(pixel_objects):
            for n in range(config.nIFO):
                pixel_indices[i, n] = int(pixel.data[n].index)
        
        # Group pixels by layer using numpy operations
        unique_layers = np.unique(pixel_layers)
        pixels_by_layer = {int(layer): np.where(pixel_layers == layer)[0] for layer in unique_layers}

        # Pre-allocate output array: (n_pixels, n_ifo, td_vec_len)
        td_vec_len = _expected_td_vec_len(config.TDSize)
        all_td_amps = np.zeros((n_pixels, config.nIFO, td_vec_len), dtype=np.float32)
        
        # Process pixels grouped by layer using JAX batch extraction
        K = int(config.TDSize)
        for layer_key, pixel_idxs in pixels_by_layer.items():
            per_ifo_inputs = td_inputs_cache.get(layer_key)
            if per_ifo_inputs is None:
                # Try adjacent key used by cWB layer-tag convention
                per_ifo_inputs = td_inputs_cache.get(layer_key - 1) or td_inputs_cache.get(layer_key + 1)
            if per_ifo_inputs is None:
                logger.warning(
                    "Missing TD input cache for layer %d, skipping %d pixels",
                    layer_key, len(pixel_idxs),
                )
                continue

            # Gather all pixel indices for this layer group — shape (n_layer_pixels, n_ifo)
            layer_pixel_indices = pixel_indices[pixel_idxs]  # (n_layer_pixels, n_ifo)

            for ifo_idx in range(config.nIFO):
                indices_np = np.asarray(layer_pixel_indices[:, ifo_idx], dtype=np.int32)
                batch_result = batch_extract_td_vecs(indices_np, per_ifo_inputs[ifo_idx], K)
                # batch_result: (n_layer_pixels, 4*K+2) float32
                all_td_amps[pixel_idxs, ifo_idx, :] = batch_result
        
        # Assign td_amp back to pixel objects using vectorized slicing
        for pidx in range(n_pixels):
            pixel_objects[pidx].td_amp = [all_td_amps[pidx, ifo_idx, :] for ifo_idx in range(config.nIFO)]

    ########################

    # prepare user parameters
    super_acor = config.Acore
    super_e2or = calculate_e2or_from_acore(super_acor, config.nIFO)

    subnet_acor = config.subacor if config.subacor > 0 else config.Acore
    subnet_e2or = calculate_e2or_from_acore(subnet_acor, config.nIFO)
    network_energy_threshold = 2 * subnet_acor * subnet_acor * config.nIFO
    if hasattr(config, "healpix") and int(config.healpix) > 0:
        healpix_order = int(config.healpix)
        min_skyres = int(getattr(config, "MIN_SKYRES_HEALPIX", healpix_order))
        if healpix_order > min_skyres:
            healpix_order = min_skyres
    else:
        healpix_order = None

    ml, FP, FX = compute_sky_delay_and_patterns(
        ifos=config.ifo,
        ref_ifo=config.refIFO,
        sample_rate=config.rateANA,
        td_size=int(config.TDSize),
        gps_time=float(strains[0].t0),
        healpix_order=healpix_order,
        n_sky=None,
    )
    n_sky = int(ml.shape[1])
    n_ifo = config.nIFO
    n_loudest = config.LOUD
    gap = config.TFgap
    Tgap = config.Tgap
    Fgap = config.Fgap
    subnet = config.subnet
    subcut = config.subcut
    subnorm = config.subnorm
    subrho = config.subrho if config.subrho > 0 else config.netRHO
    pattern = int(getattr(config, "pattern", 0))

    for fragment_cluster, lag in zip(fragment_clusters, range(n_lag)):
        logger.info(
            "-> Processing lag=%d with %d clusters",
            lag,
            len(fragment_cluster.clusters),
        )
        logger.info("   --------------------------------------------------")
        clusters = fragment_cluster.clusters

        superclusters = supercluster(clusters, 'L', gap, super_e2or, n_ifo)

        # Vectorized pixel counting for better performance
        total_pixels = sum(len(c.pixels) for c in superclusters)

        # filter out the rejected superclusters
        accepted_superclusters = [sc for sc in superclusters if sc.cluster_status <= 0]
        logger.info(
            "   super clusters|pixels      : %6d|%d",
            len(superclusters),
            total_pixels,
        )
        logger.info("   accepted superclusters     : %6d", len(accepted_superclusters))

        # if there are no accepted superclusters, return None
        if len(accepted_superclusters) == 0:
            logger.warning("No accepted superclusters after supercluster stage (lag=%d)", lag)
            return (None, td_inputs_cache) if return_td_cache else None

        selected_superclusters = _apply_subnet_cut(
            accepted_superclusters, n_loudest, ml, FP, FX,
            subnet_acor, subnet_e2or, n_ifo, n_sky,
            subnet, subcut, subnorm, subrho,
            xtalk_coeff, xtalk_lookup_table, layers,
        )

        if pattern == 0:
            # cWB order for pattern == 0: apply defragment after subnet cut
            new_superclusters = defragment(selected_superclusters, Tgap, Fgap, n_ifo)
        else:
            # cWB behavior for pattern != 0: no defragment stage
            new_superclusters = selected_superclusters

        # Vectorized pixel counting
        total_pixels = sum(len(c.pixels) for c in new_superclusters)
        logger.info(
            "   post-cut clusters|pixels   : %6d|%d",
            len(new_superclusters),
            total_pixels,
        )

        fragment_cluster.clusters = [c for c in new_superclusters if c.cluster_status <= 0]

        # Vectorized pixel counting
        total_pixels = sum(len(c.pixels) for c in fragment_cluster.clusters)
        logger.info(
            "   final clusters|pixels      : %6d|%d",
            len(fragment_cluster.clusters),
            total_pixels,
        )

        # Batch attribute updates - collect all pixels first, then update
        all_pixels = [p for c in fragment_cluster.clusters for p in c.pixels]
        for p in all_pixels:
            p.core = 1
            p.td_amp = None

    total_clusters = sum(len(fc.clusters) for fc in fragment_clusters)
    total_pixels = sum(len(c.pixels) for fc in fragment_clusters for c in fc.clusters)
    timer_stop = time.perf_counter()
    logger.info("Supercluster wrapper done")
    logger.info("total  clusters|pixels : %6d|%d", total_clusters, total_pixels)
    logger.info("----------------------------------------")
    logger.info("Supercluster time: %.2f s", timer_stop - timer_start)
    logger.info("----------------------------------------")

    return (fragment_clusters, td_inputs_cache) if return_td_cache else fragment_clusters


