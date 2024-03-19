import copy

import numpy as np

from .sub_net_cut import sub_net_cut
from .utils import get_cluster_links, calculate_statistics, get_defragment_link, \
    aggregate_clusters_from_links
from ..cwb_conversions import convert_fragment_clusters_to_netcluster, convert_sparse_series_to_sseries, \
    convert_netcluster_to_fragment_clusters
from ..likelihoodWP.likelihood import load_data_from_ifo
from ..multi_resolution_wdm import create_wdm_for_level
from ..sparse_series import sparse_table_from_fragment_clusters
from ...types.network_cluster import Cluster, ClusterMeta


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
    pixels = []
    for c_id, c in enumerate(clusters):
        for p in c.pixels:
            pixels.append([p.time / p.rate / p.layers, p.frequency * p.rate, 1 / p.rate, p.rate / 2, c_id] + [
                d.index / p.rate / p.layers for d in p.data])
    pixels = np.array(pixels)

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
    # construct the numpy pixel array with needed features
    pixels = []
    for c_id, c in enumerate(clusters):
        for p in c.pixels:
            pixels.append([p.core, p.time, p.frequency, p.rate, p.layers, p.likelihood] + [d.asnr for d in p.data])
    pixels = np.array(pixels)
    stat = calculate_statistics(pixels, atype, core, pair, nPIX, S, dF)

    pixels = []
    for c_id, c in enumerate(clusters):
        for p in c.pixels:
            pixels.append(p)

    if stat is None:
        new_supercluster = Cluster(
            pixels=pixels,
            cluster_status=1,
            cluster_meta=ClusterMeta()
        )
    else:
        new_supercluster = Cluster(
            pixels=pixels,
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
    pixels = []
    for c_id, cluster in enumerate(clusters):
        for p in cluster.pixels:
            pixels.append([p.time / p.rate / p.layers, p.frequency * p.rate, 1 / p.rate, p.rate / 2, c_id] + [
                d.index / p.rate / p.layers for d in p.data])
    pixels = np.array(pixels)

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
        clusters_temp = []
        for c_id in c_ids:
            clusters_temp.append(clusters[c_id])
        pixels = []
        for c_id, c in enumerate(clusters_temp):
            for p in c.pixels:
                pixels.append(p)

        sc = Cluster(
            pixels=pixels,
            cluster_status=0,
            cluster_meta=ClusterMeta()
        )
        superclusters.append(sc)

    return superclusters


def supercluster_wrapper(config, network, fragment_clusters, tf_maps, xtalk_coeff, xtalk_lookup_table, layers):
    # FIXME: check issue SSeries<DataType_t>::GetSTFdata : index not present in sparse table
    print("Generating sparse table from fragment clusters")
    sparse_table_list = sparse_table_from_fragment_clusters(config, tf_maps, fragment_clusters)

    skyres = config.MIN_SKYRES_HEALPIX if config.healpix > config.MIN_SKYRES_HEALPIX else 0

    ########################
    # ROOT code start
    print(f"Using cWB code to load the require data")
    if skyres > 0:
        network.update_sky_map(config, skyres)
        network.net.setAntenna()
        network.net.setDelay(config.refIFO)
        network.update_sky_mask(config, skyres)

    hot = []
    for n in range(config.nIFO):
        hot.append(network.get_ifo(n).getHoT())

    # set low-rate TD filters
    for level in config.WDM_level:
        wdm = create_wdm_for_level(config, level)
        wdm.set_td_filter(config.TDSize, 1)
        # add wavelets to network
        network.add_wavelet(wdm)

    # merge cluster
    print(f"Merging clusters with {len(fragment_clusters)} layers and {len(fragment_clusters[0])} lags")

    clusters_by_lag = []
    for j in range(int(network.nLag)):
        cluster = copy.deepcopy(fragment_clusters[0][j])
        if len(fragment_clusters) > 1:
            for fragment_cluster in fragment_clusters[1:]:
                cluster.clusters += fragment_cluster[j].clusters
        print(f"Number of clusters for lag {j}: {len(cluster.clusters)}")
        clusters_by_lag.append(cluster)

    # pwc_list = []
    # Load tdamp and convert to fragment cluster for testing
    net_clusters = [convert_fragment_clusters_to_netcluster(cluster) for cluster in clusters_by_lag]

    for n in range(config.nIFO):
        det = network.get_ifo(n)
        det.sclear()
        for sparse_table in sparse_table_list:
            det.vSS.push_back(convert_sparse_series_to_sseries(sparse_table[n]))

    fragment_clusters = []
    for lag in range(int(network.nLag)):
        print(f"Loading time-delay amp for lag {lag} of {int(network.nLag)}")
        pwc = network.get_cluster(lag)
        pwc.cpf(net_clusters[lag], False)

        if config.subacor > 0:
            network.net.acor = config.subacor
        if config.subrho > 0:
            network.net.netRHO = config.subrho

        network.set_delay_index(hot[0].rate())
        pwc.setcore(False)

        pwc.loadTDampSSE(network.net, 'a', config.BATCH, config.LOUD)

        fragment_clusters.append(convert_netcluster_to_fragment_clusters(pwc))

    print(f"cWB code finished")
    ########################

    # prepare user parameters
    acor = network.net.acor
    network_energy_threshold = 2 * acor * acor * config.nIFO
    n_sky = network.net.index.size()
    n_ifo = config.nIFO
    n_loudest = config.LOUD
    gap = config.gap
    Tgap = config.Tgap
    Fgap = config.Fgap
    e2or = network.net.e2or
    subnet = config.subnet
    subcut = config.subcut
    subnorm = config.subnorm
    subrho = config.subrho if config.subrho > 0 else network.net.netRHO
    ml, FP, FX = load_data_from_ifo(network, config.nIFO)

    for fragment_cluster, lag in zip(fragment_clusters, range(int(network.nLag))):
        print(f"Processing fragment cluster for lag {lag} with {len(fragment_cluster.clusters)} clusters")
        clusters = fragment_cluster.clusters

        superclusters = supercluster(clusters, 'L', gap, e2or, n_ifo)

        # get the total number of pixels
        total_pixels = 0
        for i, c in enumerate(superclusters):
            total_pixels += len(c.pixels)
            # print(f'supercluster {i} has {n_pix} pixels and {"rejected" if c.cluster_status > 0 else "accepted"}')

        # filter out the rejected superclusters
        accepted_superclusters = [sc for sc in superclusters if sc.cluster_status <= 0]
        print(f"Total number of superclusters: {len(superclusters)}, total pixels: {total_pixels}, "
              f"accepted clusters: {len(accepted_superclusters)}")

        # if there are no accepted superclusters, return None
        if len(accepted_superclusters) == 0:
            return None

        # defragment the superclusters
        new_superclusters = defragment(accepted_superclusters, Tgap, Fgap, n_ifo)
        # print(f"Total number of superclusters after defragment: {len(new_superclusters)}")

        # get the total number of pixels
        total_pixels = 0
        for i, c in enumerate(superclusters):
            total_pixels += len(c.pixels)
            # print(f'supercluster {i} has {n_pix} pixels and {"rejected" if c.cluster_status != 0 else "accepted"}')
        print(f"Total number of superclusters after defragment: {len(new_superclusters)}, total pixels: {total_pixels}")

        for i, c in enumerate(new_superclusters):
            # sort pixels by likelihood for down selection
            c.pixels.sort(key=lambda x: x.likelihood, reverse=True)
            # down select config.loud pixels and apply sub_net_cut
            results = sub_net_cut(c.pixels[:n_loudest], ml, FP, FX, acor, e2or, n_ifo, n_sky, subnet, subcut, subnorm, subrho,
                        xtalk_coeff, xtalk_lookup_table, layers)

            # update cluster status and print results
            if results['subnet_passed'] and results['subrho_passed'] and results['subthr_passed']:
                print(f"Cluster {i} ({len(c.pixels)} pixels) passed subnet, subrho, and subthr cut")
                c.cluster_status = -1
            else:
                log_output = f"Cluster {i} ({len(c.pixels)} pixels) failed "
                if not results['subnet_passed']:
                    log_output += f"subnet cut condition: {results['subnet_condition']}, "
                if not results['subrho_passed']:
                    log_output += f"subrho cut condition: {results['subrho_condition']}, "
                if not results['subthr_passed']:
                    log_output += f"subthr cut condition: {results['subthr_condition']}, "
                print(log_output)
                c.cluster_status = 1

        fragment_cluster.clusters = [c for c in new_superclusters if c.cluster_status <= 0]

        total_pixels = 0
        for i, c in enumerate(fragment_cluster.clusters):
            total_pixels += len(c.pixels)
        print(f"Total number of superclusters for lag {lag} after sub_net_cut: "
              f"{len(fragment_cluster.clusters)}, total pixels: {total_pixels}")

        for c in fragment_cluster.clusters:
            for p in c.pixels:
                p.core = 1
                p.td_amp = None

    ###############################
    # ROOT code start

    network.restore_skymap(config, skyres)

    ###############################
    return fragment_clusters


