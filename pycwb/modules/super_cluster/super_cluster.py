import numpy as np
from .utils import get_cluster_links, calculate_statistics, get_defragment_link, \
    aggregate_clusters_from_links
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
        # TODO: remove the cut out side of the supercluster, make it additional function
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
            cluster_rate=stat[2:],
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


