import numpy as np
from .utils import get_cluster_links, aggregate_clusters


def supercluster(cluster, gap, n_ifo):
    pixels = []
    for c_id, c in enumerate(cluster.clusters):
        for p in c.pixels:
            pixels.append([p.time / p.rate / p.layers, p.frequency * p.rate, 1 / p.rate, p.rate / 2, c_id] + [
                d.index / p.rate / p.layers for d in p.data])
    pixels = np.array(pixels)

    # get the full cluster ids
    cluster_ids = np.arange(len(cluster.clusters))

    # find links between clusters
    cluster_links = get_cluster_links(pixels, gap, n_ifo)

    # remove redundant links
    # cluster_links = remove_duplicates_sorted(cluster_links[np.lexsort((cluster_links[:, 1], cluster_links[:, 0]))])

    # aggregate clusters
    aggregated_clusters = aggregate_clusters(cluster_links)
    aggregated_clusters = [list(cluster) for cluster in aggregated_clusters]

    # find the standalone clusters
    standalone_clusters = np.array([c for c in cluster_ids if c not in np.unique(cluster_links)])

    # add standalone clusters
    aggregated_clusters = [list(c) for c in aggregated_clusters] + [[c] for c in standalone_clusters]

    return aggregated_clusters
