from numba import njit
import numpy as np
from .utils import aggregate_clusters


@njit
def get_cluster_links(pixels, gap, n_ifo):
    """

    :param pixels: list of pixels in format [0: time(s), 1: frequency(s), 2: 1/rate, 3: rate/2, 4: cluster_id, **data]
    :param gap:
    :param n_ifo:
    :return:
    """
    pixels = pixels[pixels[:, 0].argsort()]

    Tgap_base = np.max(pixels[:, 2])  # Base Tgap, inverse of the rate.
    # Update Tgap based on your gap factor.
    Tgap = Tgap_base * (1. + gap)

    cluster_links = []
    n_pixels = len(pixels)
    for i in range(n_pixels):
        p = pixels[i]
        for j in range(i + 1, n_pixels):
            q = pixels[j]

            # Check if time difference is too large, indicating no further potential neighbors.
            if q[0] - p[0] > Tgap:
                break

            # Check if they're from the same cluster or if the rate ratio is beyond threshold.
            if p[4] == q[4] or max(p[2] / q[2], q[2] / p[2]) > 3:
                continue

            # Calculate dT
            # R = p->rate + q->rate;
            R = 1 / p[2] + 1 / q[2]
            T = p[2] + q[2]
            dT = 0
            for k in range(n_ifo):
                aa = p[5 + k] - q[5 + k]
                if abs(aa) > dT:
                    dT = abs(aa)
            dT -= 0.5 * T

            # Calculate dF using half the rate difference.
            dF = abs(p[1] - q[1]) - 0.5 * R
            eps = (dT * R if dT > 0 else 0) + (dF * T if dF > 0 else 0)

            if gap >= eps:
                # create cluster link, make sure the order is correct
                # cluster_links.append([int(p[4]), int(q[4])] if p[4] < q[4] else [int(q[4]), int(p[4])])
                # Create cluster link, ensure unique pairs
                if p[4] < q[4]:
                    link = (int(p[4]), int(q[4]))
                else:
                    link = (int(q[4]), int(p[4]))

                if link not in cluster_links:  # This check is not efficient in Numba
                    cluster_links.append(link)

        # remove redundant links with numpy
    return np.array(cluster_links)


def supercluster(cluster, gap, n_ifo):
    pixels = []
    for c_id, c in enumerate(cluster.clusters):
        for p in c.pixels:
            pixels.append([p.time / p.rate / p.layers, p.frequency / p.rate, 1 / p.rate, p.rate / 2, c_id] + [
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
