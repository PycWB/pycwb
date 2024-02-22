import numpy as np
from numba.typed import List
from numba import njit, types


@njit(cache=True)
def find(parent, x):
    """Path compression find."""
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]


@njit(cache=True)
def union(parent, rank, x, y):
    """Union by rank."""
    rootX = find(parent, x)
    rootY = find(parent, y)
    if rootX != rootY:
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1


@njit(cache=True)
def aggregate_clusters(links):
    """Aggregate clusters based on provided links."""
    n = np.max(links) + 1  # Assuming links are 0-indexed.
    parent = np.arange(n)
    rank = np.zeros(n, dtype=np.int32)

    # Perform unions
    for x, y in links:
        union(parent, rank, x, y)

    # Find all unique representatives
    for i in range(n):
        parent[i] = find(parent, i)

    # Create an empty list for each unique parent
    clusters = List()
    for i in range(n):
        clusters.append(List.empty_list(types.int32))

    # Aggregate clusters
    for i in range(n):
        root = find(parent, i)
        clusters[root].append(i)

    # Filter out single-element clusters
    filtered_clusters = List()
    for cluster in clusters:
        if len(cluster) > 1:
            filtered_clusters.append(cluster)

    return filtered_clusters


@njit
def remove_duplicates_sorted(arr):
    # Assuming arr is a sorted 2D numpy array of shape (n, 2)
    n = len(arr)
    if n == 0:
        return arr  # Early return for empty array

    # Preallocate an array of the same size to store unique elements
    unique = np.empty_like(arr)
    unique[0] = arr[0]
    count = 1  # Initialize count with 1 as the first element is always unique

    for i in range(1, n):
        # Since arr is sorted, just check if the current element is different from the last unique element added
        if not np.array_equal(arr[i], unique[count - 1]):
            unique[count] = arr[i]
            count += 1

    # Trim the unique array to the correct number of unique elements found
    return unique[:count]


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
