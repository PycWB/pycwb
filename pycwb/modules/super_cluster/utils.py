import numpy as np
from numba.typed import List
from numba import jit, njit, types


# @njit
# def find(parent, x):
#     """Path compression find."""
#     if parent[x] != x:
#         parent[x] = find(parent, parent[x])
#     return parent[x]

@njit
def find(parent, x):
    root = x
    # Find the root of the tree
    while parent[root] != root:
        root = parent[root]

    # Path compression: update the parent pointers of all ancestors to point directly to the root
    while x != root:
        next = parent[x]
        parent[x] = root
        x = next

    return root

@njit
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
        clusters[root].append(np.int32(i))

    # Filter out single-element clusters
    filtered_clusters = List()
    for cluster in clusters:
        if len(cluster) > 1:
            filtered_clusters.append(cluster)

    return filtered_clusters


@njit(cache=True)
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


# @njit(cache=True)
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

    # TODO: check if it is correct to expose dF as a return value
    dF = 0

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

    return np.array(cluster_links, dtype=np.int32), dF


@njit(cache=True)
def get_defragment_link(pixels, t_gap, f_gap, n_ifo):
    pixels = pixels[pixels[:, 0].argsort()]

    Tgap = np.max(pixels[:, 2])  # Base Tgap, inverse of the rate.
    # Update Tgap based on your gap factor.
    if Tgap < t_gap:
        Tgap = t_gap

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

            if dT < t_gap and dF < f_gap:
                if p[4] < q[4]:
                    link = (int(p[4]), int(q[4]))
                else:
                    link = (int(q[4]), int(p[4]))

                if link not in cluster_links:
                    cluster_links.append(link)

    if len(cluster_links) == 0:
        return np.empty((0, 2), dtype=np.int32)
    else:
        return np.array(cluster_links, dtype=np.int32)


@njit(cache=True)
def calculate_statistics(pixels, atype, core, pair, nPIX, S, dF):
    oEo = atype == 'E' or atype == 'P'
    cT, nT, cF, nF, E = 0.0, 0.0, 0.0, 0.0, 0.0
    max_size = len(pixels)
    rate = np.zeros(max_size, dtype=np.float64)
    ampl = np.zeros(max_size, dtype=np.float64)
    powr = np.zeros(max_size, dtype=np.float64)
    sIZe = np.zeros(max_size, dtype=np.int32)
    cuts = np.zeros(max_size, dtype=np.bool_)
    like = np.zeros(max_size, dtype=np.float64)
    rate_counter = 0
    pixel_counter = 0
    for i in range(len(pixels)):
        pix = pixels[i]
        if not pix[0] and core:
            continue
        L = pix[5]  # Assuming likelihood is another property, adjust as needed.
        e = 0.0
        for a in pix[6:]:  # Assuming data values start from the 6th column
            e += abs(a) if abs(a) > 1.0 else 0.0

        a = L if atype == 'L' else e
        tt = 1.0 / pix[3]  # wavelet time resolution
        mm = pix[4]  # number of wavelet layers
        cT += int(pix[1] / mm) * a
        nT += a / tt
        cF += (pix[2] + dF) * a
        nF += a * 2.0 * tt

        insert = True
        for j in range(rate_counter):
            if rate[j] == int(pix[3] + 0.1):
                insert = False
                ampl[j] += e
                sIZe[j] += 1
                like[j] += L
                break

        if insert:
            rate[rate_counter] = int(pix[3] + 0.1)
            ampl[rate_counter] = e
            powr[rate_counter] = 0.0  # Adjust as needed
            sIZe[rate_counter] = 1
            cuts[rate_counter] = True  # Adjust as needed
            like[rate_counter] = L
            rate_counter += 1

        pixel_counter += 1
        E += e

    # Trim the arrays to the actual size used
    rate = rate[:rate_counter]
    ampl = ampl[:rate_counter]
    powr = powr[:rate_counter]
    sIZe = sIZe[:rate_counter]
    cuts = cuts[:rate_counter]
    like = like[:rate_counter]

    # cut off single level clusters coincidence between levels
    if len(rate) < pair + 1 or pixel_counter < nPIX:
        return None

    cut = True
    for i in range(len(rate)):
        if (atype == 'L' and like[i] < S) or (oEo and ampl[i] < S):
            continue
        if not pair:
            cuts[i] = cut = False
            continue
        for j in range(len(rate)):
            if (atype == 'L' and like[j] < S) or (oEo and ampl[j] < S):
                continue
            if rate[i] / 2 == rate[j] or rate[j] / 2 == rate[i]:
                cuts[i] = cuts[j] = cut = False

    if cut:
        return None

    # Select the optimal resolution
    a = -1.e99
    max = -1
    for j in range(len(rate)):
        powr[j] = ampl[j] / sIZe[j]
        if atype == 'E' and ampl[j] > a and not cuts[j]:
            max = j
            a = ampl[j]
        if atype == 'L' and like[j] > a and not cuts[j]:
            max = j
            a = like[j]
        if atype == 'P' and powr[j] > a and not cuts[j]:
            max = j
            a = powr[j]

    if a < S:
        return None

    min = -1
    a = -1.e99
    for j in range(len(rate)):
        if max == j:
            continue
        if atype == 'E' and ampl[j] < a and not cuts[j]:
            min = j
            a = ampl[j]
        if atype == 'L' and like[j] < a and not cuts[j]:
            min = j
            a = like[j]
        if atype == 'P' and powr[j] < a and not cuts[j]:
            min = j
            a = powr[j]

    cTime = cT / nT
    cFreq = cF / nF

    return [cTime, cFreq, rate[max], rate[min], E]


def aggregate_clusters_from_links(cluster_ids, cluster_links):
    # aggregate clusters
    aggregated_clusters = aggregate_clusters(cluster_links)
    aggregated_clusters = [list(cluster) for cluster in aggregated_clusters]

    flattened_aggregated_clusters = [c for cluster in aggregated_clusters for c in cluster]
    # find the standalone clusters
    standalone_clusters = np.array([c for c in cluster_ids if c not in flattened_aggregated_clusters])

    # add standalone clusters
    aggregated_clusters = [list(c) for c in aggregated_clusters] + [[c] for c in standalone_clusters]

    return aggregated_clusters
