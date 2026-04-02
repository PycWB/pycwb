from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, types
from numba.typed import List

from pycwb.types.network_cluster import Cluster
from .sub_net_cut import sub_net_cut

if TYPE_CHECKING:
    from pycwb.modules.xtalk.type import XTalk
    from pycwb.types.time_series import TimeSeries


logger = logging.getLogger(__name__)


@njit
def find(parent: np.ndarray, x: int) -> int:
    """
    Find the representative of the set containing *x* with iterative path compression.

    Parameters
    ----------
    parent : np.ndarray
        Parent array for the union-find structure.
    x : int
        Element whose root is to be found.

    Returns
    -------
    int
        Root representative of the set.
    """
    root = x
    # Walk up to the root.
    while parent[root] != root:
        root = parent[root]

    # Path compression: point every node on the path directly to root.
    while x != root:
        next_node = parent[x]
        parent[x] = root
        x = next_node

    return root


@njit
def union(parent: np.ndarray, rank: np.ndarray, x: int, y: int) -> None:
    """
    Merge the sets containing *x* and *y* using union-by-rank.

    Parameters
    ----------
    parent : np.ndarray
        Parent array for the union-find structure.
    rank : np.ndarray
        Rank array used to keep the tree shallow.
    x : int
        First element.
    y : int
        Second element.
    """
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
def aggregate_clusters(links: np.ndarray) -> List:
    """
    Build connected components from a list of (i, j) edge pairs.

    Parameters
    ----------
    links : np.ndarray of shape (N, 2), dtype int32
        Pairs of cluster indices that should be merged.

    Returns
    -------
    numba.typed.List[numba.typed.List[int32]]
        Each inner list contains the indices belonging to one component.
        Single-element components (isolated nodes) are excluded.
    """
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
def remove_duplicates_sorted(arr: np.ndarray) -> np.ndarray:
    """
    Remove consecutive duplicate rows from a sorted 2-D array.

    Parameters
    ----------
    arr : np.ndarray of shape (N, 2)
        Sorted array whose duplicate rows are to be removed.

    Returns
    -------
    np.ndarray
        View of *arr* with consecutive duplicates stripped.
    """
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


def get_cluster_links(
    pixels: np.ndarray, gap: float, n_ifo: int
) -> tuple[np.ndarray, float]:
    """
    Find (i, j) links between clusters whose pixels are close enough to merge.

    Pixels are represented as rows in *pixels* where columns are:
    ``[time_s, freq_hz, 1/rate, rate/2, cluster_id, td_index_ifo0, …]``.

    Parameters
    ----------
    pixels : np.ndarray of shape (N, 5+n_ifo)
        Per-pixel feature array sorted by time.
    gap : float
        Dimensionless overlap parameter; two pixels are linked when the
        combined time-frequency distance satisfies ``eps <= gap``.
    n_ifo : int
        Number of interferometers (determines how many td-index columns exist).

    Returns
    -------
    np.ndarray of shape (M, 2), dtype int32
        Unique pairs of cluster indices that should be merged.
    float
        Last computed frequency gap value ``dF`` (used as a cluster-level
        proxy; see FIXME in :func:`supercluster`).
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

                if link not in cluster_links:  # O(n) dedup; acceptable since cluster count is small
                    cluster_links.append(link)

    return np.array(cluster_links, dtype=np.int32), dF


@njit(cache=True)
def get_defragment_link(
    pixels: np.ndarray, t_gap: float, f_gap: float, n_ifo: int
) -> np.ndarray:
    """
    Find (i, j) links between clusters for defragmentation.

    Two clusters are linked when the time *and* frequency separations between
    their representative pixels are both within the supplied thresholds.

    Parameters
    ----------
    pixels : np.ndarray of shape (N, 5+n_ifo)
        Per-pixel feature array: ``[time_s, freq_hz, 1/rate, rate/2, cluster_id, td_index_ifo0, …]``.
    t_gap : float
        Maximum allowed time separation in seconds.
    f_gap : float
        Maximum allowed frequency separation in Hz.
    n_ifo : int
        Number of interferometers.

    Returns
    -------
    np.ndarray of shape (M, 2), dtype int32
        Unique cluster-index pairs satisfying the defragmentation condition,
        or an empty ``(0, 2)`` array when no links exist.
    """
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
def calculate_statistics(
    pixels: np.ndarray,
    atype: str,
    core: bool,
    pair: bool,
    nPIX: int,
    S: float,
    dF: float,
) -> list[float] | None:
    """
    Compute summary statistics for a merged supercluster candidate.

    Parameters
    ----------
    pixels : np.ndarray of shape (N, 6+n_ifo)
        Per-pixel feature matrix.  Columns: ``[core, time, freq, rate, layers, likelihood, asnr_ifo0, …]``.
    atype : str
        Statistics type: ``'L'`` (likelihood), ``'E'`` (energy), or ``'P'`` (power).
    core : bool
        If ``True``, skip non-core pixels.
    pair : bool
        If ``True``, require at least two resolution levels.
    nPIX : int
        Minimum pixel count; clusters with fewer pixels are rejected.
    S : float
        Minimum amplitude/likelihood threshold per resolution level.
    dF : float
        Frequency correction term added to each pixel's frequency bin.

    Returns
    -------
    list[float] or None
        ``[cTime, cFreq, rate_max, rate_min, total_energy]`` when the cluster
        passes all cuts, or ``None`` if it should be rejected.
    """
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

    # Select the optimal resolution: find the level with the highest and lowest statistic.
    a = -1.0e99
    idx_max = -1
    for j in range(len(rate)):
        powr[j] = ampl[j] / sIZe[j]
        if atype == 'E' and ampl[j] > a and not cuts[j]:
            idx_max = j
            a = ampl[j]
        if atype == 'L' and like[j] > a and not cuts[j]:
            idx_max = j
            a = like[j]
        if atype == 'P' and powr[j] > a and not cuts[j]:
            idx_max = j
            a = powr[j]

    if a < S:
        return None

    idx_min = -1
    a = -1.0e99
    for j in range(len(rate)):
        if idx_max == j:
            continue
        if atype == 'E' and ampl[j] < a and not cuts[j]:
            idx_min = j
            a = ampl[j]
        if atype == 'L' and like[j] < a and not cuts[j]:
            idx_min = j
            a = like[j]
        if atype == 'P' and powr[j] < a and not cuts[j]:
            idx_min = j
            a = powr[j]

    cTime = cT / nT
    cFreq = cF / nF

    return [cTime, cFreq, rate[idx_max], rate[idx_min], E]


def aggregate_clusters_from_links(
    cluster_ids: np.ndarray, cluster_links: np.ndarray
) -> list[list[int]]:
    """
    Merge cluster ids using the provided edge list and append singleton clusters.

    Parameters
    ----------
    cluster_ids : np.ndarray of int
        Complete set of cluster indices (including isolated nodes).
    cluster_links : np.ndarray of shape (M, 2), dtype int32
        Pairs of cluster indices to merge.

    Returns
    -------
    list[list[int]]
        Each inner list holds the cluster indices belonging to one component.
        Isolated clusters appear as single-element lists.
    """
    # Build connected components.
    aggregated_clusters = aggregate_clusters(cluster_links)
    aggregated_clusters = [list(cluster) for cluster in aggregated_clusters]

    flattened_aggregated_clusters = [c for cluster in aggregated_clusters for c in cluster]
    # find the standalone clusters
    standalone_clusters = np.array([c for c in cluster_ids if c not in flattened_aggregated_clusters])

    # add standalone clusters
    aggregated_clusters = [list(c) for c in aggregated_clusters] + [[c] for c in standalone_clusters]

    return aggregated_clusters


def extract_timeseries_data(strain: TimeSeries) -> tuple[np.ndarray, float, float]:
    """
    Extract raw data, sample rate, and start time from a strain object.

    Parameters
    ----------
    strain : TimeSeries or compatible
        Object exposing ``.data``, ``.sample_rate``, and ``.t0`` attributes.

    Returns
    -------
    values : np.ndarray
        Strain samples as float64.
    sample_rate : float
        Sampling frequency in Hz.
    start : float
        GPS start time of the time series.
    """
    values = np.asarray(strain.data, dtype=np.float64)
    sample_rate = float(strain.sample_rate)
    start = float(strain.t0)
    return values, sample_rate, start


def expected_td_vec_len(td_size: int | float) -> int:
    """
    Return the expected length of a time-delay amplitude vector.

    Parameters
    ----------
    td_size : int or float
        Number of time-delay steps (``K_td``).

    Returns
    -------
    int
        Vector length ``4 * td_size + 2``.
    """
    return 4 * int(td_size) + 2


def resolve_wdm_context(layer_tag: int | float, context_map: dict) -> object:
    """
    Look up the WDM context for *layer_tag*, falling back to ``layer_tag - 1``.

    Parameters
    ----------
    layer_tag : int or float
        WDM layer identifier (will be cast to ``int``).
    context_map : dict
        Mapping from integer layer tag to context object.

    Returns
    -------
    object
        The context object from *context_map*.

    Raises
    ------
    ValueError
        If neither ``layer_tag`` nor ``layer_tag - 1`` is present in *context_map*.
    """
    layer_tag = int(layer_tag)
    context = context_map.get(layer_tag)
    if context is not None:
        return context

    context = context_map.get(layer_tag - 1)
    if context is not None:
        return context

    raise ValueError(f"Missing WDM context for pixel layer {layer_tag}")


def apply_subnet_cut(
    superclusters: list[Cluster],
    n_loudest_local: int,
    ml_local: np.ndarray,
    FP_local: np.ndarray,
    FX_local: np.ndarray,
    acor_local: float,
    e2or_local: float,
    n_ifo_local: int,
    n_sky_local: int,
    subnet_local: float,
    subcut_local: float,
    subnorm_local: float,
    subrho_local: float,
    xtalk_local: XTalk,
) -> list[Cluster]:
    """
    Apply the sub-network cut and mark clusters that pass or fail.

    Parameters
    ----------
    superclusters : list[Cluster]
        Candidate clusters to evaluate.
    n_loudest_local : int
        Number of loudest pixels to pass to :func:`sub_net_cut`.
    ml_local : np.ndarray
        Sky-delay matrix (reduced resolution for subnet check).
    FP_local : np.ndarray
        Plus-polarisation antenna-pattern matrix.
    FX_local : np.ndarray
        Cross-polarisation antenna-pattern matrix.
    acor_local : float
        Correlation threshold.
    e2or_local : float
        Energy-to-overfit-ratio threshold.
    n_ifo_local : int
        Number of interferometers.
    n_sky_local : int
        Number of sky pixels.
    subnet_local : float
        Sub-network statistic threshold.
    subcut_local : float
        Sub-network cut threshold.
    subnorm_local : float
        Sub-network normalisation factor.
    subrho_local : float
        Sub-network SNR threshold.
    xtalk_local : XTalk
        Cross-talk catalogue instance.

    Returns
    -------
    list[Cluster]
        Clusters with ``cluster_status <= 0``, i.e. those that passed the cut.
    """
    for i, c in enumerate(superclusters):
        c.pixels.sort(key=lambda x: x.likelihood, reverse=True)
        results = sub_net_cut(
            c.pixels[:n_loudest_local], ml_local, FP_local, FX_local,
            acor_local, e2or_local, n_ifo_local, n_sky_local,
            subnet_local, subcut_local, subnorm_local, subrho_local, xtalk_local,
        )

        if results['subnet_passed'] and results['subrho_passed'] and results['subthr_passed']:
            logger.debug(
                f"Cluster {i} ({len(c.pixels)} pixels, from {c.start_time:.2f} - {c.stop_time:.2f} with freq {c.low_frequency:.2f} - {c.high_frequency:.2f} ) passed subnet, subrho, and subthr cut"
            )
            c.cluster_status = -1
        else:
            log_output = f"Cluster {i} ({len(c.pixels)} pixels, from {c.start_time:.2f} - {c.stop_time:.2f} with freq {c.low_frequency:.2f} - {c.high_frequency:.2f} ) failed "
            if not results['subnet_passed']:
                log_output += f"subnet cut condition: {results['subnet_condition']}, "
            if not results['subrho_passed']:
                log_output += f"subrho cut condition: {results['subrho_condition']}, "
            if not results['subthr_passed']:
                log_output += f"subthr cut condition: {results['subthr_condition']}, "
            logger.debug(log_output)
            c.cluster_status = 1

    return [c for c in superclusters if c.cluster_status <= 0]
