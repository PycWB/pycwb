"""Sparse row-run connectivity with every original pixel retained."""

import numpy as np
from numba import njit

from .kernels import _uf_find, _label_components_grid


@njit(cache=True)
def _join(parent, a, b):
    """Join two components using the smaller root to preserve input-order labels."""
    a = _uf_find(parent, a)
    b = _uf_find(parent, b)
    if a < b:
        parent[b] = a
    elif b < a:
        parent[a] = b


@njit(cache=True)
def label_components_runs(f_arr, t_arr, n_freq, n_time, kf, kt):
    """Match grid labels using connected runs and a sweep of their boundaries.

    A row run contains sorted pixels separated by at most kt bins. Its pixels
    are already connected. Two runs on nearby rows connect iff their closed
    time intervals are within kt bins: any internal gap is at most kt, so an
    interval overlap cannot conceal a disconnected pair. The sweep advances
    the run with the earlier end; later runs in that row start > kt past it.
    Roots and final labels follow original input order, including duplicates.
    Invalid coordinates remain isolated, as in the grid implementation.

    Parameters
    ----------
    f_arr, t_arr : numpy.ndarray
        Integer frequency and time bins in original pixel order, shape (n_pixels,).
    n_freq, n_time : int
        Grid extents used to validate each coordinate.
    kf, kt : int
        Inclusive frequency/time connection gaps in bins. Negative gaps use the
        existing grid implementation.

    Returns
    -------
    numpy.ndarray
        One-based int64 component labels in original order, including duplicates.
    """
    if kf < 0 or kt < 0:
        return _label_components_grid(f_arr, t_arr, n_freq, n_time, kf, kt)
    n_pixels = len(f_arr)
    parent = np.arange(n_pixels, dtype=np.int64)
    valid = np.empty(n_pixels, dtype=np.int64)
    keys = np.empty(n_pixels, dtype=np.int64)
    n_valid = 0
    for i in range(n_pixels):
        if 0 <= f_arr[i] < n_freq and 0 <= t_arr[i] < n_time:
            valid[n_valid] = i
            keys[n_valid] = f_arr[i] * n_time + t_arr[i]
            n_valid += 1
    order = np.argsort(keys[:n_valid])
    run_frequencies = np.empty(n_valid, dtype=np.int64)
    run_starts = np.empty(n_valid, dtype=np.int64)
    run_ends = np.empty(n_valid, dtype=np.int64)
    representative = np.empty(n_valid, dtype=np.int64)
    row_begin = np.empty(n_valid + 1, dtype=np.int64)
    n_runs = 0
    n_rows = 0
    for k in range(n_valid):
        i = valid[order[k]]
        f, t = f_arr[i], t_arr[i]
        if n_runs and f == run_frequencies[n_runs - 1] and t - run_ends[n_runs - 1] <= kt:
            _join(parent, i, representative[n_runs - 1])
            run_ends[n_runs - 1] = t
        else:
            if n_runs == 0 or f != run_frequencies[n_runs - 1]:
                row_begin[n_rows] = n_runs
                n_rows += 1
            run_frequencies[n_runs], run_starts[n_runs], run_ends[n_runs], representative[n_runs] = f, t, t, i
            n_runs += 1
    row_begin[n_rows] = n_runs
    for row in range(n_rows):
        previous = row - 1
        while previous >= 0:
            if run_frequencies[row_begin[row]] - run_frequencies[row_begin[previous]] > kf:
                break
            a, b = row_begin[row], row_begin[previous]
            while a < row_begin[row + 1] and b < row_begin[previous + 1]:
                if run_ends[a] + kt < run_starts[b]:
                    a += 1
                elif run_ends[b] + kt < run_starts[a]:
                    b += 1
                else:
                    _join(parent, representative[a], representative[b])
                    if run_ends[a] < run_ends[b]:
                        a += 1
                    elif run_ends[b] < run_ends[a]:
                        b += 1
                    else:
                        a += 1
                        b += 1
            previous -= 1
    root_labels = np.zeros(n_pixels, dtype=np.int64)
    labels = np.empty(n_pixels, dtype=np.int64)
    next_label = 1
    for i in range(n_pixels):
        root = _uf_find(parent, i)
        if root_labels[root] == 0:
            root_labels[root] = next_label
            next_label += 1
        labels[i] = root_labels[root]
    return labels
