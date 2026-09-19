"""Sparse row-run connectivity with every original pixel retained."""
import numpy as np
from numba import njit

from .kernels import _uf_find, _label_components_grid


@njit(cache=True)
def _join(parent, a, b):
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
    """
    if kf < 0 or kt < 0:
        return _label_components_grid(f_arr, t_arr, n_freq, n_time, kf, kt)
    n = len(f_arr)
    parent = np.arange(n, dtype=np.int64)
    valid = np.empty(n, dtype=np.int64)
    keys = np.empty(n, dtype=np.int64)
    nv = 0
    for i in range(n):
        if 0 <= f_arr[i] < n_freq and 0 <= t_arr[i] < n_time:
            valid[nv] = i
            keys[nv] = f_arr[i] * n_time + t_arr[i]
            nv += 1
    order = np.argsort(keys[:nv])
    rf = np.empty(nv, dtype=np.int64)
    lo = np.empty(nv, dtype=np.int64)
    hi = np.empty(nv, dtype=np.int64)
    representative = np.empty(nv, dtype=np.int64)
    row_begin = np.empty(nv + 1, dtype=np.int64)
    nr = 0
    nrow = 0
    for k in range(nv):
        i = valid[order[k]]
        f, t = f_arr[i], t_arr[i]
        if nr and f == rf[nr - 1] and t - hi[nr - 1] <= kt:
            _join(parent, i, representative[nr - 1])
            hi[nr - 1] = t
        else:
            if nr == 0 or f != rf[nr - 1]:
                row_begin[nrow] = nr
                nrow += 1
            rf[nr], lo[nr], hi[nr], representative[nr] = f, t, t, i
            nr += 1
    row_begin[nrow] = nr
    for row in range(nrow):
        previous = row - 1
        while previous >= 0:
            if rf[row_begin[row]] - rf[row_begin[previous]] > kf:
                break
            a, b = row_begin[row], row_begin[previous]
            while a < row_begin[row + 1] and b < row_begin[previous + 1]:
                if hi[a] + kt < lo[b]:
                    a += 1
                elif hi[b] + kt < lo[a]:
                    b += 1
                else:
                    _join(parent, representative[a], representative[b])
                    if hi[a] < hi[b]:
                        a += 1
                    elif hi[b] < hi[a]:
                        b += 1
                    else:
                        a += 1
                        b += 1
            previous -= 1
    root_labels = np.zeros(n, dtype=np.int64)
    labels = np.empty(n, dtype=np.int64)
    next_label = 1
    for i in range(n):
        root = _uf_find(parent, i)
        if root_labels[root] == 0:
            root_labels[root] = next_label
            next_label += 1
        labels[i] = root_labels[root]
    return labels
