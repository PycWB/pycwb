"""Index sorting with the release wavearray::waveSort tie convention.

Posterior ties can represent widely separated sky directions. Matching only
sorted values is insufficient: the release's pointer permutation determines
which tied direction is exported as the reconstructed location.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _sort_three(values, order, left, middle, right):
    if values[order[left]] > values[order[middle]]:
        order[left], order[middle] = order[middle], order[left]
    if values[order[left]] > values[order[right]]:
        order[left], order[right] = order[right], order[left]
    if values[order[middle]] > values[order[right]]:
        order[middle], order[right] = order[right], order[middle]


@njit(cache=True)
def _wave_sort(values, order, left, right):
    if left >= right:
        return
    middle = (left + right) // 2
    j = right - 1
    _sort_three(values, order, left, middle, right)
    if right - left < 3:
        return
    pivot = values[order[middle]]
    order[middle], order[j] = order[j], order[middle]
    i = left
    while True:
        i += 1
        while values[order[i]] < pivot:
            i += 1
        j -= 1
        while values[order[j]] > pivot:
            j -= 1
        if j < i:
            break
        order[i], order[j] = order[j], order[i]
    order[i], order[right - 1] = order[right - 1], order[i]
    i += 1
    if j - left > 2:
        _wave_sort(values, order, left, j)
    elif j > left:
        _sort_three(values, order, left, left + 1, j)
    if right - i > 2:
        _wave_sort(values, order, i, right)
    elif right > i:
        _sort_three(values, order, i, i + 1, right)


def wave_sort_indices(values, initial_order=None):
    """Return ascending indices, preserving cWB's nonstable tie behavior.

    The posterior's second sort starts from the first statistic sort's
    permutation, rather than restarting from sky-index order.
    """
    values = np.asarray(values)
    order = (np.arange(len(values), dtype=np.int64) if initial_order is None
             else np.array(initial_order, dtype=np.int64, copy=True))
    _wave_sort(values, order, 0, len(order) - 1)
    return order
