"""Operations on sorted, non-overlapping time intervals."""

def intersect_intervals(
    a: list[tuple[float, float]],
    b: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return the intersection of two sorted lists of ``(start, end)`` intervals.

    Both inputs must be sorted by start time and non-overlapping.

    Parameters
    ----------
    a, b : list[tuple[float, float]]
        Sorted, non-overlapping intervals.

    Returns
    -------
    list[tuple[float, float]]
        Sorted intervals representing the intersection.
    """
    result: list[tuple[float, float]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            result.append((lo, hi))
        # Advance the interval that ends first
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result
