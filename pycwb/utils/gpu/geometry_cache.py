"""Bind immutable per-trial geometry to device copies without re-uploading."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numba import cuda

MAX_ENTRIES = 2
"""Device geometry entries retained per cache; older entries are evicted."""


def geometry_key(cache: dict[tuple[int, ...], Any], *arrays: np.ndarray) -> tuple[int, ...]:
    """Return the cache key of geometry equal to ``arrays``, evicting if needed.

    Parameters
    ----------
    cache : dict
        Mapping from key to an entry whose leading items are the host arrays
        that were uploaded, in the same order as ``arrays``.
    *arrays : numpy.ndarray
        Host geometry, for example antenna patterns and delay indices.

    Returns
    -------
    tuple[int, ...]
        Either an existing key whose stored arrays are bitwise identical to
        ``arrays`` or a fresh identity-based key. When a fresh key is returned
        the cache has room for it.

    Notes
    -----
    Native code hands out fresh array views of unchanged geometry on every
    lag, so identity alone would upload one copy per lag. Bitwise comparison
    is used instead; geometry is immutable for one trial.
    """
    key = tuple(id(a) for a in arrays)
    if key in cache:
        return key
    for previous, entry in cache.items():
        if all(
            a.shape == b.shape
            and a.dtype == b.dtype
            and np.array_equal(np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))
            for a, b in zip(arrays, entry[: len(arrays)], strict=True)
        ):
            return previous
    if len(cache) >= MAX_ENTRIES:
        del cache[next(iter(cache))]
    return key


def resident_geometry(
    cache: dict[tuple[int, ...], Any], arrays: Sequence[np.ndarray], dtypes: Sequence[Any]
) -> tuple[Any, ...]:
    """Return device copies of ``arrays`` converted to ``dtypes``, cached.

    Parameters
    ----------
    cache : dict
        Cache shared by one stage instance; see :func:`geometry_key`.
    arrays : sequence of numpy.ndarray
        Host geometry arrays.
    dtypes : sequence of dtype-like
        Device dtype for each array, for example ``np.float32`` for antenna
        patterns and ``np.int32`` for delay indices.

    Returns
    -------
    tuple
        Device arrays in the order of ``arrays``. The cache entry also keeps
        references to the host arrays so their identity keys stay valid.
    """
    key = geometry_key(cache, *arrays)
    if key not in cache:
        device = tuple(
            cuda.to_device(np.ascontiguousarray(a, dtype=dtype)) for a, dtype in zip(arrays, dtypes, strict=True)
        )
        cache[key] = (*arrays, *device)
    entry = cache[key]
    return tuple(entry[len(arrays) :])
