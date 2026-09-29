"""Identity/bitwise reuse and bounded eviction of ``geometry_cache.geometry_key``."""

from __future__ import annotations

import numpy as np
import pytest

from pycwb.utils.gpu import geometry_cache
from pycwb.utils.gpu.geometry_cache import MAX_ENTRIES, geometry_key


def test_identity_key_is_returned_for_cached_arrays() -> None:
    a, b = np.ones((4, 2), np.float32), np.zeros((2, 4), np.int32)
    cache: dict = {}
    key = geometry_key(cache, a, b)
    assert key == (id(a), id(b))
    cache[key] = (a, b, "device")
    assert geometry_key(cache, a, b) == key


def test_many_fresh_equal_arrays_reuse_one_entry() -> None:
    cache: dict = {}
    for _ in range(300):
        arrays = (np.ones((16, 2), np.float32), np.zeros((2, 16), np.int32))
        key = geometry_key(cache, *arrays)
        if key not in cache:
            cache[key] = (*arrays, object())
    assert len(cache) == 1


def test_fresh_views_of_the_same_geometry_hit_the_cache() -> None:
    base = np.arange(32, dtype=np.float32).reshape(16, 2)
    cache: dict = {}
    key = geometry_key(cache, base)
    cache[key] = (base, "device")
    assert geometry_key(cache, base.copy()) == key
    assert geometry_key(cache, base[:, :]) == key
    assert geometry_key(cache, np.asfortranarray(base)) == key


def test_distinct_geometry_is_bounded_and_not_reused() -> None:
    cache: dict = {}
    for i in range(50):
        a = np.full((16, 2), i, np.float32)
        key = geometry_key(cache, a)
        assert key not in cache
        cache[key] = (a, object())
        assert len(cache) <= MAX_ENTRIES
    assert MAX_ENTRIES == 2


def test_eviction_removes_the_oldest_entry() -> None:
    cache: dict = {}
    arrays = [np.full(3, i, np.float64) for i in range(3)]
    keys = []
    for a in arrays[:2]:
        key = geometry_key(cache, a)
        cache[key] = (a, object())
        keys.append(key)
    third = geometry_key(cache, arrays[2])
    assert keys[0] not in cache
    assert keys[1] in cache
    assert third not in cache
    cache[third] = (arrays[2], object())
    assert set(cache) == {keys[1], third}


def test_shape_dtype_and_bits_all_matter() -> None:
    base = np.arange(6, dtype=np.float32)
    cache: dict = {}
    key = geometry_key(cache, base)
    cache[key] = (base, object())
    assert geometry_key(cache, base.reshape(2, 3)) != key
    assert geometry_key(cache, base.astype(np.float64)) != key
    changed = base.copy()
    changed[-1] = np.nextafter(changed[-1], np.inf)
    assert geometry_key(cache, changed) != key


def test_signed_zero_and_nan_payloads_are_distinct() -> None:
    a, b = np.array([0.0], np.float32), np.array([-0.0], np.float32)
    cache = {(id(a),): (a, object())}
    assert geometry_key(cache, b) != (id(a),)
    nan_a = np.array([np.nan], np.float64)
    nan_b = nan_a.copy()
    cache = {(id(nan_a),): (nan_a, object())}
    # Identical NaN payloads are bitwise equal, so the entry is reused.
    assert geometry_key(cache, nan_b) == (id(nan_a),)
    nan_c = nan_a.view(np.uint64) ^ np.uint64(1)
    assert geometry_key(cache, nan_c.view(np.float64)) != (id(nan_a),)


def test_multi_array_geometry_requires_all_arrays_equal() -> None:
    fp, fx, ml = np.ones((4, 2), np.float32), np.zeros((4, 2), np.float32), np.zeros((2, 4), np.int32)
    cache: dict = {}
    key = geometry_key(cache, fp, fx, ml)
    cache[key] = (fp, fx, ml, "d1", "d2", "d3")
    assert geometry_key(cache, fp.copy(), fx.copy(), ml.copy()) == key
    other = ml.copy()
    other[0, 0] = 1
    assert geometry_key(cache, fp.copy(), fx.copy(), other) != key


def test_resident_geometry_uses_geometry_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Device upload is stubbed so the caching contract is checked without a GPU."""
    uploads: list[np.ndarray] = []

    def fake_to_device(array: np.ndarray) -> str:
        uploads.append(array)
        return f"device{len(uploads)}"

    monkeypatch.setattr(geometry_cache.cuda, "to_device", fake_to_device)
    fp = np.ones((4, 2), np.float64)
    ml = np.zeros((2, 4), np.int64)
    cache: dict = {}
    first = geometry_cache.resident_geometry(cache, (fp, ml), (np.float32, np.int32))
    assert first == ("device1", "device2")
    assert uploads[0].dtype == np.float32 and uploads[1].dtype == np.int32
    second = geometry_cache.resident_geometry(cache, (fp.copy(), ml.copy()), (np.float32, np.int32))
    assert second == first
    assert len(uploads) == 2
    assert len(cache) == 1
    other = geometry_cache.resident_geometry(cache, (fp * 2, ml), (np.float32, np.int32))
    assert other == ("device3", "device4")
    assert len(cache) == 2
