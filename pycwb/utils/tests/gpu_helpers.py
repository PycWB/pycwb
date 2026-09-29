"""Bitwise comparison helpers and synthetic input builders shared by the tests."""

from __future__ import annotations

from typing import Any

import numpy as np


def assert_same_bits(actual: Any, expected: Any) -> None:
    """Assert two array-likes are bitwise identical after exact float widening.

    Float arrays are widened to float64 (exact for float32) and compared as
    ``uint64`` so that signed zeros and NaN payloads are distinguished. Integer
    and boolean arrays are compared for equality of shape and value.

    Parameters
    ----------
    actual : array-like
        Value produced by the GPU wrapper under test.
    expected : array-like
        Value produced by the CPU oracle.
    """
    x = np.asarray(actual)
    y = np.asarray(expected)
    assert x.shape == y.shape, (x.shape, y.shape)
    if x.dtype.kind == "f" or y.dtype.kind == "f":
        assert x.dtype.kind == "f" and y.dtype.kind == "f", (x.dtype, y.dtype)
        xb = np.ascontiguousarray(x, dtype=np.float64).view(np.uint64).ravel()
        yb = np.ascontiguousarray(y, dtype=np.float64).view(np.uint64).ravel()
        mismatch = np.flatnonzero(xb != yb)
        assert mismatch.size == 0, (
            f"{mismatch.size} differing values; first at flat index {mismatch[0]}: "
            f"actual={x.ravel()[mismatch[0]]!r} expected={y.ravel()[mismatch[0]]!r}"
        )
    else:
        assert x.dtype.kind == y.dtype.kind, (x.dtype, y.dtype)
        np.testing.assert_array_equal(x, y)


def assert_same_tuple_bits(actual: tuple, expected: tuple) -> None:
    """Assert two result tuples agree element by element with :func:`assert_same_bits`."""
    assert len(actual) == len(expected), (len(actual), len(expected))
    for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
        try:
            assert_same_bits(a, b)
        except AssertionError as error:
            raise AssertionError(f"tuple element {index}: {error}") from error


def sky_geometry(rng: np.random.Generator, n_sky: int, n_ifo: int, n_delay: int) -> tuple:
    """Return synthetic ``(FP, FX, ml)`` float32 antenna patterns and int32 delays."""
    FP = rng.normal(size=(n_sky, n_ifo)).astype(np.float32)
    FX = rng.normal(size=(n_sky, n_ifo)).astype(np.float32)
    half = n_delay // 2
    ml = rng.integers(-half, half + 1, size=(n_ifo, n_sky), dtype=np.int32)
    return FP, FX, ml


def pixel_packet(rng: np.random.Generator, n_pix: int, n_ifo: int, n_delay: int, scale: float = 3.0) -> tuple:
    """Return synthetic ``(rms, td00, td90)`` float32 arrays for one packet."""
    rms = rng.uniform(0.1, 1.0, size=(n_pix, n_ifo)).astype(np.float32)
    td00 = rng.normal(0.0, scale, size=(n_delay, n_ifo, n_pix)).astype(np.float32)
    td90 = rng.normal(0.0, scale, size=td00.shape).astype(np.float32)
    return rms, td00, td90


def selection_cache(
    rng: np.random.Generator,
    n_ifo: int,
    n_freq: int = 9,
    n_time: int = 64,
    edge_bins: int = 2,
    rate: float = 16.0,
) -> dict[str, Any]:
    """Build a synthetic native selection cache with the keys the selectors read.

    Energies straddle the threshold ``3.0`` and its hard cap ``6.0`` exactly in
    one frequency row so that the clipping branches are exercised.
    """
    maps = rng.uniform(-1.0, 8.0, size=(n_ifo, n_freq, n_time))
    maps[:, 3, :6] = 0.0
    maps[0, 3, :6] = [3.0, np.nextafter(3.0, 0.0), np.nextafter(3.0, 4.0), 6.0, np.nextafter(6.0, 0.0), 6.5]
    valid_start = edge_bins
    valid_stop = n_time - edge_bins
    return {
        "arrays_stack": np.ascontiguousarray(maps, dtype=np.float64),
        "n_ifo": n_ifo,
        "n_freq": n_freq,
        "n_time": n_time,
        "dt": 1.0 / rate,
        "rate": rate,
        "edge_bins": edge_bins,
        "valid_start": valid_start,
        "valid_stop": valid_stop,
        "nn_valid": valid_stop - valid_start,
        "ib": 1,
        "ie": n_freq - 1,
        "start": 1000.0,
        "stop": 1000.0 + n_time / rate,
        "f_low": 16.0,
        "f_high": 8.0 * (n_freq - 1),
        "shift_bins_by_lag": None,
    }
