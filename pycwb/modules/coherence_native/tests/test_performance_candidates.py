"""Exact comparisons for opt-in coherence performance candidates."""

from pycwb.constants.execution_profile import ExecutionProfile

import dataclasses
import numpy as np
import pytest

from pycwb.modules.coherence_native.clustering import cluster_pixels
from pycwb.modules.coherence_native.kernels import (
    _align_threshold_map_numba,
    _align_threshold_map_preindexed_numba,
)


@pytest.mark.parametrize("n_ifo", [1, 2, 3])
@pytest.mark.parametrize("has_veto", [False, True])
def test_preindexed_support_exact(n_ifo, has_veto):
    rng = np.random.default_rng(42)
    for nvalid in [0, 1, 7, 31]:
        arrays = rng.uniform(0, 20, (n_ifo, 9, nvalid + 4))
        shifts = rng.integers(-100, 100, n_ifo)
        veto = rng.integers(0, 2, nvalid + 4, dtype=np.int16)
        args = (arrays, shifts, 2, nvalid, veto, has_veto, 2, 2, 6, 6.0, 12.0)
        expected = _align_threshold_map_numba(*args)
        actual = _align_threshold_map_preindexed_numba(*args)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)


def assert_tree_equal(a, b):
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
    elif dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):
            assert_tree_equal(getattr(a, f.name), getattr(b, f.name))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    elif isinstance(a, (float, np.floating)):
        # Rejected clusters retain unset NaN metadata in both paths.
        assert float(a).hex() == float(b).hex()
    else:
        assert a == b


@pytest.mark.parametrize("thresholds", [(0.0, 0.0), (4.5, 0.1), (float("inf"), 0.1), (float("nan"), float("nan"))])
@pytest.mark.parametrize("empty", [False, True])
def test_early_cuts_exact(thresholds, empty):
    rng = np.random.default_rng(19)
    mask = rng.random((20, 60)) < (0.0 if empty else 0.12)
    f, t = np.nonzero(mask)
    en = rng.uniform(0.01, 30, (len(f), 2))
    candidates = dict(
        mask=mask,
        frequency=f,
        time=t,
        pix_det_energy=en,
        pix_det_index=np.column_stack([t * 20 + f, t * 20 + f]),
        energy=en.sum(axis=1),
        layers=20,
        rate=32.0,
    )
    reference = cluster_pixels(candidates, kt=2, kf=3)
    reference.select("subrho", thresholds[0])
    reference.select("subnet", thresholds[1])
    reference.remove_rejected()
    actual = cluster_pixels(candidates, kt=2, kf=3, select_subrho=thresholds[0], select_subnet=thresholds[1])
    assert_tree_equal(reference, actual)


@pytest.mark.parametrize("return_rejected", [False, True])
def test_pipeline_early_cuts_preserve_rejected_contract(monkeypatch, return_rejected):
    from types import SimpleNamespace
    from pycwb.modules.coherence_native import pipeline

    candidates = dict(
        mask=np.ones((5, 5), dtype=bool),
        frequency=np.array([1]),
        time=np.array([1]),
        pix_det_energy=np.array([[1.0, 1.0]]),
        pix_det_index=np.array([[6, 6]]),
        energy=np.array([2.0]),
        layers=5,
        rate=32.0,
    )
    setup = dict(
        tf_maps=[None],
        Eo=1.0,
        job_seg=SimpleNamespace(n_lag=1, lag_shifts=[[0.0, 0.0]]),
        pattern=10,
        segEdge=0,
        level=2,
        select_subrho=100.0,
        select_subnet=0.1,
    )
    monkeypatch.setattr(pipeline, "select_network_pixels", lambda **kw: candidates)
    setup["execution_profile"] = ExecutionProfile(coherence_early_cuts=False)
    reference = pipeline.coherence_single_lag([setup], 0, return_rejected=return_rejected)
    setup["execution_profile"] = ExecutionProfile(coherence_early_cuts=True)
    actual = pipeline.coherence_single_lag([setup], 0, return_rejected=return_rejected)
    assert_tree_equal(reference, actual)
    assert len(actual[0].clusters) == int(return_rejected)
