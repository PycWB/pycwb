from pycwb.config.processing import ExecutionProfile
import numpy as np
import pytest

from pycwb.modules.coherence_native.kernels import _label_components_grid
from pycwb.modules.coherence_native.run_clustering import label_components_runs


@pytest.mark.parametrize("kf,kt", [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (1, 2), (2, 2), (3, 2)])
def test_all_small_masks(kf, kt):
    # Exhaustive shapes include holes, diagonal contacts, and thin bridges.
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)]).reshape(3, 3)
        f, t = np.nonzero(mask)
        args = (f, t, 3, 3, kf, kt)
        np.testing.assert_array_equal(label_components_runs(*args), _label_components_grid(*args))


def test_permutations_duplicates_and_invalid_coordinates():
    rng = np.random.default_rng(98)
    for _ in range(150):
        f = rng.integers(-1, 15, 150)
        t = rng.integers(-1, 35, 150)
        kf, kt = rng.integers(0, 6, 2)
        args = (f, t, 14, 34, kf, kt)
        np.testing.assert_array_equal(label_components_runs(*args), _label_components_grid(*args))


def test_large_filled_region_and_nested_boundaries():
    mask = np.ones((50, 60), dtype=bool)
    mask[10:40, 10:50] = False
    mask[20:30, 20:40] = True
    for bridge in [False, True]:
        if bridge:
            mask[25, 8:22] = True
        f, t = np.nonzero(mask)
        args = (f, t, 50, 60, 3, 2)
        np.testing.assert_array_equal(label_components_runs(*args), _label_components_grid(*args))


@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("return_rejected", [False, True])
def test_cluster_products_and_rejected_contract(monkeypatch, early, return_rejected):
    from types import SimpleNamespace
    import importlib

    coherence_module = importlib.import_module("pycwb.modules.coherence_native.coherence")
    from .test_performance_candidates import assert_tree_equal

    rng = np.random.default_rng(105)
    mask = rng.random((15, 50)) < 0.2
    f, t = np.nonzero(mask)
    en = rng.uniform(0, 20, (len(f), 2))
    candidates = dict(
        mask=mask,
        frequency=f,
        time=t,
        pix_det_energy=en,
        pix_det_index=np.column_stack([t * 15 + f, t * 15 + f]),
        energy=en.sum(axis=1),
        layers=15,
        rate=32.0,
    )
    setup = dict(
        tf_maps=[None],
        Eo=1.0,
        job_seg=SimpleNamespace(n_lag=1, lag_shifts=[[0.0, 0.0]]),
        pattern=10,
        segEdge=0,
        level=2,
        select_subrho=5.0,
        select_subnet=0.1,
    )
    monkeypatch.setattr(coherence_module, "select_network_pixels", lambda **kw: candidates)
    setup["execution_profile"] = ExecutionProfile(coherence_early_cuts=early, cluster_runs=False)
    reference = coherence_module.coherence_single_lag([setup], 0, return_rejected=return_rejected)
    setup["execution_profile"] = ExecutionProfile(coherence_early_cuts=early, cluster_runs=True)
    actual = coherence_module.coherence_single_lag([setup], 0, return_rejected=return_rejected)
    assert_tree_equal(reference, actual)
