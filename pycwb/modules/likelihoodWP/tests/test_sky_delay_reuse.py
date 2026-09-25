import numpy as np
import pytest
from pycwb.modules.likelihoodWP.tests.sky_scan_reference.sky_scan import scan_sky_for_best_fit
from pycwb.modules.likelihoodWP.sky_scan_delay import make_delay_groups, scan_sky_grouped_delays, delay_groups_for_grid


def assert_exact(a, b):
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes()
    elif isinstance(a, tuple):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_exact(x, y)
    elif isinstance(a, (float, np.floating)):
        assert float(a).hex() == float(b).hex()
    else:
        assert a == b


@pytest.mark.parametrize("n_ifo", [1, 2, 3])
@pytest.mark.parametrize("mask_kind", ["all", "subset", "reverse", "duplicates"])
@pytest.mark.parametrize("n_pix", [1, 7, 64])
@pytest.mark.parametrize("threshold", [1.0, 1.0e9])
def test_grouped_scan_exact_and_inputs_unchanged(n_ifo, mask_kind, n_pix, threshold):
    rng = np.random.default_rng(123)
    n_sky = 24
    fp = rng.normal(size=(n_sky, n_ifo)).astype("f4")
    fx = rng.normal(size=(n_sky, n_ifo)).astype("f4")
    rms = rng.uniform(0.2, 1, (n_pix, n_ifo)).astype("f4")
    td0 = rng.normal(10, 2, (5, n_ifo, n_pix)).astype("f4")
    td90 = rng.normal(8, 2, td0.shape).astype("f4")
    patterns = rng.integers(-2, 3, (n_ifo, 4))
    ml = patterns[:, np.arange(n_sky) % 4].copy()
    indices = np.arange(n_sky, dtype="i8")
    if mask_kind == "subset":
        indices = np.array([17, 1, 9, 22], dtype="i8")
    if mask_kind == "reverse":
        indices = indices[::-1].copy()
    if mask_kind == "duplicates":
        indices = np.array([17, 1, 17, 9, 22], dtype="i8")
    reg = np.array([0.7, 1.0, 0.0], dtype="f4")
    args = (n_ifo, n_pix, n_sky, fp, fx, rms, td0, td90, ml, reg, 0.0, 0.5, threshold, indices)
    snapshots = [x.copy() for x in args if isinstance(x, np.ndarray)]
    expected = scan_sky_for_best_fit(*args)
    actual = scan_sky_grouped_delays(*args, *make_delay_groups(ml))
    assert_exact(actual, expected)
    for x, y in zip([x for x in args if isinstance(x, np.ndarray)], snapshots):
        assert_exact(x, y)


def test_grouping_and_tie_order():
    ml = np.array([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]])
    order, offsets = make_delay_groups(ml)
    np.testing.assert_array_equal(np.sort(order), np.arange(6))
    for a, b in zip(offsets[:-1], offsets[1:]):
        for i in order[a:b]:
            np.testing.assert_array_equal(ml[:, i], ml[:, order[a]])
    n_sky, n_pix = 6, 3
    fp = np.ones((n_sky, 2), dtype="f4")
    fx = np.tile(np.array([0.2, 0.8], dtype="f4"), (n_sky, 1))
    rms = np.ones((n_pix, 2), dtype="f4")
    td = np.ones((3, 2, n_pix), dtype="f4") * 10
    ml = np.zeros((2, n_sky), dtype="i8")
    indices = np.array([5, 1, 3, 0], dtype="i8")
    args = (2, n_pix, n_sky, fp, fx, rms, td, td, ml, np.array([0.7, 1, 0], dtype="f4"), 0.0, 0.5, 1.0, indices)
    assert_exact(scan_sky_grouped_delays(*args, *make_delay_groups(ml)), scan_sky_for_best_fit(*args))


def test_cache_grid_identity_and_separate_coarse_geometry():
    setup = {}
    main = np.array([[0, 1, 0], [1, 0, 1]])
    coarse = np.array([[0, 1], [1, 0]])
    order, offsets = delay_groups_for_grid(setup, main)
    assert delay_groups_for_grid(setup, main)[0] is order
    other = delay_groups_for_grid(setup, coarse, True)
    assert len(other[0]) == 2
    assert delay_groups_for_grid(setup, main)[1] is offsets
    replacement = main.copy()
    replacement[:, 0] = 2
    updated = delay_groups_for_grid(setup, replacement)
    assert updated[0] is not order
    for actual, expected in zip(updated, make_delay_groups(replacement)):
        np.testing.assert_array_equal(actual, expected)
