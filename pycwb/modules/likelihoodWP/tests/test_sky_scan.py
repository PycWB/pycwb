import numpy as np
import pytest
from pycwb.modules.likelihoodWP.tests.sky_scan_reference.sky_scan import scan_sky_for_best_fit
from pycwb.modules.likelihoodWP.sky_delay_groups import make_delay_groups
from pycwb.modules.likelihoodWP.sky_scan import scan_sky_kernel


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
    actual = scan_sky_kernel(*args, *make_delay_groups(ml))
    assert_exact(actual, expected)
    from pycwb.modules.likelihoodWP.sky_scan import scan_sky

    for reuse in (False, True):
        assert_exact(
            scan_sky((fp, fx, ml), (rms, td0, td90), (reg, 0.0, 0.5, threshold, indices), reuse_delays=reuse), expected
        )
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
    assert_exact(scan_sky_kernel(*args, *make_delay_groups(ml)), scan_sky_for_best_fit(*args))


@pytest.mark.parametrize("n_ifo", [1, 2, 3])
@pytest.mark.parametrize("n_pix", [1, 8, 65])
@pytest.mark.parametrize("inactive", [False, True])
def test_helpers_overwrite_poisoned_and_reused_buffers(n_ifo, n_pix, inactive):
    from pycwb.modules.likelihoodWP.tests.sky_scan_reference.dpf import dpf_np_loops_vec as compute_dpf
    from pycwb.modules.likelihoodWP.tests.sky_scan_reference.sky_stat import load_data_from_td as compute_pixel_energy_and_mask, avx_GW_ps as project_signal_packet, avx_ort_ps as orthogonalize_quadratures, avx_stat_ps as compute_coherent_statistics
    from pycwb.modules.likelihoodWP.dpf import compute_dpf_into
    from pycwb.modules.likelihoodWP.sky_kernels import (
        project_signal_packet_into,
        orthogonalize_quadratures_into,
        compute_coherent_statistics_into,
    )

    from pycwb.modules.likelihoodWP import dpf as allocating_dpf, sky_kernels as allocating_stat

    rng = np.random.default_rng(984)

    def buf(shape):
        return np.full(shape, np.nan, dtype=np.float32)

    dpf_buffers = (buf((n_pix, n_ifo)), buf((n_pix, n_ifo)), *(buf(n_pix) for _ in range(5)))
    gw_buffers = (*(buf(n_pix) for _ in range(5)), buf((n_ifo, n_pix)), buf((n_ifo, n_pix)))
    ort_buffers = tuple(buf(n_pix) for _ in range(4))
    stat_buffers = tuple(buf(n_pix) for _ in range(3))
    # First call starts with NaNs; subsequent calls reuse different-direction data.
    for rep in range(3):
        fp0 = rng.normal(size=n_ifo).astype("f4")
        fx0 = rng.normal(size=n_ifo).astype("f4")
        rms = rng.uniform(0.2, 1, (n_pix, n_ifo)).astype("f4")
        p = rng.normal(10, 2, (n_ifo, n_pix)).astype("f4")
        q = rng.normal(8, 2, p.shape).astype("f4")
        dpf = compute_dpf(fp0, fx0, rms)
        assert_exact(allocating_dpf.compute_dpf(fp0, fx0, rms), dpf)
        assert_exact(compute_dpf_into(fp0, fx0, rms, dpf_buffers), dpf)
        _, _, energy, mask = compute_pixel_energy_and_mask(p, q, 1.0e9 if inactive else 1.0)
        args = (p, q, dpf[1], dpf[2], dpf[3], dpf[4], dpf[7], energy, mask, np.array([0.7, 1, 0], dtype="f4"))
        gw = project_signal_packet(*args)
        assert_exact(allocating_stat.project_signal_packet(*args), gw)
        assert_exact(project_signal_packet_into(*args, gw_buffers), gw)
        args = (gw[1], gw[2], gw[3])
        ort = orthogonalize_quadratures(*args)
        assert_exact(allocating_stat.orthogonalize_quadratures(*args), ort)
        assert_exact(orthogonalize_quadratures_into(*args, ort_buffers), ort)
        args = (p, q, gw[1], gw[2], ort[1], ort[2], gw[3])
        assert_exact(compute_coherent_statistics_into(*args, stat_buffers), compute_coherent_statistics(*args))
        assert_exact(allocating_stat.compute_coherent_statistics(*args), compute_coherent_statistics(*args))


def test_singleton_cache_is_separate_from_grouped_and_coarse():
    from pycwb.modules.likelihoodWP.sky_delay_groups import delay_groups_for_grid

    ml = np.zeros((2, 6), dtype=np.int64)
    setup = {}
    grouped = delay_groups_for_grid(setup, ml)
    singleton = delay_groups_for_grid(setup, ml, reuse_delays=False)
    assert len(grouped[1]) == 2
    np.testing.assert_array_equal(singleton[0], np.arange(6))
    np.testing.assert_array_equal(singleton[1], np.arange(7))
    assert delay_groups_for_grid(setup, ml, reuse_delays=False)[0] is singleton[0]
    assert delay_groups_for_grid(setup, ml)[0] is grouped[0]
    assert delay_groups_for_grid(setup, ml, True, False)[0] is not singleton[0]


@pytest.mark.parametrize("reuse", [False, True])
def test_real_input_golden_fixture(reuse):
    from pathlib import Path
    from pycwb.modules.likelihoodWP.sky_scan import scan_sky

    with np.load(Path(__file__).with_name("data") / "sky_scan_preconsolidation.npz") as fixture:
        args = tuple(
            fixture[f"arg_{i}"].item() if fixture[f"arg_{i}"].ndim == 0 else fixture[f"arg_{i}"] for i in range(14)
        )
        expected = tuple(
            fixture[f"expected_{i}"].item() if fixture[f"expected_{i}"].ndim == 0 else fixture[f"expected_{i}"]
            for i in range(13)
        )
    assert_exact(scan_sky((args[3], args[4], args[8]), args[5:8], args[9:14], reuse_delays=reuse), expected)
