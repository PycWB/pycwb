import numpy as np

from pycwb.modules.super_cluster_native.sub_net_cut import (
    mra_statistics,
    mra_statistics_from_td,
    optimze_sky_loc,
    optimze_sky_loc_from_td,
    sub_net_cut,
    sub_net_cut_from_pixel_arrays,
)
from pycwb.modules.super_cluster_native.super_cluster import supercluster
from pycwb.modules.super_cluster_native.utils import (
    _top_loudest_indices,
    calculate_statistics,
    calculate_statistics_arrays,
    get_cluster_links,
    get_defragment_link,
)
from pycwb.types.network_cluster import Cluster
from pycwb.types.pixel_arrays import PixelArrays


def _reference_cluster_links(pixels, gap, n_ifo):
    pixels = pixels[pixels[:, 0].argsort()]
    t_gap = np.max(pixels[:, 2]) * (1.0 + gap)
    d_f = 0.0
    links = []
    for i, p in enumerate(pixels):
        for q in pixels[i + 1:]:
            if q[0] - p[0] > t_gap:
                break
            if p[4] == q[4] or max(p[2] / q[2], q[2] / p[2]) > 3:
                continue
            r = 1.0 / p[2] + 1.0 / q[2]
            t = p[2] + q[2]
            d_t = max(abs(p[5 + k] - q[5 + k]) for k in range(n_ifo)) - 0.5 * t
            d_f = abs(p[1] - q[1]) - 0.5 * r
            eps = (d_t * r if d_t > 0.0 else 0.0) + (d_f * t if d_f > 0.0 else 0.0)
            if gap >= eps:
                link = (int(min(p[4], q[4])), int(max(p[4], q[4])))
                if link not in links:
                    links.append(link)
    if not links:
        return np.empty((0, 2), dtype=np.int32), d_f
    return np.array(links, dtype=np.int32), d_f


def _reference_defrag_links(pixels, t_gap, f_gap, n_ifo):
    pixels = pixels[pixels[:, 0].argsort()]
    search_t_gap = max(np.max(pixels[:, 2]), t_gap)
    links = []
    for i, p in enumerate(pixels):
        for q in pixels[i + 1:]:
            if q[0] - p[0] > search_t_gap:
                break
            if p[4] == q[4] or max(p[2] / q[2], q[2] / p[2]) > 3:
                continue
            r = 1.0 / p[2] + 1.0 / q[2]
            t = p[2] + q[2]
            d_t = max(abs(p[5 + k] - q[5 + k]) for k in range(n_ifo)) - 0.5 * t
            d_f = abs(p[1] - q[1]) - 0.5 * r
            if d_t < t_gap and d_f < f_gap:
                link = (int(min(p[4], q[4])), int(max(p[4], q[4])))
                if link not in links:
                    links.append(link)
    if not links:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(links, dtype=np.int32)


def _pixel_matrix_case(kind):
    n_ifo = 2
    if kind == "none":
        return np.array(
            [
                [0.0, 0.0, 0.01, 50.0, 0.0, 0.0, 0.0],
                [1.0, 500.0, 0.01, 50.0, 1.0, 0.0, 0.0],
                [2.0, 900.0, 0.01, 50.0, 2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ), n_ifo
    if kind == "dense":
        return np.array(
            [
                [0.00, 10.0, 0.1, 5.0, 0.0, 1.00, 1.00],
                [0.01, 10.0, 0.1, 5.0, 1.0, 1.00, 1.00],
                [0.02, 10.0, 0.1, 5.0, 2.0, 1.00, 1.00],
                [0.03, 10.0, 0.1, 5.0, 0.0, 1.00, 1.00],
            ],
            dtype=np.float64,
        ), n_ifo
    return np.array(
        [
            [0.00, 10.0, 0.10, 5.0, 0.0, 1.00, 1.00],
            [0.01, 10.0, 0.20, 2.5, 1.0, 1.02, 1.01],
            [0.02, 30.0, 0.10, 5.0, 2.0, 1.03, 1.05],
            [0.03, 10.0, 0.40, 1.25, 3.0, 1.04, 1.01],
        ],
        dtype=np.float64,
    ), n_ifo


def test_link_kernels_match_reference_cases():
    for kind in ("none", "dense", "mixed"):
        pixels, n_ifo = _pixel_matrix_case(kind)
        expected_links, expected_df = _reference_cluster_links(pixels, 1.0, n_ifo)
        actual_links, actual_df = get_cluster_links(pixels, 1.0, n_ifo)
        np.testing.assert_array_equal(actual_links, expected_links)
        assert actual_df == expected_df

        expected_defrag = _reference_defrag_links(pixels, 0.5, 50.0, n_ifo)
        actual_defrag = get_defragment_link(pixels, 0.5, 50.0, n_ifo)
        np.testing.assert_array_equal(actual_defrag, expected_defrag)


def _make_pixel_arrays(
    *,
    time,
    frequency,
    rate=None,
    layers=None,
    likelihood=None,
    asnr=None,
    td_amp_dense=None,
):
    n_pix = len(time)
    n_ifo = 2
    if rate is None:
        rate = np.full(n_pix, 32.0, dtype=np.float32)
    if layers is None:
        layers = np.full(n_pix, 16, dtype=np.int32)
    if likelihood is None:
        likelihood = np.full(n_pix, 5.0, dtype=np.float32)
    if asnr is None:
        asnr = np.full((n_ifo, n_pix), 2.0, dtype=np.float32)
    return PixelArrays.from_arrays(
        time=np.asarray(time, dtype=np.int32),
        frequency=np.asarray(frequency, dtype=np.int32),
        layers=np.asarray(layers, dtype=np.int32),
        rate=np.asarray(rate, dtype=np.float32),
        likelihood=np.asarray(likelihood, dtype=np.float32),
        asnr=np.asarray(asnr, dtype=np.float32),
        noise_rms=np.ones((n_ifo, n_pix), dtype=np.float32),
        pixel_index=np.zeros((n_ifo, n_pix), dtype=np.int32),
        td_amp_dense=td_amp_dense,
        n_ifo=n_ifo,
    )


def test_statistics_array_kernel_matches_public_matrix_wrapper():
    pa = _make_pixel_arrays(
        time=np.array([16, 32, 48, 64]),
        frequency=np.array([2, 2, 3, 3]),
        rate=np.array([32, 32, 64, 64], dtype=np.float32),
        layers=np.array([16, 16, 32, 32], dtype=np.int32),
        likelihood=np.array([3.0, 4.0, 8.0, 9.0], dtype=np.float32),
    )
    matrix = np.column_stack(
        [
            pa.core.astype(np.float64),
            pa.time.astype(np.float64),
            pa.frequency.astype(np.float64),
            pa.rate.astype(np.float64),
            pa.layers.astype(np.float64),
            pa.likelihood.astype(np.float64),
            pa.asnr.T.astype(np.float64),
        ]
    )
    expected = calculate_statistics(matrix, "L", False, False, 3, 1.0, 0.0)
    actual = calculate_statistics_arrays(
        pa.core,
        pa.time.astype(np.float64),
        pa.frequency.astype(np.float64),
        pa.rate.astype(np.float64),
        pa.layers.astype(np.float64),
        pa.likelihood.astype(np.float64),
        pa.asnr.astype(np.float64),
        "L",
        False,
        False,
        3,
        1.0,
        0.0,
    )
    assert actual[0] is True
    np.testing.assert_allclose(np.array(actual[1:]), np.array(expected))


def test_supercluster_grouping_and_rejection_paths():
    linked = [
        Cluster(pixel_arrays=_make_pixel_arrays(time=[16, 32], frequency=[2, 2])),
        Cluster(pixel_arrays=_make_pixel_arrays(time=[16, 32], frequency=[2, 2])),
    ]
    merged = supercluster(linked, "L", 1.0, 1.0, 2, mini_pix=3)
    assert len(merged) == 1
    assert merged[0].cluster_status == 0
    assert len(merged[0].pixel_arrays) == 4

    rejected = supercluster(
        [
            Cluster(pixel_arrays=_make_pixel_arrays(time=[16], frequency=[2])),
            Cluster(pixel_arrays=_make_pixel_arrays(time=[16], frequency=[2])),
        ],
        "L",
        1.0,
        1.0,
        2,
        mini_pix=3,
    )
    assert len(rejected) == 1
    assert rejected[0].cluster_status == 1

    standalone = [
        Cluster(pixel_arrays=_make_pixel_arrays(time=[16], frequency=[2])),
        Cluster(pixel_arrays=_make_pixel_arrays(time=[16000], frequency=[50])),
    ]
    assert supercluster(standalone, "L", 0.0, 1.0, 2) is standalone


class _EmptyXTalk:
    @staticmethod
    def _empty(n_pix):
        return np.zeros((n_pix, 2), dtype=np.int32), np.zeros((0, 8), dtype=np.float32)

    def get_xtalk_pixels(self, pixels, check=True):
        return self._empty(len(pixels))

    def get_xtalk_pixels_from_arrays(self, layers, times, check=True):
        return self._empty(len(layers))


class _RaisingXTalk(_EmptyXTalk):
    def get_xtalk_pixels(self, pixels, check=True):
        raise AssertionError("xtalk should not be evaluated")

    def get_xtalk_pixels_from_arrays(self, layers, times, check=True):
        raise AssertionError("xtalk should not be evaluated")


def test_subnet_kernels_without_td_energy_match_reference_kernels():
    rng = np.random.default_rng(17)
    n_ifo = 2
    n_pix = 5
    n_sky = 4
    ndelay = 7
    offset = ndelay // 2

    td00 = np.ascontiguousarray(rng.normal(size=(ndelay, n_ifo, n_pix)).astype(np.float32))
    td90 = np.ascontiguousarray(rng.normal(size=(ndelay, n_ifo, n_pix)).astype(np.float32))
    td_energy = np.ascontiguousarray(td00 * td00 + td90 * td90)
    rms = np.ascontiguousarray(rng.uniform(0.2, 1.0, size=(n_pix, n_ifo)).astype(np.float32))
    fp = np.ascontiguousarray(rng.normal(size=(n_sky, n_ifo)).astype(np.float32))
    fx = np.ascontiguousarray(rng.normal(size=(n_sky, n_ifo)).astype(np.float32))
    ml = np.ascontiguousarray(
        rng.integers(-offset, offset + 1, size=(n_ifo, n_sky), dtype=np.int32)
    )

    network_energy_threshold = np.float32(0.05)
    e2or = 0.1
    subcut = -1.0

    expected_sky = optimze_sky_loc(
        n_ifo, n_pix, n_sky, fp, fx, rms, td00, td90, td_energy,
        ml, network_energy_threshold, e2or, subcut,
    )
    actual_sky = optimze_sky_loc_from_td(
        n_ifo, n_pix, n_sky, fp, fx, rms, td00, td90,
        ml, network_energy_threshold, e2or, subcut,
    )
    np.testing.assert_allclose(np.array(actual_sky), np.array(expected_sky), rtol=1e-6, atol=1e-6)

    xtalk_lookup = np.zeros((n_pix, 2), dtype=np.int32)
    xtalk = np.zeros((0, 8), dtype=np.float32)
    l_max = int(expected_sky[0])
    expected_mra = mra_statistics(
        n_ifo, n_pix, fp, fx, rms, td00, td90, td_energy, ml,
        network_energy_threshold, e2or, subcut, xtalk, xtalk_lookup, l_max,
    )
    actual_mra = mra_statistics_from_td(
        n_ifo, n_pix, fp, fx, rms, td00, td90, ml,
        network_energy_threshold, e2or, subcut, xtalk, xtalk_lookup, l_max,
    )
    np.testing.assert_allclose(np.array(actual_mra), np.array(expected_mra), rtol=1e-6, atol=1e-6)


def test_subnet_cut_skips_xtalk_and_mra_when_suball_already_fails():
    rng = np.random.default_rng(19)
    td_amp = rng.normal(size=(4, 2, 10)).astype(np.float32)
    pa = _make_pixel_arrays(
        time=np.array([16, 32, 48, 64]),
        frequency=np.array([2, 3, 4, 5]),
        likelihood=np.array([5.0, 4.0, 3.0, 2.0], dtype=np.float32),
        td_amp_dense=td_amp,
    )
    ml = np.array([[0, 1, -1], [0, -1, 1]], dtype=np.int32)
    fp = np.array([[0.8, 0.4, -0.1], [0.2, -0.3, 0.9]], dtype=np.float32)
    fx = np.array([[0.1, -0.7, 0.5], [0.6, 0.2, -0.4]], dtype=np.float32)
    timing = {}

    result = sub_net_cut_from_pixel_arrays(
        pa,
        np.arange(len(pa), dtype=np.int64),
        ml,
        fp.T.copy(),
        fx.T.copy(),
        0.1,
        0.1,
        2,
        3,
        2.0,
        -1.0,
        0.0,
        -1.0,
        _RaisingXTalk(),
        arrays_prepared=True,
        timing=timing,
    )

    assert result["subnet_passed"] is False
    assert result["subrho_passed"] is True
    assert result["subthr_passed"] is True
    assert timing.get("sky_scan", 0.0) > 0.0
    assert timing.get("xtalk", 0.0) == 0.0
    assert timing.get("mra", 0.0) == 0.0


def test_top_loudest_tie_safe_and_prepared_subnet_path_matches_wrapper():
    rng = np.random.default_rng(11)
    td_amp = rng.normal(size=(6, 2, 10)).astype(np.float32)
    pa = _make_pixel_arrays(
        time=np.array([16, 32, 48, 64, 80, 96]),
        frequency=np.array([2, 3, 4, 5, 6, 7]),
        likelihood=np.array([5.0, 4.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32),
        td_amp_dense=td_amp,
    )
    full_sort_top = np.argsort(-pa.likelihood)[:2]
    top_idx = _top_loudest_indices(pa.likelihood, 2)
    np.testing.assert_array_equal(top_idx, full_sort_top)

    ml = np.array([[0, 1, -1], [0, -1, 1]], dtype=np.int32)
    fp = np.array([[0.8, 0.4, -0.1], [0.2, -0.3, 0.9]], dtype=np.float32)
    fx = np.array([[0.1, -0.7, 0.5], [0.6, 0.2, -0.4]], dtype=np.float32)
    xtalk = _EmptyXTalk()

    top_pa = pa[full_sort_top]
    expected = sub_net_cut(
        top_pa,
        ml,
        fp,
        fx,
        0.1,
        0.1,
        2,
        3,
        -1.0,
        -1.0,
        0.0,
        -1.0,
        xtalk,
    )
    actual = sub_net_cut_from_pixel_arrays(
        pa,
        top_idx,
        ml,
        fp.T.copy(),
        fx.T.copy(),
        0.1,
        0.1,
        2,
        3,
        -1.0,
        -1.0,
        0.0,
        -1.0,
        xtalk,
        arrays_prepared=True,
    )
    assert actual == expected
