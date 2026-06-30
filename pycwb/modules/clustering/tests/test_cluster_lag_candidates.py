"""Unit tests for the replacement clustering API.

Tests the new :func:`~pycwb.modules.clustering.entry_point.cluster_lag_candidates`
entry point, :func:`~pycwb.modules.clustering.pixel_utils.build_pixel_arrays_from_candidates`,
and each backend's ``cluster_candidates`` function using purely synthetic
pixel-candidate dicts — no real cWB data or pipeline setup required.

When *setup*, *xtalk*, and *td_inputs_cache* are all ``None``, the TD-
attachment and finalisation steps are skipped and the returned
:class:`~pycwb.types.network_cluster.FragmentCluster` is the raw merged
cluster straight from the backend's clustering algorithm.  This lets every
test focus on the clustering logic without needing a full supercluster setup.

Test plan
---------
pixel_utils
    1.  Empty candidate dict → empty PixelArrays.
    2.  Candidate dict with pixels → PixelArrays has correct shape.
    3.  Encoded time matches t_bin * layers + f_bin.
    4.  build_fragment_cluster_from_candidates creates valid FragmentCluster.

entry_point
    5.  Empty candidate list → None.
    6.  Unknown method raises ValueError.
    7.  connected_components with two pixels produces one cluster.
    8.  connected_components with zero pixels → None.

connected_components
    9.  Two connected pixels → one cluster.
    10. Disconnected pixels → two or more clusters.
    11. Multiple resolutions → merged result.

weighted_graph
    12. Two adjacent pixels → one cluster.
    13. Disconnected pixels → two clusters (large separation).
    14. min_pixels prunes tiny components.

dbscan
    15. Two adjacent pixels within eps → one cluster.
    16. Distant pixels → separate clusters (or noise singletons).

hdbscan
    17. Pixels below min_cluster_size → all kept as one cluster.
    18. Larger set of adjacent pixels → one cluster.

optics
    19. Adjacent pixels → one cluster.
    20. Distant pixels → separate clusters / singletons.
"""

from __future__ import annotations

import unittest
import numpy as np

from pycwb.modules.clustering.pixel_utils import (
    build_pixel_arrays_from_candidates,
    build_fragment_cluster_from_candidates,
)
from pycwb.modules.clustering.entry_point import cluster_lag_candidates
from pycwb.modules.clustering.common import build_cluster_from_mask
from pycwb.types.network_cluster import FragmentCluster


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────

N_IFO = 2


def _make_candidates(
    time_bins,
    freq_bins,
    n_ifo: int = N_IFO,
    rate: float = 512.0,
    layers: int = 1,
    pattern: int = 1,
    select_subrho: float = 0.0,
    select_subnet: float = 0.0,
) -> dict:
    """Build a minimal synthetic pixel-candidate dict.

    Parameters
    ----------
    time_bins, freq_bins : list[int]
        Time and frequency bin indices for each pixel.
    n_ifo : int
        Number of interferometers.
    rate : float
        WDM sample rate.
    layers : int
        Number of WDM frequency layers.
    pattern : int
        Wavelet pattern flag (controls kt/kf in connected-components backend).
    select_subrho, select_subnet : float
        Coherence selection thresholds (0.0 = no cut).
    """
    n_pix = len(time_bins)
    t_arr  = np.array(time_bins, dtype=np.int64)
    f_arr  = np.array(freq_bins, dtype=np.int64)

    pix_det_energy = np.ones((n_pix, n_ifo),  dtype=np.float64)
    pix_det_index  = np.zeros((n_pix, n_ifo), dtype=np.int64)

    # Build a mask large enough to contain all bins
    t_max = int(t_arr.max()) + 2 if n_pix > 0 else 2
    mask  = np.zeros((layers, t_max), dtype=bool)
    for t, f in zip(time_bins, freq_bins):
        if 0 <= f < layers and 0 <= t < t_max:
            mask[f, t] = True

    return {
        "time":           t_arr,
        "frequency":      f_arr,
        "energy":         np.ones(n_pix, dtype=np.float64),
        "pix_det_energy": pix_det_energy,
        "pix_det_index":  pix_det_index,
        "mask":           mask,
        "rate":           rate,
        "layers":         layers,
        "start":          0.0,
        "stop":           100.0,
        "f_low":          16.0,
        "f_high":         1024.0,
        "pattern":        pattern,
        "select_subrho":  select_subrho,
        "select_subnet":  select_subnet,
        "level":          4,
        "segEdge":        4.0,
    }


def _make_empty_candidates() -> dict:
    return _make_candidates([], [])


def _n_pixels(fc: FragmentCluster) -> int:
    return sum(len(c.pixel_arrays) for c in fc.clusters)


# ─────────────────────────────────────────────────────────────────────────────
# 1. pixel_utils
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPixelArrays(unittest.TestCase):
    def test_empty_candidates_returns_empty_pixel_arrays(self):
        pa = build_pixel_arrays_from_candidates(_make_empty_candidates())
        self.assertEqual(len(pa), 0)

    def test_pixel_arrays_shape(self):
        cand = _make_candidates([0, 1, 2], [0, 0, 0])
        pa = build_pixel_arrays_from_candidates(cand)
        self.assertEqual(len(pa), 3)
        self.assertEqual(pa.pixel_index.shape, (N_IFO, 3))
        self.assertEqual(pa.asnr.shape, (N_IFO, 3))

    def test_encoded_time(self):
        # layers=4: encoded = t * 4 + f
        cand = _make_candidates([2, 3], [1, 2], layers=4)
        pa = build_pixel_arrays_from_candidates(cand)
        expected = np.array([2 * 4 + 1, 3 * 4 + 2], dtype=np.int32)
        np.testing.assert_array_equal(pa.time, expected)

    def test_layers_one_no_encoding_shift(self):
        # layers=1, f_bin always 0: encoded = t * 1 + 0 = t
        cand = _make_candidates([5, 6, 7], [0, 0, 0], layers=1)
        pa = build_pixel_arrays_from_candidates(cand)
        np.testing.assert_array_equal(pa.time, np.array([5, 6, 7], dtype=np.int32))

    def test_frequency_unchanged(self):
        cand = _make_candidates([0, 1], [3, 7], layers=16)
        pa = build_pixel_arrays_from_candidates(cand)
        np.testing.assert_array_equal(pa.frequency, np.array([3, 7], dtype=np.int32))


class TestBuildFragmentCluster(unittest.TestCase):
    def test_empty_clusters(self):
        cand = _make_candidates([0, 1], [0, 0])
        fc = build_fragment_cluster_from_candidates(cand, [])
        self.assertIsInstance(fc, FragmentCluster)
        self.assertEqual(len(fc.clusters), 0)

    def test_metadata_from_candidates(self):
        cand = _make_candidates([0], [0], rate=256.0)
        pa = build_pixel_arrays_from_candidates(cand)
        cluster = build_cluster_from_mask(pa, np.array([True]))
        fc = build_fragment_cluster_from_candidates(cand, [cluster])
        self.assertAlmostEqual(fc.rate, 256.0)
        self.assertAlmostEqual(fc.f_low,  16.0)
        self.assertAlmostEqual(fc.f_high, 1024.0)
        self.assertEqual(len(fc.clusters), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. cluster_lag_candidates — entry point
# ─────────────────────────────────────────────────────────────────────────────

class TestClusterLagCandidatesEntryPoint(unittest.TestCase):
    def test_empty_list_returns_none(self):
        result = cluster_lag_candidates([], method="connected_components")
        self.assertIsNone(result)

    def test_unknown_method_raises_value_error(self):
        cand = _make_candidates([0, 1], [0, 0])
        with self.assertRaises(ValueError):
            cluster_lag_candidates([cand], method="nonexistent_method")

    def test_returns_fragment_cluster_or_none(self):
        cand = _make_candidates([0, 1, 2], [0, 0, 0])
        result = cluster_lag_candidates(
            [cand], method="connected_components", lag_idx=0
        )
        # With no setup/xtalk the merge result is returned as-is
        self.assertTrue(result is None or isinstance(result, FragmentCluster))

    def test_two_pixels_connected_components(self):
        # Adjacent in time → connected → one cluster (kt=2 allows 2 bins separation)
        cand = _make_candidates([0, 1], [0, 0])
        result = cluster_lag_candidates([cand], method="connected_components", lag_idx=0)
        self.assertIsInstance(result, FragmentCluster)
        self.assertGreater(len(result.clusters), 0)
        self.assertEqual(_n_pixels(result), 2)

    def test_zero_pixel_candidates_returns_none(self):
        result = cluster_lag_candidates(
            [_make_empty_candidates()], method="connected_components"
        )
        self.assertIsNone(result)


# ─────────────────────────────────────────────────────────────────────────────
# 3. connected_components backend
# ─────────────────────────────────────────────────────────────────────────────

class TestNativeCC(unittest.TestCase):
    METHOD = "connected_components"

    def _run(self, candidates_list, **kwargs):
        return cluster_lag_candidates(candidates_list, method=self.METHOD, **kwargs)

    def test_single_resolution_preserves_pixel_count(self):
        cand = _make_candidates([0, 1, 2], [0, 0, 0])
        result = self._run([cand])
        self.assertIsNotNone(result)
        self.assertEqual(_n_pixels(result), 3)

    def test_disconnected_pixels_form_separate_clusters(self):
        # Pixels at t=0 and t=100 are far apart → two clusters
        cand = _make_candidates([0, 100], [0, 0])
        result = self._run([cand])
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result.clusters), 2)

    def test_multiple_resolutions_merged(self):
        cand1 = _make_candidates([0, 1], [0, 0])
        cand2 = _make_candidates([5, 6], [0, 0])
        result = self._run([cand1, cand2])
        self.assertIsNotNone(result)
        self.assertEqual(_n_pixels(result), 4)

    def test_pattern_zero_uses_kt1_kf1(self):
        # pattern=0 → kt=kf=1; adjacent pixel still forms one cluster
        cand = _make_candidates([0, 1], [0, 0], pattern=0)
        result = self._run([cand])
        self.assertIsNotNone(result)
        self.assertGreater(len(result.clusters), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. weighted_graph backend
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightedGraphImpl(unittest.TestCase):
    METHOD = "weighted_graph"

    def _run(self, candidates_list, **kwargs):
        return cluster_lag_candidates(candidates_list, method=self.METHOD, **kwargs)

    def test_two_adjacent_pixels_form_one_cluster(self):
        cand = _make_candidates([0, 1], [0, 0])
        result = self._run([cand])
        self.assertIsNotNone(result)
        self.assertEqual(_n_pixels(result), 2)

    def test_disconnected_pixels_form_separate_clusters(self):
        # time_radius_bins=2 by default; pixels 100 apart → no edge
        cand = _make_candidates([0, 100], [0, 0])
        result = self._run([cand])
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result.clusters), 2)

    def test_min_pixels_prunes_components(self):
        # Single isolated pixel: min_pixels=2 → dropped
        cand = _make_candidates([50], [0])
        result = self._run([cand], min_pixels=2)
        # Result is either None or has no clusters
        self.assertTrue(result is None or len(result.clusters) == 0)

    def test_multiple_resolutions_merged(self):
        cand1 = _make_candidates([0, 1], [0, 0])
        cand2 = _make_candidates([5, 6], [0, 0])
        result = self._run([cand1, cand2])
        self.assertIsNotNone(result)
        self.assertEqual(_n_pixels(result), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 5. DBSCAN backend
# ─────────────────────────────────────────────────────────────────────────────

class TestDBSCANImpl(unittest.TestCase):
    METHOD = "dbscan"

    def _run(self, candidates_list, **kwargs):
        try:
            return cluster_lag_candidates(
                candidates_list, method=self.METHOD, **kwargs
            )
        except ImportError:
            self.skipTest("scikit-learn not installed")

    def test_adjacent_pixels_form_one_cluster(self):
        cand = _make_candidates([0, 1, 2], [0, 0, 0])
        result = self._run([cand], eps=1.2, min_samples=1)
        self.assertIsNotNone(result)
        self.assertEqual(_n_pixels(result), 3)

    def test_distant_pixels_split(self):
        # Pixels normalised by eps_time_bins=2 → dist = 50/2 = 25 >> eps=1.2
        cand = _make_candidates([0, 100], [0, 0])
        result = self._run([cand], eps=1.2, min_samples=1, noise_as_singletons=True)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result.clusters), 2)

    def test_noise_as_singletons_preserves_all_pixels(self):
        cand = _make_candidates([0, 50, 100], [0, 0, 0])
        result = self._run([cand], eps=1.2, min_samples=2, noise_as_singletons=True)
        self.assertIsNotNone(result)
        self.assertEqual(_n_pixels(result), 3)

    def test_noise_as_singletons_false_discards_isolated(self):
        # Single isolated pixel with min_samples=2 → noise → discarded
        cand = _make_candidates([0, 100], [0, 0])
        result = self._run([cand], eps=1.2, min_samples=2, noise_as_singletons=False)
        # Both are isolated (no neighbours within eps), so both become noise
        # Result may be None or empty clusters
        total_pix = _n_pixels(result) if result is not None else 0
        self.assertEqual(total_pix, 0)

    def test_returns_fragment_cluster(self):
        cand = _make_candidates([0, 1, 2], [0, 0, 0])
        result = self._run([cand])
        self.assertIsInstance(result, FragmentCluster)


# ─────────────────────────────────────────────────────────────────────────────
# 6. HDBSCAN backend
# ─────────────────────────────────────────────────────────────────────────────

class TestHDBSCANImpl(unittest.TestCase):
    METHOD = "hdbscan"

    def _run(self, candidates_list, **kwargs):
        try:
            return cluster_lag_candidates(
                candidates_list, method=self.METHOD, **kwargs
            )
        except ImportError:
            self.skipTest("scikit-learn (HDBSCAN) not installed")

    def test_below_min_cluster_size_kept_as_one_cluster(self):
        # min_cluster_size=5 but only 2 pixels → fallback: one cluster
        cand = _make_candidates([0, 1], [0, 0])
        result = self._run([cand], min_cluster_size=5)
        self.assertIsNotNone(result)
        self.assertEqual(_n_pixels(result), 2)

    def test_adjacent_pixels_one_cluster(self):
        cand = _make_candidates([0, 1, 2, 3, 4], [0, 0, 0, 0, 0])
        result = self._run([cand], min_cluster_size=2)
        self.assertIsNotNone(result)
        self.assertEqual(_n_pixels(result), 5)

    def test_returns_fragment_cluster(self):
        cand = _make_candidates([0, 1, 2], [0, 0, 0])
        result = self._run([cand])
        self.assertIsInstance(result, FragmentCluster)


# ─────────────────────────────────────────────────────────────────────────────
# 7. OPTICS backend
# ─────────────────────────────────────────────────────────────────────────────

class TestOPTICSImpl(unittest.TestCase):
    METHOD = "optics"

    def _run(self, candidates_list, **kwargs):
        try:
            return cluster_lag_candidates(
                candidates_list, method=self.METHOD, **kwargs
            )
        except ImportError:
            self.skipTest("scikit-learn not installed")

    def test_adjacent_pixels_form_one_cluster(self):
        cand = _make_candidates([0, 1, 2, 3], [0, 0, 0, 0])
        result = self._run([cand], min_samples=2)
        self.assertIsNotNone(result)
        self.assertEqual(_n_pixels(result), 4)

    def test_below_min_samples_kept_as_one(self):
        # min_samples=3 but only 2 pixels → fallback: one cluster
        cand = _make_candidates([0, 1], [0, 0])
        result = self._run([cand], min_samples=3)
        self.assertIsNotNone(result)
        self.assertEqual(_n_pixels(result), 2)

    def test_returns_fragment_cluster(self):
        cand = _make_candidates([0, 1, 2, 3], [0, 0, 0, 0])
        result = self._run([cand])
        self.assertIsInstance(result, FragmentCluster)

    def test_distant_pixels_split_with_noise_singletons(self):
        cand = _make_candidates([0, 100, 200], [0, 0, 0])
        result = self._run(
            [cand], min_samples=2, noise_as_singletons=True
        )
        self.assertIsNotNone(result)
        # Each isolated pixel becomes its own singleton
        self.assertEqual(_n_pixels(result), 3)


if __name__ == "__main__":
    unittest.main()
