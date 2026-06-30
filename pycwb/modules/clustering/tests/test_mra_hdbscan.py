"""Tests for the additive multi-resolution HDBSCAN backend."""

from __future__ import annotations

import builtins
import unittest
from unittest.mock import patch

import numpy as np

from pycwb.modules.clustering.entry_point import cluster_lag_candidates
from pycwb.types.network_cluster import FragmentCluster


N_IFO = 2


class _FakeHDBSCAN:
    labels = None
    seen_X = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_predict(self, X):
        type(self).seen_X = X
        if type(self).labels is None:
            return np.zeros(X.shape[0], dtype=np.int32)
        return np.asarray(type(self).labels, dtype=np.int32)


def _make_candidates(
    time_bins,
    freq_bins,
    *,
    rate: float,
    layers: int,
    level: int,
    detector_energy=None,
    energy=None,
) -> dict:
    n_pix = len(time_bins)
    t_arr = np.asarray(time_bins, dtype=np.int64)
    f_arr = np.asarray(freq_bins, dtype=np.int64)

    if detector_energy is None:
        pix_det_energy = np.ones((n_pix, N_IFO), dtype=np.float64)
    else:
        pix_det_energy = np.asarray(detector_energy, dtype=np.float64)

    if energy is None:
        energy_arr = pix_det_energy.sum(axis=1) if n_pix else np.zeros(0)
    else:
        energy_arr = np.asarray(energy, dtype=np.float64)

    pix_det_index = np.zeros((n_pix, N_IFO), dtype=np.int64)
    t_max = int(t_arr.max()) + 2 if n_pix else 2
    mask = np.zeros((layers, t_max), dtype=bool)
    for t_bin, f_bin in zip(t_arr, f_arr):
        if 0 <= f_bin < layers and 0 <= t_bin < t_max:
            mask[f_bin, t_bin] = True

    return {
        "time": t_arr,
        "frequency": f_arr,
        "energy": energy_arr,
        "pix_det_energy": pix_det_energy,
        "pix_det_index": pix_det_index,
        "mask": mask,
        "rate": rate,
        "layers": layers,
        "start": 0.0,
        "stop": 100.0,
        "f_low": 16.0,
        "f_high": 1024.0,
        "pattern": 1,
        "select_subrho": 0.0,
        "select_subnet": 0.0,
        "level": level,
        "segEdge": 4.0,
    }


def _n_pixels(fragment_cluster: FragmentCluster) -> int:
    return sum(len(c.pixel_arrays) for c in fragment_cluster.clusters)


class TestMRAHDBSCAN(unittest.TestCase):
    METHOD = "mra_hdbscan"

    def setUp(self):
        _FakeHDBSCAN.labels = None
        _FakeHDBSCAN.seen_X = None

    def _run(self, candidates_list, **kwargs):
        try:
            return cluster_lag_candidates(candidates_list, method=self.METHOD, **kwargs)
        except ImportError:
            self.skipTest("scikit-learn HDBSCAN not installed")

    def test_pooled_labels_convert_to_mixed_resolution_clusters(self):
        fine = _make_candidates([10, 11], [4, 4], rate=512.0, layers=16, level=5)
        coarse = _make_candidates([5, 6], [8, 8], rate=256.0, layers=16, level=4)
        _FakeHDBSCAN.labels = [0, 1, 0, 1]

        with patch("pycwb.modules.clustering.mra_hdbscan.impl._HDBSCAN", _FakeHDBSCAN):
            result = self._run([fine, coarse], min_cluster_size=2)

        self.assertIsInstance(result, FragmentCluster)
        self.assertEqual(len(result.clusters), 2)
        self.assertEqual(_n_pixels(result), 4)
        for cluster in result.clusters:
            self.assertEqual(set(cluster.pixel_arrays.rate.tolist()), {256.0, 512.0})

    def test_feature_matrix_contains_scaled_mra_columns(self):
        fine = _make_candidates([10], [4], rate=512.0, layers=16, level=5)
        coarse = _make_candidates([5], [8], rate=256.0, layers=16, level=4)

        with patch("pycwb.modules.clustering.mra_hdbscan.impl._HDBSCAN", _FakeHDBSCAN):
            result = self._run(
                [fine, coarse],
                min_cluster_size=2,
                level_weight=0.5,
                log_energy_weight=0.25,
                detector_balance_weight=0.5,
            )

        self.assertIsInstance(result, FragmentCluster)
        self.assertIsNotNone(_FakeHDBSCAN.seen_X)
        self.assertEqual(_FakeHDBSCAN.seen_X.shape, (2, 5))

    def test_noise_as_singletons_preserves_noise_pixels(self):
        cand = _make_candidates([0, 10, 20], [1, 1, 1], rate=512.0, layers=16, level=5)
        _FakeHDBSCAN.labels = [-1, -1, -1]

        with patch("pycwb.modules.clustering.mra_hdbscan.impl._HDBSCAN", _FakeHDBSCAN):
            result = self._run([cand], min_cluster_size=2, noise_as_singletons=True)

        self.assertIsInstance(result, FragmentCluster)
        self.assertEqual(len(result.clusters), 3)
        self.assertEqual(_n_pixels(result), 3)

    def test_noise_as_singletons_false_discards_noise_pixels(self):
        cand = _make_candidates([0, 10, 20], [1, 1, 1], rate=512.0, layers=16, level=5)
        _FakeHDBSCAN.labels = [-1, -1, -1]

        with patch("pycwb.modules.clustering.mra_hdbscan.impl._HDBSCAN", _FakeHDBSCAN):
            result = self._run([cand], min_cluster_size=2, noise_as_singletons=False)

        self.assertIsNone(result)

    def test_too_few_pixels_fallback_keeps_one_cluster(self):
        cand = _make_candidates([0, 1], [1, 1], rate=512.0, layers=16, level=5)
        result = self._run([cand], min_cluster_size=5)

        self.assertIsInstance(result, FragmentCluster)
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(_n_pixels(result), 2)

    def test_empty_input_returns_none(self):
        empty = _make_candidates([], [], rate=512.0, layers=16, level=5)
        self.assertIsNone(self._run([empty]))

    def test_mra_path_does_not_call_native_supercluster(self):
        fine = _make_candidates([10], [4], rate=512.0, layers=16, level=5)
        coarse = _make_candidates([5], [8], rate=256.0, layers=16, level=4)
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "pycwb.modules.super_cluster_native.super_cluster":
                raise AssertionError("mra_hdbscan imported native supercluster")
            return real_import(name, *args, **kwargs)

        with patch("pycwb.modules.clustering.mra_hdbscan.impl._HDBSCAN", _FakeHDBSCAN):
            with patch("builtins.__import__", side_effect=guarded_import) as mocked_import:
                result = self._run([fine, coarse], min_cluster_size=2)

        self.assertIsInstance(result, FragmentCluster)
        imported_names = [call.args[0] for call in mocked_import.call_args_list if call.args]
        self.assertNotIn("pycwb.modules.super_cluster_native.super_cluster", imported_names)


if __name__ == "__main__":
    unittest.main()