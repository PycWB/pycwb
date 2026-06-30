"""Tests for the additive multi-resolution weighted-graph backend."""

from __future__ import annotations

import unittest
import builtins
from unittest.mock import patch

import numpy as np

from pycwb.modules.clustering.entry_point import cluster_lag_candidates
from pycwb.types.network_cluster import FragmentCluster


N_IFO = 2


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


class TestMRAWeightedGraph(unittest.TestCase):
    METHOD = "mra_weighted_graph"

    def _run(self, candidates_list, **kwargs):
        try:
            return cluster_lag_candidates(candidates_list, method=self.METHOD, **kwargs)
        except ImportError:
            self.skipTest("scipy not installed")

    def test_coarse_pixel_bridges_disconnected_fine_pixels(self):
        fine = _make_candidates(
            [10, 12], [4, 4], rate=512.0, layers=16, level=5,
        )
        coarse = _make_candidates(
            [5], [8], rate=256.0, layers=16, level=4,
        )

        result = self._run(
            [fine, coarse],
            time_radius_bins=0.0,
            freq_radius_bins=0.0,
            resolution_penalty=0.0,
            min_edge_weight=0.01,
        )

        self.assertIsInstance(result, FragmentCluster)
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(_n_pixels(result), 3)

    def test_disabling_cross_resolution_edges_keeps_bridge_split(self):
        fine = _make_candidates(
            [10, 12], [4, 4], rate=512.0, layers=16, level=5,
        )
        coarse = _make_candidates(
            [5], [8], rate=256.0, layers=16, level=4,
        )

        result = self._run(
            [fine, coarse],
            time_radius_bins=0.0,
            freq_radius_bins=0.0,
            enable_cross_resolution_edges=False,
            min_edge_weight=0.01,
        )

        self.assertIsInstance(result, FragmentCluster)
        self.assertEqual(len(result.clusters), 3)
        self.assertEqual(_n_pixels(result), 3)

    def test_detector_energy_mismatch_suppresses_bridge(self):
        fine = _make_candidates(
            [10], [4], rate=512.0, layers=16, level=5,
            detector_energy=[[100.0, 0.0]],
        )
        coarse = _make_candidates(
            [5], [8], rate=256.0, layers=16, level=4,
            detector_energy=[[0.0, 100.0]],
        )

        result = self._run(
            [fine, coarse],
            time_radius_bins=0.0,
            freq_radius_bins=0.0,
            detector_similarity_weight=1.0,
            energy_similarity_weight=0.0,
            resolution_penalty=0.0,
            min_edge_weight=0.01,
        )

        self.assertIsInstance(result, FragmentCluster)
        self.assertEqual(len(result.clusters), 2)

    def test_mixed_rates_and_layers_survive_output_cluster(self):
        fine = _make_candidates(
            [10], [2], rate=512.0, layers=16, level=5,
        )
        coarse = _make_candidates(
            [5], [4], rate=256.0, layers=8, level=4,
        )

        result = self._run(
            [fine, coarse],
            time_radius_bins=0.0,
            freq_radius_bins=0.0,
            resolution_penalty=0.0,
            min_edge_weight=0.01,
        )

        self.assertIsInstance(result, FragmentCluster)
        self.assertEqual(len(result.clusters), 1)
        pa = result.clusters[0].pixel_arrays
        self.assertEqual(set(pa.rate.tolist()), {256.0, 512.0})
        self.assertEqual(set(pa.layers.tolist()), {8, 16})

    def test_empty_input_returns_none(self):
        empty = _make_candidates([], [], rate=512.0, layers=16, level=5)
        self.assertIsNone(self._run([empty]))

    def test_mra_path_does_not_call_native_supercluster(self):
        fine = _make_candidates(
            [10], [4], rate=512.0, layers=16, level=5,
        )
        coarse = _make_candidates(
            [5], [8], rate=256.0, layers=16, level=4,
        )

        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "pycwb.modules.super_cluster_native.super_cluster":
                raise AssertionError("mra_weighted_graph imported native supercluster")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import) as mocked_import:
            result = self._run(
                [fine, coarse],
                time_radius_bins=0.0,
                freq_radius_bins=0.0,
                resolution_penalty=0.0,
                min_edge_weight=0.01,
            )

        self.assertIsInstance(result, FragmentCluster)
        imported_names = [call.args[0] for call in mocked_import.call_args_list if call.args]
        self.assertNotIn("pycwb.modules.super_cluster_native.super_cluster", imported_names)


if __name__ == "__main__":
    unittest.main()