"""Unit tests for the Phase 3 clustering entry point and backends.

All tests use synthetic :class:`~pycwb.types.pixel_arrays.PixelArrays`
and :class:`~pycwb.types.network_cluster.FragmentCluster` objects so that
no real cWB data or a full pipeline run is required.

Test plan
---------
1.  Empty input is handled without error by the entry point.
2.  ``connected_components`` backend is a true identity pass.
3.  Unknown method raises :class:`ValueError` with a clear message.
4.  ``weighted_graph`` backend returns valid FragmentCluster objects.
5.  ``weighted_graph`` backend output passes pool_accepted_pixels.
6.  ``weighted_graph`` backend preserves total pixel count (tight cluster).
7.  ``weighted_graph`` backend splits disconnected pixels.
8.  ``weighted_graph`` backend splits by energy balance.
9.  ``weighted_graph`` backend respects min_pixels.
10. Multiple resolutions handled correctly.
11. Rejected clusters preserved by weighted_graph.
12-17. DBSCAN backend: basic validity, connectivity, splits, noise, min_pixels, rejected.
18-23. HDBSCAN backend: basic validity, connectivity, min_cluster_size fallback, noise, rejected.
24-29. OPTICS backend: basic validity, connectivity, splits, noise, min_pixels, rejected.
"""

import unittest
import numpy as np

from pycwb.types.pixel_arrays import PixelArrays
from pycwb.types.network_cluster import Cluster, ClusterMeta, FragmentCluster
from pycwb.modules.clustering.cluster import cluster_fragment_clusters
from pycwb.modules.clustering.common import pool_accepted_pixels


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

N_IFO = 2


def _make_pixel_arrays(time_bins, freq_bins, likelihood=None, asnr_ratio=None):
    """Build a synthetic :class:`PixelArrays` from coordinate lists.

    Parameters
    ----------
    time_bins  : list[int]
    freq_bins  : list[int]
    likelihood : list[float] | None  — defaults to 1.0 per pixel
    asnr_ratio : list[float] | None  — fraction of energy in IFO 0
                                        (0→all IFO1, 1→all IFO0).
                                        Defaults to 0.5 (balanced).
    """
    n_pix = len(time_bins)
    likelihood = likelihood or [1.0] * n_pix
    asnr_ratio = asnr_ratio or [0.5] * n_pix

    asnr = np.zeros((N_IFO, n_pix), dtype=np.float32)
    for k, r in enumerate(asnr_ratio):
        # Energy proportional to r vs (1-r); asnr ≡ sqrt(energy) here
        asnr[0, k] = np.sqrt(r)
        asnr[1, k] = np.sqrt(1.0 - r)

    return PixelArrays.from_arrays(
        time        = np.array(time_bins,  dtype=np.int32),
        frequency   = np.array(freq_bins,  dtype=np.int32),
        layers      = np.ones(n_pix,       dtype=np.int32),
        rate        = np.ones(n_pix,       dtype=np.float32) * 512.0,
        noise_rms   = np.ones((N_IFO, n_pix), dtype=np.float32),
        pixel_index = np.zeros((N_IFO, n_pix), dtype=np.int32),
        n_ifo       = N_IFO,
        likelihood  = np.array(likelihood, dtype=np.float32),
        asnr        = asnr,
        a_90        = asnr.copy(),
    )


def _make_cluster(pa):
    return Cluster(
        pixel_arrays=pa,
        cluster_meta=ClusterMeta(energy=float(pa.likelihood.sum())),
        cluster_status=0,
    )


def _make_fragment_cluster(clusters):
    fc = FragmentCluster(
        rate=512.0, start=0.0, stop=1200.0,
        f_low=16.0, f_high=1024.0,
    )
    fc.clusters = clusters
    return fc


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEntryPointEmptyInput(unittest.TestCase):
    def test_empty_list_returns_empty(self):
        result = cluster_fragment_clusters([], method="connected_components")
        self.assertEqual(result, [])

    def test_empty_list_weighted_graph(self):
        result = cluster_fragment_clusters([], method="weighted_graph")
        self.assertEqual(result, [])


class TestConnectedComponentsIdentity(unittest.TestCase):
    def _make_simple_fragment(self):
        pa = _make_pixel_arrays([0, 1, 2], [0, 0, 0])
        fc = _make_fragment_cluster([_make_cluster(pa)])
        return [fc]

    def test_returns_same_object(self):
        fcs = self._make_simple_fragment()
        result = cluster_fragment_clusters(fcs, method="connected_components")
        # Identity pass: the returned list IS the same object
        self.assertIs(result, fcs)

    def test_cluster_count_unchanged(self):
        fcs = self._make_simple_fragment()
        result = cluster_fragment_clusters(fcs, method="connected_components")
        self.assertEqual(len(result[0].clusters), len(fcs[0].clusters))

    def test_pixel_count_unchanged(self):
        fcs = self._make_simple_fragment()
        n_before = sum(len(c.pixel_arrays) for c in fcs[0].clusters)
        result = cluster_fragment_clusters(fcs, method="connected_components")
        n_after = sum(len(c.pixel_arrays) for c in result[0].clusters)
        self.assertEqual(n_before, n_after)


class TestEntryPointUnknownMethod(unittest.TestCase):
    def test_raises_value_error(self):
        pa = _make_pixel_arrays([0], [0])
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        with self.assertRaises(ValueError) as ctx:
            cluster_fragment_clusters(fcs, method="nonexistent_method")
        self.assertIn("nonexistent_method", str(ctx.exception))
        self.assertIn("connected_components", str(ctx.exception))


class TestWeightedGraphBasic(unittest.TestCase):
    def _run_wg(self, fragment_clusters, **kwargs):
        return cluster_fragment_clusters(
            fragment_clusters, method="weighted_graph", **kwargs
        )

    def test_returns_list_of_fragment_clusters(self):
        pa = _make_pixel_arrays([0, 1, 2], [0, 0, 0])
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run_wg(fcs)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], FragmentCluster)

    def test_fragment_metadata_preserved(self):
        pa = _make_pixel_arrays([0, 1], [0, 0])
        fc = _make_fragment_cluster([_make_cluster(pa)])
        fc.rate = 1024.0
        fc.f_low = 32.0
        fc.f_high = 512.0
        result = self._run_wg([fc])
        self.assertAlmostEqual(result[0].rate,  1024.0)
        self.assertAlmostEqual(result[0].f_low,  32.0)
        self.assertAlmostEqual(result[0].f_high, 512.0)

    def test_pool_accepted_pixels_works_on_output(self):
        pa = _make_pixel_arrays([0, 1, 2], [5, 5, 6])
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run_wg(fcs)
        # Should not raise
        pooled, origin = pool_accepted_pixels(result[0])
        self.assertGreater(len(pooled), 0)

    def test_empty_fragment_cluster(self):
        fc = _make_fragment_cluster([])
        result = self._run_wg([fc])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].clusters), 0)


class TestWeightedGraphConnectivity(unittest.TestCase):
    def _run_wg(self, fcs, **kwargs):
        return cluster_fragment_clusters(fcs, method="weighted_graph", **kwargs)

    def test_preserves_pixel_count_tight_cluster(self):
        """Pixels within radius 1 of each other should form one component."""
        # 3 pixels in a line, adjacent in time
        pa = _make_pixel_arrays(
            [10, 11, 12], [20, 20, 20],
            asnr_ratio=[0.5, 0.5, 0.5],   # identical balance → no pruning
        )
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run_wg(fcs, time_radius_bins=1, freq_radius_bins=1, min_edge_weight=0.01)
        n_out = sum(len(c.pixel_arrays) for c in result[0].clusters if c.cluster_status <= 0)
        self.assertEqual(n_out, 3)

    def test_splits_disconnected_pixels(self):
        """Pixels far apart in TF space should form separate components."""
        # Two groups of pixels with a large gap between them
        pa = _make_pixel_arrays(
            [0, 1,  100, 101],
            [0, 0,  0,   0],
        )
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run_wg(fcs, time_radius_bins=2, freq_radius_bins=2, min_edge_weight=0.01)
        n_clusters = sum(1 for c in result[0].clusters if c.cluster_status <= 0)
        self.assertEqual(n_clusters, 2)

    def test_splits_by_energy_balance(self):
        """Two adjacent pixels with very different energy balance should be split.

        We set energy_balance_wt very high and min_edge_weight just above the
        expected attenuated weight so the edge is pruned.
        """
        # Two pixels right next to each other in TF but with opposite balance
        pa = _make_pixel_arrays(
            [10, 11],
            [20, 20],
            asnr_ratio=[0.95, 0.05],   # IFO0-dominated vs IFO1-dominated
        )
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        # Balance diff = 0.90, exp(-50 * 0.90) ≈ 0, so weight → 0 → pruned
        result = self._run_wg(
            fcs,
            time_radius_bins=2,
            freq_radius_bins=2,
            energy_balance_wt=50.0,
            min_edge_weight=0.05,
        )
        n_clusters = sum(1 for c in result[0].clusters if c.cluster_status <= 0)
        self.assertEqual(n_clusters, 2)

    def test_min_pixels_drops_small_components(self):
        """Components with fewer than min_pixels should be discarded."""
        # Two isolated single pixels
        pa = _make_pixel_arrays([0, 100], [0, 0])
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run_wg(
            fcs,
            time_radius_bins=1,
            min_edge_weight=0.01,
            min_pixels=2,   # single-pixel components are dropped
        )
        n_clusters = sum(1 for c in result[0].clusters if c.cluster_status <= 0)
        self.assertEqual(n_clusters, 0)

    def test_multiple_resolutions(self):
        """Entry point handles a list with multiple FragmentClusters."""
        pa1 = _make_pixel_arrays([0, 1], [0, 0])
        pa2 = _make_pixel_arrays([5, 6], [10, 10])
        fcs = [
            _make_fragment_cluster([_make_cluster(pa1)]),
            _make_fragment_cluster([_make_cluster(pa2)]),
        ]
        result = cluster_fragment_clusters(fcs, method="weighted_graph")
        self.assertEqual(len(result), 2)
        for fc in result:
            self.assertIsInstance(fc, FragmentCluster)


class TestWeightedGraphRejectedClusters(unittest.TestCase):
    def test_rejected_clusters_preserved(self):
        """Clusters with cluster_status > 0 are kept as-is in the output."""
        pa_good = _make_pixel_arrays([0, 1, 2], [0, 0, 0])
        pa_bad  = _make_pixel_arrays([50, 51], [0, 0])

        c_good = _make_cluster(pa_good)
        c_bad  = _make_cluster(pa_bad)
        c_bad.cluster_status = 1   # rejected

        fcs = [_make_fragment_cluster([c_good, c_bad])]
        result = cluster_fragment_clusters(fcs, method="weighted_graph")

        rejected = [c for c in result[0].clusters if c.cluster_status > 0]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(len(rejected[0].pixel_arrays), 2)


# ─────────────────────────────────────────────────────────────────────────────
# DBSCAN backend tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDBSCANBasic(unittest.TestCase):
    def _run(self, fcs, **kwargs):
        return cluster_fragment_clusters(fcs, method="dbscan", **kwargs)

    def test_returns_fragment_clusters(self):
        pa = _make_pixel_arrays([0, 1, 2], [0, 0, 0])
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run(fcs)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], FragmentCluster)

    def test_empty_input(self):
        result = cluster_fragment_clusters([], method="dbscan")
        self.assertEqual(result, [])

    def test_empty_fragment_cluster(self):
        fc = _make_fragment_cluster([])
        result = self._run([fc])
        self.assertEqual(len(result[0].clusters), 0)

    def test_pool_works_on_output(self):
        pa = _make_pixel_arrays([0, 1, 2], [5, 5, 5])
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run(fcs)
        pooled, _ = pool_accepted_pixels(result[0])
        self.assertGreater(len(pooled), 0)

    def test_tight_cluster_preserved(self):
        """Adjacent pixels should land in one cluster with default eps."""
        pa = _make_pixel_arrays([10, 11, 12], [20, 20, 20])
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run(fcs)
        n_accepted = sum(1 for c in result[0].clusters if c.cluster_status <= 0)
        # All 3 pixels should be in one cluster (or possibly singletons if
        # eps is too small, but with default eps=1.2 they connect).
        n_pix_out = sum(len(c.pixel_arrays) for c in result[0].clusters if c.cluster_status <= 0)
        self.assertEqual(n_pix_out, 3)

    def test_splits_disconnected_pixels(self):
        """Pixels far apart in TF space should form separate clusters."""
        pa = _make_pixel_arrays([0, 1, 100, 101], [0, 0, 0, 0])
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run(fcs)
        n_clusters = sum(1 for c in result[0].clusters if c.cluster_status <= 0)
        self.assertGreaterEqual(n_clusters, 2)

    def test_min_pixels_drops_small(self):
        """Single-pixel clusters should be dropped when min_pixels=2."""
        # Two isolated pixels (very far apart → separate singleton clusters)
        pa = _make_pixel_arrays([0, 1000], [0, 0])
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run(fcs, min_samples=2, noise_as_singletons=False, min_pixels=2)
        # With min_samples=2, both isolated pixels are noise → discarded
        n_pix_out = sum(len(c.pixel_arrays) for c in result[0].clusters if c.cluster_status <= 0)
        self.assertEqual(n_pix_out, 0)

    def test_rejected_clusters_preserved(self):
        pa_good = _make_pixel_arrays([0, 1], [0, 0])
        pa_bad  = _make_pixel_arrays([50, 51], [0, 0])
        c_good = _make_cluster(pa_good)
        c_bad  = _make_cluster(pa_bad)
        c_bad.cluster_status = 1
        fcs = [_make_fragment_cluster([c_good, c_bad])]
        result = self._run(fcs)
        rejected = [c for c in result[0].clusters if c.cluster_status > 0]
        self.assertEqual(len(rejected), 1)

    def test_metadata_preserved(self):
        pa = _make_pixel_arrays([0, 1], [0, 0])
        fc = _make_fragment_cluster([_make_cluster(pa)])
        fc.rate = 2048.0
        fc.f_low = 64.0
        result = self._run([fc])
        self.assertAlmostEqual(result[0].rate, 2048.0)
        self.assertAlmostEqual(result[0].f_low, 64.0)


# ─────────────────────────────────────────────────────────────────────────────
# HDBSCAN backend tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHDBSCANBasic(unittest.TestCase):
    def _run(self, fcs, **kwargs):
        return cluster_fragment_clusters(fcs, method="hdbscan", **kwargs)

    def test_returns_fragment_clusters(self):
        pa = _make_pixel_arrays([0, 1, 2, 3], [0, 0, 0, 0])
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run(fcs, min_cluster_size=2)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], FragmentCluster)

    def test_empty_input(self):
        result = cluster_fragment_clusters([], method="hdbscan")
        self.assertEqual(result, [])

    def test_empty_fragment_cluster(self):
        fc = _make_fragment_cluster([])
        result = self._run([fc])
        self.assertEqual(len(result[0].clusters), 0)

    def test_too_few_pixels_returns_one_cluster(self):
        """When n_pix < min_cluster_size, pixels kept as one cluster."""
        pa = _make_pixel_arrays([0], [0])
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run(fcs, min_cluster_size=5)
        n_pix_out = sum(len(c.pixel_arrays) for c in result[0].clusters if c.cluster_status <= 0)
        self.assertEqual(n_pix_out, 1)

    def test_tight_cluster_preserved(self):
        """A dense row of pixels should land in one cluster."""
        pa = _make_pixel_arrays([10, 11, 12, 13, 14], [20, 20, 20, 20, 20])
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run(fcs, min_cluster_size=2)
        n_pix_out = sum(len(c.pixel_arrays) for c in result[0].clusters if c.cluster_status <= 0)
        self.assertEqual(n_pix_out, 5)

    def test_splits_disconnected_pixels(self):
        """Pixel groups far apart in TF should form separate clusters."""
        pa = _make_pixel_arrays([0, 1, 100, 101], [0, 0, 0, 0])
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run(fcs, min_cluster_size=2)
        n_clusters = sum(1 for c in result[0].clusters if c.cluster_status <= 0)
        self.assertGreaterEqual(n_clusters, 2)

    def test_rejected_clusters_preserved(self):
        pa_good = _make_pixel_arrays([0, 1, 2, 3], [0, 0, 0, 0])
        pa_bad  = _make_pixel_arrays([50, 51], [0, 0])
        c_good = _make_cluster(pa_good)
        c_bad  = _make_cluster(pa_bad)
        c_bad.cluster_status = 1
        fcs = [_make_fragment_cluster([c_good, c_bad])]
        result = self._run(fcs, min_cluster_size=2)
        rejected = [c for c in result[0].clusters if c.cluster_status > 0]
        self.assertEqual(len(rejected), 1)

    def test_metadata_preserved(self):
        pa = _make_pixel_arrays([0, 1, 2, 3], [0, 0, 0, 0])
        fc = _make_fragment_cluster([_make_cluster(pa)])
        fc.rate = 4096.0
        result = self._run([fc], min_cluster_size=2)
        self.assertAlmostEqual(result[0].rate, 4096.0)


# ─────────────────────────────────────────────────────────────────────────────
# OPTICS backend tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOPTICSBasic(unittest.TestCase):
    def _run(self, fcs, **kwargs):
        return cluster_fragment_clusters(fcs, method="optics", **kwargs)

    def test_returns_fragment_clusters(self):
        pa = _make_pixel_arrays([0, 1, 2, 3], [0, 0, 0, 0])
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run(fcs, min_samples=2)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], FragmentCluster)

    def test_empty_input(self):
        result = cluster_fragment_clusters([], method="optics")
        self.assertEqual(result, [])

    def test_empty_fragment_cluster(self):
        fc = _make_fragment_cluster([])
        result = self._run([fc])
        self.assertEqual(len(result[0].clusters), 0)

    def test_too_few_pixels_returns_one_cluster(self):
        """When n_pix < min_samples, pixels kept as one cluster."""
        pa = _make_pixel_arrays([0], [0])
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run(fcs, min_samples=3)
        n_pix_out = sum(len(c.pixel_arrays) for c in result[0].clusters if c.cluster_status <= 0)
        self.assertEqual(n_pix_out, 1)

    def test_tight_cluster_preserved(self):
        """Dense adjacent pixels should cluster together."""
        pa = _make_pixel_arrays([10, 11, 12, 13, 14], [20, 20, 20, 20, 20])
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run(fcs, min_samples=2)
        n_pix_out = sum(len(c.pixel_arrays) for c in result[0].clusters if c.cluster_status <= 0)
        self.assertEqual(n_pix_out, 5)

    def test_splits_disconnected_pixels(self):
        """Pixels far apart should form separate clusters."""
        pa = _make_pixel_arrays([0, 1, 100, 101], [0, 0, 0, 0])
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run(fcs, min_samples=2, max_eps=2.0)
        n_clusters = sum(1 for c in result[0].clusters if c.cluster_status <= 0)
        self.assertGreaterEqual(n_clusters, 2)

    def test_noise_as_singletons(self):
        """Noise pixels are kept as single-pixel clusters by default."""
        # With min_samples=3, isolated single pixels become noise
        pa = _make_pixel_arrays([0, 500], [0, 0])
        fcs = [_make_fragment_cluster([_make_cluster(pa)])]
        result = self._run(fcs, min_samples=2, noise_as_singletons=True)
        n_pix_out = sum(len(c.pixel_arrays) for c in result[0].clusters if c.cluster_status <= 0)
        self.assertEqual(n_pix_out, 2)

    def test_rejected_clusters_preserved(self):
        pa_good = _make_pixel_arrays([0, 1, 2, 3], [0, 0, 0, 0])
        pa_bad  = _make_pixel_arrays([50, 51], [0, 0])
        c_good = _make_cluster(pa_good)
        c_bad  = _make_cluster(pa_bad)
        c_bad.cluster_status = 1
        fcs = [_make_fragment_cluster([c_good, c_bad])]
        result = self._run(fcs, min_samples=2)
        rejected = [c for c in result[0].clusters if c.cluster_status > 0]
        self.assertEqual(len(rejected), 1)

    def test_metadata_preserved(self):
        pa = _make_pixel_arrays([0, 1, 2, 3], [0, 0, 0, 0])
        fc = _make_fragment_cluster([_make_cluster(pa)])
        fc.f_high = 512.0
        result = self._run([fc], min_samples=2)
        self.assertAlmostEqual(result[0].f_high, 512.0)


if __name__ == "__main__":
    unittest.main()
