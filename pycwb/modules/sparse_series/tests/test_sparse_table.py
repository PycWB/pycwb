"""Tests for pycwb.modules.sparse_series — sparse table generation."""
import pytest
from unittest.mock import MagicMock, patch
from pycwb.modules.sparse_series.sparse_table import (
    _sparse_table_from_fragment_cluster,
    sparse_table_from_fragment_clusters,
)


def _make_config(**overrides):
    """Build a mock Config with required attributes."""
    defaults = {
        "nproc": 1,
        "nRES": 1,
        "nIFO": 2,
        "TDSize": 100,
        "max_delay": 0.01,
        "WDM_level": [0],
    }
    defaults.update(overrides)
    config = MagicMock()
    for k, v in defaults.items():
        setattr(config, k, v)
    return config


class TestSparseTableFromFragmentCluster:
    """Tests for _sparse_table_from_fragment_cluster — single-cluster helper."""

    @patch("pycwb.modules.sparse_series.sparse_table.create_wdm_for_level")
    @patch("pycwb.modules.sparse_series.sparse_table.SparseTimeFrequencySeries")
    def test_returns_list_of_sparse_series(self, MockSTFS, mock_create_wdm):
        """Should return one SparseTimeFrequencySeries per IFO."""
        config = _make_config(nIFO=3, WDM_level=[2])
        mock_wdm = MagicMock()
        mock_create_wdm.return_value = mock_wdm

        mock_tf_map = MagicMock()
        mock_fragment = MagicMock()

        mock_stfs = MagicMock()
        mock_stfs.from_fragment_cluster.return_value = mock_stfs
        MockSTFS.return_value = mock_stfs

        result = _sparse_table_from_fragment_cluster(
            (config, [mock_tf_map, mock_tf_map, mock_tf_map], 0, mock_fragment)
        )

        assert len(result) == 3
        assert mock_create_wdm.called

    @patch("pycwb.modules.sparse_series.sparse_table.create_wdm_for_level")
    @patch("pycwb.modules.sparse_series.sparse_table.SparseTimeFrequencySeries")
    def test_single_ifo(self, MockSTFS, mock_create_wdm):
        """Single IFO should return single-element list."""
        config = _make_config(nIFO=1, WDM_level=[0])
        mock_create_wdm.return_value = MagicMock()

        mock_stfs = MagicMock()
        mock_stfs.from_fragment_cluster.return_value = mock_stfs
        MockSTFS.return_value = mock_stfs

        result = _sparse_table_from_fragment_cluster(
            (config, [MagicMock()], 0, MagicMock())
        )
        assert len(result) == 1


class TestSparseTableFromFragmentClusters:
    """Tests for sparse_table_from_fragment_clusters — batch helper."""

    @patch("pycwb.modules.sparse_series.sparse_table._sparse_table_from_fragment_cluster")
    def test_sequential_mode(self, mock_single):
        """In sequential (non-parallel) mode, should call helper for each cluster."""
        config = _make_config(nRES=3)
        mock_single.return_value = [MagicMock()]

        tf_maps = [MagicMock(), MagicMock()]
        fragments = [MagicMock(), MagicMock(), MagicMock()]

        result = sparse_table_from_fragment_clusters(
            config, tf_maps, fragments, parallel=False
        )

        assert len(result) == 3
        assert mock_single.call_count == 3

    def test_empty_fragments_returns_empty(self):
        """Empty fragment list should return empty list."""
        config = _make_config()
        result = sparse_table_from_fragment_clusters(
            config, [MagicMock()], [], parallel=False
        )
        assert result == []
