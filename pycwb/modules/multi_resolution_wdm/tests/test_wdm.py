"""Tests for pycwb.modules.multi_resolution_wdm — WDM creation and validation."""
import pytest
from unittest.mock import MagicMock, patch
from pycwb.modules.multi_resolution_wdm.wdm import create_wdm_set, create_wdm_for_level


def _make_config(**overrides):
    """Build a mock Config with required attributes."""
    defaults = {
        "rateANA": 4096,
        "segEdge": 8.0,
        "TDSize": 100,
        "l_high": 7,
        "l_low": 0,
        "WDM_beta_order": 2,
        "WDM_precision": 4,
        "nRES": 8,
        "nIFO": 2,
        "max_delay": 0.01,
        "WDM_level": [0, 1, 2, 3, 4, 5, 6, 7],
    }
    defaults.update(overrides)
    config = MagicMock()
    for k, v in defaults.items():
        setattr(config, k, v)
    return config


class TestCreateWdmForLevel:
    """Tests for create_wdm_for_level — single-level WDM factory + validation."""

    @patch("pycwb.modules.multi_resolution_wdm.wdm.WDM")
    def test_creates_wdm_for_positive_level(self, MockWDM):
        """Positive level should create WDM with layers=2**level."""
        config = _make_config()
        # Set m_H to a value that makes wdmFLen < segEdge (so no error)
        mock_wdm = MagicMock()
        mock_wdm.m_H = 100.0  # wdmFLen = 100/4096 ≈ 0.024 < segEdge=8.0
        MockWDM.return_value = mock_wdm

        wdm = create_wdm_for_level(config, level=3)
        MockWDM.assert_called_once_with(8, 8, 2, 4)  # layers=2^3=8
        assert wdm is mock_wdm

    @patch("pycwb.modules.multi_resolution_wdm.wdm.WDM")
    def test_level_zero_has_zero_layers(self, MockWDM):
        """Level 0 should have layers=0."""
        config = _make_config()
        mock_wdm = MagicMock()
        mock_wdm.m_H = 4.0  # wdmFLen = 4/4096 < segEdge
        MockWDM.return_value = mock_wdm

        wdm = create_wdm_for_level(config, level=0)
        MockWDM.assert_called_once_with(0, 0, 2, 4)

    @patch("pycwb.modules.multi_resolution_wdm.wdm.WDM")
    def test_filter_too_long_raises(self, MockWDM):
        """When WDM filter length > segEdge, ValueError should be raised."""
        config = _make_config(rateANA=100, segEdge=0.001)  # tiny segEdge

        # Mock WDM so m_H is large
        mock_wdm = MagicMock()
        mock_wdm.m_H = 1000.0  # filter length = 1000/100 = 10 sec > 0.001 segEdge
        MockWDM.return_value = mock_wdm

        with pytest.raises(ValueError, match="Filter length"):
            create_wdm_for_level(config, level=0)

    @patch("pycwb.modules.multi_resolution_wdm.wdm.WDM")
    def test_segEdge_too_small_for_td_raises(self, MockWDM):
        """When segEdge < 1.5 * TD length, ValueError should be raised."""
        # For level=0: rate = 100, TDSize/rate = 200/100 = 2.0
        # 1.5 * 2.0 = 3.0, int(3.0 + 0.5) = 3
        # segEdge=2 < 3 → should raise
        config = _make_config(rateANA=100, segEdge=2.0, TDSize=200)

        mock_wdm = MagicMock()
        mock_wdm.m_H = 0.1  # wdmFLen = 0.1/100 = 0.001 < segEdge (passes filter check)
        MockWDM.return_value = mock_wdm

        with pytest.raises(ValueError, match="segEdge must be"):
            create_wdm_for_level(config, level=0)


class TestCreateWdmSet:
    """Tests for create_wdm_set — multi-level WDM list builder."""

    @patch("pycwb.modules.multi_resolution_wdm.wdm.create_wdm_for_level")
    def test_creates_correct_number_of_wdms(self, mock_create):
        """Should create (l_high - l_low + 1) WDMs."""
        config = _make_config(l_high=3, l_low=0)
        mock_create.return_value = MagicMock()

        result = create_wdm_set(config)
        assert len(result) == 4  # levels 3, 2, 1, 0
        assert mock_create.call_count == 4

    @patch("pycwb.modules.multi_resolution_wdm.wdm.create_wdm_for_level")
    def test_levels_are_reversed(self, mock_create):
        """Levels should go from l_high down to l_low."""
        config = _make_config(l_high=5, l_low=2)
        mock_create.return_value = MagicMock()

        create_wdm_set(config)
        # Should be called with levels 5, 4, 3, 2 (in that order)
        calls = [call.args[1] for call in mock_create.call_args_list]
        assert calls == [5, 4, 3, 2]

    @patch("pycwb.modules.multi_resolution_wdm.wdm.create_wdm_for_level")
    def test_single_level(self, mock_create):
        """When l_high == l_low, only one WDM is created."""
        config = _make_config(l_high=2, l_low=2)
        mock_create.return_value = MagicMock()

        result = create_wdm_set(config)
        assert len(result) == 1
        mock_create.assert_called_once_with(config, 2)
