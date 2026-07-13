"""Tests for pycwb.modules.workflow_utils.trigger_utils — trigger folder and saving."""
import pytest
from pycwb.modules.workflow_utils.trigger_utils import create_single_trigger_folder


class TestCreateSingleTriggerFolder:
    """Tests for create_single_trigger_folder — path construction."""

    def test_returns_correct_path_format(self):
        """Should return path with expected structure."""
        from unittest.mock import MagicMock
        job_seg = MagicMock()
        job_seg.index = 5
        job_seg.trial_idx = 2

        event = MagicMock()
        event.stop = [1234567890.0]
        event.hash_id = 98765

        path = create_single_trigger_folder(
            "/work", "triggers", job_seg, (event, None, None)
        )
        expected = "/work/triggers/trigger_5_2_1234567890.0_98765"
        assert path == expected

    def test_different_job_segment_values(self):
        """Path should reflect the given index and trial_idx."""
        from unittest.mock import MagicMock
        job_seg = MagicMock()
        job_seg.index = 0
        job_seg.trial_idx = 0

        event = MagicMock()
        event.stop = [100.0]
        event.hash_id = 1

        path = create_single_trigger_folder(
            "/base", "evts", job_seg, (event, None, None)
        )
        assert path == "/base/evts/trigger_0_0_100.0_1"

    def test_negative_stop_value(self):
        """Negative GPS stop time should appear in path."""
        from unittest.mock import MagicMock
        job_seg = MagicMock()
        job_seg.index = 1
        job_seg.trial_idx = 3

        event = MagicMock()
        event.stop = [-50.5]
        event.hash_id = 42

        path = create_single_trigger_folder(
            "/a", "b", job_seg, (event, None, None)
        )
        assert "-50.5" in path
        assert "42" in path
