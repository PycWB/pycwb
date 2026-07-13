"""Unit tests for pycwb.modules.statistics.eff_plot."""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for testing

from pycwb.modules.statistics.eff_plot import hrss50_bar_plot


class TestHrss50BarPlot:
    """Tests for the standalone hrss50 bar-plot function."""

    @pytest.fixture
    def sample_data(self):
        """Build a minimal dataset for plotting."""
        data_dict1 = {
            'SG4Q9': [1e-22, 1e-23],
            'SG5Q10': [2e-22, 2e-23],
            'SG4Q10': [1.5e-22, 1.5e-23],
        }
        data_dict2 = {
            'SG4Q9': [8e-23, 8e-24],
            'SG5Q10': [1.8e-22, 1.8e-23],
            'SG4Q10': [1.3e-22, 1.3e-23],
        }
        return [((data_dict1, 'run1'), (data_dict2, 'run2'))]

    def test_plot_runs_without_error(self, tmp_path, sample_data):
        """Plot should run without raising an exception."""
        output_dir = str(tmp_path)
        data_sets = sample_data[0]
        # Should not raise
        hrss50_bar_plot(
            list(data_sets),
            output_dir=output_dir,
            filename='test_hrss50.png'
        )
        # Check file was created
        import os
        assert os.path.exists(os.path.join(output_dir, 'test_hrss50.png'))

    def test_plot_with_wf_selection(self, tmp_path, sample_data):
        """Plot with explicit waveform selection should work."""
        output_dir = str(tmp_path)
        data_sets = sample_data[0]
        hrss50_bar_plot(
            list(data_sets),
            wf_selections=['SG4Q9', 'SG5Q10'],
            output_dir=output_dir,
            filename='test_hrss50_subset.png'
        )
        import os
        assert os.path.exists(os.path.join(output_dir, 'test_hrss50_subset.png'))

    def test_plot_accepts_empty_data(self, tmp_path):
        """Plot with empty dataset should not crash."""
        empty_dict = {}
        hrss50_bar_plot(
            [({}, 'empty_run')],
            wf_selections=[],
            output_dir=str(tmp_path),
            filename='test_empty.png'
        )
        import os
        assert os.path.exists(os.path.join(str(tmp_path), 'test_empty.png'))
