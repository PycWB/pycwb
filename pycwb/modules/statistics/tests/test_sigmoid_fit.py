"""Unit tests for pycwb.modules.statistics.sigmoid_fit."""

import numpy as np
import pytest
from pycwb.modules.statistics.sigmoid_fit import logNfit, fit, estimate_hrss


class TestLogNfit:
    """Tests for the vectorized sigmoid function."""

    def test_vectorized_output(self):
        """logNfit should return an array for array input."""
        x = np.linspace(-25, -18, 100)  # log10(hrss) values
        result = logNfit(x, -22.0, 0.3, 1.0, 1.0, 0)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)

    def test_scalar_output(self):
        """logNfit returns a numpy array even for scalar input (vectorized)."""
        result = logNfit(-22.0, -22.0, 0.3, 1.0, 1.0, 0)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 0  # 0-d array for scalar input

    def test_range_zero_to_one(self):
        """Output should be in [0, 1] for reasonable inputs."""
        x = np.linspace(-25, -18, 200)  # log10(hrss)
        result = logNfit(x, -22.0, 0.3, 1.0, 1.0, 0)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_at_hrss50_par4_0(self):
        """At hrss50 with par4=0, efficiency should be 0.5."""
        # log10(hrss50) = -22.0, so x = -22.0 and par0 = -22.0
        result = logNfit(-22.0, -22.0, 0.3, 1.0, 1.0, 0)
        assert float(result) == pytest.approx(0.5, abs=1e-10)

    def test_at_hrss50_par4_1(self):
        """At hrss50 with par4=1, efficiency should also be 0.5."""
        result = logNfit(-22.0, -22.0, 0.3, 1.0, 1.0, 1)
        assert float(result) == pytest.approx(0.5, abs=1e-10)

    def test_monotonic_par4_0(self):
        """Efficiency should be monotonically increasing with hrss for par4=0."""
        x = np.linspace(-25, -18, 100)  # log10(hrss)
        result = logNfit(x, -22.0, 0.3, 1.0, 1.0, 0)
        assert np.all(np.diff(result) >= 0)

    def test_high_hrss_near_one(self):
        """At very high hrss (large log10 x), efficiency should approach 1."""
        # With par1=0.08 (min sigma), the sigmoid is steeper and approaches 1 faster
        result = logNfit(-10.0, -22.0, 0.08, 1.0, 2.5, 0)
        assert float(result) > 0.99

    def test_low_hrss_near_zero(self):
        """At very low hrss (very negative log10 x), efficiency should approach 0."""
        result = logNfit(-30.0, -22.0, 0.3, 1.0, 1.0, 0)
        assert float(result) < 0.01


class TestFit:
    """Tests for the Minuit-based sigmoid fitting."""

    def test_fit_perfect_sigmoid(self):
        """Fit should recover parameters from synthetic sigmoid data."""
        # Generate data from known sigmoid (x in log10 space)
        xdata = np.linspace(-25, -18, 50)
        true_params = (-22.0, 0.3, 1.0, 1.0, 0)
        ydata = logNfit(xdata, *true_params)
        # Add tiny noise to avoid perfect-fit edge cases
        rng = np.random.RandomState(42)
        ydata = ydata + rng.normal(0, 1e-6, size=len(ydata))

        result = fit(xdata, ydata)

        chi2, hrss50, hrssEr, sigma, betam, betap, flag = result
        assert chi2 < 0.1  # should be a good fit
        assert hrss50 == pytest.approx(10 ** true_params[0], rel=0.1)
        assert sigma == pytest.approx(true_params[1], rel=0.5)
        assert flag == 0  # par4=0 should win since we generated with par4=0

    def test_fit_returns_list_of_7(self):
        """fit() should return a list of 7 values."""
        xdata = np.linspace(-25, -18, 30)
        ydata = logNfit(xdata, -22.0, 0.3, 1.0, 1.0, 0)
        result = fit(xdata, ydata)
        assert len(result) == 7

    def test_fit_with_debug_false(self):
        """fit(debug=False) should not raise."""
        xdata = np.linspace(-25, -18, 30)
        ydata = logNfit(xdata, -22.0, 0.3, 1.0, 1.0, 0)
        result = fit(xdata, ydata, debug=False)
        assert result[1] > 0  # hrss50 should be positive


class TestEstimateHrss:
    """Tests for hrss estimation from fitted parameters."""

    def test_estimate_hrss50(self):
        """Estimating at 0.5 should recover hrss50."""
        xdata = np.linspace(-25, -18, 50)
        ydata = logNfit(xdata, -22.0, 0.3, 1.0, 1.0, 0)
        result = fit(xdata, ydata)
        hrss50, sigma, betam, betap, flag = result[1], result[3], result[4], result[5], result[6]

        estimated = estimate_hrss([hrss50, sigma, betam, betap, flag],
                                  xlim=(-25, -18), target_dp=0.5)
        assert estimated == pytest.approx(hrss50, rel=0.05)

    def test_estimate_out_of_range_returns_nan(self):
        """If target_dp is unreachable within xlim, should return NaN."""
        xdata = np.linspace(-25, -18, 50)
        ydata = logNfit(xdata, -22.0, 0.3, 1.0, 1.0, 0)
        result = fit(xdata, ydata)
        hrss50, sigma, betam, betap, flag = result[1], result[3], result[4], result[5], result[6]

        # Target 0.9999 is unreachable within xlim (-30, -25)
        estimated = estimate_hrss([hrss50, sigma, betam, betap, flag],
                                  xlim=(-30, -25), target_dp=0.9999)
        assert np.isnan(estimated)
