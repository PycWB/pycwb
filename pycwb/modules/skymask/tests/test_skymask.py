"""Tests for pycwb.modules.skymask — angular distance and coordinate conversion."""
import numpy as np
import pytest
from pycwb.modules.skymask.skymask import _angular_distance_deg, _cwb_to_geo


class TestAngularDistance:
    """Tests for _angular_distance_deg — spherical angular separation."""

    def test_same_point_zero_distance(self):
        """Angular distance from a point to itself should be 0 (within tolerance)."""
        d = _angular_distance_deg(45.0, 30.0, 45.0, 30.0)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_antipodal_points(self):
        """Antipodal points should be ~180 degrees apart."""
        d = _angular_distance_deg(0.0, 0.0, 180.0, 0.0)
        assert d == pytest.approx(180.0, abs=1e-6)

    def test_north_to_south_pole(self):
        """Distance from North Pole to South Pole along a meridian."""
        d = _angular_distance_deg(0.0, 90.0, 0.0, -90.0)
        assert d == pytest.approx(180.0, abs=1e-6)

    def test_equator_90_degrees(self):
        """Two points separated by 90 degrees on the equator."""
        d = _angular_distance_deg(0.0, 0.0, 90.0, 0.0)
        assert d == pytest.approx(90.0, abs=1e-6)

    def test_known_distance(self):
        """Paris (48.9N, 2.3E) to New York (40.7N, 74.0W) ~5837 km ~52.5 deg."""
        d = _angular_distance_deg(2.3, 48.9, -74.0, 40.7)
        assert 52.0 < d < 53.0

    def test_array_input(self):
        """Should work with numpy array inputs (scalar or broadcast)."""
        d = _angular_distance_deg(
            np.array([0.0, 10.0]),
            np.array([0.0, 0.0]),
            np.array([90.0, 100.0]),
            np.array([0.0, 0.0]),
        )
        assert d.shape == (2,)
        assert d[0] == pytest.approx(90.0, abs=1e-6)

    def test_clips_cos_to_valid_range(self):
        """Numerical noise should not produce NaN — cos_d is clipped to [-1,1]."""
        d = _angular_distance_deg(0.0, 0.0, 0.0, 0.0)
        assert not np.isnan(d)
        assert d >= 0


class TestCwbToGeo:
    """Tests for _cwb_to_geo — cWB coordinate conversion."""

    def test_phi_below_180(self):
        """Phi <= 180 stays the same, theta becomes 90 - theta."""
        phi_geo, theta_geo = _cwb_to_geo(45.0, 30.0)
        assert phi_geo == pytest.approx(45.0)
        assert theta_geo == pytest.approx(60.0)

    def test_phi_above_180_wraps(self):
        """Phi > 180 wraps to negative (phi - 360)."""
        phi_geo, theta_geo = _cwb_to_geo(200.0, 10.0)
        assert phi_geo == pytest.approx(-160.0)
        assert theta_geo == pytest.approx(80.0)

    def test_phi_exactly_180(self):
        """Exactly 180 stays 180."""
        phi_geo, theta_geo = _cwb_to_geo(180.0, 0.0)
        assert phi_geo == pytest.approx(180.0)
        assert theta_geo == pytest.approx(90.0)

    def test_phi_zero(self):
        """Zero phi stays zero."""
        phi_geo, theta_geo = _cwb_to_geo(0.0, 45.0)
        assert phi_geo == pytest.approx(0.0)
        assert theta_geo == pytest.approx(45.0)

    def test_negative_theta(self):
        """Negative theta (southern hemisphere)."""
        phi_geo, theta_geo = _cwb_to_geo(100.0, -20.0)
        assert phi_geo == pytest.approx(100.0)
        assert theta_geo == pytest.approx(110.0)

    def test_array_inputs(self):
        """Array inputs should produce array outputs."""
        phi_geo, theta_geo = _cwb_to_geo(
            np.array([50.0, 200.0, 180.0]),
            np.array([30.0, 10.0, 0.0]),
        )
        assert phi_geo.shape == (3,)
        assert theta_geo.shape == (3,)
        assert phi_geo[0] == pytest.approx(50.0)
        assert phi_geo[1] == pytest.approx(-160.0)
        assert phi_geo[2] == pytest.approx(180.0)
