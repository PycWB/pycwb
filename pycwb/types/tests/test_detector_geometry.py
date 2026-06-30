"""Unit tests for :mod:`pycwb.types.detector` geometry functions.

Tests cover all pure-geometry methods and functions:
- ``Detector`` init, properties, coordinate transforms, antenna patterns
- ``DetectorNetwork`` init, parsing, antenna pattern computation
- Module-level functions (GMST, arm response, earth-centered vectors, etc.)
"""

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from pycwb.types.detector import (
    Detector,
    DetectorNetwork,
    _build_sky_directions,
    calculate_e2or_from_acore,
    compute_sky_delay_and_patterns,
    earth_centered_vectors,
    get_max_delay,
    gmst_accurate,
    single_arm_frequency_response,
)
from pycwb.utils import geometry as geometry_utils
from pycwb.utils import network as network_utils
from pycwb.utils.geometry import (
    cartesian_to_spherical,
    local_to_earth_centered,
    spherical_to_cartesian,
)


# ---------------------------------------------------------------------------
# Leaf geometry utilities
# ---------------------------------------------------------------------------

class TestGeometryUtilities:
    """Direct tests for :mod:`pycwb.utils.geometry`."""

    @pytest.mark.parametrize(
        ("lat", "lon", "local_vec", "expected"),
        [
            (0.0, 0.0, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            (0.0, 0.0, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (0.0, 0.0, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
            (0.0, math.pi / 2, (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        ],
    )
    def test_local_to_earth_centered_basis(self, lat, lon, local_vec, expected):
        east, north, up = local_vec
        result = local_to_earth_centered(lat, lon, east, north, up)
        assert result == pytest.approx(np.array(expected), abs=1e-12)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-12)

    def test_local_to_earth_centered_normalizes_non_unit_input(self):
        result = local_to_earth_centered(0.0, 0.0, 2.0, 0.0, 0.0)
        assert result == pytest.approx(np.array([0.0, 1.0, 0.0]), abs=1e-12)

    @pytest.mark.parametrize(
        ("ra", "dec", "expected"),
        [
            (0.0, 0.0, (1.0, 0.0, 0.0)),
            (math.pi / 2, 0.0, (0.0, 1.0, 0.0)),
            (0.0, math.pi / 2, (0.0, 0.0, 1.0)),
            (0.0, -math.pi / 2, (0.0, 0.0, -1.0)),
        ],
    )
    def test_spherical_to_cartesian_cardinal_directions(self, ra, dec, expected):
        result = spherical_to_cartesian(ra, dec)
        assert result == pytest.approx(np.array(expected), abs=1e-12)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize(
        ("coords", "expected_ra", "expected_dec"),
        [
            ((1.0, 0.0, 0.0), 0.0, 0.0),
            ((0.0, 1.0, 0.0), math.pi / 2, 0.0),
            ((0.0, 0.0, 2.0), 0.0, math.pi / 2),
            ((0.0, 0.0, -2.0), 0.0, -math.pi / 2),
        ],
    )
    def test_cartesian_to_spherical_cardinal_directions(
        self, coords, expected_ra, expected_dec
    ):
        ra, dec = cartesian_to_spherical(*coords)
        assert ra == pytest.approx(expected_ra, abs=1e-12)
        assert dec == pytest.approx(expected_dec, abs=1e-12)

    @pytest.mark.parametrize(
        ("ra", "dec"),
        [
            (0.4, -0.7),
            (2.1, 0.3),
            (5.7, 1.2),
        ],
    )
    def test_spherical_cartesian_round_trip(self, ra, dec):
        x, y, z = spherical_to_cartesian(ra, dec)
        ra_back, dec_back = cartesian_to_spherical(x, y, z)
        ra_delta = (ra_back - ra + math.pi) % (2 * math.pi) - math.pi
        assert ra_delta == pytest.approx(0.0, abs=1e-12)
        assert dec_back == pytest.approx(dec, abs=1e-12)

    def test_network_coordinate_helpers_are_reexports(self):
        assert network_utils.local_to_earth_centered is geometry_utils.local_to_earth_centered
        assert network_utils.spherical_to_cartesian is geometry_utils.spherical_to_cartesian
        assert network_utils.cartesian_to_spherical is geometry_utils.cartesian_to_spherical


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def h1():
    return Detector("H1")


@pytest.fixture
def l1():
    return Detector("L1")


@pytest.fixture
def v1():
    return Detector("V1")


@pytest.fixture
def h1l1_network(h1, l1):
    return DetectorNetwork(detectors=[h1, l1])


@pytest.fixture
def h1l1v1_network(h1, l1, v1):
    return DetectorNetwork(detectors=[h1, l1, v1])


# ---------------------------------------------------------------------------
# Detector — init & properties
# ---------------------------------------------------------------------------

class TestDetectorInit:
    """Tests for Detector construction and basic properties."""

    def test_init_from_name_h1(self, h1):
        assert h1.name == "H1"
        assert h1.full_name == "LHO_4k"
        assert isinstance(h1.latitude, float)
        assert isinstance(h1.longitude, float)

    def test_init_from_name_l1(self, l1):
        assert l1.name == "L1"
        assert l1.full_name == "LLO_4k"

    def test_init_from_name_v1(self, v1):
        assert v1.name == "V1"
        assert v1.full_name == "VIRGO"

    def test_init_unknown_name_raises(self):
        """Unknown name without explicit params should raise."""
        with pytest.raises((KeyError, AttributeError)):
            Detector("ZZ9")

    def test_init_explicit_params(self):
        d = Detector(
            name="TEST",
            full_name="Test Detector",
            latitude=0.5,
            longitude=-1.2,
            altitude=100.0,
            x_azimuth=1.0,
            x_altitude=0.0,
            x_midpoint=2000.0,
            y_azimuth=2.5,
            y_altitude=0.0,
            y_midpoint=2000.0,
        )
        assert d.name == "TEST"
        assert d.latitude == 0.5
        assert d.vertex_vec_earth_centered is not None

    def test_x_length(self, h1):
        assert h1.x_length == pytest.approx(h1.x_midpoint * 2)
        assert h1.x_length > 0

    def test_y_length(self, h1):
        assert h1.y_length == pytest.approx(h1.y_midpoint * 2)
        assert h1.y_length > 0

    def test_x_vec(self, h1):
        v = h1.x_vec
        assert isinstance(v, np.ndarray)
        assert v.shape == (3,)
        # Should be a unit vector (or close to it)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)

    def test_y_vec(self, h1):
        v = h1.y_vec
        assert isinstance(v, np.ndarray)
        assert v.shape == (3,)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)

    def test_vertex_vec(self, h1):
        v = h1.vertex_vec
        assert isinstance(v, np.ndarray)
        assert v.shape == (3,)
        # Should be at approximately Earth radius
        r = np.linalg.norm(v)
        assert 6.3e6 < r < 6.4e6  # Earth radius in meters

    def test_vertex_vec_earth_centered(self, h1):
        v = h1.vertex_vec_earth_centered
        assert isinstance(v, np.ndarray)
        assert v.shape == (3,)

    def test_response_tensors(self, h1):
        assert h1.response.shape == (3, 3)
        assert h1.x_response.shape == (3, 3)
        assert h1.y_response.shape == (3, 3)
        # Response tensor should be symmetric
        assert np.allclose(h1.response, h1.response.T)

    def test_h1_l1_different_positions(self, h1, l1):
        """H1 and L1 should be at different ECEF positions."""
        dist = np.linalg.norm(
            h1.vertex_vec_earth_centered - l1.vertex_vec_earth_centered
        )
        assert dist > 1000.0  # more than 1 km apart


# ---------------------------------------------------------------------------
# Detector — coordinate transforms
# ---------------------------------------------------------------------------

class TestDetectorCoordinates:
    """Tests for coordinate conversion methods."""

    def test_get_cartesian_components(self):
        """Known alt/az/lat/lon should produce unit vector."""
        v = Detector.get_cartesian_components(0.0, 0.0, 0.5, 0.0)
        assert v.shape == (3,)
        # Vertical arm at the equator pointing north → should be mainly in z
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)

    def test_geodetic_to_geocentric_equator(self):
        """At equator, geodetic ≈ geocentric."""
        x, y, z = Detector.geodetic_to_geocentric(0.0, 0.0, 0.0)
        assert z == pytest.approx(0.0, abs=1.0)

    def test_geodetic_to_geocentric_north_pole(self):
        """At north pole, x≈0, y≈0, z≈R_earth."""
        x, y, z = Detector.geodetic_to_geocentric(
            math.pi / 2, 0.0, 0.0
        )
        assert abs(x) < 1.0
        assert abs(y) < 1.0
        assert 6.3e6 < z < 6.4e6

    def test_time_rotated(self, h1):
        """Time rotation should change longitude."""
        rotated = h1.time_rotated(43200.0)  # 12 hours = 180 degrees
        assert rotated.longitude != h1.longitude
        # After 24h, should be back to ~same longitude (mod 2π)
        rotated_full = h1.time_rotated(86400.0)
        assert abs(
            (rotated_full.longitude % (2 * math.pi))
            - (h1.longitude % (2 * math.pi))
        ) < 1e-10

    def test_time_rotated_preserves_vertex(self, h1):
        """time_rotated changes longitude but pre-computed ECEF vectors
        are cached from __init__ and not recomputed."""
        rotated = h1.time_rotated(43200.0, name="H1_rotated")
        assert rotated.name == "H1_rotated"
        # Longitude should change significantly
        assert abs(rotated.longitude - h1.longitude) > 1.0


# ---------------------------------------------------------------------------
# Detector — arm endpoint geometry
# ---------------------------------------------------------------------------

class TestDetectorArmEndpoints:
    def test_x_arm_endpoint_shape(self, h1):
        ep = h1.get_x_arm_endpoint_in_geo(100)
        assert ep.shape == (2,)
        # Latitude should be within [-π/2, π/2]
        assert -math.pi / 2 <= ep[1] <= math.pi / 2

    def test_y_arm_endpoint_shape(self, h1):
        ep = h1.get_y_arm_endpoint_in_geo(100)
        assert ep.shape == (2,)

    def test_arms_are_different(self, h1):
        """X and Y arm endpoints should point in different directions."""
        x_ep = h1.get_x_arm_endpoint_in_geo(500)
        y_ep = h1.get_y_arm_endpoint_in_geo(500)
        dist = np.linalg.norm(x_ep - y_ep)
        assert dist > 0.01  # > ~0.5 degrees


# ---------------------------------------------------------------------------
# Detector — antenna patterns & time delay
# ---------------------------------------------------------------------------

class TestDetectorAntennaPattern:
    """Tests for Detector.atenna_pattern and related methods."""

    def test_static_pattern_scalar(self, h1):
        """Basic F+/Fx at zenith should be finite."""
        t_gps = 1261873618.0
        fplus, fcross = h1.atenna_pattern(0.0, 0.0, 0.0, t_gps)
        assert np.isfinite(fplus)
        assert np.isfinite(fcross)
        # F+/Fx values should be within [-1, 1] for static response
        assert -1.0 <= fplus <= 1.0
        assert -1.0 <= fcross <= 1.0

    def test_static_pattern_array(self, h1):
        """Vectorized call with arrays."""
        t_gps = 1261873618.0
        ra = np.linspace(0, 2 * np.pi, 10)
        dec = np.zeros(10)
        pol = np.zeros(10)
        fplus, fcross = h1.atenna_pattern(ra, dec, pol, t_gps)
        assert fplus.shape == (10,)
        assert fcross.shape == (10,)
        assert np.all(np.abs(fplus) <= 1.0)
        assert np.all(np.abs(fcross) <= 1.0)

    def test_frequency_response_zero_freq(self, h1):
        """f=0 should match static response."""
        t_gps = 1261873618.0
        fp_static, fx_static = h1.atenna_pattern(0.5, 0.3, 0.1, t_gps)
        fp_freq, fx_freq = h1.atenna_pattern(
            0.5, 0.3, 0.1, t_gps, frequency=0
        )
        assert fp_static == pytest.approx(fp_freq)
        assert fx_static == pytest.approx(fx_freq)

    def test_frequency_response_nonzero(self, h1):
        """Non-zero frequency should give a valid (complex) result."""
        t_gps = 1261873618.0
        fp, fx = h1.atenna_pattern(
            0.5, 0.3, 0.1, t_gps, frequency=100.0
        )
        # With frequency, response may be complex
        assert np.isfinite(fp)
        assert np.isfinite(fx)

    def test_vector_polarization(self, h1):
        """Vector polarization type should return fx, fy."""
        t_gps = 1261873618.0
        fx, fy = h1.atenna_pattern(
            0.5, 0.3, 0.1, t_gps, polarization_type="vector"
        )
        assert np.isfinite(fx)
        assert np.isfinite(fy)

    def test_scalar_polarization(self, h1):
        """Scalar polarization type should return fb, fl."""
        t_gps = 1261873618.0
        fb, fl = h1.atenna_pattern(
            0.5, 0.3, 0.1, t_gps, polarization_type="scalar"
        )
        assert np.isfinite(fb)
        assert np.isfinite(fl)

    def test_optimal_orientation(self, h1):
        ra, dec = h1.optimal_orientation(1261873618.0)
        assert 0.0 <= ra < 2 * math.pi
        assert -math.pi / 2 <= dec <= math.pi / 2

    def test_time_delay_h1_l1(self, h1, l1):
        """Time delay between H1 and L1 should be ~10ms."""
        t_gps = 1261873618.0
        dt = h1.time_delay_from_detector(l1, 0.0, 0.0, t_gps)
        # Light travel time Hanford→Livingston ≈ 10 ms
        assert -0.1 < dt < 0.1
        assert abs(dt) < 0.02  # ~10ms max

    def test_time_delay_symmetry(self, h1, l1):
        """dt(H1←L1) ≈ -dt(L1←H1)."""
        t_gps = 1261873618.0
        d1 = h1.time_delay_from_detector(l1, 0.5, 0.3, t_gps)
        d2 = l1.time_delay_from_detector(h1, 0.5, 0.3, t_gps)
        assert d1 == pytest.approx(-d2)

    def test_project_wave_shape(self, h1):
        """project_wave returns a TimeSeries with same length as input."""
        from pycwb.types.time_series import TimeSeries

        hp = TimeSeries(data=np.random.randn(100), dt=1.0 / 256, t0=1261873618.0)
        hc = TimeSeries(data=np.random.randn(100), dt=1.0 / 256, t0=1261873618.0)
        strain = h1.project_wave(hp, hc, 0.5, 0.3, 0.1)
        assert len(strain.data) == 100

    def test_compute_detector_tensor(self, h1):
        D, xv, yv = h1.compute_detector_tensor()
        assert D.shape == (3, 3)
        assert xv.shape == (3,)
        assert yv.shape == (3,)
        # Detector tensor should be symmetric and traceless
        assert np.allclose(D, D.T)
        assert abs(np.trace(D)) < 1e-10
        # Unit vectors
        assert np.linalg.norm(xv) == pytest.approx(1.0, abs=1e-6)
        assert np.linalg.norm(yv) == pytest.approx(1.0, abs=1e-6)
        # Arms should be roughly orthogonal
        assert abs(np.dot(xv, yv)) < 2e-6  # ~90 degrees for H1

    def test_compute_antenna_pattern_for_grid(self, h1):
        """Grid antenna pattern should return two 2D arrays."""
        n_lon, n_lat = 36, 18
        lon_grid, lat_grid = np.meshgrid(
            np.linspace(0, 2 * np.pi, n_lon),
            np.linspace(0, np.pi, n_lat),
        )
        fp, fx = h1.compute_antenna_pattern_for_grid(lat_grid, lon_grid)
        assert fp.shape == (n_lat, n_lon)
        assert fx.shape == (n_lat, n_lon)
        assert np.all(np.abs(fp) <= 1.0)
        assert np.all(np.abs(fx) <= 1.0)


# ---------------------------------------------------------------------------
# DetectorNetwork
# ---------------------------------------------------------------------------

class TestDetectorNetwork:
    def test_init_from_string(self):
        net = DetectorNetwork("H1L1")
        assert len(net.detectors) == 2

    def test_init_from_list(self):
        net = DetectorNetwork(ifos=["H1", "L1", "V1"])
        assert len(net.detectors) == 3

    def test_init_from_detectors(self, h1, l1, v1):
        net = DetectorNetwork(detectors=[h1, l1, v1])
        assert len(net.detectors) == 3

    def test_init_empty(self):
        net = DetectorNetwork()
        assert len(net.detectors) == 0

    def test_add_detector_string(self, h1l1_network):
        h1l1_network.add_detector("V1")
        assert len(h1l1_network.detectors) == 3

    def test_add_detector_object(self, h1l1_network, v1):
        h1l1_network.add_detector(v1)
        assert len(h1l1_network.detectors) == 3

    def test_add_detectors_string(self):
        net = DetectorNetwork()
        net.add_detectors("H1L1V1")
        assert len(net.detectors) == 3

    def test_add_detectors_list(self):
        net = DetectorNetwork()
        net.add_detectors(["H1", "L1"])
        assert len(net.detectors) == 2

    def test_parse_detector_codes(self):
        result = DetectorNetwork._parse_detector_codes("H1L1V1")
        assert result == ["H1", "L1", "V1"]

    def test_parse_detector_codes_mixed(self):
        result = DetectorNetwork._parse_detector_codes("ABCH1XYZL1")
        assert result == ["H1", "L1"]

    def test_parse_invalid_raises(self):
        with pytest.raises(ValueError):
            DetectorNetwork._parse_detector_codes("XXXX")

    def test_get_detector_info(self, h1l1_network):
        info = h1l1_network._get_detector_info()
        assert len(info) == 2
        assert info[0]["code"] == "H1"
        assert "lat" in info[0]
        assert "lon" in info[0]

    def test_create_sky_grid(self):
        lon_grid, lat_grid, n_lon, n_lat = DetectorNetwork._create_sky_grid(2)
        assert lon_grid.shape == (360, 720)
        assert lat_grid.shape == (360, 720)
        assert n_lon == 720
        assert n_lat == 360

    def test_compute_antenna_patterns(self, h1l1_network):
        lon_grid, lat_grid, _, _ = DetectorNetwork._create_sky_grid(1)
        detectors = h1l1_network._get_detector_info()
        fp, fx = DetectorNetwork._compute_antenna_patterns(
            lat_grid, lon_grid, detectors
        )
        n_lat, n_lon = lat_grid.shape
        n_det = len(detectors)
        assert fp.shape == (n_lat, n_lon, n_det)
        assert fx.shape == (n_lat, n_lon, n_det)
        assert np.all(np.abs(fp) <= 1.0)

    def test_compute_polarization_quantity(self):
        val = DetectorNetwork._compute_polarization_quantity(
            1.0, 0.5, 0.2, 3, 3
        )
        assert np.isfinite(val)
        assert val >= 0

    def test_polarization_quantity_unsupported(self):
        with pytest.raises(ValueError):
            DetectorNetwork._compute_polarization_quantity(1, 1, 1, 99, 2)

    def test_compute_reference_max(self, h1l1_network):
        lon_grid, lat_grid, _, _ = DetectorNetwork._create_sky_grid(1)
        detectors = h1l1_network._get_detector_info()
        fp, fx = DetectorNetwork._compute_antenna_patterns(
            lat_grid, lon_grid, detectors
        )
        scales = np.ones(len(detectors))
        ref_max = DetectorNetwork._compute_reference_max(fp, fx, scales)
        assert ref_max > 0

    def test_compute_antenna_pattern(self, h1l1_network):
        lon_grid, lat_grid, _, _ = DetectorNetwork._create_sky_grid(1)
        detectors = h1l1_network._get_detector_info()
        fp, fx = DetectorNetwork._compute_antenna_patterns(
            lat_grid, lon_grid, detectors
        )
        scales = np.ones(len(detectors))
        pattern, pattern_max = DetectorNetwork._compute_antenna_pattern(
            fp, fx, 3, scales
        )
        assert pattern.shape == lat_grid.shape
        assert pattern_max > 0

    def test_compute_arm_endpoints(self):
        det_info = {
            "lat": 0.81079526383,
            "lon": -2.08405676917,
            "x_az": 5.65487724844,
            "x_alt": -0.0006195,
            "y_az": 4.08408092164,
            "y_alt": 0.0000125,
        }
        (x_lon, x_lat), (y_lon, y_lat) = DetectorNetwork._compute_arm_endpoints(
            det_info
        )
        assert -180 <= x_lon <= 180
        assert -90 <= x_lat <= 90
        assert -180 <= y_lon <= 180
        assert -90 <= y_lat <= 90


# ---------------------------------------------------------------------------
# Module-level geometry functions
# ---------------------------------------------------------------------------

class TestGMSTAccurate:
    def test_gmst_finite(self):
        gmst = gmst_accurate(1261873618.0)
        assert np.isfinite(gmst)
        assert 0.0 <= gmst < 2 * math.pi

    def test_gmst_increases_with_time(self):
        g1 = gmst_accurate(0.0)
        g2 = gmst_accurate(3600.0)  # 1 hour later
        # GMST increases by ~15 deg per hour
        assert g2 > g1

    def test_gmst_24h_periodicity(self):
        """GMST should advance by ~2π over one sidereal day (~86164s)."""
        g1 = gmst_accurate(0.0)
        g2 = gmst_accurate(86164.0905)  # one sidereal day
        # g2 ≈ g1 + 2π  (mod 2π they should be close)
        diff = (g2 - g1) % (2 * math.pi)
        # diff should be small (near 0) or near 2π due to rounding
        assert min(diff, abs(diff - 2 * math.pi)) < 0.01


class TestSingleArmFrequencyResponse:
    def test_zero_frequency(self):
        """At f=0 the formula has a 0/0 singularity; test f→0 limit."""
        # f=0 is a singularity (division by zero), use very small f instead
        r_small = single_arm_frequency_response(1e-10, 0.0, 4000.0)
        assert abs(r_small - 1.0) < 1e-6

    def test_normal_incidence(self):
        """Normal incidence (n=1) special case."""
        r = single_arm_frequency_response(100.0, 1.0, 4000.0)
        assert np.isfinite(r)

    def test_grazing_incidence(self):
        """Grazing incidence (n=0)."""
        r = single_arm_frequency_response(100.0, 0.0, 4000.0)
        assert np.isfinite(r)


class TestEarthCenteredVectors:
    def test_output_keys(self):
        result = earth_centered_vectors(
            -2.08405676917, 0.81079526383,
            yangle=5.65487724844, xangle=4.08408092164,
            height=142.554,
        )
        assert "loc_vec" in result
        assert "x_vec" in result
        assert "y_vec" in result
        assert "response" in result
        assert "x_response" in result
        assert "y_response" in result

    def test_loc_vec_on_earth_surface(self):
        result = earth_centered_vectors(-2.084, 0.811, height=0)
        r = np.linalg.norm(result["loc_vec"])
        assert 6.36e6 < r < 6.39e6  # ~Earth radius (varies with latitude)

    def test_right_angle_default(self):
        """When xangle is None, should assume right-angle detector."""
        result = earth_centered_vectors(
            -2.084, 0.811, yangle=4.0, height=0,
        )
        assert result["x_vec"] is not None
        assert result["y_vec"] is not None

    def test_response_symmetric(self):
        result = earth_centered_vectors(-2.084, 0.811, height=0)
        resp = result["response"]
        assert resp.shape == (3, 3)
        assert np.allclose(resp, resp.T)


class TestGetMaxDelay:
    def test_none_returns_zero(self):
        assert get_max_delay(None) == 0.0

    def test_empty_returns_zero(self):
        assert get_max_delay([]) == 0.0

    def test_single_detector_returns_zero(self):
        assert get_max_delay([Detector("H1")]) == 0.0


class TestNetworkMaxDelay:
    def test_empty_or_single_ifo_returns_zero(self):
        assert network_utils.max_delay([]) == 0.0
        assert network_utils.max_delay(["H1"]) == 0.0

    def test_h1_l1_delay(self):
        assert network_utils.max_delay(["H1", "L1"]) == pytest.approx(
            0.010013, rel=1e-4
        )


class TestImportBoundaries:
    def test_detector_import_does_not_load_plotting_stack(self):
        code = """
import json
import sys

from pycwb.types.detector import Detector

Detector("H1")
modules = [
    "matplotlib.pyplot",
    "cartopy",
    "plotly",
    "pycwb.modules.plot",
    "pycwb.modules.plot.detector_antenna",
    "pycwb.modules.plot.detector_globe",
]
print(json.dumps({name: name in sys.modules for name in modules}, sort_keys=True))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        loaded = json.loads(result.stdout.strip().splitlines()[-1])
        assert not any(loaded.values()), loaded


class TestCalculateE2orFromAcore:
    def test_less_than_two_ifo(self):
        """n_ifo < 2 should return max(0, acore)."""
        assert calculate_e2or_from_acore(0.5, 1) == 0.5
        assert calculate_e2or_from_acore(-1.0, 1) == 0.0

    def test_three_ifo(self):
        e2or = calculate_e2or_from_acore(2.0, 3)
        assert e2or > 0
        assert np.isfinite(e2or)

    def test_monotonic_in_acore(self):
        """Larger acore → larger e2or."""
        e1 = calculate_e2or_from_acore(1.0, 3)
        e2 = calculate_e2or_from_acore(2.0, 3)
        assert e2 > e1


class TestBuildSkyDirections:
    def test_fibonacci_default(self):
        ra, dec = _build_sky_directions(100)
        assert ra.shape == (100,)
        assert dec.shape == (100,)
        assert np.all(np.abs(dec) <= math.pi / 2)

    def test_healpix(self):
        """With valid healpix_order, returns HEALPix grid."""
        try:
            import healpy  # noqa: F401
        except ImportError:
            pytest.skip("healpy not available")
        ra, dec = _build_sky_directions(100, healpix_order=2)
        # nside = 2**order = 4, npix = 12 * nside**2 = 192
        assert ra.size == 192

    def test_minimum_n_sky(self):
        ra, dec = _build_sky_directions(0)
        assert ra.shape == (1,)


class TestComputeSkyDelayAndPatterns:
    def test_basic_call(self):
        ml, FP, FX = compute_sky_delay_and_patterns(
            ["H1", "L1"], "H1", 256.0, 64, 1261873618.0, n_sky=100,
        )
        n_ifo, n_sky = 2, 100
        assert ml.shape == (n_ifo, n_sky)
        assert FP.shape == (n_ifo, n_sky)
        assert FX.shape == (n_ifo, n_sky)
        assert ml.dtype == np.int32
        assert FP.dtype == np.float64

    def test_delays_within_bounds(self):
        ml, _, _ = compute_sky_delay_and_patterns(
            ["H1", "L1"], "H1", 256.0, 64, 1261873618.0, n_sky=100,
        )
        assert np.all(np.abs(ml) <= 64)

    def test_patterns_within_unit_range(self):
        _, FP, FX = compute_sky_delay_and_patterns(
            ["H1", "L1"], "H1", 256.0, 64, 1261873618.0, n_sky=100,
        )
        assert np.all(np.abs(FP) <= 1.0)
        assert np.all(np.abs(FX) <= 1.0)

    def test_empty_ifos_raises(self):
        with pytest.raises(ValueError):
            compute_sky_delay_and_patterns(
                [], "H1", 256.0, 64, 0.0,
            )

    def test_detector_objects_accepted(self, h1, l1):
        ml, FP, FX = compute_sky_delay_and_patterns(
            [h1, l1], "H1", 256.0, 64, 1261873618.0, n_sky=50,
        )
        assert ml.shape == (2, 50)
