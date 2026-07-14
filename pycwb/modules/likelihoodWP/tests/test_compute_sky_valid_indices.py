import unittest
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from pycwb.modules.likelihoodWP.sky_mask import (
    _read_healpix_map,
    compute_sky_valid_indices,
    sky_valid_indices_for_cluster,
)
from pycwb.utils.skymap_coord import convert_phi_to_ra, gmst_cwb


def _uniform_sky(n_sky=200):
    """Return (ra, dec) for n_sky uniformly distributed directions in radians."""
    idx = np.arange(n_sky, dtype=np.float64)
    z = 1.0 - 2.0 * (idx + 0.5) / float(n_sky)
    ra = (np.pi * (3.0 - np.sqrt(5.0)) * idx) % (2.0 * np.pi)
    dec = np.arcsin(np.clip(z, -1.0, 1.0))
    return ra, dec


def _angular_sep(ra1, dec1, ra2, dec2):
    """Great-circle angular separation in radians."""
    cos_d = (np.sin(dec1) * np.sin(dec2) +
             np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
    return np.arccos(np.clip(cos_d, -1.0, 1.0))


class TestComputeSkyValidIndices_NoneConfig(unittest.TestCase):
    """sky_mask_config = None → all indices returned."""

    def test_returns_all_indices(self):
        ra, dec = _uniform_sky(100)
        result = compute_sky_valid_indices(ra, dec, None)
        np.testing.assert_array_equal(result, np.arange(100, dtype=np.int64))

    def test_dtype_is_int64(self):
        ra, dec = _uniform_sky(50)
        result = compute_sky_valid_indices(ra, dec, None)
        self.assertEqual(result.dtype, np.int64)

    def test_single_direction(self):
        ra, dec = np.array([1.0]), np.array([0.5])
        result = compute_sky_valid_indices(ra, dec, None)
        np.testing.assert_array_equal(result, np.array([0], dtype=np.int64))


class TestComputeSkyValidIndices_UniformAllSky(unittest.TestCase):
    """type=UniformAllSky → all indices, same as None."""

    def test_returns_all_indices(self):
        ra, dec = _uniform_sky(80)
        cfg = {'type': 'UniformAllSky'}
        result = compute_sky_valid_indices(ra, dec, cfg)
        np.testing.assert_array_equal(result, np.arange(80, dtype=np.int64))

    def test_dtype_is_int64(self):
        ra, dec = _uniform_sky(10)
        result = compute_sky_valid_indices(ra, dec, {'type': 'UniformAllSky'})
        self.assertEqual(result.dtype, np.int64)


class TestComputeSkyValidIndices_Patch(unittest.TestCase):
    """type=Patch — circular cap selection."""

    # Centre at (RA, Dec) = (1.0, 0.3) rad
    CENTER_PHI = 1.0
    CENTER_THETA = 0.3
    RADIUS_RAD = 0.5  # radians

    def _config_rad(self, radius=None):
        return {
            'type': 'Patch',
            'patch': {
                'center': {'phi': self.CENTER_PHI, 'theta': self.CENTER_THETA},
                'radius': radius if radius is not None else self.RADIUS_RAD,
                'unit': 'rad',
            },
        }

    def _config_deg(self):
        return {
            'type': 'Patch',
            'patch': {
                'center': {
                    'phi': np.degrees(self.CENTER_PHI),
                    'theta': np.degrees(self.CENTER_THETA),
                },
                'radius': np.degrees(self.RADIUS_RAD),
                'unit': 'deg',
            },
        }

    def setUp(self):
        self.ra, self.dec = _uniform_sky(300)

    def test_all_valid_directions_within_radius(self):
        result = compute_sky_valid_indices(self.ra, self.dec, self._config_rad())
        for idx in result:
            sep = _angular_sep(self.ra[idx], self.dec[idx],
                               self.CENTER_PHI, self.CENTER_THETA)
            self.assertLessEqual(sep, self.RADIUS_RAD + 1e-10,
                                 msg=f"Index {idx} is outside the patch radius")

    def test_no_valid_direction_outside_radius(self):
        result = compute_sky_valid_indices(self.ra, self.dec, self._config_rad())
        valid_set = set(result.tolist())
        for i in range(len(self.ra)):
            sep = _angular_sep(self.ra[i], self.dec[i],
                               self.CENTER_PHI, self.CENTER_THETA)
            if sep <= self.RADIUS_RAD:
                self.assertIn(i, valid_set,
                              msg=f"Index {i} should be valid but is missing")

    def test_deg_and_rad_give_same_result(self):
        result_rad = compute_sky_valid_indices(self.ra, self.dec, self._config_rad())
        result_deg = compute_sky_valid_indices(self.ra, self.dec, self._config_deg())
        np.testing.assert_array_equal(result_rad, result_deg)

    def test_zero_radius_picks_nearest(self):
        # Use an exact grid point as center so arccos gives exactly 0.0 <= 0.0.
        center_phi = float(self.ra[5])
        center_theta = float(self.dec[5])
        cfg = {'type': 'Patch', 'patch': {
            'center': {'phi': center_phi, 'theta': center_theta},
            'radius': 0.0, 'unit': 'rad'}}
        result = compute_sky_valid_indices(self.ra, self.dec, cfg)
        # Exactly the matching grid index should be selected
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result[0]), 5)

    def test_full_sky_radius_returns_all(self):
        result = compute_sky_valid_indices(self.ra, self.dec, self._config_rad(radius=np.pi))
        self.assertEqual(len(result), len(self.ra))

    def test_dtype_is_int64(self):
        result = compute_sky_valid_indices(self.ra, self.dec, self._config_rad())
        self.assertEqual(result.dtype, np.int64)

    def test_result_is_sorted(self):
        result = compute_sky_valid_indices(self.ra, self.dec, self._config_rad())
        # np.where returns indices in ascending order
        self.assertTrue(np.all(result[:-1] <= result[1:]))


class TestComputeSkyValidIndices_Fixed(unittest.TestCase):
    """type=Fixed — single nearest sky direction."""

    def setUp(self):
        self.ra, self.dec = _uniform_sky(200)

    def _config(self, phi, theta, unit='rad'):
        return {
            'type': 'Fixed',
            'coordinates': {
                'sky_loc': {'phi': phi, 'theta': theta},
                'unit': unit,
            },
        }

    def test_returns_exactly_one_index(self):
        result = compute_sky_valid_indices(self.ra, self.dec, self._config(1.0, 0.2))
        self.assertEqual(len(result), 1)

    def test_returns_nearest_direction(self):
        target_phi, target_theta = 1.0, 0.2
        result = compute_sky_valid_indices(self.ra, self.dec,
                                           self._config(target_phi, target_theta))
        seps = _angular_sep(self.ra, self.dec, target_phi, target_theta)
        expected = int(np.argmin(seps))
        self.assertEqual(int(result[0]), expected)

    def test_deg_and_rad_give_same_result(self):
        phi_rad, theta_rad = 2.0, -0.4
        result_rad = compute_sky_valid_indices(self.ra, self.dec,
                                               self._config(phi_rad, theta_rad, 'rad'))
        result_deg = compute_sky_valid_indices(
            self.ra, self.dec,
            self._config(np.degrees(phi_rad), np.degrees(theta_rad), 'deg'),
        )
        self.assertEqual(int(result_rad[0]), int(result_deg[0]))

    def test_dtype_is_int64(self):
        result = compute_sky_valid_indices(self.ra, self.dec, self._config(0.0, 0.0))
        self.assertEqual(result.dtype, np.int64)

    def test_exact_match_direction(self):
        # If a sky direction is exactly at the requested position, it must be selected.
        ra = np.array([0.0, 1.0, 2.0])
        dec = np.array([0.0, 0.5, -0.5])
        result = compute_sky_valid_indices(ra, dec, self._config(1.0, 0.5))
        self.assertEqual(int(result[0]), 1)


class TestComputeSkyValidIndices_UnknownType(unittest.TestCase):
    """Unknown type → ValueError."""

    def test_raises_value_error(self):
        ra, dec = _uniform_sky(10)
        with self.assertRaises(ValueError):
            compute_sky_valid_indices(ra, dec, {'type': 'Galaxy'})


class TestComputeSkyValidIndices_EmptySky(unittest.TestCase):
    """Edge case: n_sky = 0."""

    def test_none_config_empty(self):
        ra, dec = np.array([]), np.array([])
        result = compute_sky_valid_indices(ra, dec, None)
        self.assertEqual(len(result), 0)

    def test_patch_empty_sky(self):
        ra, dec = np.array([]), np.array([])
        cfg = {
            'type': 'Patch',
            'patch': {'center': {'phi': 0.0, 'theta': 0.0}, 'radius': 1.0, 'unit': 'rad'},
        }
        result = compute_sky_valid_indices(ra, dec, cfg)
        self.assertEqual(len(result), 0)


if __name__ == '__main__':
    unittest.main()


def test_fixed_icrs_uses_cwb_ra_to_phi_sign():
    gps = 1261873618.0
    gmst = gmst_cwb(gps)
    config = {
        'type': 'Fixed',
        'coordsys': 'icrs',
        'coordinates': {'ra': f'{np.degrees(gmst)} deg', 'dec': '0 deg'},
    }
    result = compute_sky_valid_indices(
        np.array([0.0, gmst]), np.array([0.0, 0.0]), config, t_ref=gps
    )
    np.testing.assert_array_equal(result, np.array([0], dtype=np.int64))


def test_icrs_mask_requires_reference_time():
    config = {
        'type': 'Fixed',
        'coordsys': 'icrs',
        'coordinates': {'ra': '120 deg', 'dec': '0 deg'},
    }
    with pytest.raises(ValueError, match='requires a GPS reference time'):
        compute_sky_valid_indices(np.array([0.0]), np.array([0.0]), config)


def test_custom_icrs_indexes_phi_plus_cwb_gmst():
    import healpy as hp

    gps = 1261873618.0
    nside = 8
    phi_geo = np.array([0.0])
    latitude = np.array([0.0])
    map_ra = convert_phi_to_ra(phi_geo, gps, gmst_model='cwb')
    correct_pixel = int(hp.ang2pix(nside, np.array([np.pi / 2.0]), map_ra)[0])
    raw_pixel = int(hp.ang2pix(nside, np.array([np.pi / 2.0]), phi_geo)[0])
    assert correct_pixel != raw_pixel

    skymap = np.zeros(hp.nside2npix(nside))
    skymap[correct_pixel] = 1.0
    config = {
        'type': 'Custom',
        'coordsys': 'icrs',
        'custom': {
            'healpix_map': 'mock.fits',
            'nside': nside,
            'ordering': 'ring',
            'threshold': 0.0,
        },
    }
    _read_healpix_map.cache_clear()
    with patch('healpy.read_map', return_value=skymap):
        result = compute_sky_valid_indices(
            phi_geo, latitude, config, t_ref=gps
        )
    np.testing.assert_array_equal(result, np.array([0], dtype=np.int64))


def test_icrs_mask_is_recomputed_at_cluster_time():
    segment_start = 1261873618.0
    cluster_offset = 600.0
    event_gps = segment_start + cluster_offset
    target_ra = gmst_cwb(event_gps)
    config = {
        'type': 'Fixed',
        'coordsys': 'icrs',
        'coordinates': {'ra': f'{np.degrees(target_ra)} deg', 'dec': '0 deg'},
    }
    phi_at_segment_start = float(
        (target_ra - gmst_cwb(segment_start)) % (2.0 * np.pi)
    )
    setup = {
        'sky_mask_config': config,
        'segment_start_gps': segment_start,
        'phi_geo_arr': np.array([0.0, phi_at_segment_start]),
        'latitude_arr': np.array([0.0, 0.0]),
        'sky_valid_indices': np.array([1], dtype=np.int64),
    }
    cluster = SimpleNamespace(cluster_time=cluster_offset)
    result = sky_valid_indices_for_cluster(setup, cluster)
    np.testing.assert_array_equal(result, np.array([0], dtype=np.int64))


def test_cwb_semantic_mask_converts_colatitude_to_latitude():
    config = {
        'type': 'Fixed',
        'coordsys': 'cwb',
        'coordinates': {'phi_geo': '0 deg', 'theta_cwb': '60 deg'},
    }
    result = compute_sky_valid_indices(
        np.array([0.0, 0.0]),
        np.deg2rad(np.array([30.0, -30.0])),
        config,
    )
    np.testing.assert_array_equal(result, np.array([0], dtype=np.int64))
