"""Coordinate-contract and cWB parity tests."""

import math

import numpy as np
import pytest
from astropy import units as u

from pycwb.utils.skymap_coord import (
    convert_cwb_to_geo,
    convert_geo_to_cwb,
    convert_phi_to_ra,
    convert_ra_to_phi,
    gmst_astropy,
    gmst_cwb,
    parse_angle_quantity,
    parse_sky_coordinates,
    validate_user_sky_config,
)


def _cwb_core_phi_ra_deg(phi_deg, gps, inverse=False):
    """Literal Python transcription of cwb-core/skymap.hh::phiRA."""
    t = (gps - 630763213.0) / 86400.0 / 36525.0
    sidereal = (-6.2e-6 * t + 0.093104) * t * t + 67310.54841
    sidereal += 8640184.812866 * t + 3155760000.0 * t
    gmst = 360.0 * sidereal / 86400.0
    if inverse:
        gmst = -gmst
    return (phi_deg + gmst) % 360.0


@pytest.mark.parametrize("gps", [0.0, 630763213.0, 1126259462.0, 1261873618.0])
def test_gmst_and_phi_ra_match_cwb_core(gps):
    phi = 123.456
    expected_ra = _cwb_core_phi_ra_deg(phi, gps)
    actual_ra = math.degrees(convert_phi_to_ra(math.radians(phi), gps))
    # Python converts the cWB degree polynomial through radians, introducing a
    # sub-nanodegree roundoff relative to C++ fmod in degrees.
    assert actual_ra == pytest.approx(expected_ra, abs=1e-9)

    expected_phi = _cwb_core_phi_ra_deg(expected_ra, gps, inverse=True)
    actual_phi = math.degrees(convert_ra_to_phi(math.radians(expected_ra), gps))
    assert actual_phi == pytest.approx(expected_phi, abs=1e-9)
    assert math.degrees(gmst_cwb(gps)) == pytest.approx(
        _cwb_core_phi_ra_deg(0.0, gps), abs=1e-9
    )


def test_cwb_geo_round_trip_at_poles():
    phi = np.array([0.0, np.pi / 2.0, 3.0 * np.pi / 2.0])
    theta = np.array([0.0, np.pi / 2.0, np.pi])
    longitude, latitude = convert_cwb_to_geo(phi, theta)
    phi_back, theta_back = convert_geo_to_cwb(longitude, latitude)
    np.testing.assert_allclose(phi_back, phi, atol=1e-15)
    np.testing.assert_allclose(theta_back, theta, atol=1e-15)


def test_astropy_projection_model_preserves_earth_fixed_hour_angle():
    gps = 1261873618.0
    phi_geo = 1.2
    ra = float(convert_phi_to_ra(phi_geo, gps, gmst_model="astropy"))
    gha = (gmst_astropy(gps) - ra + np.pi) % (2.0 * np.pi) - np.pi
    assert gha == pytest.approx(-phi_geo, abs=1e-12)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("180 deg", np.pi), ("3.141592653589793 rad", np.pi), (30 * u.arcmin, np.pi / 360.0)],
)
def test_parse_angle_quantity(value, expected):
    assert parse_angle_quantity(value, name="angle") == pytest.approx(expected)


@pytest.mark.parametrize("value", [120.0, "5 s", 2 * u.m])
def test_parse_angle_quantity_rejects_ambiguous_or_nonangular(value):
    with pytest.raises(ValueError):
        parse_angle_quantity(value, name="angle")


@pytest.mark.parametrize(
    ("coordsys", "values"),
    [
        ("icrs", {"ra": "120 deg", "dec": "-30 deg"}),
        ("geo", {"longitude": "-70 deg", "latitude": "30 deg"}),
        ("cwb", {"phi_geo": "290 deg", "theta_cwb": "60 deg"}),
    ],
)
def test_semantic_coordinate_names_crosscheck_frame(coordsys, values):
    first, second = parse_sky_coordinates(values, coordsys)
    assert np.isfinite(first)
    assert np.isfinite(second)


def test_semantic_coordinate_names_reject_wrong_frame():
    with pytest.raises(ValueError, match="expected keys"):
        parse_sky_coordinates({"ra": "1 rad", "dec": "0 rad"}, "cwb")


def test_semantic_coordinates_reject_legacy_aliases_mixed_in():
    with pytest.raises(ValueError, match="mixes coordinate frames"):
        parse_sky_coordinates(
            {"ra": "1 rad", "dec": "0 rad", "phi": 1.0, "theta": 0.0},
            "icrs",
        )


@pytest.mark.parametrize(
    ("coordsys", "values", "message"),
    [
        ("icrs", {"ra": "361 deg", "dec": "0 deg"}, "ra must be within"),
        (
            "geo",
            {"longitude": "181 deg", "latitude": "0 deg"},
            "longitude must be within",
        ),
        (
            "cwb",
            {"phi_geo": "-1 deg", "theta_cwb": "90 deg"},
            "phi_geo must be within",
        ),
    ],
)
def test_semantic_coordinates_validate_longitude_range(coordsys, values, message):
    with pytest.raises(ValueError, match=message):
        parse_sky_coordinates(values, coordsys)


def test_yaml_validation_requires_explicit_coordsys_for_semantic_names():
    config = {
        "type": "Patch",
        "patch": {
            "center": {"ra": "120 deg", "dec": "-30 deg"},
            "radius": "5 deg",
        },
    }
    with pytest.raises(ValueError, match="explicit coordsys"):
        validate_user_sky_config(
            config, context="sky_mask", default_coordsys="geo"
        )


def test_yaml_validation_accepts_semantic_quantity_contract():
    validate_user_sky_config(
        {
            "type": "Patch",
            "coordsys": "icrs",
            "patch": {
                "center": {"ra": "120 deg", "dec": "-30 deg"},
                "radius": "5 deg",
            },
        },
        context="sky_mask",
        default_coordsys="geo",
    )


def test_yaml_validation_rejects_unknown_frame_even_for_all_sky():
    with pytest.raises(ValueError, match="not recognized"):
        validate_user_sky_config(
            {"type": "UniformAllSky", "coordsys": "mars"},
            context="sky_mask",
            default_coordsys="geo",
        )


def test_yaml_validation_rejects_patch_radius_over_hemisphere_limit():
    with pytest.raises(ValueError, match="radius must be within"):
        validate_user_sky_config(
            {
                "type": "Patch",
                "coordsys": "icrs",
                "patch": {
                    "center": {"ra": "120 deg", "dec": "-30 deg"},
                    "radius": "181 deg",
                },
            },
            context="sky_mask",
            default_coordsys="geo",
        )
