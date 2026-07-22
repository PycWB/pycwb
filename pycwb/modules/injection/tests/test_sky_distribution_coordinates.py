import numpy as np
import pytest

from pycwb.modules.injection.sky_distribution import (
    distribute_injections_on_sky,
    generate_sky_distribution,
)


def test_fixed_icrs_quantity_coordinates():
    first, second = generate_sky_distribution(
        {
            "type": "Fixed",
            "coordsys": "icrs",
            "coordinates": {"ra": "120 deg", "dec": "-30 deg"},
        },
        3,
    )
    np.testing.assert_allclose(first, np.deg2rad(120.0))
    np.testing.assert_allclose(second, np.deg2rad(-30.0))


def test_fixed_cwb_quantity_coordinates_remain_colatitude():
    first, second = generate_sky_distribution(
        {
            "type": "Fixed",
            "coordsys": "cwb",
            "coordinates": {"phi_geo": "290 deg", "theta_cwb": "60 deg"},
        },
        2,
    )
    np.testing.assert_allclose(first, np.deg2rad(290.0))
    np.testing.assert_allclose(second, np.deg2rad(60.0))


def test_uniform_cwb_uses_colatitude_range():
    _, theta_cwb = generate_sky_distribution(
        {"type": "UniformAllSky", "coordsys": "cwb"}, 1000
    )
    assert np.all(theta_cwb >= 0.0)
    assert np.all(theta_cwb <= np.pi)


def test_distribution_coordsys_is_case_normalized():
    injections = [{}]
    distribute_injections_on_sky(
        injections, (np.array([1.0]), np.array([0.2])),
        shuffle=False, coordsys="ICRS",
    )
    assert injections == [{"ra": 1.0, "dec": 0.2}]


def test_semantic_coordinates_require_quantity_units():
    with pytest.raises(ValueError, match="must include an angular unit"):
        generate_sky_distribution(
            {
                "type": "Fixed",
                "coordsys": "geo",
                "coordinates": {"longitude": 10.0, "latitude": 20.0},
            },
            1,
        )
