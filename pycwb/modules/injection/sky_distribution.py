"""Generate and attach injection sky distributions."""

from __future__ import annotations

import logging
import warnings

import numpy as np
from astropy import units as u

from pycwb.utils.skymap_coord import (
    COORDINATE_KEYS,
    convert_dec_to_theta,
    convert_theta_to_dec,
    normalize_coordinate_system,
    parse_angle_quantity,
    parse_sky_coordinates,
)

try:
    import healpy as hp
except ImportError:  # pragma: no cover - optional dependency
    hp = None

logger = logging.getLogger(__name__)


def generate_sky_distribution(sky_distribution_config, n_samples):
    """Generate two radian coordinate arrays in the configured frame."""
    dist_type = sky_distribution_config["type"]
    coordsys = normalize_coordinate_system(sky_distribution_config.get("coordsys", "icrs"))
    logger.info(
        "Generating sky distribution type=%s, coordsys=%s, n=%d",
        dist_type, coordsys, n_samples,
    )

    if dist_type == "existing":
        unit_value = sky_distribution_config.get("unit")
        try:
            table_unit = u.Unit(unit_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Existing sky tables require an explicit angular unit") from exc
        if not table_unit.is_equivalent(u.rad):
            raise ValueError(f"Existing sky-table unit must be angular, got {table_unit}")

        expected_columns = list(COORDINATE_KEYS[coordsys])
        columns = sky_distribution_config.get("columns")
        if columns is None:
            warnings.warn(
                "Existing sky tables without semantic 'columns' are deprecated; "
                f"declare columns: {expected_columns}",
                DeprecationWarning,
                stacklevel=2,
            )
        elif list(columns) != expected_columns:
            raise ValueError(
                f"Existing sky-table columns {columns!r} do not match "
                f"coordsys={coordsys!r}; expected {expected_columns!r}"
            )
        first, second = np.loadtxt(sky_distribution_config["existing_path"], unpack=True)
        scale = (1.0 * table_unit).to_value(u.rad)
        return np.asarray(first) * scale, np.asarray(second) * scale

    if dist_type == "UniformAllSky":
        first = np.random.uniform(0.0, 2.0 * np.pi, n_samples)
        latitude = np.arcsin(2.0 * np.random.uniform(0, 1, n_samples) - 1.0)
        second = convert_dec_to_theta(latitude) if coordsys == "cwb" else latitude
        return first, second

    if dist_type == "Patch":
        patch = sky_distribution_config["patch"]
        if "center" not in patch or "radius" not in patch:
            raise ValueError("Patch requires center and radius")
        legacy_unit = patch.get("unit")
        center_first, center_second = parse_sky_coordinates(
            patch["center"],
            coordsys,
            context="sky_distribution.patch.center",
            legacy_unit=legacy_unit,
        )
        is_legacy = "phi" in patch["center"] and "theta" in patch["center"]
        radius = parse_angle_quantity(
            patch["radius"],
            name="sky_distribution.patch.radius",
            legacy_unit=legacy_unit,
            allow_legacy_numeric=is_legacy,
        )
        if not 0.0 <= radius <= np.pi:
            raise ValueError("Patch radius must be within [0 deg, 180 deg]")

        center_latitude = (
            convert_theta_to_dec(center_second) if coordsys == "cwb" else center_second
        )
        first, latitude = _sample_uniform_sky_area_rad(
            center_first, center_latitude, radius, n_samples
        )
        second = convert_dec_to_theta(latitude) if coordsys == "cwb" else latitude
        return first, second

    if dist_type == "Fixed":
        coordinates = sky_distribution_config["coordinates"]
        if "sky_loc" in coordinates:
            warnings.warn(
                "coordinates.sky_loc is deprecated; put semantic coordinate keys "
                "directly under coordinates",
                DeprecationWarning,
                stacklevel=2,
            )
            values = coordinates["sky_loc"]
        else:
            values = coordinates
        first, second = parse_sky_coordinates(
            values,
            coordsys,
            context="sky_distribution.coordinates",
            legacy_unit=coordinates.get("unit"),
        )
        return np.repeat(first, n_samples), np.repeat(second, n_samples)

    if dist_type == "Custom":
        if hp is None:
            raise ImportError("healpy is required for Custom HEALPix sky distribution")
        custom = sky_distribution_config["custom"]
        ordering = str(custom.get("ordering", "ring")).lower()
        if ordering not in {"ring", "nested"}:
            raise ValueError("custom.ordering must be 'ring' or 'nested'")
        nest = ordering == "nested"
        skymap = np.asarray(hp.read_map(custom["healpix_map"], nest=nest), dtype=float)
        map_nside = int(hp.get_nside(skymap))
        nside = int(custom.get("nside", map_nside))
        if nside != map_nside:
            raise ValueError("custom.nside does not match the loaded HEALPix map")

        weights = np.clip(skymap, 0.0, None)
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("Custom HEALPix sky distribution has no positive probability")
        indices = np.random.choice(len(weights), size=n_samples, p=weights / total)
        theta_hp, first = hp.pix2ang(nside, indices, nest=nest)
        latitude = np.pi / 2.0 - theta_hp
        second = convert_dec_to_theta(latitude) if coordsys == "cwb" else latitude
        return first, second

    raise ValueError(f"Unknown distribution type: {dist_type}")


def distribute_injections_on_sky(injections, sky_locations, shuffle=True, coordsys="icrs"):
    """Attach radian sky coordinates to injection dictionaries."""
    coordsys = normalize_coordinate_system(coordsys)
    first = np.asarray(sky_locations[0])
    second = np.asarray(sky_locations[1])
    if len(injections) != len(first) or len(injections) != len(second):
        raise ValueError("The number of injections and sky locations must be the same")

    if shuffle:
        coordinates = np.column_stack((first, second))
        np.random.shuffle(coordinates)
        first, second = coordinates[:, 0], coordinates[:, 1]

    if coordsys == "icrs":
        for i, injection in enumerate(injections):
            injection["ra"] = float(first[i])
            injection["dec"] = float(second[i])
    else:
        for i, injection in enumerate(injections):
            injection["sky_loc"] = [float(first[i]), float(second[i])]
            injection["coordsys"] = coordsys


def _sample_uniform_sky_area_rad(longitude_center, latitude_center, radius, n_samples=1):
    """Sample uniformly in a spherical cap; all arguments/results are radians."""
    cos_alpha = np.random.uniform(np.cos(radius), 1.0, n_samples)
    alpha = np.arccos(cos_alpha)
    beta = np.random.uniform(0.0, 2.0 * np.pi, n_samples)

    x = np.sin(alpha) * np.cos(beta)
    y = np.sin(alpha) * np.sin(beta)
    z = np.cos(alpha)

    sin_lat = np.sin(latitude_center)
    cos_lat = np.cos(latitude_center)
    sin_lon = np.sin(longitude_center)
    cos_lon = np.cos(longitude_center)

    x_new = sin_lat * cos_lon * x - sin_lon * y + cos_lat * cos_lon * z
    y_new = sin_lat * sin_lon * x + cos_lon * y + cos_lat * sin_lon * z
    z_new = -cos_lat * x + sin_lat * z

    longitude = np.arctan2(y_new, x_new) % (2.0 * np.pi)
    latitude = np.arcsin(np.clip(z_new, -1.0, 1.0))
    return longitude, latitude


def sample_uniform_sky_area(phi_center, theta_center, radius, n_samples=1):
    """Backward-compatible degree API returning ``(longitude, latitude)``."""
    longitude, latitude = _sample_uniform_sky_area_rad(
        np.deg2rad(phi_center),
        np.deg2rad(theta_center),
        np.deg2rad(radius),
        n_samples,
    )
    return np.array([np.rad2deg(longitude), np.rad2deg(latitude)])
