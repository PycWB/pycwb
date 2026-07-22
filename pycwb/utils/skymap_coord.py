"""Sky-coordinate conversion and user-angle parsing utilities.

Canonical internal angles are radians.  Coordinate names have one meaning:

* ICRS: ``ra``, ``dec``
* geographic/Earth fixed: ``longitude``, ``latitude``
* cWB Earth fixed: ``phi_geo``, ``theta_cwb`` (co-latitude)

The cWB sidereal-time polynomial is retained explicitly for numerical parity
with ``cwb-core/skymap.hh``.  Astropy GMST is available separately for physical
detector projection.  Callers must choose one model for a complete round trip.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping

import numpy as np
from astropy import units as u

logger = logging.getLogger(__name__)

TWO_PI = 2.0 * np.pi
EPOCH_J2000_0_GPS = 630763213.0

COORDINATE_KEYS = {
    "icrs": ("ra", "dec"),
    "geo": ("longitude", "latitude"),
    "cwb": ("phi_geo", "theta_cwb"),
}
_ALL_SEMANTIC_KEYS = frozenset(key for pair in COORDINATE_KEYS.values() for key in pair)


def normalize_coordinate_system(coordinate_system: str | None, *, default: str = "icrs") -> str:
    """Return a validated lower-case coordinate-system name."""
    value = default if coordinate_system is None else str(coordinate_system).lower()
    if value not in COORDINATE_KEYS:
        supported = ", ".join(sorted(COORDINATE_KEYS))
        raise ValueError(
            f"Coordinate system {coordinate_system!r} is not recognized; "
            f"supported values are: {supported}"
        )
    return value


def parse_angle_quantity(
    value,
    *,
    name: str,
    legacy_unit: str | u.UnitBase | None = None,
    allow_legacy_numeric: bool = False,
) -> float:
    """Parse one angular value and return radians.

    New user-facing values must be an Astropy ``Quantity`` or a quantity string
    such as ``"120 deg"``.  Bare numeric values are accepted only for an
    explicitly enabled legacy path with a supplied unit.
    """
    if isinstance(value, u.Quantity):
        quantity = value
    elif isinstance(value, str):
        try:
            quantity = u.Quantity(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an angular quantity, got {value!r}") from exc
    elif allow_legacy_numeric and legacy_unit is not None and np.isscalar(value):
        try:
            quantity = float(value) * u.Unit(legacy_unit)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid legacy angular unit {legacy_unit!r} for {name}") from exc
    else:
        raise ValueError(
            f"{name} must include an angular unit, for example '120 deg' or "
            f"'2.094 rad'; bare numeric angles are ambiguous"
        )

    if not quantity.unit.is_equivalent(u.rad):
        raise ValueError(f"{name} must have angular units, got {quantity.unit}")
    return float(quantity.to_value(u.rad))


def parse_sky_coordinates(
    values: Mapping,
    coordinate_system: str,
    *,
    context: str = "sky coordinates",
    legacy_unit: str | u.UnitBase | None = None,
    allow_legacy: bool = True,
) -> tuple[float, float]:
    """Parse and validate a semantic sky-coordinate pair.

    The key pair is cross-checked against ``coordinate_system``.  The legacy
    ``phi``/``theta`` pair remains available temporarily when ``allow_legacy``
    is true; it requires a detached legacy unit and emits a deprecation warning.
    """
    if not isinstance(values, Mapping):
        raise ValueError(f"{context} must be a mapping")

    coordsys = normalize_coordinate_system(coordinate_system)
    expected = COORDINATE_KEYS[coordsys]
    present_semantic = _ALL_SEMANTIC_KEYS.intersection(values)
    present_legacy = {"phi", "theta"}.intersection(values)
    used_legacy = False

    if all(key in values for key in expected):
        unexpected = present_semantic.difference(expected)
        if unexpected or present_legacy:
            raise ValueError(
                f"{context} mixes coordinate frames: coordsys={coordsys!r} requires "
                f"{expected}, but also found {sorted(unexpected | present_legacy)}"
            )
        first = parse_angle_quantity(values[expected[0]], name=f"{context}.{expected[0]}")
        second = parse_angle_quantity(values[expected[1]], name=f"{context}.{expected[1]}")
    elif present_semantic:
        raise ValueError(
            f"{context} does not match coordsys={coordsys!r}: expected keys "
            f"{expected}, found {sorted(present_semantic)}"
        )
    elif allow_legacy and "phi" in values and "theta" in values:
        used_legacy = True
        if legacy_unit is None:
            raise ValueError(
                f"Legacy {context} phi/theta values require an explicit angular unit"
            )
        warnings.warn(
            f"Legacy {context} keys 'phi'/'theta' and detached 'unit' are deprecated; "
            f"use {expected[0]!r}/{expected[1]!r} with unit-bearing values",
            DeprecationWarning,
            stacklevel=2,
        )
        first = parse_angle_quantity(
            values["phi"], name=f"{context}.phi", legacy_unit=legacy_unit,
            allow_legacy_numeric=True,
        )
        second = parse_angle_quantity(
            values["theta"], name=f"{context}.theta", legacy_unit=legacy_unit,
            allow_legacy_numeric=True,
        )
    else:
        raise ValueError(
            f"{context} with coordsys={coordsys!r} requires keys "
            f"{expected[0]!r} and {expected[1]!r}"
        )

    if coordsys == "icrs":
        if not used_legacy and not 0.0 <= first <= TWO_PI:
            raise ValueError(f"{context}.ra must be within [0 deg, 360 deg]")
        first %= TWO_PI
        if not -np.pi / 2.0 <= second <= np.pi / 2.0:
            raise ValueError(f"{context}.dec must be within [-90 deg, 90 deg]")
    elif coordsys == "geo":
        if not used_legacy and not -np.pi <= first <= np.pi:
            raise ValueError(
                f"{context}.longitude must be within [-180 deg, 180 deg]"
            )
        first = (first + np.pi) % TWO_PI - np.pi
        if not -np.pi / 2.0 <= second <= np.pi / 2.0:
            raise ValueError(f"{context}.latitude must be within [-90 deg, 90 deg]")
    else:
        if not used_legacy and not 0.0 <= first <= TWO_PI:
            raise ValueError(f"{context}.phi_geo must be within [0 deg, 360 deg]")
        first %= TWO_PI
        if not 0.0 <= second <= np.pi:
            raise ValueError(f"{context}.theta_cwb must be within [0 deg, 180 deg]")

    return first, second


def validate_user_sky_config(config, *, context: str, default_coordsys: str) -> None:
    """Validate frame names and angular quantities in one YAML sky block.

    Legacy ``phi/theta`` plus detached ``unit`` remains accepted with warnings.
    New semantic key sets require an explicit ``coordsys`` so the name-derived
    frame and declared frame can be cross-checked.
    """
    if not config:
        return
    dist_type = config.get("type", "UniformAllSky")
    coordsys = normalize_coordinate_system(config.get("coordsys", default_coordsys))
    if dist_type == "UniformAllSky":
        return

    def _require_explicit_coordsys(values):
        if _ALL_SEMANTIC_KEYS.intersection(values) and "coordsys" not in config:
            raise ValueError(
                f"{context} uses semantic coordinate names and therefore requires "
                "an explicit coordsys for cross-validation"
            )

    if dist_type == "Patch":
        patch = config.get("patch", {})
        center = patch.get("center", {})
        _require_explicit_coordsys(center)
        legacy_unit = patch.get("unit")
        parse_sky_coordinates(
            center, coordsys, context=f"{context}.patch.center",
            legacy_unit=legacy_unit,
        )
        is_legacy = "phi" in center and "theta" in center
        radius = parse_angle_quantity(
            patch.get("radius"), name=f"{context}.patch.radius",
            legacy_unit=legacy_unit, allow_legacy_numeric=is_legacy,
        )
        if not 0.0 <= radius <= np.pi:
            raise ValueError(f"{context}.patch.radius must be within [0 deg, 180 deg]")
        return

    if dist_type == "Fixed":
        coordinates = config.get("coordinates", {})
        values = coordinates.get("sky_loc", coordinates)
        _require_explicit_coordsys(values)
        parse_sky_coordinates(
            values, coordsys, context=f"{context}.coordinates",
            legacy_unit=coordinates.get("unit"),
        )
        return

    if dist_type == "existing":
        unit = config.get("unit")
        try:
            parsed_unit = u.Unit(unit)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}.unit must be an angular unit") from exc
        if not parsed_unit.is_equivalent(u.rad):
            raise ValueError(f"{context}.unit must be angular, got {parsed_unit}")
        columns = config.get("columns")
        if columns is not None and list(columns) != list(COORDINATE_KEYS[coordsys]):
            raise ValueError(
                f"{context}.columns do not match coordsys={coordsys!r}; "
                f"expected {list(COORDINATE_KEYS[coordsys])}"
            )
        return

    if dist_type == "Custom":
        custom = config.get("custom", {})
        if not isinstance(custom, Mapping) or not custom.get("healpix_map"):
            raise ValueError(f"{context}.custom.healpix_map is required")
        ordering = str(custom.get("ordering", "ring")).lower()
        if ordering not in {"ring", "nested"}:
            raise ValueError(f"{context}.custom.ordering must be 'ring' or 'nested'")
        if "nside" in custom:
            nside = int(custom["nside"])
            if nside <= 0 or nside & (nside - 1):
                raise ValueError(f"{context}.custom.nside must be a positive power of two")
        return

    raise ValueError(f"Unknown {context} type: {dist_type!r}")


def gmst_cwb(gps_time):
    """Return cWB-compatible GMST in radians.

    This is a direct unit-converted transcription of
    ``cwb-core/skymap.hh::skymap::phiRA``.
    """
    gps = np.asarray(gps_time, dtype=np.float64)
    t = (gps - EPOCH_J2000_0_GPS) / 86400.0 / 36525.0
    sidereal_time_sec = (
        (-6.2e-6 * t + 0.093104) * t * t
        + 67310.54841
        + 8640184.812866 * t
        + 3155760000.0 * t
    )
    result = np.deg2rad((360.0 * sidereal_time_sec / 86400.0) % 360.0)
    return float(result) if result.ndim == 0 else result


def gmst_astropy(gps_time: float) -> float:
    """Return Astropy mean sidereal time in radians for physical projection."""
    from astropy.time import Time

    return float(
        Time(float(gps_time), format="gps", scale="utc", location=(0, 0))
        .sidereal_time("mean")
        .rad
    )


def _gmst_for_model(gps_time, gmst_model: str):
    model = str(gmst_model).lower()
    if model == "cwb":
        return gmst_cwb(gps_time)
    if model == "astropy":
        return gmst_astropy(gps_time)
    raise ValueError("gmst_model must be 'cwb' or 'astropy'")


def convert_phi_to_ra(phi, gps_time, *, gmst_model: str = "cwb"):
    """Convert Earth-fixed longitude to ICRS RA in radians."""
    return (np.asarray(phi) + _gmst_for_model(gps_time, gmst_model)) % TWO_PI


def convert_ra_to_phi(ra, gps_time, *, gmst_model: str = "cwb"):
    """Convert ICRS RA to Earth-fixed cWB longitude in radians."""
    return (np.asarray(ra) - _gmst_for_model(gps_time, gmst_model)) % TWO_PI


def convert_theta_to_dec(theta):
    """Convert cWB co-latitude to latitude/declination in radians."""
    return np.pi / 2.0 - np.asarray(theta)


def convert_dec_to_theta(dec):
    """Convert latitude/declination to cWB co-latitude in radians."""
    return np.pi / 2.0 - np.asarray(dec)


def convert_cwb_to_geo(phi_geo, theta_cwb):
    """Convert cWB longitude/co-latitude to geographic lon/lat radians."""
    longitude = (np.asarray(phi_geo) + np.pi) % TWO_PI - np.pi
    latitude = convert_theta_to_dec(theta_cwb)
    return longitude, latitude


def convert_geo_to_cwb(longitude, latitude):
    """Convert geographic lon/lat to cWB longitude/co-latitude radians."""
    phi_geo = np.asarray(longitude) % TWO_PI
    theta_cwb = convert_dec_to_theta(latitude)
    return phi_geo, theta_cwb


def convert_to_celestial_coordinates(
    first,
    second,
    gps_time: float,
    coordinate_system: str | None = None,
    *,
    gmst_model: str = "cwb",
):
    """Convert a radian coordinate pair to ICRS ``(RA, Dec)``.

    For detector projection from Earth-fixed input, pass
    ``gmst_model='astropy'`` so the conversion uses the same sidereal-time
    implementation as :meth:`pycwb.types.detector.Detector.atenna_pattern`.
    """
    coordsys = normalize_coordinate_system(coordinate_system)
    if coordsys == "icrs":
        return np.asarray(first), np.asarray(second)
    if coordsys == "geo":
        return convert_phi_to_ra(first, gps_time, gmst_model=gmst_model), np.asarray(second)
    return (
        convert_phi_to_ra(first, gps_time, gmst_model=gmst_model),
        convert_theta_to_dec(second),
    )


def gmst_rad(gps_time):
    """Backward-compatible alias for :func:`gmst_cwb`."""
    return gmst_cwb(gps_time)


__all__ = [
    "COORDINATE_KEYS",
    "convert_cwb_to_geo",
    "convert_dec_to_theta",
    "convert_geo_to_cwb",
    "convert_phi_to_ra",
    "convert_ra_to_phi",
    "convert_theta_to_dec",
    "convert_to_celestial_coordinates",
    "gmst_astropy",
    "gmst_cwb",
    "gmst_rad",
    "normalize_coordinate_system",
    "parse_angle_quantity",
    "parse_sky_coordinates",
    "validate_user_sky_config",
]
