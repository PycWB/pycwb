"""Sky masks for the Earth-fixed native likelihood grid."""

from __future__ import annotations

import logging
import warnings
from functools import lru_cache

import numpy as np

from pycwb.utils.skymap_coord import (
    convert_phi_to_ra,
    convert_ra_to_phi,
    convert_theta_to_dec,
    normalize_coordinate_system,
    parse_angle_quantity,
    parse_sky_coordinates,
)

logger = logging.getLogger(__name__)


def sky_mask_requires_event_time(sky_mask_config) -> bool:
    """Return whether a configured mask rotates relative to the cWB grid."""
    if not sky_mask_config or sky_mask_config.get("type", "UniformAllSky") == "UniformAllSky":
        return False
    return normalize_coordinate_system(sky_mask_config.get("coordsys", "geo")) == "icrs"


def _mask_coordsys(sky_mask_config) -> str:
    if "coordsys" not in sky_mask_config:
        warnings.warn(
            "Restricted sky masks without explicit coordsys are deprecated; "
            "legacy behavior is coordsys='geo'",
            DeprecationWarning,
            stacklevel=3,
        )
    return normalize_coordinate_system(sky_mask_config.get("coordsys", "geo"))


def _to_earth_fixed(first, second, coordsys: str, t_ref: float | None, *, context: str):
    if coordsys == "icrs":
        if t_ref is None:
            raise ValueError(f"{context} with coordsys='icrs' requires a GPS reference time")
        return float(convert_ra_to_phi(first, t_ref, gmst_model="cwb")), second
    if coordsys == "geo":
        return first % (2.0 * np.pi), second
    return first % (2.0 * np.pi), float(convert_theta_to_dec(second))


def _legacy_values(container):
    """Unwrap the old coordinates.sky_loc object."""
    if "sky_loc" in container:
        warnings.warn(
            "coordinates.sky_loc is deprecated; put semantic coordinate keys "
            "directly under coordinates",
            DeprecationWarning,
            stacklevel=3,
        )
        return container["sky_loc"]
    return container


@lru_cache(maxsize=16)
def _read_healpix_map(path: str, nest: bool):
    import healpy as hp

    return np.asarray(hp.read_map(path, nest=nest), dtype=np.float64)


def compute_sky_valid_indices(
    phi_geo_arr,
    latitude_arr,
    sky_mask_config,
    t_ref=None,
):
    """Return valid Earth-fixed sky-grid indices for a configured mask.

    Parameters are radians.  ``phi_geo_arr`` is cWB Earth-fixed longitude and
    ``latitude_arr`` is geographic latitude (numerically equal to declination,
    but not an ICRS coordinate until longitude is rotated at a GPS epoch).

    The ICRS transformation matches ``cwb-core/skymap.hh`` and the celestial
    mask loop in ``cwb-core/network.cc``: ``RA = phi_geo + GMST(trigger_time)``.
    """
    phi_geo_arr = np.asarray(phi_geo_arr, dtype=np.float64)
    latitude_arr = np.asarray(latitude_arr, dtype=np.float64)
    if phi_geo_arr.shape != latitude_arr.shape:
        raise ValueError("Sky longitude and latitude arrays must have identical shapes")
    n_sky = len(phi_geo_arr)

    if sky_mask_config is None:
        return np.arange(n_sky, dtype=np.int64)

    dist_type = sky_mask_config.get("type", "UniformAllSky")
    if dist_type == "UniformAllSky":
        return np.arange(n_sky, dtype=np.int64)

    coordsys = _mask_coordsys(sky_mask_config)
    logger.info("Applying sky mask type=%s coordsys=%s at GPS=%s", dist_type, coordsys, t_ref)

    if dist_type == "Patch":
        patch = sky_mask_config["patch"]
        legacy_unit = patch.get("unit", "rad")
        first, second = parse_sky_coordinates(
            patch["center"], coordsys, context="sky_mask.patch.center",
            legacy_unit=legacy_unit,
        )
        is_legacy = "phi" in patch["center"] and "theta" in patch["center"]
        radius = parse_angle_quantity(
            patch["radius"], name="sky_mask.patch.radius",
            legacy_unit=legacy_unit, allow_legacy_numeric=is_legacy,
        )
        if not 0.0 <= radius <= np.pi:
            raise ValueError("sky_mask.patch.radius must be within [0 deg, 180 deg]")
        center_phi, center_latitude = _to_earth_fixed(
            first, second, coordsys, t_ref, context="sky_mask.patch"
        )
        cos_d = (
            np.sin(latitude_arr) * np.sin(center_latitude)
            + np.cos(latitude_arr) * np.cos(center_latitude)
            * np.cos(phi_geo_arr - center_phi)
        )
        valid = np.where(np.arccos(np.clip(cos_d, -1.0, 1.0)) <= radius)[0]
        return valid.astype(np.int64)

    if dist_type == "Fixed":
        coordinates = sky_mask_config["coordinates"]
        values = _legacy_values(coordinates)
        first, second = parse_sky_coordinates(
            values, coordsys, context="sky_mask.coordinates",
            legacy_unit=coordinates.get("unit", "rad"),
        )
        fixed_phi, fixed_latitude = _to_earth_fixed(
            first, second, coordsys, t_ref, context="sky_mask.coordinates"
        )
        if n_sky == 0:
            return np.array([], dtype=np.int64)
        cos_d = (
            np.sin(latitude_arr) * np.sin(fixed_latitude)
            + np.cos(latitude_arr) * np.cos(fixed_latitude)
            * np.cos(phi_geo_arr - fixed_phi)
        )
        nearest = int(np.argmax(np.clip(cos_d, -1.0, 1.0)))
        return np.array([nearest], dtype=np.int64)

    if dist_type == "Custom":
        import healpy as hp

        if coordsys == "icrs" and t_ref is None:
            raise ValueError("sky_mask Custom with coordsys='icrs' requires a GPS reference time")
        custom = sky_mask_config["custom"]
        ordering = str(custom.get("ordering", "ring")).lower()
        if ordering not in {"ring", "nested"}:
            raise ValueError("sky_mask.custom.ordering must be 'ring' or 'nested'")
        nest = ordering == "nested"
        skymap = _read_healpix_map(str(custom["healpix_map"]), nest)
        map_nside = int(hp.get_nside(skymap))
        nside = int(custom.get("nside", map_nside))
        if nside != map_nside or len(skymap) != hp.nside2npix(nside):
            raise ValueError("sky_mask.custom.nside does not match the loaded HEALPix map")

        map_longitude = phi_geo_arr
        if coordsys == "icrs":
            map_longitude = convert_phi_to_ra(
                phi_geo_arr, t_ref, gmst_model="cwb"
            )
        theta_hp = np.pi / 2.0 - latitude_arr
        pixels = hp.ang2pix(nside, theta_hp, map_longitude, nest=nest)
        threshold = float(custom.get("threshold", 0.0))
        return np.where(skymap[pixels] > threshold)[0].astype(np.int64)

    raise ValueError(
        f"Unknown sky_mask type: {dist_type!r}; supported types are "
        "UniformAllSky, Patch, Fixed, Custom"
    )


def sky_valid_indices_for_cluster(setup, cluster, *, use_big_grid: bool = False):
    """Return mask indices at the cluster GPS time, matching cWB's sky loop."""
    mask_config = setup.get("sky_mask_config")
    cached_key = "sky_valid_indices_big" if use_big_grid else "sky_valid_indices"
    cached = setup.get(cached_key)
    if not sky_mask_requires_event_time(mask_config):
        return cached

    segment_start = setup.get("segment_start_gps")
    if segment_start is None:
        raise ValueError(
            "An ICRS sky mask requires segment_start_gps so it can be evaluated "
            "at the cluster time"
        )
    cluster_offset = float(getattr(cluster, "cluster_time", 0.0) or 0.0)
    if cluster_offset == 0.0:
        meta = getattr(cluster, "cluster_meta", None)
        cluster_offset = float(getattr(meta, "c_time", 0.0) or 0.0)
    trigger_gps = float(segment_start) + cluster_offset

    if use_big_grid:
        phi_geo_arr = setup.get("phi_geo_arr_big_cluster")
        latitude_arr = setup.get("latitude_arr_big_cluster")
    else:
        phi_geo_arr = setup.get("phi_geo_arr", setup.get("ra_arr"))
        latitude_arr = setup.get("latitude_arr", setup.get("dec_arr"))
    return compute_sky_valid_indices(
        phi_geo_arr, latitude_arr, mask_config, t_ref=trigger_gps
    )


__all__ = [
    "compute_sky_valid_indices",
    "sky_mask_requires_event_time",
    "sky_valid_indices_for_cluster",
]
