"""
Sky mask utilities for restricting the likelihood sky scan.

This module is intentionally kept free of heavy pycwb imports so that it can
be loaded (and tested) without cartopy, ROOT, or numba being available.
"""
from __future__ import annotations

import logging

import numpy as np

from pycwb.types.detector import gmst_accurate


logger = logging.getLogger(__name__)


def compute_sky_valid_indices(ra_arr, dec_arr, sky_mask_config, t_ref=None):
    """
    Compute an array of valid sky indices from a sky mask configuration.

    Uses the same format as the injection ``sky_distribution`` config block so
    that the same YAML syntax can restrict the sky scan without any CLI option.

    Parameters
    ----------
    ra_arr : np.ndarray
        Right ascension of sky directions in radians, shape (n_sky,).
    dec_arr : np.ndarray
        Declination of sky directions in radians, shape (n_sky,).
    sky_mask_config : dict or None
        Sky mask config dict (``sky_mask`` key in user YAML).  Supported
        ``type`` values mirror the injection sky distribution:

        * ``UniformAllSky`` (default / ``None``) — no restriction.
        * ``Patch`` — circular cap on the sphere; requires ``patch.center.phi``
          (RA), ``patch.center.theta`` (Dec), ``patch.radius``, ``patch.unit``
          (``"rad"`` or ``"deg"``).
        * ``Fixed`` — single nearest sky direction; requires
          ``coordinates.sky_loc.phi``, ``coordinates.sky_loc.theta``,
          ``coordinates.unit``.
        * ``Custom`` — threshold a HEALPix probability map; requires
          ``custom.healpix_map`` (path), ``custom.nside``, and optionally
          ``custom.threshold`` (default 0).
    t_ref : float
        Reference GPS time used to convert ICRS right ascension to geocentric
        longitude when ``coordsys`` is set to ``"icrs"``. If ``None``, no
        time-dependent conversion is applied.

    Returns
    -------
    np.ndarray
        1-D int64 array of valid sky indices (subset of ``range(n_sky)``).
        Returns ``np.arange(n_sky)`` when no mask is applied.
    """
    ra_arr = np.asarray(ra_arr, dtype=np.float64)
    dec_arr = np.asarray(dec_arr, dtype=np.float64)
    n_sky = len(ra_arr)

    if sky_mask_config is None:
        return np.arange(n_sky, dtype=np.int64)

    dist_type = sky_mask_config.get('type', 'UniformAllSky')
    logger.info("Applying sky mask of type: %s", dist_type)

    coordsys = sky_mask_config.get('coordsys', 'geo')
    if dist_type == 'UniformAllSky':
        return np.arange(n_sky, dtype=np.int64)

    elif dist_type == 'Patch':
        patch = sky_mask_config['patch']
        unit = patch.get('unit', 'rad')
        center_phi = float(patch['center']['phi'])    # RA
        center_theta = float(patch['center']['theta'])  # Dec
        radius = float(patch['radius'])
        if unit == 'deg':
            center_phi = np.deg2rad(center_phi)
            center_theta = np.deg2rad(center_theta)
            radius = np.deg2rad(radius)
        # Convert ICRS right ascension to geocentric longitude using GMST
        # at the reference time of the analyzed strain data.
        if coordsys.lower() == 'icrs':
            if t_ref is not None:
                gmst = gmst_accurate(t_ref)
                center_phi = (center_phi - gmst) % (2.0 * np.pi)  # RA -> GEO phi
        # Angular separation via dot-product formula (numerically stable)
        cos_d = (np.sin(dec_arr) * np.sin(center_theta) +
                 np.cos(dec_arr) * np.cos(center_theta) * np.cos(ra_arr - center_phi))
        cos_d = np.clip(cos_d, -1.0, 1.0)
        valid = np.where(np.arccos(cos_d) <= radius)[0].astype(np.int64)
        logger.info(
            "Sky mask Patch: center (phi=%.4f rad, theta=%.4f rad), radius=%.4f rad "
            "-> %d / %d directions valid",
            center_phi, center_theta, radius, len(valid), n_sky,
        )
        return valid

    elif dist_type == 'Fixed':
        coords = sky_mask_config['coordinates']
        unit = coords.get('unit', 'rad')
        phi = float(coords['sky_loc']['phi'])
        theta = float(coords['sky_loc']['theta'])
        if unit == 'deg':
            phi = np.deg2rad(phi)
            theta = np.deg2rad(theta)

        # Convert ICRS right ascension to geocentric longitude using GMST
        # at the reference time of the analyzed strain data.
        if coordsys.lower() == 'icrs':
            if t_ref is not None:
                gmst = gmst_accurate(t_ref)
                phi = (phi - gmst) % (2.0 * np.pi)  # RA -> GEO phi
        cos_d = (np.sin(dec_arr) * np.sin(theta) +
                 np.cos(dec_arr) * np.cos(theta) * np.cos(ra_arr - phi))
        cos_d = np.clip(cos_d, -1.0, 1.0)
        nearest = int(np.argmin(np.arccos(cos_d)))
        logger.info("Sky mask Fixed: nearest direction index %d", nearest)
        return np.array([nearest], dtype=np.int64)

    elif dist_type == 'Custom':
        import healpy as hp
        custom = sky_mask_config['custom']
        skymap = hp.read_map(custom['healpix_map'])
        nside = int(custom['nside'])
        threshold = float(custom.get('threshold', 0.0))
        theta_hp = np.pi / 2.0 - dec_arr  # co-latitude for healpy
        pixels = hp.ang2pix(nside, theta_hp, ra_arr)
        valid = np.where(skymap[pixels] > threshold)[0].astype(np.int64)
        logger.info(
            "Sky mask Custom: %d / %d valid directions (threshold=%.4f)",
            len(valid), n_sky, threshold,
        )
        return valid

    else:
        raise ValueError(
            f"Unknown sky_mask type: {dist_type!r}. "
            "Supported: UniformAllSky, Patch, Fixed, Custom"
        )
