"""Coordinate transformation utilities for pycWB.

Pure NumPy geometry functions with no external dependencies beyond numpy.
Used by both ``pycwb.types.detector`` and ``pycwb.utils.network`` to avoid
import cycles.
"""

import numpy as np


def local_to_earth_centered(lat, lon, east, north, up):
    """
    Transform local horizon frame to Earth-centered Cartesian coordinates.

    Parameters
    ----------
    lat : float
        Latitude in radians
    lon : float
        Longitude in radians
    east : float
        East component in local frame
    north : float
        North component in local frame
    up : float
        Up component in local frame

    Returns
    -------
    np.ndarray
        Unit vector in Earth-centered coordinates
    """
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    # Local frame basis vectors in Earth-centered coords
    east_vec = np.array([-sin_lon, cos_lon, 0.0])
    north_vec = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up_vec = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])

    vec_earth = east * east_vec + north * north_vec + up * up_vec
    return vec_earth / np.linalg.norm(vec_earth)


def spherical_to_cartesian(ra, dec):
    """Convert spherical (ra, dec) to Cartesian unit vector.

    Parameters
    ----------
    ra : float
        Right ascension in radians.
    dec : float
        Declination in radians.

    Returns
    -------
    np.ndarray
        Unit vector [x, y, z].
    """
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.array([x, y, z])


def cartesian_to_spherical(x, y, z):
    """Convert Cartesian unit vector to spherical (ra, dec).

    Parameters
    ----------
    x, y, z : float
        Cartesian components.

    Returns
    -------
    ra : float
        Right ascension in radians.
    dec : float
        Declination in radians.
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    # Right ascension follows the package-wide ICRS contract [0, 2*pi).
    ra = np.arctan2(y, x) % (2.0 * np.pi)
    dec = np.arcsin(z / r)
    return ra, dec
