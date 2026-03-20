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
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.array([x, y, z])


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    ra = np.arctan2(y, x)
    dec = np.arcsin(z / r)
    return ra, dec


def max_delay(ifos):
    """Maximum light-travel time between any detector pair (seconds).

    Uses the simple geometric baseline: ``|r_i - r_j| / c``, consistent
    with the cWB ``getDelay("MAX")`` approach.
    """
    from astropy import constants
    from pycwb.types.detector import Detector

    c = float(constants.c.value)
    max_d = 0.0
    for i in range(len(ifos)):
        for j in range(i + 1, len(ifos)):
            d1 = Detector(ifos[i])
            d2 = Detector(ifos[j])
            baseline = np.linalg.norm(
                d1.vertex_vec_earth_centered - d2.vertex_vec_earth_centered
            )
            max_d = max(max_d, baseline / c)
    return max_d
