from pycbc.detector import Detector
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
    # find all combinations of detectors
    ifo_pairs = [(ifos[i], ifos[j]) for i in range(len(ifos)) for j in range(i + 1, len(ifos))]

    # find the maximum delay between any two detectors
    max_delay = 0
    for ifo1, ifo2 in ifo_pairs:
        det1 = Detector(ifo1)
        det2 = Detector(ifo2)

        # find the orientation of the detectors
        v1 = np.array(det1.optimal_orientation(0))
        v2 = np.array(det2.optimal_orientation(0))
        v1_cart = spherical_to_cartesian(v1[0], v1[1])
        v2_cart = spherical_to_cartesian(v2[0], v2[1])
        v3_cart = v2_cart - v1_cart
        v3 = cartesian_to_spherical(v3_cart[0], v3_cart[1], v3_cart[2])

        delay = abs(det1.time_delay_from_detector(det2, v3[0], v3[1], 0))

        max_delay = max(max_delay, delay)

    return max_delay
