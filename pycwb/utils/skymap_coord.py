import numpy as np
import logging 

logger = logging.getLogger(__name__)


def convert_cwb_to_geo(phi_deg, theta_deg):
    # for single point
    if isinstance(phi_deg, float) and isinstance(theta_deg, float):
        phi_deg = phi_deg - 360 if phi_deg > 180 else phi_deg
        theta_deg = -(theta_deg - 90)

        return phi_deg, theta_deg

    phi_deg[phi_deg > 180] -= 360
    theta_deg = -(theta_deg - 90)

    return phi_deg, theta_deg


def convert_to_celestial_coordinates(phi,theta,gps_time,coordinate_system = None): 
        """
        Covert given coordinate system to celestial (RA, Dec) coordinates and save them in the injection dictionary
        """
        if coordinate_system is None: 
            logger.info(f'No coordinate system provided, the input "phi" and "theta" will be interpreted as "right ascension" and "declination" respectively')
            coordinate_system = 'icrs'

        if coordinate_system == 'icrs': 
            right_ascension = phi
            declination = theta

        elif coordinate_system == 'geo':
            right_ascension = convert_phi_to_ra(phi, gps_time) 
            declination = theta 

        elif coordinate_system == 'cwb':
            phi_deg, theta_deg = convert_cwb_to_geo(np.rad2deg(phi), np.rad2deg(theta))
            right_ascension = convert_phi_to_ra(np.deg2rad(np.deg2rad(phi_deg)), gps_time)
            declination = np.deg2rad(theta_deg) 

        else:  
            raise ValueError(f"Coordinate system {coordinate_system} not recognized. Supported systems are 'geo', 'cwb' and 'celestial'.")

        return right_ascension, declination

def convert_phi_to_ra(ph_rad, gps_time, inverse=False):
    """
    Convert between local azimuth-like angle `phi` and sky-fixed RA, in radians.
    If inverse=False: returns RA(t) in radians
    If inverse=True: returns azimuth `phi` in radians
    """
    gmst = gmst_rad(gps_time)
    if inverse:
        gmst = -gmst
    ra = (ph_rad + gmst) % (2 * np.pi) 
    return ra 


def gmst_rad(gps_time):
    """Compute GMST (Greenwich Mean Sidereal Time) in radians."""
    EPOCH_J2000_GPS = 630763213.0  # GPS time at J2000.0

    t = (gps_time - EPOCH_J2000_GPS) / 86400.0 / 36525.0  # Julian centuries from J2000
    sidereal_time_sec = ((-6.2e-6 * t + 0.093104) * t**2 +
                         67310.54841 +
                         8640184.812866 * t +
                         3155760000.0 * t)
    gmst_deg = (360.0 * sidereal_time_sec / 86400.0) % 360.0
    gmst_rad = np.deg2rad(gmst_deg)
    return gmst_rad