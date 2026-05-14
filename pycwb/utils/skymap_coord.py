"""
Coordinate system conventions:

CELESTIAL (ICRS)
- ra, dec ([0, 360], [-90, 90])
GEO
- phi, theta ([-180, 180], [-90, 90])
CWB
- phi, theta ([0, 360], [0, 180])
"""

import numpy as np
import logging 

logger = logging.getLogger(__name__)

def convert_to_celestial_coordinates(phi: float | np.ndarray, theta: float | np.ndarray, gps_time: float, coordinate_system: str = None):
        """
        Convert given coordinate system to celestial (RA, Dec) coordinates and return them in radians.

        Parameters:
        ----------
        phi: float or np.ndarray
            The phi coordinate(s) [rad] in the input coordinate system.
        theta: float or np.ndarray
            The theta coordinate(s) [rad] in the input coordinate system.
        gps_time: float
            The GPS time corresponding to the coordinates, used for time-dependent transformations.
        coordinate_system: str, optional
            The coordinate system of the input phi and theta. Supported values are 'geo', 'cwb', and 'icrs'. 
            If None, defaults to 'icrs' (celestial coordinates).
        
        Returns:
        -------
        right_ascension: float or np.ndarray
            The right ascension(s) in radians.
        declination: float or np.ndarray
            The declination(s) in radians.
        """

        if coordinate_system is None: 
            logger.info(f'No coordinate system provided, the input "phi" and "theta" will be interpreted as "right ascension" and "declination" respectively')
            coordinate_system = 'icrs'

        if coordinate_system.lower() == 'icrs': 
            right_ascension = phi
            declination = theta
        
        elif coordinate_system.lower() == 'geo':
            # FIXME: convert_phi_to_ra can take geo as an input only for forward conversion
            # phi_cwb, _ = convert_cwb_to_geo(phi, 0., inverse=True)
            right_ascension = convert_phi_to_ra(phi, gps_time)
            declination = theta

        elif coordinate_system.lower() == 'cwb':
            right_ascension = convert_phi_to_ra(phi, gps_time)
            declination = convert_theta_to_dec(theta)

        else:  
            raise ValueError(f"Coordinate system {coordinate_system} not recognized. Supported systems are 'geo', 'cwb' and 'celestial'.")

        return right_ascension, declination

def convert_phi_to_ra(phi, gps_time, inverse=False):
    """
    Convert between cWB phi and RA by rotating by GMST. Both in radians.

    Parameters
    ----------
    phi : float or np.ndarray
        cWB phi [rad] if inverse=False, or RA [rad] if inverse=True.
    gps_time : float
        GPS time used to compute GMST for the rotation to RA.
    inverse : bool, optional
        If False (default): phi(CWB) -> RA. If True: RA -> phi(CWB).

    Returns
    -------
    float or np.ndarray
        RA [0, 2π) [rad] if inverse=False, or phi(CWB) [0, 2π) [rad] if inverse=True.

    Notes
    -----
    Forward (inverse=False) tolerates geo phi as input because the modular
    arithmetic handles negative values correctly. Inverse always returns [0, 2π) (cWB convention).
    """
    gmst = gmst_rad(gps_time)
    if inverse:
        gmst = -gmst
    ra = (phi + gmst) % (2 * np.pi)
    return ra

def convert_ra_to_phi(ra, gps_time):
    return convert_phi_to_ra(ra, gps_time, inverse=True)

def convert_theta_to_dec(theta):
    return  -(theta - np.pi / 2)

def convert_dec_to_theta(dec):
    return convert_theta_to_dec(dec)

def convert_cwb_to_geo(phi, theta, inverse=False):
    """
    Convert between cWB and geo angle conventions, in radians.

    Parameters
    ----------
    phi : float or np.ndarray
        cWB phi [rad] if inverse=False, or geo phi [rad] if inverse=True.
    theta : float or np.ndarray
        cWB theta [rad] if inverse=False, or Dec [rad] if inverse=True.
    inverse : bool, optional
        If False (default): cWB -> geo. If True: geo -> cWB.

    Returns
    -------
    phi : float or np.ndarray
        geo phi [-π, π) [rad] if inverse=False, or cWB phi [0, 2π) [rad] if inverse=True.
    theta : float or np.ndarray
        Dec [-π/2, π/2] [rad] if inverse=False, or cWB theta [0, π] [rad] if inverse=True.
    """
    if inverse:
        # geo -> cwb
        phi = phi % (2 * np.pi)
        theta = -(theta - np.pi / 2)
        return phi, theta

    # cwb -> geo
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    theta = convert_theta_to_dec(theta)
    return phi, theta

def convert_geo_to_cwb(phi, theta):
    return convert_cwb_to_geo(phi, theta, inverse=True)

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