import math
from dataclasses import dataclass
from copy import deepcopy
from pycwb.constants.physics_constants import LAL_EARTHFLAT, LAL_REARTH_SI


@dataclass
class Detector:
    """
    Class for storing detector information.

    Parameters
    ----------
    name : str
        detector name
    """
    name: str
    full_name: str
    latitude: float
    longitude: float
    altitude: float
    x_azimuth: float
    x_altitude: float
    y_azimuth: float
    y_altitude: float

    def __init__(self, name):
        super().__init__(name)

    @staticmethod
    def get_cartesian_components(alt, az, lat, lon):
        """
        Calculate the Cartesian components of a vector based on Altitude, Azimuth, Latitude, and Longitude.

        Parameters:
            alt (float): Altitude angle in radians.
            az (float): Azimuth angle in radians.
            lat (float): Latitude in radians.
            lon (float): Longitude in radians.
        """
        cosAlt = math.cos(alt)
        sinAlt = math.sin(alt)
        cosAz = math.cos(az)
        sinAz = math.sin(az)
        cosLat = math.cos(lat)
        sinLat = math.sin(lat)
        cosLon = math.cos(lon)
        sinLon = math.sin(lon)

        uNorth = cosAlt * cosAz
        uEast = cosAlt * sinAz
        # uUp == sinAlt
        uRho = -sinLat * uNorth + cosLat * sinAlt
        # uLambda == uEast

        u = [cosLon * uRho - sinLon * uEast,
             sinLon * uRho + cosLon * uEast,
             cosLat * uNorth + sinLat * sinAlt]

        return u

    @staticmethod
    def geodetic_to_geocentric(latitude, longitude, elevation):
        """
        Convert geodetic coordinates (latitude, longitude, elevation) to geocentric Cartesian coordinates (X, Y, Z).

        Parameters:
            latitude (float): Latitude in radians.
            longitude (float): Longitude in radians.
            elevation (float): Elevation above the Earth's surface in meters.

        Returns:
            tuple: A tuple containing the Cartesian coordinates (X, Y, Z) in meters.
        """
        # Intermediate calculations
        fFac = 1.0 - LAL_EARTHFLAT
        fFac *= fFac

        cosP = math.cos(latitude)
        sinP = math.sin(latitude)

        c = math.sqrt(1.0 / (cosP * cosP + fFac * sinP * sinP))
        print(c)
        s = fFac * c
        c = (LAL_REARTH_SI * c + elevation) * cosP
        s = (LAL_REARTH_SI * s + elevation) * sinP
        print(c)
        # Cartesian coordinates
        x = c * math.cos(longitude)
        y = c * math.sin(longitude)
        z = s

        return x, y, z

    def time_rotated(self, time):
        """
        Equivalent detector position for a delayed time.

        Parameters:
            time (float): Time in seconds.

        Returns:
            Detector: A new detector object with the equivalent position for the delayed time.
        """
        # Rotate the detector
        new_lon = self.longitude + 2 * math.pi * time / 86400

        # copy the detector
        new_detector = deepcopy(self)
        new_detector.longitude = new_lon

        return new_detector
