import math
import numpy as np
from dataclasses import dataclass
from copy import deepcopy
from pycwb.constants.physics_constants import LAL_EARTHFLAT, LAL_REARTH_SI
from pycwb.constants.detectors import DETECTORS
import plotly.graph_objects as go


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

    def __init__(self, name, full_name=None, latitude=None, longitude=None, altitude=None,
                 x_azimuth=None, x_altitude=None, y_azimuth=None, y_altitude=None):
        """
        Initialize the Detector object with either a name or specific parameters.
        If a name is provided, it will look up the detector information from the DETECTORS dictionary.
        If specific parameters are provided, they will be used to initialize the detector.
        """
        if name in DETECTORS:
            self.name = name
            self.full_name = DETECTORS[name]["name"]
            self.latitude = DETECTORS[name]["lat"]
            self.longitude = DETECTORS[name]["lon"]
            self.altitude = DETECTORS[name]["elevation"]
            self.x_azimuth = DETECTORS[name]["x"]["az"]
            self.x_altitude = DETECTORS[name]["x"]["alt"]
            self.y_azimuth = DETECTORS[name]["y"]["az"]
            self.y_altitude = DETECTORS[name]["y"]["alt"]
        elif all(param is not None for param in [name, full_name, latitude, longitude, altitude,
                                                  x_azimuth, x_altitude, y_azimuth, y_altitude]):
            self.name = name
            self.full_name = full_name
            self.latitude = latitude
            self.longitude = longitude
            self.altitude = altitude
            self.x_azimuth = x_azimuth
            self.x_altitude = x_altitude
            self.y_azimuth = y_azimuth
            self.y_altitude = y_altitude

    @property
    def x_arm_direction_in_cartesian(self):
        """
        Get the Cartesian components of the X arm of the detector.
        
        Returns:
            list: A list containing the Cartesian components [X, Y, Z] of the X arm.
        """
        return self.get_cartesian_components(self.x_altitude, self.x_azimuth, self.latitude, self.longitude)
    
    @property
    def y_arm_direction_in_cartesian(self):
        """
        Get the Cartesian components of the Y arm of the detector.
        
        Returns:
            list: A list containing the Cartesian components [X, Y, Z] of the Y arm.
        """
        return self.get_cartesian_components(self.y_altitude, self.y_azimuth, self.latitude, self.longitude)
    
    @property
    def vertex_location_in_cartesian(self):
        """
        Get the Cartesian components of the detector's vertex.
        
        Returns:
            tuple: A tuple containing the Cartesian coordinates (X, Y, Z) of the detector's vertex.
        """
        return self.geodetic_to_geocentric(self.latitude, self.longitude, self.altitude)

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

        return np.array(u, dtype=np.float64)

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
        s = fFac * c
        c = (LAL_REARTH_SI * c + elevation) * cosP
        s = (LAL_REARTH_SI * s + elevation) * sinP
        # Cartesian coordinates
        x = c * math.cos(longitude)
        y = c * math.sin(longitude)
        z = s

        return np.array([x, y, z], dtype=np.float64)

    def time_rotated(self, time, name=None):
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

        if name is not None:
            new_detector.name = name

        return new_detector

    def get_x_arm_endpoint_in_geo(self, distance_km=100):
        """
        Get the X arm direction in longitude and latitude coordinates.

        Returns:
            tuple: A tuple containing the longitude and latitude of the X arm direction.
        """
        delta = distance_km / (LAL_REARTH_SI / 1000)

        lat1 = self.latitude
        lon1 = self.longitude
        azimuth = self.x_azimuth

        # Compute destination point
        lat2 = np.arcsin(np.sin(lat1) * np.cos(delta) +
                        np.cos(lat1) * np.sin(delta) * np.cos(azimuth))
        
        lon2 = lon1 + np.arctan2(np.sin(azimuth) * np.sin(delta) * np.cos(lat1),
                                np.cos(delta) - np.sin(lat1) * np.sin(lat2))

        return np.array([lon2, lat2], dtype=np.float64)
    
    def get_y_arm_endpoint_in_geo(self, distance_km=100):
        """
        Get the Y arm direction in longitude and latitude coordinates.

        Returns:
            tuple: A tuple containing the longitude and latitude of the Y arm direction.
        """
        delta = distance_km / (LAL_REARTH_SI / 1000)

        lat1 = self.latitude
        lon1 = self.longitude
        azimuth = self.y_azimuth

        # Compute destination point
        lat2 = np.arcsin(np.sin(lat1) * np.cos(delta) +
                        np.cos(lat1) * np.sin(delta) * np.cos(azimuth))
        
        lon2 = lon1 + np.arctan2(np.sin(azimuth) * np.sin(delta) * np.cos(lat1),
                                np.cos(delta) - np.sin(lat1) * np.sin(lat2))

        return np.array([lon2, lat2], dtype=np.float64)
    
    def plot_on_globe(self, fig=False, 
                      distance_km=500,
                      projection_type='orthographic',
                      width=800, height=800, color='red'):
        """
        Plot the detector's X and Y arms on a globe using Plotly.

        Parameters:
            fig (go.Figure or bool): A Plotly figure to add the detector arms to. If False, a new figure is created.
            distance_km (float): The distance in kilometers for the arm endpoints.
            projection_type (str): The type of globe projection to use (e.g., 'orthographic').
            width (int): Width of the figure.
            height (int): Height of the figure.
            color (str): Color of the arm lines.

        Example usage:
        >>> ifo_H1 = Detector('H1')
        >>> ifo_L1 = Detector('L1')
        >>> ifo_V1 = Detector('V1')
        >>> fig = ifo_H1.plot_on_globe()
        >>> ifo_L1.plot_on_globe(fig=fig)
        >>> ifo_V1.plot_on_globe(fig=fig)
        >>> fig.show()

        Returns:
            go.Figure: A Plotly figure with the detector arms plotted on a globe.
        """
        if fig is False:
            fig = go.Figure()

            fig.update_layout(
                geo=dict(
                    showland=True,
                    showcountries=False,
                    lataxis_showgrid=True,
                    lonaxis_showgrid=True,
                    projection_type=projection_type
                ),
                width=width,
                height=height
            )

        x_arm_endpoint = np.rad2deg(self.get_x_arm_endpoint_in_geo(distance_km))
        y_arm_endpoint = np.rad2deg(self.get_y_arm_endpoint_in_geo(distance_km))
        longitude = np.rad2deg(self.longitude)
        latitude = np.rad2deg(self.latitude)

        fig.add_trace(go.Scattergeo(
            lon=[longitude, x_arm_endpoint[0]],
            lat=[latitude, x_arm_endpoint[1]],
            mode='lines',
            line=dict(width=2, color=color),
            marker=dict(size=5),
            name=f'{self.name} X Arm',
        ))

        fig.add_trace(go.Scattergeo(
            lon=[longitude, y_arm_endpoint[0]],
            lat=[latitude, y_arm_endpoint[1]],
            mode='lines',
            line=dict(width=2, color=color),
            marker=dict(size=5),
            name=f'{self.name} Y Arm'
        ))

        return fig