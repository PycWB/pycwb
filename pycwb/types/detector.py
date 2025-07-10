import math
import numpy as np
from dataclasses import dataclass
from copy import deepcopy
from astropy.time import Time
from astropy import constants, coordinates, units
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.units.si import sday, meter
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
    x_midpoint: float
    y_azimuth: float
    y_altitude: float
    y_midpoint: float
    vertex_vec_earth_centered: np.ndarray = None
    x_vec_earth_centered: np.ndarray = None
    y_vec_earth_centered: np.ndarray = None
    x_response: np.ndarray = None
    y_response: np.ndarray = None
    response: np.ndarray = None
    reference_time: float = None

    def __init__(self, name, full_name=None, latitude=None, longitude=None, altitude=None,
                 x_azimuth=None, x_altitude=None, x_midpoint=None,
                 y_azimuth=None, y_altitude=None, y_midpoint=None,
                 reference_time=1126259462.0):
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
            self.x_midpoint = DETECTORS[name]["x"]["midpoint"]
            self.y_midpoint = DETECTORS[name]["y"]["midpoint"]

        elif all(param is not None for param in [name, full_name, latitude, longitude, altitude,
                                                  x_azimuth, x_altitude, x_midpoint,
                                                  y_azimuth, y_altitude, y_midpoint]):
            self.name = name
            self.full_name = full_name
            self.latitude = latitude
            self.longitude = longitude
            self.altitude = altitude
            self.x_azimuth = x_azimuth
            self.x_altitude = x_altitude
            self.x_midpoint = x_midpoint
            self.y_azimuth = y_azimuth
            self.y_altitude = y_altitude
            self.y_midpoint = y_midpoint

        ifo_vecs = earth_centered_vectors(
            self.longitude, self.latitude,
            yangle=self.y_azimuth, xangle=self.x_azimuth,
            height=self.altitude,
            xaltitude=self.x_altitude, yaltitude=self.y_altitude
        )

        self.vertex_vec_earth_centered = ifo_vecs['loc_vec']
        self.x_vec_earth_centered = ifo_vecs['x_vec']
        self.y_vec_earth_centered = ifo_vecs['y_vec']
        self.x_response = ifo_vecs['x_response']
        self.y_response = ifo_vecs['y_response']
        self.response = ifo_vecs['response']
        self.reference_time = reference_time
        
    @property
    def x_length(self):
        """
        Get the length of the X arm of the detector.
        
        Returns:
            float: The length of the X arm in meters.
        """
        return self.x_midpoint * 2
    
    @property
    def y_length(self):
        """
        Get the length of the Y arm of the detector.
        
        Returns:
            float: The length of the Y arm in meters.
        """
        return self.y_midpoint * 2

    @property
    def x_vec(self):
        """
        Get the Cartesian components of the X arm of the detector.
        
        Returns:
            list: A list containing the Cartesian components [X, Y, Z] of the X arm.
        """
        return self.get_cartesian_components(self.x_altitude, self.x_azimuth, self.latitude, self.longitude)
    
    @property
    def y_vec(self):
        """
        Get the Cartesian components of the Y arm of the detector.
        
        Returns:
            list: A list containing the Cartesian components [X, Y, Z] of the Y arm.
        """
        return self.get_cartesian_components(self.y_altitude, self.y_azimuth, self.latitude, self.longitude)
    
    @property
    def vertex_vec(self):
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
    
    def gmst_estimate(self, gps_time):
        if self.reference_time is None:
            return gmst_accurate(gps_time)

        gmst_reference = gmst_accurate(self.reference_time)
        dphase = (gps_time - self.reference_time) / float(sday.si.scale) * (2.0 * np.pi)
        gmst = (gmst_reference + dphase) % (2.0 * np.pi)
        return gmst
    
    # TODO: to check
    def atenna_pattern(self, right_ascension, declination, polarization, t_gps,
                      frequency=0,
                      polarization_type='tensor'):
        """
        Return the detector response.

        Parameters
        ----------
        right_ascension: float or numpy.ndarray
            The right ascension of the source in radians.
        declination: float or numpy.ndarray
            The declination of the source in radians.
        polarization: float or numpy.ndarray
            The polarization angle of the source in radians.
        t_gps: float or lal.LIGOTimeGPS
            The GPS time of the observation.
        frequency: float, optional
            The frequency of the gravitational wave signal. Default is 0.
        polarization_type: str, optional
            The type of gravitational wave polarizations. Options are 'tensor', 'vector', or 'scalar'.
            Default is 'tensor'.

        Returns
        -------
        fplus(default) or fx or fb : float or numpy.ndarray
            The plus or vector-x or breathing polarization factor for this sky location / orientation
        fcross(default) or fy or fl : float or numpy.ndarray
            The cross or vector-y or longitudnal polarization factor for this sky location / orientation
        """
        t_gps = float(t_gps)
        gha = self.gmst_estimate(t_gps) - right_ascension

        cosgha = np.cos(gha)
        singha = np.sin(gha)
        cosdec = np.cos(declination)
        sindec = np.sin(declination)
        cospsi = np.cos(polarization)
        sinpsi = np.sin(polarization)

        if frequency:
            e0 = cosdec * cosgha
            e1 = cosdec * -singha
            e2 = np.sin(declination)
            nhat = np.array([e0, e1, e2], dtype=object)

            nx = nhat.dot(self.x_vec_earth_centered)
            ny = nhat.dot(self.y_vec_earth_centered)

            rx = single_arm_frequency_response(frequency, nx,
                                               self.x_length)
            ry = single_arm_frequency_response(frequency, ny,
                                               self.y_length)
            resp = ry * self.y_response - rx * self.x_response
            ttype = np.complex128
        else:
            resp = self.response
            ttype = np.float64

        x0 = -cospsi * singha - sinpsi * cosgha * sindec
        x1 = -cospsi * cosgha + sinpsi * singha * sindec
        x2 =  sinpsi * cosdec

        x = np.array([x0, x1, x2], dtype=object)
        dx = resp.dot(x)

        y0 =  sinpsi * singha - cospsi * cosgha * sindec
        y1 =  sinpsi * cosgha + cospsi * singha * sindec
        y2 =  cospsi * cosdec

        y = np.array([y0, y1, y2], dtype=object)
        dy = resp.dot(y)

        if polarization_type != 'tensor':
            z0 = -cosdec * cosgha
            z1 = cosdec * singha
            z2 = -sindec
            z = np.array([z0, z1, z2], dtype=object)
            dz = resp.dot(z)

        if polarization_type == 'tensor':
            if hasattr(dx, 'shape'):
                fplus = (x * dx - y * dy).sum(axis=0).astype(ttype)
                fcross = (x * dy + y * dx).sum(axis=0).astype(ttype)
            else:
                fplus = (x * dx - y * dy).sum()
                fcross = (x * dy + y * dx).sum()
            return fplus, fcross

        elif polarization_type == 'vector':
            if hasattr(dx, 'shape'):
                fx = (z * dx + x * dz).sum(axis=0).astype(ttype)
                fy = (z * dy + y * dz).sum(axis=0).astype(ttype)
            else:
                fx = (z * dx + x * dz).sum()
                fy = (z * dy + y * dz).sum()

            return fx, fy

        elif polarization_type == 'scalar':
            if hasattr(dx, 'shape'):
                fb = (x * dx + y * dy).sum(axis=0).astype(ttype)
                fl = (z * dz).sum(axis=0)
            else:
                fb = (x * dx + y * dy).sum()
                fl = (z * dz).sum()
            return fb, fl

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

def gmst_accurate(gps_time):
    gmst = Time(gps_time, format='gps', scale='utc',
                location=(0, 0)).sidereal_time('mean').rad
    return gmst

# Copied from pycbc.detector.single_arm_frequency_response
# Notation matches
# Eq 4 of https://link.aps.org/accepted/10.1103/PhysRevD.96.084004
def single_arm_frequency_response(f, n, arm_length):
    """ The relative amplitude factor of the arm response due to
    signal delay. This is relevant where the long-wavelength
    approximation no longer applies)
    """
    n = np.clip(n, -0.999, 0.999)
    phase = arm_length / constants.c.value * 2.0j * np.pi * f
    a = 1.0 / 4.0 / phase
    b = (1 - np.exp(-phase * (1 - n))) / (1 - n)
    c = np.exp(-2.0 * phase) * (1 - np.exp(phase * (1 + n))) / (1 + n)
    return a * (b - c) * 2.0  # We'll make this relative to the static resp

# Adapted from pycbc.detector.add_detector_on_earth
def earth_centered_vectors(longitude, latitude,
                        yangle=0, xangle=None, height=0,
                        xaltitude=0, yaltitude=0):
    """ Add a new detector on the earth

    Parameters
    ----------

    name: str
        two-letter name to identify the detector
    longitude: float
        Longitude in radians using geodetic coordinates of the detector
    latitude: float
        Latitude in radians using geodetic coordinates of the detector
    yangle: float
        Azimuthal angle of the y-arm (angle drawn from pointing north)
    xangle: float
        Azimuthal angle of the x-arm (angle drawn from point north). If not set
        we assume a right angle detector following the right-hand rule.
    xaltitude: float
        The altitude angle of the x-arm measured from the local horizon.
    yaltitude: float
        The altitude angle of the y-arm measured from the local horizon.
    height: float
        The height in meters of the detector above the standard
        reference ellipsoidal earth
    """
    if xangle is None:
        # assume right angle detector if no separate xarm direction given
        xangle = yangle + np.pi / 2.0

    # baseline response of a single arm pointed in the -X direction
    resp = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 0]])
    rm2 = rotation_matrix(-longitude * units.rad, 'z')
    rm1 = rotation_matrix(-1.0 * (np.pi / 2.0 - latitude) * units.rad, 'y')
    
    # Calculate response in earth centered coordinates
    # by rotation of response in coordinates aligned
    # with the detector arms
    resps = []
    vecs = []
    for angle, azi in [(yangle, yaltitude), (xangle, xaltitude)]:
        rm0 = rotation_matrix(angle * units.rad, 'z')
        rmN = rotation_matrix(-azi *  units.rad, 'y')
        rm = rm2 @ rm1 @ rm0 @ rmN
        # apply rotation
        resps.append(rm @ resp @ rm.T / 2.0)
        vecs.append(rm @ np.array([-1, 0, 0]))

    full_resp = (resps[0] - resps[1])
    loc = coordinates.EarthLocation.from_geodetic(longitude * units.rad,
                                                latitude * units.rad,
                                                height=height*units.meter)
    loc = np.array([loc.x.value, loc.y.value, loc.z.value])
    
    return {
        'loc_vec': loc,
        'x_vec': vecs[1],
        'y_vec': vecs[0],
        'x_response': resps[1],
        'y_response': resps[0],
        'response': full_resp,
    }
