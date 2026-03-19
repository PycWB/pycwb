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
from pycwb.utils.network import local_to_earth_centered
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm
from scipy.special import gammaincc, gammainccinv


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

    def optimal_orientation(self, t_gps):
        """
        Return the sky position (ra, dec) that maximises the antenna response.

        The optimal orientation is found by evaluating the combined antenna
        response ``sqrt(F+² + Fx²)`` on a sky grid at zero polarisation and
        returning the (ra, dec) of the maximum.

        Parameters
        ----------
        t_gps : float
            GPS time of the observation.

        Returns
        -------
        ra : float
            Right ascension in radians.
        dec : float
            Declination in radians.
        """
        n_ra, n_dec = 360, 180
        ra_vals = np.linspace(0, 2 * np.pi, n_ra, endpoint=False)
        dec_vals = np.linspace(-np.pi / 2, np.pi / 2, n_dec)

        best_power = -1.0
        best_ra, best_dec = 0.0, 0.0
        for dec in dec_vals:
            fplus, fcross = self.atenna_pattern(ra_vals, dec, 0.0, t_gps)
            power = np.asarray(fplus) ** 2 + np.asarray(fcross) ** 2
            idx = np.argmax(power)
            if power[idx] > best_power:
                best_power = power[idx]
                best_ra = float(ra_vals[idx])
                best_dec = float(dec)
        return best_ra, best_dec

    def time_delay_from_detector(self, other_det, ra, dec, t_gps):
        """
        Return the time delay from *other_det* to *self* for a signal
        arriving from sky position (ra, dec) in **equatorial** coordinates.

        ``dt > 0`` means the signal arrives at *self* **after** *other_det*.

        .. note::

           This method takes *equatorial* (RA, dec) and converts via GMST,
           matching the pycbc convention.  The cWB workflow instead uses
           Earth-fixed (geographic) sky coordinates and skips the GMST
           rotation — see :func:`compute_sky_delay_and_patterns` for the
           cWB-consistent bulk computation.

        Parameters
        ----------
        other_det : Detector
            The other detector.
        ra : float
            Right ascension in radians (equatorial).
        dec : float
            Declination in radians.
        t_gps : float
            GPS time (used to rotate Earth-fixed frame to equatorial).

        Returns
        -------
        float
            Time delay in seconds.
        """
        gmst = self.gmst_estimate(float(t_gps))
        # Source direction in Earth-centered coordinates
        gha = gmst - ra
        n_hat = np.array([
            np.cos(dec) * np.cos(gha),
            np.cos(dec) * (-np.sin(gha)),
            np.sin(dec),
        ])
        d_vec = self.vertex_vec_earth_centered - other_det.vertex_vec_earth_centered
        return float(np.dot(d_vec, n_hat) / constants.c.value)

    def project_wave(self, hp, hc, ra, dec, polarization):
        """
        Project plus/cross polarisations onto the detector response.

        ``h(t) = F+ * hp(t) + Fx * hc(t)``

        The GPS epoch is taken from *hp* (attribute ``t0`` for pycwb
        TimeSeries, ``start_time`` for pycbc TimeSeries, or ``t0`` for gwpy).

        Parameters
        ----------
        hp : TimeSeries
            Plus polarisation time series.
        hc : TimeSeries
            Cross polarisation time series.
        ra : float
            Right ascension in radians.
        dec : float
            Declination in radians.
        polarization : float
            Polarisation angle in radians.

        Returns
        -------
        pycwb.types.time_series.TimeSeries
            Projected detector strain.
        """
        from pycwb.types.time_series import TimeSeries

        hp_ts = TimeSeries.from_input(hp)
        hc_ts = TimeSeries.from_input(hc)

        t_gps = float(hp_ts.t0)
        fplus, fcross = self.atenna_pattern(ra, dec, polarization, t_gps)

        projected = fplus * hp_ts.data + fcross * hc_ts.data
        return TimeSeries(data=projected, dt=hp_ts.dt, t0=hp_ts.t0)

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

    def compute_detector_tensor(self):
        """
        Compute the detector response tensor in Earth-centered coordinates.
        
        Returns
        -------
        D : np.ndarray
            3x3 detector tensor matrix
        x_vec : np.ndarray
            Unit vector along x-arm in Earth-centered coordinates
        y_vec : np.ndarray
            Unit vector along y-arm in Earth-centered coordinates
        """
        lat_det = self.latitude
        lon_det = self.longitude
        
        x_az, x_alt = self.x_azimuth, self.x_altitude
        y_az, y_alt = self.y_azimuth, self.y_altitude
        
        # Local horizon frame components for x-arm
        x_east = np.sin(x_az)
        x_north = np.cos(x_az)
        x_up = np.sin(x_alt)
        
        # Local horizon frame components for y-arm
        y_east = np.sin(y_az)
        y_north = np.cos(y_az)
        y_up = np.sin(y_alt)
        
        # Normalize
        x_norm = np.sqrt(x_east**2 + x_north**2 + x_up**2)
        y_norm = np.sqrt(y_east**2 + y_north**2 + y_up**2)
        x_east /= x_norm
        x_north /= x_norm
        x_up /= x_norm
        y_east /= y_norm
        y_north /= y_norm
        y_up /= y_norm
        
        # Convert to Earth-centered coordinates
        x_vec = local_to_earth_centered(lat_det, lon_det, x_east, x_north, x_up)
        y_vec = local_to_earth_centered(lat_det, lon_det, y_east, y_north, y_up)
        
        # Detector tensor: D = 0.5 * (x ⊗ x - y ⊗ y)
        D = 0.5 * (np.outer(x_vec, x_vec) - np.outer(y_vec, y_vec))
        
        return D, x_vec, y_vec
    
    def compute_antenna_pattern_for_grid(self, theta_grid, phi_grid):
        """
        Compute F+ and Fx antenna patterns for this detector on a sky grid.
        
        Parameters
        ----------
        theta_grid : np.ndarray
            2D array of sky position theta (polar angle, 0 to pi)
        phi_grid : np.ndarray
            2D array of sky position phi (azimuthal angle, 0 to 2*pi)
            
        Returns
        -------
        F_plus : np.ndarray
            Plus polarization antenna pattern
        F_cross : np.ndarray
            Cross polarization antenna pattern
        """
        n_lat, n_lon = theta_grid.shape
        
        sin_theta, cos_theta = np.sin(theta_grid), np.cos(theta_grid)
        sin_phi, cos_phi = np.sin(phi_grid), np.cos(phi_grid)
        
        # Wave direction unit vector
        n_x = sin_theta * cos_phi
        n_y = sin_theta * sin_phi
        n_z = cos_theta
        
        # Polarization basis vectors
        e_theta_x = cos_theta * cos_phi
        e_theta_y = cos_theta * sin_phi
        e_theta_z = -sin_theta
        
        e_phi_x = -sin_phi
        e_phi_y = cos_phi
        e_phi_z = 0.0
        
        # Normalize
        e_theta_norm = np.sqrt(e_theta_x**2 + e_theta_y**2 + e_theta_z**2)
        e_theta_x /= e_theta_norm
        e_theta_y /= e_theta_norm
        e_theta_z /= e_theta_norm
        
        e_phi_norm = np.sqrt(e_phi_x**2 + e_phi_y**2 + e_phi_z**2)
        e_phi_x /= e_phi_norm
        e_phi_y /= e_phi_norm
        e_phi_z /= e_phi_norm
        
        # Get detector tensor
        D, _, _ = self.compute_detector_tensor()
        
        # Compute D:e_theta⊗e_theta
        D_e_theta_e_theta = (
            e_theta_x * (D[0, 0] * e_theta_x + D[0, 1] * e_theta_y + D[0, 2] * e_theta_z) +
            e_theta_y * (D[1, 0] * e_theta_x + D[1, 1] * e_theta_y + D[1, 2] * e_theta_z) +
            e_theta_z * (D[2, 0] * e_theta_x + D[2, 1] * e_theta_y + D[2, 2] * e_theta_z)
        )
        
        # Compute D:e_phi⊗e_phi
        D_e_phi_e_phi = (
            e_phi_x * (D[0, 0] * e_phi_x + D[0, 1] * e_phi_y + D[0, 2] * e_phi_z) +
            e_phi_y * (D[1, 0] * e_phi_x + D[1, 1] * e_phi_y + D[1, 2] * e_phi_z) +
            e_phi_z * (D[2, 0] * e_phi_x + D[2, 1] * e_phi_y + D[2, 2] * e_phi_z)
        )
        
        # Compute D:e_theta⊗e_phi
        D_e_theta_e_phi = (
            e_theta_x * (D[0, 0] * e_phi_x + D[0, 1] * e_phi_y + D[0, 2] * e_phi_z) +
            e_theta_y * (D[1, 0] * e_phi_x + D[1, 1] * e_phi_y + D[1, 2] * e_phi_z) +
            e_theta_z * (D[2, 0] * e_phi_x + D[2, 1] * e_phi_y + D[2, 2] * e_phi_z)
        )
        
        F_plus = D_e_theta_e_theta - D_e_phi_e_phi
        F_cross = 2.0 * D_e_theta_e_phi
        
        return F_plus, F_cross
    
    def draw_antenna_pattern(self, polarization=3, palette='turbo',
                           resolution=2, projection='rectilinear',
                           display_world_map=True, add_title=True,
                           ax=None, vmin=0.0, vmax=None):
        """
        Draw antenna pattern for this detector.
        
        Parameters
        ----------
        polarization : int
            0 -> |Fx| (DPF)
            1 -> |F+| (DPF)
            2 -> |Fx|/|F+| (DPF)
            3 -> sqrt(|F+|^2+|Fx|^2) (DPF)
            4 -> |Fx|^2 (DPF)
            5 -> |F+|^2 (DPF)
        palette : str
            Matplotlib colormap name
        resolution : int
            Sky map resolution (1=low, 2=medium, 4=high)
        projection : str
            Map projection: 'hammer', 'mollweide', 'rectilinear', 'sinusoidal'
        display_world_map : bool
            Whether to display world map background
        add_title : bool
            Whether to add title to plot
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on
        vmin : float
            Minimum value for colorbar
        vmax : float, optional
            Maximum value for colorbar (auto if None)
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        # Create sky grid
        n_lon = 360 * resolution
        n_lat = 180 * resolution
        lon_rad = np.linspace(0, 2 * np.pi, n_lon)
        lat_rad = np.linspace(0, np.pi, n_lat)
        lon_grid, lat_grid = np.meshgrid(lon_rad, lat_rad)
        
        # Compute antenna patterns
        F_plus, F_cross = self.compute_antenna_pattern_for_grid(lat_grid, lon_grid)
        
        # Compute requested polarization pattern
        if polarization == 0:
            pattern = np.abs(F_cross)
        elif polarization == 1:
            pattern = np.abs(F_plus)
        elif polarization == 2:
            with np.errstate(divide='ignore', invalid='ignore'):
                pattern = np.abs(F_cross) / np.abs(F_plus)
                pattern[~np.isfinite(pattern)] = 0
        elif polarization == 3:
            pattern = np.sqrt(F_plus**2 + F_cross**2)
        elif polarization == 4:
            pattern = F_cross**2
        elif polarization == 5:
            pattern = F_plus**2
        else:
            raise ValueError(f"Unsupported polarization: {polarization}")
        
        # Create figure if needed
        if ax is None:
            if projection.lower() == 'hammer':
                fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': 'aitoff'})
            elif projection.lower() == 'mollweide':
                fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': 'mollweide'})
            elif projection.lower() == 'sinusoidal':
                projection_ccrs = ccrs.Sinusoidal(central_longitude=0)
                fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': projection_ccrs})
            else:
                projection_ccrs = ccrs.PlateCarree()
                fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': projection_ccrs})
        else:
            fig = ax.get_figure()
        
        # Convert coordinates for plotting
        lon_deg = np.degrees(lon_grid) - 180
        lat_deg = 90 - np.degrees(lat_grid)
        pattern_flipped = np.flipud(pattern)
        
        # Set vmax if not provided
        if vmax is None:
            vmax = np.max(pattern)
        
        # Plot pattern
        if projection.lower() in ['hammer', 'mollweide']:
            lon_plot = lon_deg * np.pi / 180
            lat_plot = lat_deg * np.pi / 180
            im = ax.pcolormesh(lon_plot, lat_plot, pattern_flipped,
                              cmap=palette, shading='auto', vmin=vmin, vmax=vmax)
            ax.grid(True, linestyle='--', alpha=0.5)
        else:
            im = ax.pcolormesh(lon_deg, lat_deg, pattern_flipped,
                              transform=ccrs.PlateCarree(),
                              cmap=palette, shading='auto', vmin=vmin, vmax=vmax)
        
        # Add world map
        if display_world_map and projection.lower() not in ['hammer', 'mollweide']:
            try:
                ax.coastlines(linewidth=0.5, alpha=0.7)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5, linestyle=':')
                ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
                ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.1)
                
                if projection.lower() != 'sinusoidal':
                    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                     linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.xlabel_style = {'size': 8}
                    gl.ylabel_style = {'size': 8}
            except Exception:
                ax.grid(True, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        else:
            ax.grid(True, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        
        # Plot detector site
        lat_deg_det = np.degrees(self.latitude)
        lon_deg_det = np.degrees(self.longitude)
        
        if projection.lower() in ['hammer', 'mollweide']:
            lon_plot = np.radians(lon_deg_det)
            if lon_plot > np.pi:
                lon_plot -= 2 * np.pi
            lat_plot = np.radians(lat_deg_det)
            ax.plot(lon_plot, lat_plot, 'k.', markersize=10, markeredgewidth=2, transform=ax.transData)
            ax.text(lon_plot + 0.05, lat_plot + 0.05, self.name,
                   fontsize=12, fontweight='bold', ha='left', va='bottom', transform=ax.transData)
        else:
            ax.plot(lon_deg_det, lat_deg_det, 'k.', markersize=10,
                   transform=ccrs.PlateCarree(), markeredgewidth=2)
            ax.text(lon_deg_det + 1, lat_deg_det + 1, self.name,
                   transform=ccrs.PlateCarree(), fontsize=12, fontweight='bold', ha='left', va='bottom')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, shrink=0.45)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Pattern Value', fontsize=11)
        
        # Title
        if add_title:
            polarization_names = {
                0: r"$|F_x|$ (DPF)",
                1: r"$|F_+|$ (DPF)",
                2: r"$|F_x|/|F_+|$ (DPF)",
                3: r"$\sqrt{|F_+|^2 + |F_x|^2}$ (DPF)",
                4: "$|F_x|^2$ (DPF)",
                5: "$|F_+|^2$ (DPF)"
            }
            title = f"Detector: {self.name} - {polarization_names.get(polarization, f'Polarization {polarization}')}"
            ax.set_title(title, fontsize=14, pad=20)
        
        plt.tight_layout()
        return fig, ax


class DetectorNetwork:
    """
    Network of detectors with utilities to compute and plot network antenna patterns.

    Parameters
    ----------
    ifos : list[str] | str | None
        List of detector codes (e.g., ["H1", "L1", "V1"]) or a string (e.g., "H1L1V1").
    detectors : list[Detector] | None
        Optional list of Detector objects to initialize the network.
    """

    def __init__(self, ifos=None, detectors=None):
        self.detectors = []
        if detectors:
            for det in detectors:
                self.add_detector(det)
        if ifos:
            self.add_detectors(ifos)

    def add_detector(self, detector):
        if isinstance(detector, Detector):
            self.detectors.append(detector)
        elif isinstance(detector, str):
            self.detectors.append(Detector(detector))
        else:
            raise TypeError("Detector must be a Detector instance or detector code string")

    def add_detectors(self, ifos):
        if isinstance(ifos, str):
            ifo_list = self._parse_detector_codes(ifos)
        else:
            ifo_list = list(ifos)
        for ifo in ifo_list:
            self.add_detector(ifo)

    @staticmethod
    def _parse_detector_codes(network_str):
        code_mapping = {
            'H1': 'H1', 'L1': 'L1', 'G1': 'G1', 'V1': 'V1',
            'T1': 'T1', 'H2': 'H2', 'A1': 'A1', 'O1': 'O1',
            'N1': 'N1', 'E1': 'E1', 'A2': 'A2', 'J1': 'J1',
            'I1': 'I1', 'K1': 'K1', 'E2': 'E2', 'E3': 'E3', 'E0': 'E0'
        }
        detector_codes = [code for code in code_mapping if code in network_str]
        if not detector_codes:
            raise ValueError("No valid detectors found in network string")
        return detector_codes

    def _get_detector_info(self):
        detectors = []
        for det in self.detectors:
            det_info = DETECTORS.get(det.name)
            if not det_info:
                continue
            detectors.append({
                'code': det.name,
                'name': det.full_name,
                'lat': det_info['lat'],
                'lon': det_info['lon'],
                'x_alt': det_info['x']['alt'],
                'x_az': det_info['x']['az'],
                'y_alt': det_info['y']['alt'],
                'y_az': det_info['y']['az']
            })
        return detectors

    @staticmethod
    def _create_sky_grid(resolution):
        n_lon = 360 * resolution
        n_lat = 180 * resolution
        lon_rad = np.linspace(0, 2 * np.pi, n_lon)
        lat_rad = np.linspace(0, np.pi, n_lat)
        lon_grid, lat_grid = np.meshgrid(lon_rad, lat_rad)
        return lon_grid, lat_grid, n_lon, n_lat

    @classmethod
    def _compute_detector_tensor(cls, detector):
        lat_det = detector['lat']
        lon_det = detector['lon']
        x_az, x_alt = detector['x_az'], detector['x_alt']
        y_az, y_alt = detector['y_az'], detector['y_alt']
        x_east = np.sin(x_az)
        x_north = np.cos(x_az)
        x_up = np.sin(x_alt)
        y_east = np.sin(y_az)
        y_north = np.cos(y_az)
        y_up = np.sin(y_alt)
        x_norm = np.sqrt(x_east**2 + x_north**2 + x_up**2)
        y_norm = np.sqrt(y_east**2 + y_north**2 + y_up**2)
        x_east /= x_norm
        x_north /= x_norm
        x_up /= x_norm
        y_east /= y_norm
        y_north /= y_norm
        y_up /= y_norm
        x_vec = local_to_earth_centered(lat_det, lon_det, x_east, x_north, x_up)
        y_vec = local_to_earth_centered(lat_det, lon_det, y_east, y_north, y_up)
        D = 0.5 * (np.outer(x_vec, x_vec) - np.outer(y_vec, y_vec))
        return D, x_vec, y_vec

    @classmethod
    def _compute_antenna_patterns(cls, theta_grid, phi_grid, detectors):
        n_lat, n_lon = theta_grid.shape
        n_detectors = len(detectors)
        F_plus = np.zeros((n_lat, n_lon, n_detectors))
        F_cross = np.zeros((n_lat, n_lon, n_detectors))
        sin_theta, cos_theta = np.sin(theta_grid), np.cos(theta_grid)
        sin_phi, cos_phi = np.sin(phi_grid), np.cos(phi_grid)
        e_theta_x = cos_theta * cos_phi
        e_theta_y = cos_theta * sin_phi
        e_theta_z = -sin_theta
        e_phi_x = -sin_phi
        e_phi_y = cos_phi
        e_phi_z = 0.0
        e_theta_norm = np.sqrt(e_theta_x**2 + e_theta_y**2 + e_theta_z**2)
        e_theta_x /= e_theta_norm
        e_theta_y /= e_theta_norm
        e_theta_z /= e_theta_norm
        e_phi_norm = np.sqrt(e_phi_x**2 + e_phi_y**2 + e_phi_z**2)
        e_phi_x /= e_phi_norm
        e_phi_y /= e_phi_norm
        e_phi_z /= e_phi_norm
        for i, detector in enumerate(detectors):
            D, _, _ = cls._compute_detector_tensor(detector)
            D_e_theta_e_theta = (
                e_theta_x * (D[0, 0] * e_theta_x + D[0, 1] * e_theta_y + D[0, 2] * e_theta_z) +
                e_theta_y * (D[1, 0] * e_theta_x + D[1, 1] * e_theta_y + D[1, 2] * e_theta_z) +
                e_theta_z * (D[2, 0] * e_theta_x + D[2, 1] * e_theta_y + D[2, 2] * e_theta_z)
            )
            D_e_phi_e_phi = (
                e_phi_x * (D[0, 0] * e_phi_x + D[0, 1] * e_phi_y + D[0, 2] * e_phi_z) +
                e_phi_y * (D[1, 0] * e_phi_x + D[1, 1] * e_phi_y + D[1, 2] * e_phi_z) +
                e_phi_z * (D[2, 0] * e_phi_x + D[2, 1] * e_phi_y + D[2, 2] * e_phi_z)
            )
            D_e_theta_e_phi = (
                e_theta_x * (D[0, 0] * e_phi_x + D[0, 1] * e_phi_y + D[0, 2] * e_phi_z) +
                e_theta_y * (D[1, 0] * e_phi_x + D[1, 1] * e_phi_y + D[1, 2] * e_phi_z) +
                e_theta_z * (D[2, 0] * e_phi_x + D[2, 1] * e_phi_y + D[2, 2] * e_phi_z)
            )
            F_plus[:, :, i] = D_e_theta_e_theta - D_e_phi_e_phi
            F_cross[:, :, i] = 2.0 * D_e_theta_e_phi
        return F_plus, F_cross

    @staticmethod
    def _compute_polarization_quantity(gp, gx, gI, polarization, n_detectors):
        gR = (gp - gx) / 2.0
        gr = (gp + gx) / 2.0
        gc = np.sqrt(gR**2 + gI**2)
        if polarization == 0:
            val = gr - gc
            return np.sqrt(np.abs(val)) if np.abs(val) > 1e-8 else 0.0
        if polarization == 1:
            return np.sqrt(gr + gc)
        if polarization == 2:
            denom = gr + gc
            if denom > 1e-8:
                val = (gr - gc) / denom
                return np.sqrt(val) if val > 0 else 0.0
            return 0.0
        if polarization == 3:
            return np.sqrt(2.0 * gr / n_detectors)
        if polarization == 4:
            val = gr - gc
            return val if np.abs(val) > 1e-8 else 0.0
        if polarization == 5:
            return gr + gc
        raise ValueError(f"Unsupported polarization: {polarization}")

    @staticmethod
    def _compute_reference_max(F_plus, F_cross, scales):
        n_lat, n_lon, n_detectors = F_plus.shape
        F_plus_flat = F_plus.reshape(-1, n_detectors)
        F_cross_flat = F_cross.reshape(-1, n_detectors)
        gp = np.sum(scales * F_plus_flat**2, axis=1)
        gx = np.sum(scales * F_cross_flat**2, axis=1)
        gI = np.sum(scales * F_plus_flat * F_cross_flat, axis=1)
        gR = (gp - gx) / 2.0
        gr = (gp + gx) / 2.0
        gc = np.sqrt(gR**2 + gI**2)
        pattern_values = np.sqrt(gr + gc)
        return np.max(pattern_values)

    @classmethod
    def _compute_antenna_pattern(cls, F_plus, F_cross, polarization, scales):
        n_lat, n_lon, n_detectors = F_plus.shape
        F_plus_flat = F_plus.reshape(-1, n_detectors)
        F_cross_flat = F_cross.reshape(-1, n_detectors)
        gp = np.sum(scales * F_plus_flat**2, axis=1)
        gx = np.sum(scales * F_cross_flat**2, axis=1)
        gI = np.sum(scales * F_plus_flat * F_cross_flat, axis=1)
        if polarization == 1:
            gR = (gp - gx) / 2.0
            gr = (gp + gx) / 2.0
            gc = np.sqrt(gR**2 + gI**2)
            pattern_values = np.sqrt(gr + gc)
        else:
            pattern_values = np.array([
                cls._compute_polarization_quantity(gp_i, gx_i, gI_i, polarization, n_detectors)
                for gp_i, gx_i, gI_i in zip(gp, gx, gI)
            ])
        pattern = pattern_values.reshape(n_lat, n_lon)
        pattern_max = np.max(pattern)
        return pattern, pattern_max

    @staticmethod
    def _create_plot_figure(projection):
        if projection.lower() == 'hammer':
            fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': 'aitoff'})
        elif projection.lower() == 'mollweide':
            fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': 'mollweide'})
        elif projection.lower() == 'sinusoidal':
            projection_ccrs = ccrs.Sinusoidal(central_longitude=0)
            fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': projection_ccrs})
        else:
            projection_ccrs = ccrs.PlateCarree()
            fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': projection_ccrs})
        return fig, ax

    @staticmethod
    def _plot_pattern(ax, pattern, lon_deg, lat_deg, palette, projection, vmin=0.0, vmax=None):
        pattern_flipped = np.flipud(pattern)
        if projection.lower() in ['hammer', 'mollweide']:
            lon_plot = lon_deg * np.pi / 180
            lat_plot = lat_deg * np.pi / 180
            im = ax.pcolormesh(lon_plot, lat_plot, pattern_flipped,
                              cmap=palette, shading='auto', vmin=vmin, vmax=vmax)
            ax.grid(True, linestyle='--', alpha=0.5)
        else:
            im = ax.pcolormesh(lon_deg, lat_deg, pattern_flipped,
                              transform=ccrs.PlateCarree(),
                              cmap=palette, shading='auto', vmin=vmin, vmax=vmax)
        return im

    @staticmethod
    def _add_world_map(ax, display_world_map, projection):
        if not display_world_map:
            return
        if projection.lower() not in ['hammer', 'mollweide']:
            try:
                ax.coastlines(linewidth=0.5, alpha=0.7)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5, linestyle=':')
                ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
                ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.1)
                if projection.lower() != 'sinusoidal':
                    gl = ax.gridlines(
                        crs=ccrs.PlateCarree(),
                        draw_labels=True,
                        linewidth=0.5,
                        color='gray',
                        alpha=0.5,
                        linestyle='--'
                    )
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.xlabel_style = {'size': 8}
                    gl.ylabel_style = {'size': 8}
            except Exception:
                ax.grid(True, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        else:
            ax.grid(True, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

    @classmethod
    def _compute_arm_endpoints(cls, detector, arm_length_factor=8.0):
        lat, lon = detector['lat'], detector['lon']
        _, x_vec, y_vec = cls._compute_detector_tensor(detector)
        sin_lat, cos_lat = np.sin(lat), np.cos(lat)
        sin_lon, cos_lon = np.sin(lon), np.cos(lon)
        det_x = cos_lat * cos_lon
        det_y = cos_lat * sin_lon
        det_z = sin_lat
        arm_length = arm_length_factor * np.pi / 180.0
        normal = np.array([det_x, det_y, det_z])

        def project_to_tangent(v, n):
            return v - np.dot(v, n) * n

        x_tangent = project_to_tangent(x_vec, normal)
        y_tangent = project_to_tangent(y_vec, normal)
        x_tangent_norm = np.linalg.norm(x_tangent)
        y_tangent_norm = np.linalg.norm(y_tangent)
        if x_tangent_norm > 0:
            x_tangent /= x_tangent_norm
        if y_tangent_norm > 0:
            y_tangent /= y_tangent_norm

        def point_on_sphere(center, direction, angle):
            axis = np.cross(center, direction)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 0:
                axis /= axis_norm
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                return (cos_a * center + sin_a * np.cross(axis, center) +
                        (1 - cos_a) * np.dot(axis, center) * axis)
            return center

        x_end_cart = point_on_sphere(normal, x_tangent, arm_length)
        y_end_cart = point_on_sphere(normal, y_tangent, arm_length)

        def cart_to_sph(x, y, z):
            r = np.sqrt(x**2 + y**2 + z**2)
            lat_sph = np.arcsin(z / r)
            lon_sph = np.arctan2(y, x)
            return lat_sph, lon_sph

        x_lat_sph, x_lon_sph = cart_to_sph(*x_end_cart)
        y_lat_sph, y_lon_sph = cart_to_sph(*y_end_cart)
        x_lat_deg = np.degrees(x_lat_sph)
        x_lon_deg = np.degrees(x_lon_sph)
        y_lat_deg = np.degrees(y_lat_sph)
        y_lon_deg = np.degrees(y_lon_sph)
        if x_lon_deg > 180:
            x_lon_deg -= 360
        elif x_lon_deg < -180:
            x_lon_deg += 360
        if y_lon_deg > 180:
            y_lon_deg -= 360
        elif y_lon_deg < -180:
            y_lon_deg += 360
        return (x_lon_deg, x_lat_deg), (y_lon_deg, y_lat_deg)

    @classmethod
    def _plot_detector_sites(cls, ax, detectors, projection):
        for det in detectors:
            lat_deg = np.degrees(det['lat'])
            lon_deg = np.degrees(det['lon'])
            (x_lon, x_lat), (y_lon, y_lat) = cls._compute_arm_endpoints(det, arm_length_factor=6.0)
            if projection.lower() in ['hammer', 'mollweide']:
                lon_plot = np.radians(lon_deg)
                if lon_plot > np.pi:
                    lon_plot -= 2 * np.pi
                lat_plot = np.radians(lat_deg)
                x_lon_plot = np.radians(x_lon)
                if x_lon_plot > np.pi:
                    x_lon_plot -= 2 * np.pi
                x_lat_plot = np.radians(x_lat)
                y_lon_plot = np.radians(y_lon)
                if y_lon_plot > np.pi:
                    y_lon_plot -= 2 * np.pi
                y_lat_plot = np.radians(y_lat)
                ax.plot(lon_plot, lat_plot, 'k.', markersize=10,
                       markeredgewidth=2, transform=ax.transData)
                ax.plot([lon_plot, x_lon_plot], [lat_plot, x_lat_plot],
                       'k-', linewidth=2.0, transform=ax.transData)
                ax.plot([lon_plot, y_lon_plot], [lat_plot, y_lat_plot],
                       'k-', linewidth=2.0, transform=ax.transData)
                ax.text(lon_plot + 0.05, lat_plot + 0.05, det['code'],
                       fontsize=12, fontweight='bold', ha='left', va='bottom',
                       transform=ax.transData)
            else:
                ax.plot(lon_deg, lat_deg, 'k.', markersize=10,
                       transform=ccrs.PlateCarree(), markeredgewidth=2)
                ax.plot([lon_deg, x_lon], [lat_deg, x_lat],
                       'k-', linewidth=2.0, transform=ccrs.PlateCarree())
                ax.plot([lon_deg, y_lon], [lat_deg, y_lat],
                       'k-', linewidth=2.0, transform=ccrs.PlateCarree())
                ax.text(lon_deg + 1, lat_deg + 1, det['code'],
                       transform=ccrs.PlateCarree(), fontsize=12,
                       fontweight='bold', ha='left', va='bottom')

    def draw_antenna_pattern(self, polarization=3, palette='turbo',
                             resolution=2, projection='rectilinear',
                             display_world_map=True, add_title=True,
                             uniform_colorbar=True, ax=None, detector_scales=None):
        """
        Draw antenna pattern for the detector network.
        """
        detectors = self._get_detector_info()
        if not detectors:
            raise ValueError("No valid detectors found")

        scales = np.ones(len(detectors))
        if detector_scales is not None:
            if isinstance(detector_scales, dict):
                scales = np.array([detector_scales.get(det['code'], 1.0) for det in detectors])
            elif isinstance(detector_scales, (list, np.ndarray)):
                if len(detector_scales) == len(detectors):
                    scales = np.array(detector_scales)

        lon_grid, lat_grid, _, _ = self._create_sky_grid(resolution)
        theta_grid = lat_grid
        phi_grid = lon_grid

        F_plus, F_cross = self._compute_antenna_patterns(theta_grid, phi_grid, detectors)

        reference_max = None
        if uniform_colorbar:
            reference_max = self._compute_reference_max(F_plus, F_cross, scales)

        pattern, pattern_max = self._compute_antenna_pattern(F_plus, F_cross, polarization, scales)

        if ax is None:
            fig, ax = self._create_plot_figure(projection)
        else:
            fig = ax.get_figure()

        lon_deg = np.degrees(lon_grid) - 180
        lat_deg = 90 - np.degrees(lat_grid)

        vmax = reference_max if uniform_colorbar and reference_max is not None else pattern_max
        im = self._plot_pattern(ax, pattern, lon_deg, lat_deg, palette, projection, vmin=0.0, vmax=vmax)

        self._add_world_map(ax, display_world_map, projection)

        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, shrink=0.45)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Pattern Value', fontsize=11)

        self._plot_detector_sites(ax, detectors, projection)

        if add_title:
            polarization_names = {
                0: r"$|F_x|$ (DPF)",
                1: r"$|F_+|$ (DPF)",
                2: r"$|F_x|/|F_+|$ (DPF)",
                3: r"$\sqrt{|F_+|^2 + |F_x|^2}$ (DPF)",
                4: "|F_x|^2 (DPF)",
                5: "|F_+|^2 (DPF)"
            }
            title = f"Network: {' '.join([d['code'] for d in detectors])} - {polarization_names.get(polarization, f'Polarization {polarization}')}"
            ax.set_title(title, fontsize=14, pad=20)

        plt.tight_layout()
        return fig, ax


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


def get_max_delay(detectors) -> float:
    """
    Return the network maximum delay using detector-level `tau` values.

    This is a Python-native equivalent of cWB `getDelay("MAX")`: it scans the
    available detector delay maps and returns the largest absolute value among
    global extrema.

    Parameters
    ----------
    detectors : sequence | object | None
        Detector-like object(s). Each detector may expose:
        - `tau.max()` / `tau.min()` (ROOT-like map), or
        - an array-like `tau` (optionally via `tau.data`).

    Returns
    -------
    float
        Maximum absolute delay in seconds. Returns `0.0` when insufficient
        detector data is available.
    """
    if detectors is None:
        return 0.0

    # Accept a single detector object for convenience.
    if not isinstance(detectors, (list, tuple)):
        detectors = [detectors]

    if len(detectors) < 2:
        return 0.0

    max_tau = -np.inf
    min_tau = np.inf

    for detector in detectors:
        tau = getattr(detector, "tau", None)
        if tau is None:
            continue

        # Fast path for ROOT-like maps that already provide min/max methods.
        if hasattr(tau, "max") and hasattr(tau, "min") and callable(tau.max) and callable(tau.min):
            local_max = float(tau.max())
            local_min = float(tau.min())
        else:
            # Generic path for numpy-compatible arrays.
            tau_values = np.asarray(getattr(tau, "data", tau), dtype=float)
            if tau_values.size == 0:
                continue
            local_max = float(np.max(tau_values))
            local_min = float(np.min(tau_values))

        if local_max > max_tau:
            max_tau = local_max
        if local_min < min_tau:
            min_tau = local_min

    if not np.isfinite(max_tau) or not np.isfinite(min_tau):
        return 0.0

    return float(max(abs(max_tau), abs(min_tau)))


def calculate_e2or_from_acore(acore: float, n_ifo: int) -> float:
    """
    Compute cWB-like subnetwork energy threshold (`e2or`) from `Acore`.

    This follows the C++ relation in `network::setAcore`:
      p = 1 - Gamma(nIFO, acore^2 * nIFO)
      e2or = iGamma(nIFO - 1, p)

    Here `Gamma` / `iGamma` are implemented via regularized upper incomplete
    gamma and its inverse (`gammaincc`, `gammainccinv`).

    Parameters
    ----------
    acore : float
        Core pixel amplitude threshold.
    n_ifo : int
        Number of detectors in network.

    Returns
    -------
    float
        Subnetwork energy threshold `e2or`.
    """
    n_ifo = int(max(1, n_ifo))
    if n_ifo < 2:
        return float(max(0.0, acore))

    prob = float(gammaincc(float(n_ifo), float(acore) * float(acore) * float(n_ifo)))
    prob = float(np.clip(prob, 1.0e-12, 1.0 - 1.0e-12))
    return float(gammainccinv(float(n_ifo - 1), prob))


def _build_sky_directions(n_sky: int, healpix_order: int | None = None):
    """Build sky directions as (ra, dec) arrays in radians."""
    if healpix_order is not None and int(healpix_order) > 0:
        try:
            import healpy as hp
            nside = 2 ** int(healpix_order)
            npix = hp.nside2npix(nside)
            theta, phi = hp.pix2ang(nside, np.arange(npix, dtype=np.int64), nest=False)
            ra = np.asarray(phi, dtype=np.float64)
            dec = (np.pi / 2.0) - np.asarray(theta, dtype=np.float64)
            return ra, dec
        except Exception:
            pass

    n_sky = int(max(1, n_sky))
    idx = np.arange(n_sky, dtype=np.float64)
    z = 1.0 - 2.0 * (idx + 0.5) / float(n_sky)
    phi = (np.pi * (3.0 - np.sqrt(5.0)) * idx) % (2.0 * np.pi)
    dec = np.arcsin(np.clip(z, -1.0, 1.0))
    ra = phi
    return ra.astype(np.float64), dec.astype(np.float64)


def compute_sky_delay_and_patterns(ifos, ref_ifo, sample_rate, td_size, gps_time,
                                   healpix_order=None, n_sky=None):
    """
    Compute pure-Python sky delay indices and antenna patterns.

    Returns arrays compatible with `load_data_from_ifo` output:
      - `ml`: int32 delay index, shape `(nIFO, nSky)`
      - `FP`: float64 plus pattern, shape `(nIFO, nSky)`
      - `FX`: float64 cross pattern, shape `(nIFO, nSky)`
    """
    detector_objs = [Detector(ifo) if isinstance(ifo, str) else ifo for ifo in ifos]
    if len(detector_objs) == 0:
        raise ValueError("No detectors provided")

    n_ifo = len(detector_objs)
    rate = float(sample_rate)
    td_size = int(max(1, td_size))
    if n_sky is None:
        if healpix_order is not None and int(healpix_order) > 0:
            n_sky = int(12 * (2 ** int(healpix_order)) ** 2)
        else:
            n_sky = 3072

    ra, dec = _build_sky_directions(n_sky=n_sky, healpix_order=healpix_order)
    n_sky = int(ra.size)

    ref_idx = 0
    for i, det in enumerate(detector_objs):
        if det.name == ref_ifo:
            ref_idx = i
            break

    sin_dec = np.sin(dec)
    cos_dec = np.cos(dec)
    cos_ra = np.cos(ra)
    sin_ra = np.sin(ra)
    n_hat = np.vstack((cos_dec * cos_ra, cos_dec * sin_ra, sin_dec)).T

    c_light = float(constants.c.value)
    ref_pos = detector_objs[ref_idx].vertex_vec_earth_centered

    ml = np.zeros((n_ifo, n_sky), dtype=np.int32)
    FP = np.zeros((n_ifo, n_sky), dtype=np.float64)
    FX = np.zeros((n_ifo, n_sky), dtype=np.float64)

    # CWB maps HEALPix pixel angles to the Earth-fixed (geographic) frame:
    #   phi_hp  = geographic longitude  (NOT equatorial RA)
    #   theta_hp = geographic colatitude (same Z-axis as equatorial)
    #
    # The time-delay n_hat is already correct:
    #   n_hat = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
    # is valid as a geographic/ECEF unit vector and is dotted against
    # the ECEF detector positions — no GMST rotation needed.
    #
    # For antenna patterns, atenna_pattern computes
    #   gha = GMST(t) - ra_arg
    # We want gha = -phi_geo (CWB convention), so we pass
    #   ra_arg = GMST(t) + phi_geo = GMST(t) + ra
    # The GMST cancels: gha = GMST - (GMST + phi_geo) = -phi_geo
    # Result is independent of the actual GPS time value chosen.
    gmst_rad = gmst_accurate(float(gps_time))
    ra_eff = ra + gmst_rad  # effective equatorial RA that gives gha = -phi_geo

    for i, det in enumerate(detector_objs):
        dt = np.einsum('ij,j->i', n_hat, det.vertex_vec_earth_centered - ref_pos) / c_light
        delay_idx = np.rint(dt * rate).astype(np.int32)
        ml[i] = np.clip(delay_idx, -td_size, td_size)

        f_plus, f_cross = det.atenna_pattern(ra_eff, dec, 0.0, float(gps_time))
        FP[i] = np.asarray(f_plus, dtype=np.float64)
        FX[i] = np.asarray(f_cross, dtype=np.float64)

    return ml, FP, FX
