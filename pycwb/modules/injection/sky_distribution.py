import numpy as np
import logging
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Galactic
import astropy.time as time
import healpy as hp  # Optional for custom maps

logger = logging.getLogger(__name__)

#TODO: Modify ra dec to phi theta  

def generate_sky_distribution(sky_distribution_config, n_samples):
    """
    Generate a sky distribution of n_samples points. The supported distribution types are:
    - existing: Load phi/Theta from existing data
    - random_all_sky: Randomly distribute points in the whole sky
    - patch: Generate points in a small circle around the center
    - fixed: Fixed phi/Theta for all injections
    - custom: Sample from a HEALPix map

    :param sky_distribution_config: The sky distribution configuration
    :param n_samples: The number of samples
    :return: The RA and Dec of the sky distribution
    """
    dist_type = sky_distribution_config['type']
    logger.info(f"Generating sky distribution of type: {dist_type} with {n_samples} samples")
    
    if dist_type == "existing":
        logger.info("Loading existing sky distribution data from: "
                    f"{sky_distribution_config['existing_path']}")
        # Load RA/Dec from existing data (example)
        unit = sky_distribution_config.get('unit', None)
        if unit not in ['rad', 'deg']:
            raise ValueError("Must specify the unit of the existing data, either 'rad' or 'deg'")
        phi, theta = np.loadtxt(sky_distribution_config['existing_path'], unpack=True)

        if unit == 'deg':
            logger.info("Converting phi/theta from degrees to radians")
            phi = np.deg2rad(phi)
            theta = np.deg2rad(theta)
        return phi, theta
    
    elif dist_type == "UniformAllSky":
        logger.info("Generating uniform distribution across the entire sky")
        phi = np.random.uniform(0, 360, n_samples)
        theta = np.degrees(np.arcsin(2*np.random.uniform(0, 1, n_samples) - 1))
        return np.deg2rad(phi), np.deg2rad(theta)
    
    elif dist_type == "Patch":
        logger.info("Generating points in a small circular patch around a center RA/Dec")
        logger.info("Patch generation uses degree internally, and will return RA/Dec in radians for injection.")
        # parameters check
        unit = sky_distribution_config['patch'].get('unit', None)
        if unit not in ['rad', 'deg']:
            raise ValueError("Must specify the unit of the patch, either 'rad' or 'deg'")
        if 'center' not in sky_distribution_config['patch'] or 'radius' not in sky_distribution_config['patch']:
            raise ValueError("Must specify the center and radius of the patch")
        if 'phi' not in sky_distribution_config['patch']['center'] or 'theta' not in sky_distribution_config['patch']['center']:
            raise ValueError("Must specify the center RA and Dec of the patch")

        center_phi = sky_distribution_config['patch']['center']['phi']
        center_theta = sky_distribution_config['patch']['center']['theta']
        radius = sky_distribution_config['patch']['radius']

        if unit == 'rad':
            logger.info(f"Center phi: {center_phi} radians, Center theta: {center_theta} radians, Radius: {radius} radians")
            logger.info("Converting center Phi/Theta and radius from radians to degrees")
            center_phi = np.rad2deg(center_phi)
            center_theta = np.rad2deg(center_theta)
            radius = np.rad2deg(radius) #try radius 0 
       
        logger.info(f"Center Phi: {center_phi} degrees, Center Theta: {center_theta} degrees, Radius: {radius} degrees")
        logger.info("Generating points in a small circular patch around a center Phi/Theta")
        # Generate points in a small circle around the center
        phi, theta = sample_uniform_sky_area(center_phi, center_theta, radius, n_samples)
        return np.deg2rad(phi), np.deg2rad(theta)
    
    elif dist_type == "Fixed": 
        unit = sky_distribution_config["coordinates"].get('unit', None)
        if unit not in ['rad', 'deg']:
            raise ValueError("Must specify the unit of the patch, either 'rad' or 'deg'")
        
        phi = sky_distribution_config['coordinates']['sky_loc']['phi']
        theta = sky_distribution_config['coordinates']['sky_loc']['theta'] 
        if unit == 'deg': 
            logger.info("Converting phi/theta from degrees to radians")
            phi = np.deg2rad(phi)
            theta = np.deg2rad(theta)
        return np.repeat(phi, n_samples), np.repeat(theta, n_samples)
    
    elif dist_type == "Custom":
        logger.info(f"Sampling from a custom HEALPix map: {sky_distribution_config['custom']['healpix_map']}")
        # Example: Sample from a HEALPix map
        skymap = hp.read_map(sky_distribution_config['custom']['healpix_map'])
        nside = sky_distribution_config['custom']['nside']
        npix = hp.nside2npix(nside)
        indices = np.random.choice(npix, size=n_samples, p=skymap)
        theta, phi = hp.pix2ang(nside, indices)
        phi = np.degrees(phi)
        theta = np.degrees(np.pi/2 - theta)
        return np.deg2rad(phi), np.deg2rad(theta)
    
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")
    

def distribute_injections_on_sky(injections, sky_locations, shuffle=True, coordsys='icrs'):
    """
    Randomly distribute injections in the sky locations. If the coordinate system is ICRS, the sky locations are saved in ra and dec attributes. 
    Otherwise, the sky locations are saved in sky_loc attribute and the coordsys attribute is added for later conversion with gps_time.

    :param injections: The list of injections
    :param sky_locations: The list of sky locations (ra, dec)
    :param shuffle: Shuffle the sky locations before distributing the injections
    :param coordsys: The coordinate system of the sky locations
    """
    logger.info(f"Distributing {len(injections)} injections on sky locations with coordsys: {coordsys}")
    phi = sky_locations[0]
    theta = sky_locations[1]

    if len(injections) != len(phi) or len(injections) != len(theta):
        raise ValueError("The number of injections and sky locations must be the same.")
    
    # distributed_injections = []

    # shuffle the sky locations
    if shuffle:
        np.random.shuffle(phi)
        np.random.shuffle(theta)

    if coordsys == 'icrs':
        # If the coordinate system is ICRS, save the sky location directly in ra and dec attributes
        for i, inj in enumerate(injections):
            inj['ra'] = phi[i]
            inj['dec'] = theta[i]
    else:
        # If the coordinate system is not ICRS, save the sky location in sky_loc attribute and add the coordsys for later conversion
        for i, inj in enumerate(injections):
            inj['sky_loc'] = [phi[i], theta[i]]
            inj['coordsys'] = coordsys



def sample_uniform_sky_area(phi_center, theta_center, radius, n_samples=1):
    """
    Sample points uniformly in area within a circular patch on the celestial sphere.

    Parameters:
    ra_center (float): Right Ascension (RA) of the center in degrees.
    dec_center (float): Declination (Dec) of the center in degrees.
    radius (float): Radius of the circular patch in degrees.
    n_samples (int): Number of points to sample.

    Returns:
    np.ndarray: (n_samples, 2) array with sampled (RA, Dec) coordinates in degrees.
    """
    # Convert degrees to radians 
    phi_center = np.radians(phi_center) #B4: RA center
    theta_center = np.radians(theta_center) #B4: Dec center
    radius = np.radians(radius)
    
    # Sample uniformly in cos(θ) and φ for area uniformity
    cos_alpha = np.random.uniform(np.cos(radius), 1, n_samples) #renamed theta -> alpha 
    alpha = np.arccos(cos_alpha)  # Convert back to theta    #renamed theta -> alpha 
    beta = np.random.uniform(0, 2 * np.pi, n_samples)         #renamed phi -> beta

    # Convert to Cartesian coordinates
    x = np.sin(alpha) * np.cos(beta)
    y = np.sin(alpha) * np.sin(beta)
    z = np.cos(alpha)

    # Rotate from (0,0,1) to (RA_center, Dec_center)
    sin_theta = np.sin(theta_center)
    cos_theta = np.cos(theta_center)
    sin_phi = np.sin(phi_center)
    cos_phi = np.cos(phi_center)

    x_new = sin_theta * cos_phi * x - sin_phi * y + cos_theta * cos_phi * z
    y_new = sin_theta * sin_phi * x + cos_phi * y + cos_theta * sin_phi * z
    z_new = - cos_theta * x + sin_theta * z

    # Convert back to RA and Dec
    theta_samples = np.arcsin(z_new)
    phi_samples = np.arctan2(y_new, x_new)

    # Normalize RA to [0, 360) degrees
    phi_samples = np.degrees(phi_samples) % 360
    theta_samples = np.degrees(theta_samples)

    return np.array([phi_samples, theta_samples])
