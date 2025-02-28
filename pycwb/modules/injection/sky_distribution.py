import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Galactic
import astropy.time as time
import healpy as hp  # Optional for custom maps

def generate_sky_distribution(sky_distribution_config, n_samples):
    """
    Generate a sky distribution of n_samples points. The supported distribution types are:
    - existing: Load RA/Dec from existing data
    - random_all_sky: Randomly distribute points in the whole sky
    - patch: Generate points in a small circle around the center
    - custom: Sample from a HEALPix map

    :param sky_distribution_config: The sky distribution configuration
    :param n_samples: The number of samples
    :return: The RA and Dec of the sky distribution
    """
    dist_type = sky_distribution_config['type']
    
    if dist_type == "existing":
        # Load RA/Dec from existing data (example)
        ra, dec = np.loadtxt(sky_distribution_config['existing_path'], unpack=True)
        return ra, dec
    
    elif dist_type == "UniformAllSky":
        ra = np.random.uniform(0, 360, n_samples)
        dec = np.degrees(np.arcsin(2*np.random.uniform(0, 1, n_samples) - 1))
        return ra, dec
    
    elif dist_type == "Patch":
        center_ra, center_dec = sky_distribution_config['patch']['center']
        radius = sky_distribution_config['patch']['radius']
        
        # Generate points in a small circle around the center
        ra, dec = sample_uniform_sky_area(center_ra, center_dec, radius, n_samples)
        return ra, dec
    
    elif dist_type == "Custom":
        # Example: Sample from a HEALPix map
        skymap = hp.read_map(sky_distribution_config['custom']['healpix_map'])
        nside = sky_distribution_config['custom']['nside']
        npix = hp.nside2npix(nside)
        indices = np.random.choice(npix, size=n_samples, p=skymap)
        theta, phi = hp.pix2ang(nside, indices)
        ra = np.degrees(phi)
        dec = np.degrees(np.pi/2 - theta)
        return ra, dec
    
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
    ra = sky_locations[0]
    dec = sky_locations[1]

    if len(injections) != len(ra) or len(injections) != len(dec):
        raise ValueError("The number of injections and sky locations must be the same.")
    
    # distributed_injections = []

    # shuffle the sky locations
    if shuffle:
        np.random.shuffle(ra)
        np.random.shuffle(dec)

    if coordsys == 'icrs':
        # If the coordinate system is ICRS, save the sky location directly in ra and dec attributes
        for i, inj in enumerate(injections):
            inj['ra'] = ra[i]
            inj['dec'] = dec[i]
    else:
        # If the coordinate system is not ICRS, save the sky location in sky_loc attribute and add the coordsys for later conversion
        for i, inj in enumerate(injections):
            inj['sky_loc'] = [ra[i], dec[i]]
            inj['coordsys'] = coordsys



def sample_uniform_sky_area(ra_center, dec_center, radius, n_samples=1):
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
    ra_center = np.radians(ra_center)
    dec_center = np.radians(dec_center)
    radius = np.radians(radius)
    
    # Sample uniformly in cos(θ) and φ for area uniformity
    cos_theta = np.random.uniform(np.cos(radius), 1, n_samples)
    theta = np.arccos(cos_theta)  # Convert back to theta
    phi = np.random.uniform(0, 2 * np.pi, n_samples)

    # Convert to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Rotate from (0,0,1) to (RA_center, Dec_center)
    sin_dec = np.sin(dec_center)
    cos_dec = np.cos(dec_center)
    sin_ra = np.sin(ra_center)
    cos_ra = np.cos(ra_center)

    x_new = cos_dec * cos_ra * x - sin_ra * y + sin_dec * cos_ra * z
    y_new = cos_dec * sin_ra * x + cos_ra * y + sin_dec * sin_ra * z
    z_new = -sin_dec * x + cos_dec * z

    # Convert back to RA and Dec
    dec_samples = np.arcsin(z_new)
    ra_samples = np.arctan2(y_new, x_new)

    # Normalize RA to [0, 360) degrees
    ra_samples = np.degrees(ra_samples) % 360
    dec_samples = np.degrees(dec_samples)

    return np.array([ra_samples, dec_samples])