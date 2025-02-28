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
        frame = sky_distribution_config['patch']['frame']
        
        # Generate points in a small circle around the center
        skycoord = SkyCoord(ra=center_ra*u.deg, dec=center_dec*u.deg, frame='icrs')
        ra, dec = skycoord.represent_as('unitsphericalcircle').sample(n_samples, radius=radius*u.deg)
        return ra.to(u.deg).value, dec.to(u.deg).value
    
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
    

def distribute_injections(injections, sky_locations, coordsys='icrs'):
    """
    Randomly distribute injections in the sky locations. If the coordinate system is ICRS, the sky locations are saved in ra and dec attributes. 
    Otherwise, the sky locations are saved in sky_loc attribute and the coordsys attribute is added for later conversion with gps_time.

    :param injections: The list of injections
    :param sky_locations: The list of sky locations (ra, dec)
    :param coordsys: The coordinate system of the sky locations
    :return: The distributed injections
    """
    ra = sky_locations[0]
    dec = sky_locations[1]

    if len(injections) != len(ra) or len(injections) != len(dec):
        raise ValueError("The number of injections and sky locations must be the same.")
    
    distributed_injections = []
    # shuffle the sky locations
    ra = np.random.shuffle(ra)
    dec = np.random.shuffle(dec)

    if coordsys == 'icrs':
        # If the coordinate system is ICRS, save the sky location directly in ra and dec attributes
        for inj, sky_loc in zip(injections, sky_locations):
            inj['ra'] = sky_loc[0]
            inj['dec'] = sky_loc[1]
            distributed_injections.append(inj)
    else:
        # If the coordinate system is not ICRS, save the sky location in sky_loc attribute and add the coordsys for later conversion
        for inj, sky_loc in zip(injections, sky_locations):
            inj['sky_loc'] = sky_loc
            inj['coordsys'] = coordsys
            distributed_injections.append(inj)
    
    return distributed_injections

