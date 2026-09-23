"""Reference injection-arrival metadata (cWB injection::fill_in convention)."""
import numpy as np


def network_reference_times(centroids, snr_squared, delays):
    """Anchor detector arrivals on cWB's SNR^4-weighted network centroid.

    detector::ISNR stores energy (SNR squared); injection::fill_in squares
    ISNR again. Geometrical offsets are relative to the first detector.
    Work in relative GPS time to avoid loss of precision in weighted sums.
    """
    times=np.asarray(centroids,dtype=float)
    energy=np.asarray(snr_squared,dtype=float)
    delay=np.asarray(delays,dtype=float)
    if times.shape!=energy.shape or times.shape!=delay.shape or not len(times):
        raise ValueError('Injection timing needs equally sized detector arrays')
    if not (np.isfinite(times).all() and np.isfinite(energy).all() and np.isfinite(delay).all()) or np.any(energy<0) or energy.max()<=0:
        raise ValueError('Injection timing requires finite detector times and positive network energy')
    weights=(energy/energy.max())**2
    center=times[0]+np.sum((times-times[0])*weights)/weights.sum()
    return (center+(delay-delay[0])).tolist()


def cwb_arrival_times(centroids,snr_squared,injection,ifos,config):
    """Use the same geometry and coordinate convention as strain generation."""
    from pycwb.types.detector import Detector
    from pycwb.utils.skymap_coord import convert_to_celestial_coordinates, normalize_coordinate_system
    gps=injection['gps_time']
    coords=normalize_coordinate_system(injection.get('coordsys','icrs'))
    if coords!='icrs':
        ra,dec=convert_to_celestial_coordinates(*injection['sky_loc'],gps,coords,gmst_model='astropy')
    else:
        ra,dec=injection.get('ra'),injection.get('dec')
    if ra is None or dec is None:
        raise ValueError('cWB injection-arrival metadata requires source sky coordinates')
    delays=[Detector(ifo,geometry_model=getattr(config,'detector_geometry','lal')).time_delay_from_earth_center(ra,dec,gps) for ifo in ifos]
    return network_reference_times(centroids,snr_squared,delays)
