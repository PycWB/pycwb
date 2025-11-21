import numpy as np
import pycbc.noise
import pycbc.psd
from pycbc.detector import Detector
from pycbc.types import TimeSeries
from pycbc.waveform import get_td_waveform
import lalsimulation as lalsim
import os, logging
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from pycwb.utils.module import import_function
from pycwb.utils.skymap_coord import convert_to_celestial_coordinates
from ...config import Config
from ...utils.conversions.timeseries import convert_to_pycbc_timeseries

logger = logging.getLogger(__name__)


def generate_noise(psd: str = None, f_low: float = 30.0, delta_f: float = 1.0 / 4, duration: int = 32,
                   sample_rate: float = 16384, seed: int = 1234, start_time: int = 0):
    """
    Generate noise from a given psd file or aLIGOZeroDetHighPower psd

    Parameters
    ----------
    psd : str
        path to psd file
    f_low : float
        low frequency cutoff
    delta_f : float
        frequency resolution
    duration : int
        duration of the noise
    sample_rate : float
        sample rate of the noise
    seed : int or None
        seed for the random number generator
    start_time : int
        start time of the noise

    Returns
    -------
    pycbc.types.timeseries.TimeSeries
        time series of noise
    """
    # generate noise
    flen = int(sample_rate / delta_f) + 1
    if psd:
        logger.info(f"Using psd file {psd} with f_low {f_low}, delta_f {delta_f}, flen {flen}")
        psd = pycbc.psd.from_txt(psd, flen, delta_f, f_low)
    else:
        logger.info(f"Using aLIGOZeroDetHighPower psd with f_low {f_low}, delta_f {delta_f}, flen {flen}")
        psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, f_low)

    delta_t = 1.0 / sample_rate

    # if sample rate is higher than psd provided, resize the psd to fill 0 values
    desired_length = int (1.0 / delta_t / psd.delta_f)//2+1
    if len(psd) < desired_length:
        logger.warning(f"PSD length {len(psd)} is less than desired length {desired_length}, resizing PSD")
        psd.resize(desired_length)

    t_samples = int(duration / delta_t)
    noise = pycbc.noise.noise_from_psd(t_samples, delta_t, psd, seed=seed)

    if start_time:
        noise._epoch = start_time
    # return noise
    return noise


def generate_noise_for_job_seg(job_seg, sample_rate, f_low=2.0, data=None):
    # if seeds is not provided, use None for all ifos
    logger.info(f"Generating noise for job segment {job_seg.index}")
    if 'seeds' in job_seg.noise:
        logger.info(f"Using seeds {job_seg.noise['seeds']}")
    if 'psds' in job_seg.noise:
        logger.info(f"Using psds {job_seg.noise['psds']}")
    logger.info(f"Sample rate: {sample_rate}")
    logger.info(f"Low frequency: {f_low}")

    seeds = job_seg.noise['seeds'] if 'seeds' in job_seg.noise else [None] * len(job_seg.ifos)
    psds = job_seg.noise['psds'] if 'psds' in job_seg.noise else [None] * len(job_seg.ifos)

    # generate noise for each ifo
    noises = [generate_noise(psd=psds[i],
                             f_low=f_low, sample_rate=sample_rate,
                             duration=job_seg.duration,
                             start_time=job_seg.start_time, seed=seed) for i, seed in enumerate(seeds)]

    # if there are upstream data, add noise into the data
    if data:
        data = [noises[i].add_into(data[i]) for i in range(len(seeds))]
    else:
        data = noises

    logger.info(f"Generated noise for job segment {job_seg.index}")
    return data


def project_to_detector(hp, hc, ra, dec, polarization, detectors, geocent_end_time, ref_ifo='H1'):
    """Make a h(t) strain time-series from an injection object as read from
    a sim_inspiral table, for example.

    Parameters
    -----------
    inj : injection object
        The injection object to turn into a strain h(t).
    delta_t : float
        Sample rate to make injection at.
    detector_name : string
        Name of the detector used for projecting injections.
    f_lower : {None, float}, optional
        Low-frequency cutoff for injected signals. If None, use value
        provided by each injection.
    distance_scale: {1, float}, optional
        Factor to scale the distance of an injection with. The default is
        no scaling.

    Returns
    --------
    signal : list of TimeSeries
        h(t) corresponding to the injection.
    """
    hp._epoch += geocent_end_time
    hc._epoch += geocent_end_time

    signals = []
    for ifo in detectors:
        detector = Detector(ifo)
        # compute the detector response
        signal = detector.project_wave(hp, hc, ra, dec, polarization)

        signals.append(signal)

    return signals


def save_to_gwf(signals, detectors, channel_name, out_dir, start_time, duration, label):
    """
    Save the signals to gwf files

    :param signals: signals to save
    :type signals: list[pycbc.types.timeseries.TimeSeries]
    :param detectors: list of detectors
    :type detectors: list[str]
    :param channel_name: channel name for the gwf file
    :type channel_name: str
    :param out_dir: output directory
    :type out_dir: str
    :param start_time: start time to be used in the name of the file
    :type start_time: float
    :param duration: duration to be used in the name of the file
    :type duration: float
    :param label: label to be used in the name of the file
    :type label: str
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i, detector in enumerate(detectors):
        strain = GWpyTimeSeries(
            data=signals[i].data,
            times=signals[i].sample_times,
        )
        strain.channel = f'{detector}:{channel_name}'
        strain.name = strain.channel
        strain.write(f'{out_dir}/{detector}-{label}-{int(start_time)}-{int(duration)}.gwf')


def generate_strain_from_injection(injection: dict, config: Config, sample_rate, ifos) -> list[TimeSeries]:
    """
    Generate strain from given injection parameters, the config is used to get the default values

    Parameters
    ----------
    injection : dict
        injection parameters
    config : Config
        user configuration
    sample_rate : float
        sample rate
    ifos : list[str]
        list of detectors

    Returns
    -------
    list[pycbc.types.timeseries.TimeSeries]
        strain
    """
    from warnings import warn

    # setting default values removed, PycWB core code should not handle the default values to prevent inexplicit overwrite!!!
    injection['delta_t'] = 1.0 / sample_rate
    logger.info(f'Generating injection for {ifos} with parameters: \n {injection} \n')

    ##############################
    # generating injection
    ##############################
    # check if waveform generator is specified
    if 'generator' in injection:
        generator = injection['generator']
    elif 'generator' in config.injection:
        generator = config.injection['generator']
    else:
        # ------------------------------
        # deprecated warning: the default waveform generator will be remove in the future
        warn("The default waveform generator will be removed in the future, please specify the generator in the injection parameters", DeprecationWarning)
        generator = "pycbc.waveform.get_td_waveform"
        # ------------------------------

    # check if generator is a string
    if not isinstance(generator, str):
        raise TypeError(f"Generator should be a string, e.g. 'pycbc.waveform.get_td_waveform', got {type(generator)}")

    # generate hp and hc
    logger.info(f"Generating waveform using {generator}")
    # import function
    function = import_function(generator)
    # generate waveform
    generated_data = function(**injection)

    # ------------------------------
    # backward compatibility for hp and hc, remove in the future
    if isinstance(generated_data, tuple):
        warn("Returning hp and hc as tuple is deprecated, please return as dict with keys hp and hc", DeprecationWarning)
        logger.warning("Returning hp and hc as tuple is going to be deprecated, please return as dict with keys hp and hc")
        generated_data = {
            'type': 'polarizations',
            'hp': generated_data[0],
            'hc': generated_data[1]
        }
    # ------------------------------

    if isinstance(generated_data, dict):
        if generated_data.get('type') == 'strain':
            logger.info("Strain is generated")
            generated_data.pop('type')

            # check on ifos
            logger.info(f"Checking if the ifos in the config are same as the provided ifos")
            provided_ifos = set(generated_data.keys())
            if provided_ifos != set(ifos):
                raise ValueError(f"Provided ifos {provided_ifos} are not same as the ifos {ifos} in config")
            
            # return the strain with the order of ifos in the config
            strain = [convert_to_pycbc_timeseries(generated_data.get(ifo)) for ifo in ifos]
        elif generated_data.get('type') == 'polarizations':
            generated_data.pop('type')
            logger.info(f"Polarizations {generated_data.keys()} are generated")

            # convert to pycbc timeseries
            # if more polarizations are generated, throw an error
            if set(generated_data.keys()) != {'hp', 'hc'}:
                raise NotImplementedError("Only hp and hc polarization is supported for now")

            hp = convert_to_pycbc_timeseries(generated_data.get('hp'))
            hc = convert_to_pycbc_timeseries(generated_data.get('hc'))

            #check GPS time 
            gps_end_time = injection.get('gps_time')
            if gps_end_time is None:
                raise ValueError("gps_time is required in the injection parameters")
            
            #Check Coordinate system, default to 'icrs' if not provided
            right_ascension = injection.get('ra') 
            declination = injection.get('dec')
            polarization = injection.get('pol')
            coordinate_system = injection.get('coordsys', 'icrs')

            #change coordinates to icrs 
            if coordinate_system != 'icrs': 
                logger.info(f"Converting from {coordinate_system} to celestial coordinates for injection")
                
                sky_loc = injection.get('sky_loc')
                if sky_loc is None or len(sky_loc) != 2:
                    raise ValueError(f"sky_loc = [phi,theta] is required in the injection parameters when coordinate system is {coordinate_system}, while sky_loc: {injection.get('sky_loc')}") 
                right_ascension, declination = convert_to_celestial_coordinates(sky_loc[0],sky_loc[1],injection.get('gps_time'),coordinate_system)

            if declination is None or right_ascension is None or polarization is None:
                raise ValueError(f"ra, dec and pol are required in the injection parameters, while ra: {right_ascension}, dec: {declination}, pol: {polarization}")
        
            # project to detectors 
            logger.info(f"Projecting {generated_data.keys()} to detectors {ifos}")

            strain = project_to_detector(hp, hc, right_ascension, declination, polarization, ifos, gps_end_time)
    else:
        raise ValueError(f"Unsupported return type from waveform generator: {generated_data}, should be tuple for hp and hc, dict for more polarizations or list for strain")

    return strain


def generate_injections(config, job_seg, strain=None):
    """
    A sample function to generate injection from pycbc and save it to gwf files
    with the detectors specified in the config

    :param config: user configuration
    :type config: Config
    :return: list of strains for each detector
    :rtype: list[pycbc.types.timeseries.TimeSeries]
    """
    ifos = job_seg.ifos

    # load noise
    logger.info(f'Generating noise for {ifos}')

    injected = strain

    # generate zero noise if injected is None
    if not injected:
        injected = [TimeSeries(np.zeros(int(job_seg.duration * config.inRate)),
                               delta_t=1.0 / config.inRate,
                               epoch=job_seg.start_time) for ifo in ifos]

    for injection in job_seg.injections:
        strain = generate_strain_from_injection(injection, config, injected[0].sample_rate, ifos)

        # inject signal into noise and convert to wavearray
        injected = [injected[i].add_into(strain[i]) for i in range(len(ifos))]
    return injected

# Backward compatibility
generate_injection = generate_injections