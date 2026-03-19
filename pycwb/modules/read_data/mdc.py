import numpy as np
import lalsimulation as lalsim
import os, logging
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from pycwb.types.detector import Detector
from pycwb.types.time_series import TimeSeries as PycwbTimeSeries
from pycwb.modules.noise import generate_noise as _native_generate_noise
from pycwb.utils.module import import_function
from pycwb.utils.skymap_coord import convert_to_celestial_coordinates
from ...config import Config

logger = logging.getLogger(__name__)


def generate_noise(psd: str = None, f_low: float = 30.0, delta_f: float = 1.0 / 4, duration: int = 32,
                   sample_rate: float = 16384, seed: int = 1234, start_time: int = 0):
    """
    Generate noise from a given psd file or aLIGOZeroDetHighPower psd.

    Uses the native ``pycwb.modules.noise`` module (backed by
    ``lalsimulation.SimNoise``).

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
    pycwb.types.time_series.TimeSeries
        time series of noise
    """
    return _native_generate_noise(
        psd=psd, f_low=f_low, delta_f=delta_f,
        duration=duration, sample_rate=sample_rate,
        seed=seed, start_time=start_time,
    )


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

    # generate noise for the full padded window [padded_start, padded_end] so that
    # the whitening step sees edge samples and the WDM TF map t_idx=0 maps to padded_start,
    # consistent with the real-data path (read_from_job_segment also reads the padded window).
    noises = [generate_noise(psd=psds[i],
                             f_low=f_low, sample_rate=sample_rate,
                             duration=job_seg.padded_duration,
                             start_time=job_seg.padded_start, seed=seed) for i, seed in enumerate(seeds)]

    # if there are upstream data, add noise into the data
    if data:
        data = [noises[i].inject(PycwbTimeSeries.from_input(data[i])) for i in range(len(seeds))]
    else:
        data = noises

    logger.info(f"Generated noise for job segment {job_seg.index}")
    return data


def project_to_detector(hp, hc, ra, dec, polarization, detectors, geocent_end_time, ref_ifo='H1'):
    """Project plus/cross polarisations onto a list of detectors.

    Parameters
    ----------
    hp : TimeSeries
        Plus polarisation (any supported TimeSeries type).
    hc : TimeSeries
        Cross polarisation (any supported TimeSeries type).
    ra : float
        Right ascension in radians.
    dec : float
        Declination in radians.
    polarization : float
        Polarisation angle in radians.
    detectors : list of str
        Detector names (e.g. ['H1', 'L1']).
    geocent_end_time : float
        Geocentric end time (added to the epoch of hp/hc).
    ref_ifo : str, optional
        Reference detector (unused, kept for backward compatibility).

    Returns
    -------
    list of pycwb.types.time_series.TimeSeries
        Projected h(t) for each detector.
    """
    hp_ts = PycwbTimeSeries.from_input(hp)
    hc_ts = PycwbTimeSeries.from_input(hc)
    hp_ts = PycwbTimeSeries(data=hp_ts.data, dt=hp_ts.dt, t0=hp_ts.t0 + geocent_end_time)
    hc_ts = PycwbTimeSeries(data=hc_ts.data, dt=hc_ts.dt, t0=hc_ts.t0 + geocent_end_time)

    signals = []
    for ifo in detectors:
        detector = Detector(ifo)
        signal = detector.project_wave(hp_ts, hc_ts, ra, dec, polarization)
        signals.append(signal)

    return signals


def save_to_gwf(signals, detectors, channel_name, out_dir, start_time, duration, label):
    """
    Save the signals to gwf files

    :param signals: signals to save
    :type signals: list[pycwb.types.time_series.TimeSeries]
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


def generate_strain_from_injection(injection: dict, config: Config, sample_rate, ifos) -> list[PycwbTimeSeries]:
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
    list[pycwb.types.time_series.TimeSeries]
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
            strain = [PycwbTimeSeries.from_input(generated_data.get(ifo)) for ifo in ifos]
        elif generated_data.get('type') == 'polarizations':
            generated_data.pop('type')
            logger.info(f"Polarizations {generated_data.keys()} are generated")

            # convert to pycbc timeseries
            # if more polarizations are generated, throw an error
            if set(generated_data.keys()) != {'hp', 'hc'}:
                raise NotImplementedError("Only hp and hc polarization is supported for now")

            hp = PycwbTimeSeries.from_input(generated_data.get('hp'))
            hc = PycwbTimeSeries.from_input(generated_data.get('hc'))

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
    :rtype: list[pycwb.types.time_series.TimeSeries]
    """
    ifos = job_seg.ifos

    # load noise
    logger.info(f'Generating noise for {ifos}')

    injected = strain

    # generate zero noise if injected is None
    if not injected:
        injected = [PycwbTimeSeries(data=np.zeros(int(job_seg.duration * config.inRate)),
                                    dt=1.0 / config.inRate,
                                    t0=job_seg.analyze_start) for ifo in ifos]

    for injection in job_seg.injections:
        inj_strain = generate_strain_from_injection(injection, config, injected[0].sample_rate, ifos)

        # inject signal into noise
        injected = [injected[i].inject(inj_strain[i]) for i in range(len(ifos))]
    return injected

# Backward compatibility
generate_injection = generate_injections