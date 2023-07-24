import numpy as np
import pycbc.noise
import pycbc.psd
from pycbc.detector import Detector
from pycbc.types import TimeSeries
from pycbc.waveform import get_td_waveform
import lalsimulation as lalsim
import os, logging
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from .read_data import check_and_resample
from pycwb.utils.module import import_helper
from ...utils.conversions.timeseries import convert_to_pycbc_timeseries

logger = logging.getLogger(__name__)


def generate_noise(psd: str = None, f_low: float = 30.0, delta_f: float = 1.0 / 16, duration: int = 32,
                   sample_rate: float = 4096, seed: int = 1234, start_time: int = 0):
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
        psd = pycbc.psd.from_txt(psd, flen, delta_f, f_low)
    else:
        psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, f_low)

    delta_t = 1.0 / sample_rate
    # Generate 32 seconds of noise at 4096 Hz
    t_samples = int(duration / delta_t)
    noise = pycbc.noise.noise_from_psd(t_samples, delta_t, psd, seed=seed)

    if start_time:
        noise._epoch = start_time
    # return noise
    return noise


def generate_from_pycbc(m1, m2, inclination, distance, sample_rate,
                        ra, dec, polarization, detectors, geocent_end_time,
                        spin1=[0, 0, 0], spin2=[0, 0, 0], f_ref=20.0, f_lower=20.0,
                        approximant='IMRPhenomXPHM'):
    """
    Generate hp and hx from pycbc and project the strains to detectors

    :param m1: mass of the first component
    :type m1: float
    :param m2: mass of the second component
    :type m2: float
    :param inclination: inclination of the binary
    :type inclination: float
    :param distance: distance of the binary
    :type distance: float
    :param sample_rate: sample rate
    :type sample_rate: float
    :param ra: right ascension
    :type ra: float
    :param dec: declination
    :type dec: float
    :param polarization: polarization
    :type polarization: float
    :param detectors: list of names of detectors to project to
    :type detectors: list[str]
    :param geocent_end_time: geocentric end time
    :type geocent_end_time: float
    :param spin1: spin of the first component
    :type spin1: list[float], optional
    :param spin2: spin of the second component
    :type spin2: list[float], optional
    :param f_ref: reference frequency, defaults to 20.0
    :type f_ref: float, optional
    :param f_lower: lower frequency cutoff, defaults to 20.0
    :type f_lower: float, optional
    :param approximant: approximant, defaults to 'IMRPhenomXPHM'
    :type approximant: str, optional
    :return: list of strains projected to detectors, the order is the same as the order of detectors given
    :rtype: list[pycbc.types.timeseries.TimeSeries]
    """
    try:
        order = lalsim.GetOrderFromString(approximant)
    except:
        order = -1

    name = lalsim.GetStringFromApproximant(lalsim.GetApproximantFromString(approximant))

    hp, hc = get_td_waveform(mass1=m1, mass2=m2,
                             spin1x=spin1[0], spin1y=spin1[1], spin1z=spin1[2], spin2x=spin2[0], spin2y=spin2[1],
                             spin2z=spin2[2], inclination=inclination,
                             approximant=name, phase_order=order, distance=distance, f_ref=f_ref, f_lower=f_lower,
                             delta_t=1.0 / sample_rate)

    project_to_detector(hp, hc, ra, dec, polarization, detectors, geocent_end_time)
    pass


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
    signal : float
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


def generate_injection_from_config(config):
    """
    A sample function to generate injection from pycbc and save it to gwf files
    with the detectors specified in the config

    :param config: user configuration
    :type config: Config
    :return: list of strains for each detector
    :rtype: list[pycbc.types.timeseries.TimeSeries]
    """
    # load noise
    start_time = 931158100
    noise = [generate_noise(f_low=30.0, sample_rate=1024.0, duration=600, start_time=start_time, seed=i)
             for i, ifo in enumerate(config.ifo)]

    # generate injection from pycbc
    from pycbc.waveform import get_td_waveform
    hp, hc = get_td_waveform(approximant="IMRPhenomPv3",
                             mass1=20,
                             mass2=20,
                             spin1z=0.9,
                             spin2z=0.4,
                             inclination=1.23,
                             coa_phase=2.45,
                             distance=500,
                             delta_t=1.0 / noise[0].sample_rate,
                             f_lower=20)
    declination = 0.65
    right_ascension = 4.67
    polarization = 2.34
    gps_end_time = 931158400
    from pycwb.modules.read_data import project_to_detector
    strain = project_to_detector(hp, hc, right_ascension, declination, polarization, config.ifo, gps_end_time)

    # inject signal into noise and convert to wavearray
    injected = [noise[i].add_into(strain[i]) for i in range(len(config.ifo))]

    return injected


def generate_injection(config, job_seg, strain=None):
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

    injected = None

    if strain:
        injected = strain

    if job_seg.noise:
        # load seeds from config, if not specified, use random seeds
        seeds = job_seg.noise['seeds'] if 'seeds' in job_seg.noise else [None, None]

        # generate noise
        noises = [generate_noise(f_low=2.0, sample_rate=config.inRate,
                                duration=job_seg.duration,
                                start_time=job_seg.start_time, seed=seeds[i])
                 for i, ifo in enumerate(ifos)]

        if injected:
            # inject signal into noise
            injected = [noises[i].add_into(injected[i]) for i in range(len(ifos))]
        else:
            injected = noises

    # generate zero noise if injected is None
    if injected is None:
        injected = [TimeSeries(np.zeros(int(job_seg.duration * config.inRate)), delta_t=1.0 / config.inRate)
                    for ifo in ifos]

    for injection in job_seg.injections:
        ##############################
        # setting default values
        ##############################
        if 'approximant' in injection:
            approximant = injection['approximant']
        elif 'approximant' in config.injection:
            approximant = config.injection['approximant']
        else:
            approximant = 'IMRPhenomXPHM'

        injection['approximant'] = approximant
        injection['delta_t'] = 1.0 / noises[0].sample_rate
        injection['f_lower'] = config.fLow if 'f_lower' not in injection else injection['f_lower']

        declination = injection['dec'] if 'dec' in injection else 0.0
        right_ascension = injection['ra'] if 'ra' in injection else 0.0
        polarization = injection['pol'] if 'pol' in injection else 0.0
        gps_end_time = injection['gps_time']

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
            generator = None

        # generate hp and hc
        if generator:
            logger.info(f'Using generator: {generator}')
            # import module
            module = import_helper(generator['module'], "wf_gen")
            # get function
            function = getattr(module, generator['function'])
            # generate waveform
            hp, hc = function(**injection)

            hp = convert_to_pycbc_timeseries(hp)
            hc = convert_to_pycbc_timeseries(hc)
        else:
            from pycbc.waveform import get_td_waveform
            hp, hc = get_td_waveform(**injection)

        from pycwb.modules.read_data import project_to_detector
        strain = project_to_detector(hp, hc, right_ascension, declination, polarization, ifos, gps_end_time)

        # inject signal into noise and convert to wavearray
        injected = [injected[i].add_into(strain[i]) for i in range(len(ifos))]

    return [check_and_resample(injected[i], config, i) for i in range(len(ifos))]
