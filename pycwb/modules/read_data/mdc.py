import pycbc.noise
import pycbc.psd
from pycbc.detector import Detector
from pycbc.waveform import get_td_waveform
import lalsimulation as lalsim


def generate_noise(psd: str = None, f_low: float = 30.0, delta_f: float = 1.0 / 16, duration: int = 32,
                   sample_rate: float = 4096, seed: int = 1234, start_time: int = 0):
    # generate noise
    flen = int(2048 / delta_f) + 1
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
                        spin1=[0, 0, 0], spin2=[0, 0, 0], f_ref=11.0, f_lower=11.0,
                        approximant='NRSur7dq4pseudoFourPN'):
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


def project_to_detector(hp, hc, ra, dec, polarization, detectors, geocent_end_time):
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
