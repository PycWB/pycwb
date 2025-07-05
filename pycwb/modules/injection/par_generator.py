import numpy as np
from pycwb.utils.module import import_function
import logging
logger = logging.getLogger(__name__)


def get_injection_list_from_parameters(injection):
    """
    Generate the injection list from the injection parameters.

    :param injection: The injection parameters.
    :type injection: dict
    :return: The list of injections.
    """
    # get the injection parameters
    if 'parameters' in injection:
        if isinstance(injection['parameters'], list):
            injections = injection['parameters']
        else:
            injections = [injection['parameters']]
    elif 'parameters_from_python' in injection:
        # get the injection parameters
        par_gen_func = import_function(injection['parameters_from_python']['function'])
        # call the function to get the injection parameters
        func_args = injection['parameters_from_python'].get('args', {})
        injections = par_gen_func(**func_args)

        logger.info(f"{len(injections)} injection parameters generated")

        if not isinstance(injections, list):
            raise ValueError('The function get_injection_parameters() should return a list of injection parameters')
    else:
        raise ValueError('No injection parameters specified, '
                         'please specify either parameters or parameters_from_python')
    
    return injections


def hrss_scaling(input, targeted_hrss_list):
    """
    This function scales the input waveform to the targeted hrss list.
    The scaling factor is calculated as the ratio of the targeted hrss and the input hrss.

    :param input: The input waveform
    :param targeted_hrss_list: The list of targeted hrss
    :return: The scaled waveform
    """
    hrss = np.sqrt(np.sum(input ** 2))
    return input * (targeted_hrss_list / hrss)


def snr_scaling(input, psd):
    """
    This function scales the input waveform to the targeted snr list.
    The scaling factor is calculated as the ratio of the targeted snr and the input snr.

    :param input: The input waveform
    :param psd: The power spectral density
    :return: The scaled waveform
    """
    pass


def repeat(par_list, n_repeat):
    """
    This function repeats the parameter list n times.

    :param par_list: The list of parameters
    :param n_repeat: The number of repeats
    :return: The repeated list
    """
    return [d.copy() for _ in range(n_repeat) for d in par_list]


def inc_pol_replicator(par_list, inclinations, polarizations):
    """
    This function replicates the parameter list for pairs of inclinations and polarizations.

    :param par_list: The list of parameters
    :param inclinations: The list of inclinations
    :param polarizations: The list of polarizations
    :return: The replicated list
    """
    return [dict(par, inclination=inc, polarization=pol) for par in par_list for inc, pol in zip(inclinations, polarizations)]
