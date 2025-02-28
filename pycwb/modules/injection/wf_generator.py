from pycbc.waveform import get_td_waveform
import lalsimulation as lalsim
from pycwb.utils.module import import_function
import logging

logger = logging.getLogger(__name__)


def generate_injection(parameters, generator):
    """
    Generate an injection waveform.

    :param parameters: The injection parameters
    :param generator: The generator function
    :return: The injection waveform
    """

    wf_gen_func = import_function(generator)

    hp, hc = wf_gen_func(parameters)

    # TODO: check the type and sanity of the waveform
    return hp, hc