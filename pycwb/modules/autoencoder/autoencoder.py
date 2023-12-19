"""
This module is a glitch classifier based on autoencoder neural network.
"""
from .cwb_autoencoder import AutoEncoder
import pycwb, os
import logging

logger = logging.getLogger(__name__)


def get_glitchness(config, reconstructed_waveform, sSNR, likelihood, weight_path=None):
    """
    Get glitchness of reconstructed waveform with autoencoder

    Parameters
    ----------
    config : pycwb.Config
        Configuration object
    reconstructed_waveform : list of numpy.ndarray or list of pycbc.types.timeseries.TimeSeries
        Reconstructed waveform
    sSNR : list of float
        Signal-to-noise ratio of the cluster
    likelihood : float
        Likelihood of the cluster
    weight_path : str, optional
        Path to the weight file, if not given, use the default weight file in the package

    Returns
    -------
    float
        Glitchness of the cluster
    """
    try:
        if not weight_path:
            package_abs_path = os.path.dirname(os.path.abspath(pycwb.__file__))
            weight_path = os.path.join(package_abs_path, 'vendor/autoencoder/cwb_autoencoder.h5')

        ae = AutoEncoder()
        ae.set_weights(weight_path)
        glitchness = 0
        for i in range(config.nIFO):
            glitchness += sSNR[i] * ae.get_glness(reconstructed_waveform[i], 0)

        logger.info("Glitchness: %f", (glitchness / likelihood)[0][0])
        return (glitchness / likelihood)[0][0]
    except Exception as e:
        logger.error("Error in get_glitchness: %s", e)
        return None
