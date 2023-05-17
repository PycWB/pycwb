"""
This module is a glitch classifier based on autoencoder neural network.
"""
from .cwb_autoencoder import AutoEncoder
import pycwb, os


def get_glitchness(config, data, sSNR, likelihood, weight_path=None):
    """
    Get glitchness of reconstructed waveform with autoencoder

    Parameters
    ----------
    config : pycwb.Config
        Configuration object
    data : list of numpy.ndarray or pycbc.types.timeseries.TimeSeries
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
    if not weight_path:
        package_abs_path = os.path.dirname(os.path.abspath(pycwb.__file__))
        weight_path = os.path.join(package_abs_path, 'vendor/autoencoder/cwb_autoencoder.h5')

    ae = AutoEncoder()
    ae.set_weights(weight_path)
    glitchness = 0
    for i in range(config.nIFO):
        glitchness += sSNR[i] * ae.get_glness(data[i], 0)

    return glitchness / likelihood
