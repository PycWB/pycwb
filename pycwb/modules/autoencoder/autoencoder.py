from .cwb_autoencoder import AutoEncoder
import pycwb, os


def get_glitchness(config, data, sSNR, likelihood, weight_path=None):
    if not weight_path:
        package_abs_path = os.path.dirname(os.path.abspath(pycwb.__file__))
        weight_path = os.path.join(package_abs_path, 'vendor/autoencoder/cwb_autoencoder.h5')

    ae = AutoEncoder()
    ae.set_weights(weight_path)
    glitchness = 0
    for i in range(config.nIFO):
        glitchness += sSNR[i] * ae.get_glness(data[i], 0)

    return glitchness / likelihood
