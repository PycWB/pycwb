import warnings

import numpy as np

try:
    import ROOT
except ImportError:
    ROOT = None
    warnings.warn(
        "ROOT module not found. CWB conversions will not work. This warning will be removed in future versions when ROOT is no longer a dependency.",
        ImportWarning,
        stacklevel=2,
    )


def convert_numpy_to_wavearray(data: np.array):
    """
    Convert numpy array to wavearray with python loop.

    Parameters
    ----------
    data : np.ndarray
        Input numpy array.

    Returns
    -------
    ROOT.wavearray
        Converted ROOT wavearray.
    """
    h = ROOT.wavearray(np.double)()

    for d in data:
        h.append(d)

    h.start(0.0)
    h.rate(1.0)

    return h
