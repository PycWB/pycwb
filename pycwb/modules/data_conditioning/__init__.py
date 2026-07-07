"""
pycwb.modules.data_conditioning — Native data conditioning pipeline.

Pure-Python resampling, regression (line removal), and whitening
(wavelet-based or MESA) for GW strain data. Operates per-lag on native
NumPy time series. This is the production conditioning engine.
"""

from .regression import *
from .whitening import whitening_python
from .whitening_mdc import whitening_mdc
from .PSD_correction import psd_correction_python
from .data_conditioning import *


def whitening_mesa_python(*args, **kwargs):
    from .whitening_mesa import whitening_mesa_python as _whitening_mesa_python

    return _whitening_mesa_python(*args, **kwargs)

__all__ = [
    "whitening_mesa_python",
    "whitening_python",
    "whitening_mdc",
    "psd_correction_python",
]
