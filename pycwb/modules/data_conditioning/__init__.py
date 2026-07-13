"""
pycwb.modules.data_conditioning — Native data conditioning pipeline.

Pure-Python resampling, regression (line removal), and whitening
(wavelet-based or MESA) for GW strain data. Operates per-lag on native
NumPy time series. This is the production conditioning engine.
"""

from .regression import *
from .whitening import whitening_python
from .injection_whitening import whiten_injection_strain
from .PSD_correction import psd_correction_python
from .data_conditioning import *


def whitening_mesa_python(*args, **kwargs):
    from .whitening_mesa import whitening_mesa_python as _whitening_mesa_python

    return _whitening_mesa_python(*args, **kwargs)

__all__ = [
    "whitening_mesa_python",
    "whitening_python",
    "whiten_injection_strain",
    "psd_correction_python",
]

# Compatibility alias for one release. Direct imports of the former module emit
# a DeprecationWarning.
whitening_mdc = whiten_injection_strain
__all__.append("whitening_mdc")
