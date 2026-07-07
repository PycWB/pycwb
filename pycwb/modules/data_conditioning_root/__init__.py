"""
pycwb.modules.data_conditioning_root — ROOT-backed data conditioning.

Regression and whitening using cWB C++ routines via ROOT bindings.
Supports parallel processing with multiprocessing.

.. note::
   This module depends on ROOT and is part of the legacy layer being phased out.
"""

from .regression import *
from .whitening import whitening_cwb
from .whitening_mdc import whitening_mdc
from .data_conditioning import *


def whitening_mesa(*args, **kwargs):
    from .whitening_mesa import whitening_mesa as _whitening_mesa

    return _whitening_mesa(*args, **kwargs)

__all__ = [
    "whitening_mesa",
    "whitening_cwb",
    "whitening_mdc",
]
