"""
pycwb.modules.likelihood_root — ROOT-backed coherent likelihood.

Wraps cWB's C++ ``network::likelihood`` via ROOT bindings. Iterates over
clusters, converting between native and ROOT types via ``cwb_conversions``.

.. note::
   This module depends on ROOT and is part of the legacy layer being phased out.
"""

from .likelihood import *
