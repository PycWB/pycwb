"""
pycwb.modules.coherence_root — ROOT-backed coherence engine.

Wraps cWB's C++ ``network::getNetworkPixels`` and ``network::cluster``
routines via ROOT bindings. Loops over resolution levels and lags using
ROOT WSeries/WDM objects.

.. note::
   This module depends on ROOT and is part of the legacy layer being phased out.
"""

from .coherence import *
