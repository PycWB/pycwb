"""
Temporary transition shim for likelihoodWP.utils.

.. deprecated::
    Prefer importing from ``pycwb.modules.likelihoodWP.packet_ops`` directly.
    This module is not part of the documented public API.

    Tracked for removal: see SPLIT_LIKELIHOODWP_PLAN.md Phase A2.
"""
from .packet_ops import *
from .packet_ops import __all__
