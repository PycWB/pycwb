"""
pycwb.modules.multi_resolution_wdm — Multi-resolution WDM management.

Creates and manages Wavelet Domain Model (WDM) objects for all resolution
levels. Validates filter lengths against segment edges and time-delay
sizes. Wraps ``pycwb.types.wdm.WDM``.
"""

from .wdm import *