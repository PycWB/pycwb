"""
pycwb.modules.cwb_results — Legacy cWB results parser.

Reads and summarizes cWB ``liveTime`` ROOT files, computing live-time
statistics (total seconds, losses, min/max, counts per threshold).

.. note::
   This module depends on ROOT and is part of the legacy layer being phased out.
"""

from .live_root import CwbLiveRoot, LiveRootSummary

__all__ = ["CwbLiveRoot", "LiveRootSummary"]
