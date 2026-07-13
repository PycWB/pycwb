"""
pycwb.modules.gwosc — GW Open Science Center interface.

Interfaces with the GWOSC API to retrieve public event metadata
(GPS time, detectors, science segments) and download frame files
for known GW events.
"""

from .gwosc import event_info, download_frames_files, get_cat_files, analysis_period

__all__ = [
    "event_info",
    "download_frames_files",
    "get_cat_files",
    "analysis_period",
]
