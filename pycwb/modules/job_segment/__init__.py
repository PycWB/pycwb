"""
pycwb.modules.job_segment — Analysis job segmentation.

Constructs the analysis job segmentation: reads DQ segment lists, builds
job segments with frame file selection, injection scheduling, and
super-lag generation. Handles flattening by trial index and CAT2 veto
windows.
"""

from .job_segment import *
from .job_segment import build_injection_veto_windows, intersect_intervals
from .dq_segment import *
from .dq_segment import build_cat2_veto_windows
from .super_lag import *