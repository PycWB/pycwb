"""
pycwb.modules.sparse_series — Sparse time-frequency representations.

Creates sparse TF series from fragment clusters. Extracts pixel-level
data (time, frequency, amplitude, phase) from TF maps at each resolution
level for efficient downstream processing.
"""

from .sparse_table import *