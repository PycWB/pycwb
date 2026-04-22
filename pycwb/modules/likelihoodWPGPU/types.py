"""
Data types for the GPU likelihood module.

Re-exports ``SkyStatistics`` and ``SkyMapStatistics`` from the CPU module
so callers do not need to change imports.  These dataclasses are backend-
agnostic (plain numpy arrays and scalars).
"""

from pycwb.modules.likelihoodWP.typing import SkyStatistics, SkyMapStatistics

__all__ = ["SkyStatistics", "SkyMapStatistics"]
