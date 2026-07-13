"""
pycwb.modules.xtalk — Cross-talk coefficient management.

Manages cross-talk (XTalk) catalogs: loads pre-computed crosstalk
coefficient files (binary or .npz) and provides fast Numba-accelerated
lookup of crosstalk coefficients for pixel pairs.
"""

from .type import XTalk
from .xtalk_data import check_and_download_xtalk_data
from .monster import load_catalog, read_catalog_metadata

__all__ = [
    "XTalk",
    "check_and_download_xtalk_data",
    "load_catalog",
    "read_catalog_metadata",
]
