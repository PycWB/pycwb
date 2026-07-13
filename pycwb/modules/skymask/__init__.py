"""
pycwb.modules.skymask — HEALPix sky mask generation.

Creates circular sky masks on HEALPix grids for targeted GW searches.
Converts cWB sky coordinates (phi/theta) to geographic coordinates and
fills mask pixels within a specified angular radius.
"""

from .skymask import make_sky_mask

__all__ = ["make_sky_mask"]
