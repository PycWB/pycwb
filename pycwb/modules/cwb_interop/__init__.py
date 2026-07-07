"""
pycwb.modules.cwb_interop — cWB interoperability helpers.

Creates standalone cWB working directories for direct numerical comparison
between PycWB and original cWB runs. Generates equivalent
``user_parameters.C`` configs, frame file lists, and DQ files.
"""

from .cwb_workdir_setup import create_cwb_workdir
