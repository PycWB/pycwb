"""Compatibility alias for :mod:`pycwb.modules.data_conditioning.parallel`.

New code should import the canonical module. Aliasing the module object keeps
legacy imports, process-local caches and monkeypatches on the same state.
"""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("pycwb.modules.data_conditioning.parallel")
