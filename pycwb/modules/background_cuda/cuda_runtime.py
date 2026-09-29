"""Compatibility alias for :mod:`pycwb.utils.gpu.cuda_runtime`.

New code should import the canonical module. Aliasing the module object keeps
legacy imports, process-local caches and monkeypatches on the same state.
"""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("pycwb.utils.gpu.cuda_runtime")
