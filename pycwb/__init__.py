from ._version import version, __version__, version_tuple, __version_tuple__

import warnings
try:
    import ROOT
    # load wavelet library if not loaded
    from pycwb.utils.check_ROOT import check_and_load_wavelet
    check_and_load_wavelet(ROOT)
except ImportError:
    ROOT = None
    warnings.warn(
        "ROOT module not found. CWB conversions will not work. This warning will be removed in future versions when ROOT is no longer a dependency.",
        ImportWarning,
        stacklevel=2
    )

import matplotlib as mpl
mpl.use('Agg')


