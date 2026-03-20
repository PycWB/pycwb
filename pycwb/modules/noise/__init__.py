"""
pycwb.modules.noise — Native noise generation for pycwb.

Provides coloured Gaussian noise from arbitrary PSDs without depending on PyCBC.
The underlying engine is ``lalsimulation.SimNoise`` (overlap-save method),
identical to what PyCBC uses internally.

Public API
----------
generate_noise       High-level dispatcher (routes to gaussian / non-gaussian).
load_psd             Load a two-column (freq, ASD) text file onto a uniform grid.
analytic_psd         Evaluate a named lalsimulation noise model on a uniform grid.
gaussian_noise_from_psd  Generate coloured Gaussian noise from a PSD array.

Stubs (not yet implemented)
---------------------------
non_gaussian_noise   Placeholder for non-Gaussian noise models.
inject_glitches      Placeholder for glitch injection.
"""

from .psd import load_psd, analytic_psd
from .gaussian import gaussian_noise_from_psd, generate_noise

__all__ = [
    "generate_noise",
    "load_psd",
    "analytic_psd",
    "gaussian_noise_from_psd",
]
