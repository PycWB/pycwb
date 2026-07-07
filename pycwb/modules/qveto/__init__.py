"""
pycwb.modules.qveto — Data quality veto metrics.

Computes Qveto and Qfactor from reconstructed waveforms using
zero-crossing segment-maxima analysis and time-domain energy ratios
to identify glitch-like signals.
"""

from .qveto import get_qveto

__all__ = ["get_qveto"]
