"""
pycwb.modules.autoencoder — Neural-network glitch classifier.

Computes a "glitchness" score — a per-cluster metric indicating how
glitch-like the reconstructed waveform is — to help reject
non-astrophysical triggers.
"""

from .autoencoder import get_glitchness