"""
pycwb.modules.plot — Visualization toolkit.

Provides spectrograms, 1D/2D histograms, event overlays, detector antenna
patterns, globe plots, fragment cluster visualization, and data quality
diagnostic plots.
"""

from . import event as event
from . import spectrogram as spectrogram
from .data_quality import plot_data_quality as plot_data_quality
from .detector_antenna import plot_detector_antenna_pattern as plot_detector_antenna_pattern
from .detector_antenna import plot_network_antenna_pattern as plot_network_antenna_pattern
from .detector_globe import plot_detector_on_globe as plot_detector_on_globe
from .event import plot_event_on_spectrogram as plot_event_on_spectrogram
from .fragment_cluster_viz import plot_fragment_clusters as plot_fragment_clusters
from .histograms import plot_1d_histogram as plot_1d_histogram
from .histograms import plot_2d_histogram as plot_2d_histogram
from .spectrogram import plot_spectrogram as plot_spectrogram

__all__ = [
    "event",
    "plot_data_quality",
    "plot_1d_histogram",
    "plot_2d_histogram",
    "plot_detector_antenna_pattern",
    "plot_detector_on_globe",
    "plot_event_on_spectrogram",
    "plot_fragment_clusters",
    "plot_network_antenna_pattern",
    "plot_spectrogram",
    "spectrogram",
]
