"""
Data types for the online gravitational-wave search workflow.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OnlineSegment:
    """A segment of data ready for online analysis.

    Carries pre-read detector data together with GPS metadata and wall-clock
    timing.  The ``data_payload`` spans
    ``[segment_gps_start - seg_edge, segment_gps_end + seg_edge]`` so that
    analysis modules receive the same edge-padded window they expect from
    the file-based pipeline.
    """
    index: int
    """Monotonically increasing segment counter."""

    ifos: List[str]
    """Detector names, same order as ``config.ifo``."""

    segment_gps_start: float
    """GPS start of the analysis window (excluding edge padding)."""

    segment_gps_end: float
    """GPS end of the analysis window (excluding edge padding)."""

    seg_edge: float
    """Wavelet boundary padding (seconds)."""

    sample_rate: float
    """Native sample rate of the payload data."""

    data_payload: list
    """One TimeSeries per IFO, including segEdge padding on both sides."""

    wall_time_received: float
    """``time.time()`` when the snapshot was taken."""

    stride: float = 0.0
    """Seconds of NEW data in this segment (for sliding-window mode)."""

    overlap_frac: float = 0.0
    """Fraction of data shared with the previous segment."""


@dataclass
class OnlineTrigger:
    """A trigger produced by online analysis of one segment."""

    event: object
    """``Event`` instance with all sky-localisation and significance fields."""

    cluster: object
    """``Cluster`` from the likelihood stage."""

    sky_stats: object
    """Sky-localisation statistics."""

    segment_index: int
    """``OnlineSegment.index`` of the source segment."""

    segment_gps: float
    """GPS start of the analysis window."""

    wall_time_done: float
    """``time.time()`` when analysis completed."""
