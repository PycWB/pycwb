"""
pycwb.modules.online — Streaming GW search pipeline.

Full online/streaming analysis pipeline for real-time GW burst detection.
Components include: ``DataSource`` (abstract data adapter),
``DataAcquisitionManager`` (polling daemon), ``RingBuffer`` (thread-safe
per-IFO buffer), ``TriggerHandler`` (deduplication + significance +
GraceDB alerts), ``BackgroundManager``, and ``LatencyMonitor``.
"""

from .data_source import (
    DataSource,
    NDS2DataSource,
    KafkaDataSource,
    SharedMemoryDataSource,
    create_data_source,
)
from .data_acquisition import DataAcquisitionManager
from .trigger_handler import TriggerHandler
from .background import BackgroundManager
from .deduplication import TriggerDeduplicator
from .latency_monitor import LatencyMonitor
from .ring_buffer import RingBuffer
from .significance import load_significance_model, assign_significance

__all__ = [
    "DataSource",
    "NDS2DataSource",
    "KafkaDataSource",
    "SharedMemoryDataSource",
    "create_data_source",
    "DataAcquisitionManager",
    "TriggerHandler",
    "BackgroundManager",
    "TriggerDeduplicator",
    "LatencyMonitor",
    "RingBuffer",
    "load_significance_model",
    "assign_significance",
]
