"""Old-generation cycles must be reclaimed within the configured bound."""

import gc
import weakref
from types import SimpleNamespace
import psutil
from pycwb.workflow.subflow import job_segment_output as output


class Cycle:
    def __init__(self):
        self.self = self
        self.payload = bytearray(1024 * 1024)


def old_cycle():
    value = Cycle()
    gc.collect(2)  # Promote the still-reachable object to the oldest generation.
    return weakref.ref(value)


def test_old_cycles_reclaimed_at_interval(monkeypatch):
    monkeypatch.setenv("PYCWB_GC_FULL_INTERVAL", "3")
    monkeypatch.setattr(output, "_cleanup_count", 0)
    monkeypatch.setattr(output, "_last_full_collection_rss", psutil.Process().memory_info().rss)
    reference = old_cycle()
    output._cleanup_lag_output_state()
    output._cleanup_lag_output_state()
    assert reference() is not None
    output._cleanup_lag_output_state()
    assert reference() is None


def test_memory_growth_reclaims_old_cycles_early(monkeypatch):
    monkeypatch.setenv("PYCWB_GC_FULL_INTERVAL", "16")
    monkeypatch.setattr(output, "_cleanup_count", 0)
    reference = old_cycle()
    # RSS can fall between real samples when JAX worker threads release buffers.
    # Control the observed growth while testing collection of a real old cycle.
    monkeypatch.setattr(output, "_last_full_collection_rss", 0)
    monkeypatch.setattr(
        output.psutil, "Process", lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=128 * 1024**2 + 1))
    )
    output._cleanup_lag_output_state()
    assert reference() is None


def test_cleanup_keeps_live_objects_and_automatic_gc(monkeypatch):
    monkeypatch.setenv("PYCWB_GC_FULL_INTERVAL", "16")
    monkeypatch.setattr(output, "_last_full_collection_rss", None)
    value = Cycle()
    enabled = gc.isenabled()
    output._cleanup_lag_output_state()
    assert value.self is value and len(value.payload) == 1024 * 1024
    assert gc.isenabled() == enabled
