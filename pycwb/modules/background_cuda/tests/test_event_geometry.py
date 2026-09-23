"""Cached detector geometry keeps native values, ownership and the native class."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from pycwb.modules.background_cuda import event_geometry
from pycwb.modules.background_cuda.event_geometry import CachedGeometryEvent, _geometry, cached_detector
from pycwb.modules.background_cuda.validation import leaves
from pycwb.types.detector import Detector
from pycwb.types.network_event import Event


@pytest.mark.parametrize("model", ["lal", "cwb_6.4.6.9"])
@pytest.mark.parametrize("name", ["H1", "L1"])
def test_cached_detector_matches_native_and_owns_its_arrays(model: str, name: str) -> None:
    expected = leaves(Detector(name, geometry_model=model))
    actual = cached_detector(name, geometry_model=model)
    assert leaves(actual) == expected
    actual.response[:] = np.nan
    actual.vertex_vec_earth_centered[:] = np.nan
    actual.name = "mutated"
    # The cache hands out independent deep copies, so mutation is invisible.
    assert leaves(cached_detector(name, geometry_model=model)) == expected


def test_cache_is_bounded_and_keyed_by_name_and_model() -> None:
    _geometry.cache_clear()
    for model in ("lal", "cwb_6.4.6.9"):
        for name in ("H1", "L1"):
            cached_detector(name, geometry_model=model)
            cached_detector(name, geometry_model=model)
    info = _geometry.cache_info()
    assert info.maxsize == 8
    assert info.currsize == 4
    assert info.misses == 4
    assert info.hits == 4


def test_native_event_class_is_unchanged() -> None:
    assert "output_py" in Event.__dict__
    assert CachedGeometryEvent.output_py is not Event.output_py
    # The native method keeps its local imports; only the private clone binds the cache.
    assert "Detector" not in Event.output_py.__globals__
    assert event_geometry._output.__globals__["Detector"] is cached_detector
    assert event_geometry._output.__code__.co_filename == Event.output_py.__code__.co_filename
    assert issubclass(CachedGeometryEvent, Event)
    assert Event.output_py.__code__.co_filename.endswith("network_event.py")


def test_cached_event_pickles_to_equivalent_event() -> None:
    event = CachedGeometryEvent()
    event.job_id = 4
    event.qveto = 0.3125
    result = pickle.loads(pickle.dumps(event))
    assert isinstance(result, Event)
    assert isinstance(result, CachedGeometryEvent)
    assert leaves(result) == leaves(event)
