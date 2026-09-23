"""Ownership, ordering and failure propagation of ``OverlappedSetup``."""

from __future__ import annotations

from contextlib import nullcontext
from threading import Event

import pytest

from pycwb.modules.background_cuda import setup_overlap
from pycwb.modules.background_cuda.setup_overlap import OverlappedSetup


@pytest.fixture(autouse=True)
def _cpu_only_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid touching a real JAX backend; the wrapper only needs a device scope."""
    monkeypatch.setattr(setup_overlap.jax, "devices", lambda *_: [object()])
    monkeypatch.setattr(setup_overlap.jax, "default_device", lambda _: nullcontext())


def test_overlap_and_trial_ownership() -> None:
    started, finished = Event(), Event()
    config, strains, cache, maps = object(), object(), object(), object()

    def td(c: object, s: object) -> object:
        assert c is config
        assert s is strains
        started.set()
        assert finished.wait(5), "coherence did not overlap TD"
        return cache

    def coherence(c: object, s: object, **kwargs: object) -> object:
        assert started.wait(5), "TD did not start concurrently"
        assert kwargs == {"nRMS": 3}
        finished.set()
        return maps

    setup = OverlappedSetup(coherence, td)
    for _ in range(2):
        started.clear()
        finished.clear()
        assert setup.setup_coherence(config, strains, nRMS=3) is maps
        assert setup.pending is not None
        with pytest.raises(RuntimeError, match="not consumed"):
            setup.setup_coherence(config, strains, nRMS=3)
        with pytest.raises(ValueError, match="different trial inputs"):
            setup.build_td_inputs_cache(config, object())
        with pytest.raises(ValueError, match="different trial inputs"):
            setup.build_td_inputs_cache(object(), strains)
        assert setup.build_td_inputs_cache(config, strains) is cache
        assert setup.pending is None
        with pytest.raises(RuntimeError, match="must precede"):
            setup.build_td_inputs_cache(config, strains)


def test_td_consumption_before_preparation_is_an_error() -> None:
    setup = OverlappedSetup(lambda *a, **k: object(), lambda *a: object())
    with pytest.raises(RuntimeError, match="Coherence preparation must precede"):
        setup.build_td_inputs_cache(object(), object())


@pytest.mark.parametrize("failing_stage", ["td", "coherence"])
def test_failure_joins_worker_and_does_not_publish_cache(failing_stage: str) -> None:
    td_done = Event()

    def td(*args: object) -> object:
        try:
            if failing_stage == "td":
                raise ArithmeticError("TD failed")
            return object()
        finally:
            td_done.set()

    def coherence(*args: object, **kwargs: object) -> object:
        if failing_stage == "coherence":
            raise ArithmeticError("Coherence failed")
        return object()

    setup = OverlappedSetup(coherence, td)
    with pytest.raises(ArithmeticError):
        setup.setup_coherence(object(), object())
    assert td_done.is_set()
    assert setup.pending is None


def test_td_runs_inside_cpu_device_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    entered: list[object] = []
    device = object()

    class Scope:
        def __init__(self, dev: object) -> None:
            self.dev = dev

        def __enter__(self) -> None:
            entered.append(self.dev)

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(setup_overlap.jax, "devices", lambda kind: [device] if kind == "cpu" else [])
    monkeypatch.setattr(setup_overlap.jax, "default_device", Scope)
    setup = OverlappedSetup(lambda *a, **k: "maps", lambda *a: "cache")
    assert setup.setup_coherence("cfg", "strains") == "maps"
    assert entered == [device]
    assert setup.build_td_inputs_cache("cfg", "strains") == "cache"
