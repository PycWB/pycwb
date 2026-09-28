"""Guards, transport and parent routing of opt-in worker-side output."""

from __future__ import annotations

import pickle
from types import SimpleNamespace as NS

import pytest

from pycwb.modules.background_cuda import worker_output as worker
from pycwb.modules.background_cuda.output_buffer import OutputWriter

UNSUPPORTED_FLAGS = (
    "save_waveform",
    "save_cluster",
    "save_sky_map",
    "plot_waveform",
    "plot_trigger",
    "plot_sky_map",
)


@pytest.fixture
def context() -> NS:
    return NS(config=NS(), sub_job_seg=NS(injections=[]))


@pytest.fixture
def result() -> NS:
    return NS(lag=7, events_data=[(NS(injection=False), None, None)])


def test_enabled_uses_job_config():
    assert worker.enabled() is False
    assert worker.enabled(NS(gpu={"worker_output": True})) is True
    assert worker.enabled(NS(gpu={"worker_output": False})) is False


def test_validate_accepts_catalog_only_background(context: NS) -> None:
    worker.validate(context)
    context.config = NS(**{flag: False for flag in UNSUPPORTED_FLAGS})
    worker.validate(context)


@pytest.mark.parametrize("flag", UNSUPPORTED_FLAGS)
def test_validate_rejects_saved_products(context: NS, flag: str) -> None:
    context.config = NS(**{flag: True})
    with pytest.raises(ValueError, match="catalog-only background"):
        worker.validate(context)


@pytest.mark.parametrize("injections", [[object()], ("inj",), {"a": 1}])
def test_validate_rejects_injection_jobs(context: NS, injections: object) -> None:
    context.sub_job_seg = NS(injections=injections)
    with pytest.raises(ValueError, match="catalog-only background"):
        worker.validate(context)


def test_process_rejects_before_computation(
    context: NS, result: NS, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_: object) -> None:
        raise AssertionError("reconstruction must not run")

    monkeypatch.setattr(worker.native, "_postprocess_saved_triggers", explode)
    context.config = NS(save_waveform=True)
    with pytest.raises(ValueError):
        worker.process(context, result)
    context.config = NS()
    result.events_data[0][0].injection = True
    with pytest.raises(ValueError, match="cannot process injections"):
        worker.process(context, result)


def test_process_transport_and_parent_skips_reconstruction(
    context: NS, result: NS, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple] = []

    def compute(
        output: object, lag_result: object, names: list[str]
    ) -> tuple[float, float, float]:
        calls.append((output, lag_result, names))
        return (1.0, 2.0, 0.0)

    monkeypatch.setattr(worker.native, "_postprocess_saved_triggers", compute)
    processed = worker.process(context, result)
    assert len(calls) == 1
    output, lag_result, names = calls[0]
    assert lag_result is result
    assert names == [""]
    assert output.wave_file is None and output.queue is None
    assert output.config is context.config and output.sub_job_seg is context.sub_job_seg

    transported = pickle.loads(pickle.dumps(processed))
    assert isinstance(transported, worker.ProcessedLag)
    assert transported.lag == 7
    assert transported.timings == (1.0, 2.0, 0.0)

    writer = object.__new__(OutputWriter)
    writer.context, writer.sink = context, None
    from pycwb.constants.gpu_options import gpu_options

    writer.options = gpu_options(context.config)
    seen: dict[str, object] = {}

    def fake_specialize(function: object, **bindings: object) -> object:
        seen["function"] = function
        seen["bindings"] = bindings

        def save(ctx: object, res: object) -> None:
            seen["saved"] = (ctx, res)

        return save

    monkeypatch.delenv("PYCWB_GPU_PROFILE_LAGS", raising=False)
    monkeypatch.setattr("pycwb.utils.function_binding.specialize", fake_specialize)
    writer.save(None, transported)
    assert seen["function"] is worker.native._save_lag_outputs
    assert seen["saved"] == (context, transported.result)
    skip = seen["bindings"]["_postprocess_saved_triggers"]
    assert skip(None, None, None) == (1.0, 2.0, 0.0)


def test_process_failure_does_not_return_committable_result(
    context: NS, result: NS, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_: object) -> None:
        raise RuntimeError("reconstruction failed")

    monkeypatch.setattr(worker.native, "_postprocess_saved_triggers", fail)
    with pytest.raises(RuntimeError, match="reconstruction failed"):
        worker.process(context, result)
