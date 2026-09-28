"""Check GPU processor input dispatch without creating a CUDA context."""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from pycwb.workflow.execution.cache import FrameProvider


@pytest.mark.parametrize("use_cache", [False, True])
def test_gpu_processor_preserves_direct_reads_or_accepts_provider(
    monkeypatch, use_cache
):
    from pycwb.modules.background_cuda import processor

    for key in list(os.environ):
        if key.startswith("PYCWB_GPU_"):
            monkeypatch.delenv(key)
    config = SimpleNamespace(gpu={"read_workers": 2})
    monkeypatch.setattr(
        processor,
        "jax",
        SimpleNamespace(
            config=SimpleNamespace(x64_enabled=True),
            devices=lambda backend: [backend],
            default_device=lambda device: nullcontext(),
        ),
    )
    monkeypatch.setattr(
        processor.native, "process_job_segment", lambda *args, **kwargs: kwargs
    )
    bindings = []

    def specialize(function, **values):
        bindings.append(values)
        return function

    monkeypatch.setattr(processor, "specialize", specialize)
    provider = FrameProvider(()) if use_cache else None
    result = processor.process_job_segment(".", config, object(), input_provider=provider)
    assert processor.process_job_segment.supports_input_provider
    assert result["input_provider"] is provider
    assert result["lag_processor"] is processor._process_lags
    assert bool(bindings) is not use_cache
    if bindings:
        assert set(bindings[0]) == {"read_from_job_segment"}


def test_gpu_processor_declares_explicit_worker_cpu_budget():
    from pycwb.modules.background_cuda.processor import process_job_segment

    config = SimpleNamespace(gpu={"lag_workers": 6})
    assert process_job_segment.requested_cores(config) == 6
    config.gpu = {"overlap_setup": True, "setup_workers": 3, "td_setup_workers": 2}
    assert process_job_segment.requested_cores(config) == 5


def test_executor_reserves_gpu_lag_workers(monkeypatch):
    from pycwb.workflow.execution import executor
    from pycwb.modules.background_cuda.processor import process_job_segment

    monkeypatch.setattr(executor, "available_cpus", lambda: [0, 1, 2, 3])
    config = SimpleNamespace(gpu={"lag_workers": 6}, execution={"profile": "scalable"})
    context = executor.ExecutionContext([], config, process_job_segment, ".", "catalog")
    with pytest.raises(ValueError, match="6 cores but allocation has 4"):
        executor.ScalableExecutor().execute(SimpleNamespace(requests=[]), context)
