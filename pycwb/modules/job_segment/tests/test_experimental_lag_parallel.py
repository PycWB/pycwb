"""Contracts for the opt-in experimental executors."""

from concurrent.futures import ThreadPoolExecutor
from pycwb.config.processing import execution_profile
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from pycwb.workflow.subflow import process_job_segment_parallel as parallel
from pycwb.workflow.subflow.process_job_segment_native import LagAnalysisContext


def context(workers=2, injections=None):
    return LagAnalysisContext(
        config=SimpleNamespace(
            parallel_lag_workers=workers, parallel_lag_inner_threads=1
        ),
        job_seg=None,
        sub_job_seg=SimpleNamespace(injections=injections),
        trial_idx=0,
        n_lag=4,
        coherence_setup=np.arange(16.0),
        supercluster_setup=None,
        td_inputs_cache=None,
        xtalk=None,
        likelihood_setup=None,
        nRMS=None,
        veto_windows=None,
    )


def test_copy_on_write_inputs_cannot_modify_shared_file(tmp_path, monkeypatch):
    path = tmp_path / "context.joblib"
    joblib.dump(context(), path)
    monkeypatch.setattr(parallel, "_worker_context", None)
    monkeypatch.setattr(parallel.logging, "basicConfig", lambda **kwargs: None)
    parallel._initialize_process(path, 1, tmp_path)
    array = parallel._worker_context.coherence_setup
    assert isinstance(array, np.memmap)
    array[0] = 999
    assert joblib.load(path, mmap_mode="r").coherence_setup[0] == 0


def test_completed_lags_skip_executor_and_setup(monkeypatch):
    monkeypatch.setattr(
        parallel,
        "ProcessPoolExecutor",
        lambda **kwargs: pytest.fail("unexpected executor"),
    )
    parallel.process_lags(context(), None, {0: {0, 1, 2, 3}})


def test_injections_fall_back_to_serial_and_honor_resume(monkeypatch):
    seen = []
    saved = []
    monkeypatch.setattr(
        parallel.native, "_run_lag_analysis", lambda ctx, lag: seen.append(lag) or lag
    )
    monkeypatch.setattr(
        parallel.native, "_save_lag_outputs", lambda out, result: saved.append(result)
    )
    parallel.process_lags(context(injections=[{"trial_idx": 0}]), None, {0: {1, 3}})
    assert seen == saved == [0, 2]


def test_bounded_writer_records_each_success_once(monkeypatch):
    saved = []
    monkeypatch.setattr(
        parallel.native,
        "_save_lag_outputs",
        lambda out, result: saved.append(result.lag),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        parallel._consume_bounded(
            executor, lambda lag: SimpleNamespace(lag=lag), range(20), None, 2
        )
    assert sorted(saved) == list(range(20))


def test_worker_failure_propagates_without_committing_failed_lag(monkeypatch):
    saved = []
    monkeypatch.setattr(
        parallel.native,
        "_save_lag_outputs",
        lambda out, result: saved.append(result.lag),
    )

    def analyze(lag):
        if lag == 0:
            raise RuntimeError("deliberate analysis failure")
        return SimpleNamespace(lag=lag)

    with (
        ThreadPoolExecutor(max_workers=2) as executor,
        pytest.raises(RuntimeError, match="deliberate"),
    ):
        parallel._consume_bounded(executor, analyze, range(20), None, 2)
    assert 0 not in saved


def test_processor_selection_passes_explicit_hook(monkeypatch):
    seen = []
    monkeypatch.setattr(
        parallel.native,
        "process_job_segment",
        lambda *args, **kwargs: seen.append(kwargs["lag_processor"]),
    )
    parallel.process_job_segment("run")
    assert seen == [parallel.process_lags]


def test_shared_input_scratch_removed_after_worker_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        parallel,
        "ProcessPoolExecutor",
        lambda **kwargs: ThreadPoolExecutor(max_workers=kwargs["max_workers"]),
    )

    def fail(lag):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(parallel, "_analyze_process", fail)
    with pytest.raises(RuntimeError, match="worker failed"):
        parallel.process_lags(
            context(), SimpleNamespace(working_dir=str(tmp_path)), None
        )
    assert not list(tmp_path.glob(".lag-inputs-*"))


def test_process_worker_collects_its_own_heap_without_jax(monkeypatch):
    seen = []
    monkeypatch.setattr(parallel, "_worker_context", context())
    monkeypatch.setattr(parallel.native, "_run_lag_analysis", lambda ctx, lag: lag)
    monkeypatch.setattr(
        parallel.native,
        "_cleanup_lag_output_state",
        lambda **kwargs: seen.append(kwargs),
    )
    assert parallel._analyze_process(3) == 3
    assert seen == [{"release_jax": False, "profile": execution_profile(context().config)}]


def test_analysis_only_gc_does_not_initialize_jax(monkeypatch):
    from pycwb.workflow.subflow import job_segment_output as output

    calls = []
    monkeypatch.setenv("PYCWB_GC_FULL_INTERVAL", "1")
    monkeypatch.setattr(output.gc, "collect", lambda *args: calls.append("gc"))
    monkeypatch.setattr(output, "_free_jax_buffers", lambda: calls.append("jax"))
    output._cleanup_lag_output_state(release_jax=False)
    assert calls == ["gc"]
    output._cleanup_lag_output_state()
    assert calls == ["gc", "gc", "jax"]


def _inspect_spawned_input(lag):
    """Exercise the real spawn initializer without invoking scientific kernels."""
    import os

    array = parallel._worker_context.coherence_setup
    before = float(array[0])
    array[0] = lag
    return os.getpid(), isinstance(array, np.memmap), before, float(array[0])


def test_spawned_worker_maps_inputs_once_and_isolates_writes(tmp_path):
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    path = tmp_path / "context.joblib"
    joblib.dump(context(), path, compress=0)
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=parallel._initialize_process,
        initargs=(str(path), 1, str(tmp_path)),
    ) as executor:
        first = executor.submit(_inspect_spawned_input, 7).result(timeout=60)
        second = executor.submit(_inspect_spawned_input, 9).result(timeout=60)
    assert first[0] == second[0] != os.getpid()
    assert first[1:] == (True, 0.0, 7.0)
    assert second[1:] == (True, 7.0, 9.0)
    assert joblib.load(path, mmap_mode="r").coherence_setup[0] == 0


def test_mismatched_result_is_not_saved(monkeypatch):
    monkeypatch.setattr(
        parallel.native,
        "_save_lag_outputs",
        lambda *args: pytest.fail("mismatched lag must not be saved"),
    )
    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        pytest.raises(RuntimeError, match="returned 99, expected 0"),
    ):
        parallel._consume_bounded(
            executor, lambda lag: SimpleNamespace(lag=99), [0], None, 1
        )


def test_worker_cleanup_also_runs_on_analysis_failure(monkeypatch):
    cleaned = []
    monkeypatch.setattr(parallel, "_worker_context", context())

    def fail(*args):
        raise ValueError("analysis failed")

    monkeypatch.setattr(parallel.native, "_run_lag_analysis", fail)
    monkeypatch.setattr(
        parallel.native,
        "_cleanup_lag_output_state",
        lambda **kwargs: cleaned.append(kwargs),
    )
    with pytest.raises(ValueError, match="analysis failed"):
        parallel._analyze_process(0)
    assert cleaned == [{"release_jax": False, "profile": execution_profile(context().config)}]


@pytest.mark.parametrize(
    "workers, skipped, expected", [(1, {1, 3}, [0, 2]), (6, {0, 1, 3}, [2])]
)
def test_single_effective_worker_avoids_serialization(
    workers, skipped, expected, monkeypatch
):
    saved = []
    monkeypatch.setattr(
        parallel.joblib,
        "dump",
        lambda *args, **kwargs: pytest.fail("unexpected input file"),
    )
    monkeypatch.setattr(
        parallel.native, "_run_lag_analysis", lambda ctx, lag: SimpleNamespace(lag=lag)
    )
    monkeypatch.setattr(
        parallel.native,
        "_save_lag_outputs",
        lambda out, result: saved.append(result.lag),
    )
    parallel.process_lags(context(workers=workers), None, {0: skipped})
    assert saved == expected


def test_bounded_scheduler_stops_submission_on_output_failure(monkeypatch):
    from concurrent.futures import Future

    submitted = []

    class ControlledExecutor:
        def submit(self, analyze, lag):
            future = Future()
            if lag == 0:
                future.set_result(SimpleNamespace(lag=lag))
            submitted.append(future)
            return future

    def fail_to_save(*args):
        raise OSError("output unavailable")

    monkeypatch.setattr(parallel.native, "_save_lag_outputs", fail_to_save)
    with pytest.raises(OSError, match="output unavailable"):
        parallel._consume_bounded(ControlledExecutor(), None, range(100), None, 2)
    assert 1 < len(submitted) <= 4
    assert all(future.cancelled() for future in submitted[1:])


@pytest.mark.parametrize("reuse_delays", [False, True])
def test_nogil_release_scan_preserves_cpu_results(reuse_delays):
    import numpy as np
    from pycwb.workflow.subflow import process_job_segment_nogil as nogil
    from pycwb.modules.likelihoodWP.sky_scan import scan_sky

    rng = np.random.default_rng(623)
    geometry = (rng.normal(size=(16, 2)).astype(np.float32),
                rng.normal(size=(16, 2)).astype(np.float32),
                rng.integers(-2, 3, size=(2, 16), dtype=np.int32))
    cluster = (rng.uniform(.1, 1, size=(11, 2)).astype(np.float32),
               rng.normal(size=(9, 2, 11)).astype(np.float32),
               rng.normal(size=(9, 2, 11)).astype(np.float32))
    settings = (np.array([.1, 2., 0.], np.float32), -.5, .1, .5, np.arange(16))
    expected = scan_sky(geometry, cluster, settings, reuse_delays=reuse_delays)
    actual = nogil._likelihood.__globals__["_scan_sky"](
        geometry, cluster, settings, reuse_delays=reuse_delays)
    for a, b in zip(actual, expected, strict=True):
        assert np.asarray(a).dtype == np.asarray(b).dtype
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
