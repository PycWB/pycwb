"""Exercise GPU workflow composition and native stage dispatch without a GPU."""

import ast
import os
import pickle
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pycwb.workflow.subflow import process_job_segment_gpu as gpu
from pycwb.workflow.subflow import process_job_segment_native as native
from pycwb.workflow.subflow.job_segment_stages import LagStages, PreparationStages


@pytest.fixture(autouse=True)
def clean_gpu_options(monkeypatch):
    for key in os.environ:
        if key.startswith("PYCWB_GPU_"):
            monkeypatch.delenv(key)


def test_scientific_modules_do_not_depend_on_workflow_or_legacy_package():
    root = Path(gpu.__file__).parents[2] / "modules"
    for package in ("coherence_gpu", "super_cluster_gpu", "likelihood_gpu"):
        for path in (root / package).glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                assert not any(n.startswith(("pycwb.workflow", "pycwb.modules.background_cuda")) for n in names), path


def test_spawn_entrypoints_and_output_types_remain_pickleable():
    from pycwb.workflow.subflow import process_job_segment_gpu_parallel as parallel

    for value in (parallel._initialize, parallel._analyze, gpu.process_job_segment, native.LagResult):
        assert pickle.loads(pickle.dumps(value)) is value


def test_analyzer_composes_stages_without_mutating_native(monkeypatch):
    calls = []
    selector = SimpleNamespace(sessions={})
    coherence, supercluster, likelihood = object(), object(), object()
    monkeypatch.setattr(gpu, "build_coherence", lambda config: (coherence, selector))
    monkeypatch.setattr(gpu, "build_supercluster", lambda config: supercluster)
    monkeypatch.setattr(gpu, "build_likelihood", lambda config: likelihood)
    monkeypatch.setattr(native, "_run_lag_analysis", lambda context, lag, *, stages: calls.append((context, lag, stages)))
    original = (native.coherence_single_lag, native.supercluster_single_lag, native.evaluate_cluster_likelihood)
    analyze, owner = gpu._build_analyzer()
    analyze("context", 2)
    assert owner is selector
    assert calls == [("context", 2, LagStages(coherence, supercluster, likelihood, native.Event))]
    assert original == (native.coherence_single_lag, native.supercluster_single_lag, native.evaluate_cluster_likelihood)


def test_default_factories_retain_native_numerics():
    from pycwb.modules.likelihood_gpu.likelihood import build_likelihood
    from pycwb.modules.super_cluster_gpu.super_cluster import build_supercluster

    assert build_likelihood() is native.evaluate_cluster_likelihood
    assert build_supercluster() is native.supercluster_single_lag


def test_parallel_preparation_and_overlap_are_composed(monkeypatch):
    from pycwb.modules.coherence_gpu.coherence import setup_coherence
    from pycwb.modules.data_conditioning.parallel import condition_strains
    from pycwb.modules.super_cluster_gpu.td_setup_parallel import build_td_inputs_cache

    config = SimpleNamespace(gpu=dict(condition_workers=2, setup_workers=2, td_setup_workers=2))
    stages = gpu._build_preparation(config)
    assert stages.condition_strains.func is condition_strains
    assert stages.condition_strains.keywords == {"workers": 2, "validate": False}
    assert stages.setup_coherence is setup_coherence
    assert stages.build_td_inputs_cache is build_td_inputs_cache
    config.gpu["overlap_setup"] = True
    overlapped = gpu._build_preparation(config)
    owner = overlapped.setup_coherence.__self__
    assert overlapped.build_td_inputs_cache.__self__ is owner
    assert owner.coherence is setup_coherence
    assert owner.td is build_td_inputs_cache


@pytest.mark.parametrize("failure", [False, True])
def test_serial_lags_honor_resume_release_maps_and_commit_only_on_success(monkeypatch, failure):
    from pycwb.workflow.subflow import gpu_output

    saved, closed = [], []
    selector = SimpleNamespace(sessions={1: object()})
    context = SimpleNamespace(config=SimpleNamespace(gpu={}), trial_idx=0, n_lag=3, sub_job_seg=SimpleNamespace(injections=None))
    output = object()

    def analyze(ctx, lag):
        assert ctx is context
        if failure and lag == 2:
            raise RuntimeError("analysis failed")
        return lag

    monkeypatch.setattr(gpu, "_build_analyzer", lambda config: (analyze, selector))
    monkeypatch.setattr(gpu_output, "OutputWriter", lambda ctx: SimpleNamespace(
        save=lambda ctx, result: saved.append((ctx, result)), close=lambda: closed.append(True),
    ))
    with pytest.raises(RuntimeError, match="analysis failed") if failure else nullcontext():
        gpu._process_lags(context, output, {0: {1}})
    assert saved == [(output, 0)] if failure else saved == [(output, 0), (output, 2)]
    assert closed == ([] if failure else [True])
    assert selector.sessions == {}


def test_native_lag_dispatches_explicit_stages_in_order(monkeypatch):
    calls = []
    cluster = SimpleNamespace(
        cluster_status=-1, cluster_id=1, pixel_arrays=[], start_time=0, stop_time=1,
        low_frequency=32, high_frequency=64,
    )
    event = SimpleNamespace(output_py=lambda *a, **k: calls.append("event"), long_id="event-id")
    stages = LagStages(
        lambda *a, **k: calls.append("coherence") or [cluster],
        lambda *a, **k: calls.append("supercluster") or SimpleNamespace(clusters=[cluster]),
        lambda *a, **k: (calls.append("likelihood") or cluster, "sky"),
        lambda: event,
    )
    segment = SimpleNamespace(index=4, ifos=["H1", "L1"], n_lag=1, lag_shifts=np.zeros((1, 2)),
                              shift=None, injections=None, livetime=lambda lag: 10)
    context = native.LagAnalysisContext(
        SimpleNamespace(nIFO=2), segment, segment, 0, 1, [], None, None, None, None, None, None,
    )
    # Native module globals must not run when explicit stages are supplied.
    def wrong_stage(*args, **kwargs):
        pytest.fail("used native stage instead of explicit stage")
    for name in ("coherence_single_lag", "supercluster_single_lag", "evaluate_cluster_likelihood", "Event"):
        monkeypatch.setattr(native, name, wrong_stage)
    result = native._run_lag_analysis(context, 0, stages=stages)
    assert calls == ["coherence", "supercluster", "likelihood", "event"]
    assert result.events_data == [(event, cluster, "sky")]
    assert result.progress_record["n_triggers"] == 1
    assert event.id == "event-id"


def test_native_preparation_passes_products_into_lag_pipeline(monkeypatch, tmp_path):
    """Run the real job lifecycle with cheap stages and assert every handoff."""
    from pycwb.modules.conditioning_plugins import api

    calls = []
    raw, conditioned, noise, coherence, td = [object()], [SimpleNamespace(start_time=0)], [object()], [], {}
    config = SimpleNamespace(nIFO=1, MRAcatalog="unused")
    segment = SimpleNamespace(frames=["frame"], noise=None, injections=None, ifos=["H1"],
                              n_lag=1, duration=10, index=1, veto_windows=None,
                              padded_duration=12, seg_edge=1)
    provider = object()

    def read(cfg, seg, *, input_provider):
        assert (cfg, seg, input_provider) == (config, segment, provider)
        calls.append("read")
        return raw

    def condition(cfg, data):
        assert data == raw
        calls.append("condition")
        return conditioned, noise

    def setup(cfg, strains, **kwargs):
        assert strains is conditioned and kwargs == {"job_seg": segment, "nRMS": noise}
        calls.append("coherence setup")
        return coherence

    def setup_td(cfg, strains):
        assert strains is conditioned
        calls.append("td setup")
        return td

    def lags(ctx, out, *, skip_lags):
        assert ctx.coherence_setup is coherence and ctx.td_inputs_cache is td and ctx.nRMS is noise
        assert out.sub_job_seg is segment and skip_lags == {0: set()}
        calls.append("lags")

    for name in ("print_job_info", "print_node_info", "release_memory"):
        monkeypatch.setattr(native, name, lambda *a, **k: None)
    monkeypatch.setattr(native, "check_and_resample_py", lambda data, *args: data)
    monkeypatch.setattr(api, "run_hooks", lambda *a: SimpleNamespace(
        strains=conditioned, noise_rms=noise, diagnostics=None, excluded_intervals=[],
    ))
    monkeypatch.setattr(native, "XTalk", SimpleNamespace(load=lambda path: None))
    monkeypatch.setattr(native, "setup_supercluster", lambda *args: dict(ml=None, FP=None, FX=None))
    monkeypatch.setattr(native, "prepare_likelihood_inputs", lambda *args, **kwargs: None)
    native.process_job_segment(str(tmp_path), config, segment, skip_lags={0: set()},
                               input_provider=provider, lag_processor=lags,
                               preparation_stages=PreparationStages(read, condition, setup, setup_td))
    assert calls == ["read", "condition", "coherence setup", "td setup", "lags"]


@pytest.mark.parametrize("argument", ["lag_processor", "preparation_stages"])
def test_gpu_workflow_rejects_conflicting_ownership(monkeypatch, argument):
    monkeypatch.setattr(gpu, "jax", SimpleNamespace(config=SimpleNamespace(x64_enabled=True), devices=lambda kind: [kind]))
    with pytest.raises(ValueError, match="owns"):
        gpu.process_job_segment(".", SimpleNamespace(gpu={}), object(), **{argument: None})
