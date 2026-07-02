import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from pycwb.config import Config
from pycwb.constants.user_parameters_schema import schema
from pycwb.modules.job_segment.job_segment import flatten_job_segments_by_trial
from pycwb.types.job import WaveSegment


def _not_called(*args, **kwargs):
    raise AssertionError("heavy native dependency stub should not be called")


def _stub_module(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class DummyXTalk:
    @classmethod
    def load(cls, *args, **kwargs):
        return cls()


_STUBBED_MODULES = [
    "psutil",
    "pycwb.modules.super_cluster_native",
    "pycwb.modules.super_cluster_native.super_cluster",
    "pycwb.utils.td_vector_batch",
    "pycwb.modules.xtalk.type",
    "pycwb.modules.coherence_native.coherence",
    "pycwb.modules.read_data",
    "pycwb.modules.read_data.data_check",
    "pycwb.modules.data_conditioning",
    "pycwb.modules.data_conditioning.data_conditioning",
    "pycwb.modules.cwb_interop",
    "pycwb.modules.likelihoodWP",
    "pycwb.modules.likelihoodWP.likelihood",
    "pycwb.modules.qveto",
    "pycwb.modules.qveto.qveto",
    "pycwb.modules.reconstruction",
    "pycwb.modules.workflow_utils",
    "pycwb.modules.workflow_utils.job_setup",
    "pycwb.utils.memory",
    "pycwb.workflow.subflow.postprocess_and_plots",
]
_MISSING = object()
_ORIGINAL_MODULES = {
    name: sys.modules.get(name, _MISSING)
    for name in _STUBBED_MODULES
}
_ORIGINAL_PARENT_ATTRS = {}
for _name in _STUBBED_MODULES:
    if "." not in _name:
        continue
    _parent_name, _attr = _name.rsplit(".", 1)
    _parent = sys.modules.get(_parent_name, _MISSING)
    _value = _MISSING if _parent is _MISSING else getattr(_parent, _attr, _MISSING)
    _ORIGINAL_PARENT_ATTRS[(_parent_name, _attr)] = (_parent, _value)

_stub_module(
    "psutil",
    Process=lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=0),
    ),
)
for package_name in [
    "pycwb.modules.super_cluster_native",
    "pycwb.modules.read_data",
    "pycwb.modules.data_conditioning",
    "pycwb.modules.likelihoodWP",
    "pycwb.modules.qveto",
    "pycwb.modules.workflow_utils",
]:
    package = _stub_module(package_name)
    package.__path__ = []

_stub_module(
    "pycwb.modules.super_cluster_native.super_cluster",
    setup_supercluster=_not_called,
    supercluster_single_lag=_not_called,
)
_stub_module("pycwb.utils.td_vector_batch", build_td_inputs_cache=_not_called)
_stub_module("pycwb.modules.xtalk.type", XTalk=DummyXTalk)
_stub_module(
    "pycwb.modules.coherence_native.coherence",
    setup_coherence=_not_called,
    coherence_single_lag=_not_called,
)
_stub_module(
    "pycwb.modules.read_data",
    generate_strain_from_injection=_not_called,
    generate_noise_for_job_seg=_not_called,
    read_from_job_segment=_not_called,
)
_stub_module("pycwb.modules.read_data.data_check", check_and_resample_py=_not_called)
_stub_module("pycwb.modules.data_conditioning.data_conditioning", data_conditioning=_not_called)
_stub_module("pycwb.modules.cwb_interop", create_cwb_workdir=_not_called)
_stub_module(
    "pycwb.modules.likelihoodWP.likelihood",
    likelihood=_not_called,
    setup_likelihood=_not_called,
)
_stub_module("pycwb.modules.qveto.qveto", get_qveto=_not_called)
_stub_module("pycwb.modules.reconstruction", estimate_snr=_not_called)
_stub_module(
    "pycwb.modules.workflow_utils.job_setup",
    print_job_info=lambda *args, **kwargs: None,
    print_node_info=lambda *args, **kwargs: None,
)
sys.modules["pycwb.modules.workflow_utils"].create_single_trigger_folder = _not_called
sys.modules["pycwb.modules.workflow_utils"].save_trigger = _not_called
_stub_module("pycwb.utils.memory", release_memory=lambda *args, **kwargs: None)
_stub_module(
    "pycwb.workflow.subflow.postprocess_and_plots",
    plot_trigger_flow=_not_called,
    reconstruct_waveforms_flow=_not_called,
    reconstruct_INJwaveforms_flow=_not_called,
    plot_skymap_flow=_not_called,
)


def _restore_stubbed_modules():
    for name, module in _ORIGINAL_MODULES.items():
        if module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
    for (parent_name, attr), (original_parent, original_value) in _ORIGINAL_PARENT_ATTRS.items():
        parent = sys.modules.get(parent_name)
        if parent is None:
            continue
        if original_parent is _MISSING or original_value is _MISSING:
            if hasattr(parent, attr):
                delattr(parent, attr)
        elif parent is original_parent:
            setattr(parent, attr, original_value)


def _load_native_under_test():
    module_path = Path(__file__).parents[4] / "pycwb/workflow/subflow/process_job_segment_native.py"
    spec = importlib.util.spec_from_file_location("_native_low_cost_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
        _restore_stubbed_modules()
    return module


native = _load_native_under_test()


class ListQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def _config(workers=1, inner_threads=None, nproc=1):
    return SimpleNamespace(
        parallel_lag_workers=workers,
        parallel_lag_inner_threads=inner_threads,
        nproc=nproc,
    )


def _analysis_context(*, workers=1, n_lag=1, sub_job_seg=None, config=None, veto_windows=None):
    return native.LagAnalysisContext(
        config=config or _config(workers=workers, nproc=4),
        job_seg=SimpleNamespace(index=10),
        sub_job_seg=sub_job_seg or SimpleNamespace(injections=None),
        trial_idx=0,
        n_lag=n_lag,
        coherence_setup=None,
        supercluster_setup=None,
        td_inputs_cache=None,
        xtalk=None,
        likelihood_setup=None,
        nRMS=None,
        veto_windows=veto_windows,
    )


def _output_context(queue=None):
    return native.LagOutputContext(
        working_dir=".",
        config=SimpleNamespace(),
        sub_job_seg=SimpleNamespace(ifos=["H1", "L1"]),
        catalog_file=None,
        wave_file=None,
        queue=queue,
        HoT_list=None,
        mdc_maps=None,
    )


def _raise(error):
    raise error


def _run_fake_lag_loop(monkeypatch, *, workers, n_lag=4, skip_lags=None):
    compute_calls = []
    persisted = []
    numba_threads_seen = []

    def fake_analysis(context, lag):
        compute_calls.append(lag)
        numba_threads_seen.append(context.numba_threads)
        return native.LagResult(
            lag=lag,
            lag_timer=0.0,
            time_lag=[],
            segment_lag=[],
            events_data=[],
            progress_record={
                "job_id": 10,
                "trial_idx": context.trial_idx,
                "lag_idx": lag,
                "n_triggers": 0,
                "livetime": 100.0,
                "status": "completed",
            },
        )

    def fake_save(output_context, result):
        persisted.append(result.progress_record)

    monkeypatch.setattr(native, "_run_lag_analysis", fake_analysis)
    monkeypatch.setattr(native, "_save_lag_outputs", fake_save)

    native._process_lags(
        _analysis_context(workers=workers, n_lag=n_lag),
        _output_context(),
        skip_lags=skip_lags,
    )
    return compute_calls, persisted, numba_threads_seen


def test_parallel_lag_config_defaults_are_registered():
    cfg = Config()
    assert cfg.parallel_lag_workers == 1
    assert cfg.parallel_lag_inner_threads is None
    assert schema["properties"]["parallel_lag_workers"]["default"] == 1
    assert schema["properties"]["parallel_lag_inner_threads"]["default"] is None


def test_native_workflow_import_contract_points_to_support_helpers():
    from pycwb.workflow.subflow import job_segment_output
    from pycwb.workflow.subflow import job_segment_progress
    from pycwb.workflow.subflow import job_segment_resources
    from pycwb.workflow.subflow import job_segment_veto

    default_processor = schema["properties"]["segment_processer"]["default"]
    assert default_processor == "pycwb.workflow.subflow.process_job_segment_native.process_job_segment"
    assert getattr(native, default_processor.rsplit(".", 1)[1]) is native.process_job_segment

    assert native._catalog_path is job_segment_progress._catalog_path
    assert native._record_lag_progress is job_segment_progress._record_lag_progress
    assert native._effective_veto_windows is job_segment_veto._effective_veto_windows
    assert native._lag_livetime is job_segment_veto._lag_livetime
    assert native._parallel_inner_threads is job_segment_resources._parallel_inner_threads
    assert native._free_jax_buffers is job_segment_resources._free_jax_buffers
    assert native._create_and_save_trigger_folders is job_segment_output._create_and_save_trigger_folders


def test_progress_helpers_build_metadata_and_route_queue_or_catalog(monkeypatch):
    config = SimpleNamespace(catalog_dir="catalog")
    sub_job_seg = SimpleNamespace(
        ifos=["H1", "L1"],
        lag_shifts=np.asarray([[1.5, -2.0]]),
        shift=np.asarray([0.25, -0.5]),
    )
    time_lag, segment_lag, lag_shifts = native._lag_metadata(sub_job_seg, 0)
    assert time_lag == [1.5, -2.0]
    assert segment_lag == [0.25, -0.5]
    assert lag_shifts.tolist() == [1.5, -2.0]
    assert native._catalog_path("/work", config, "catalog_1.h5") == "/work/catalog/catalog_1.h5"

    queue = ListQueue()
    record = {"job_id": 1, "trial_idx": 0, "lag_idx": 2, "n_triggers": 0}
    native._record_lag_progress("/work", config, "catalog_1.h5", queue, record)
    assert queue.items == [{"type": "progress", **record}]

    catalog_rows = []

    class FakeCatalog:
        @classmethod
        def open(cls, path):
            assert path == "/work/catalog/catalog_1.h5"
            return cls()

        def add_lag_progress(self, **kwargs):
            catalog_rows.append(kwargs)

    catalog_pkg = ModuleType("pycwb.modules.catalog")
    catalog_pkg.__path__ = []
    catalog_module = ModuleType("pycwb.modules.catalog.catalog")
    catalog_module.Catalog = FakeCatalog
    monkeypatch.setitem(sys.modules, "pycwb.modules.catalog", catalog_pkg)
    monkeypatch.setitem(sys.modules, "pycwb.modules.catalog.catalog", catalog_module)

    native._record_lag_progress("/work", config, "catalog_1.h5", None, record)
    assert catalog_rows == [record]


def test_veto_helpers_preserve_cat2_injection_and_livetime_semantics():
    config = SimpleNamespace(analyze_injection_only=True, injection_padding=1.0)
    sub_job_seg = SimpleNamespace(
        cwb_veto_windows=[(10.0, 20.0)],
        veto_windows=[(0.0, 30.0)],
        injections=[{"real_start": 12.0, "real_end": 15.0}],
        duration=100.0,
        circular_livetime=lambda lag, veto_windows=None: 88.0,
    )

    assert native._effective_veto_windows(config, sub_job_seg) == [(10.0, 20.0)]
    assert native._injection_aware_veto_windows(
        config,
        sub_job_seg,
        [(0.0, 13.0), (14.0, 30.0)],
    ) == [(11.0, 13.0), (14.0, 16.0)]
    assert native._lag_livetime(SimpleNamespace(sub_job_seg=sub_job_seg, veto_windows=[]), 0) == 88.0


def test_resource_helpers_compute_inner_threads_and_restore_numba(monkeypatch):
    assert native._parallel_inner_threads(SimpleNamespace(parallel_lag_inner_threads=5), 2) == 5
    assert native._parallel_inner_threads(
        SimpleNamespace(parallel_lag_inner_threads=None, nproc=7),
        3,
    ) == 2
    assert native._parallel_inner_threads(
        SimpleNamespace(parallel_lag_inner_threads=None, nproc=0),
        3,
    ) == 1

    state = {"threads": 8}

    def set_num_threads(value):
        state["threads"] = value

    fake_numba = SimpleNamespace(
        config=SimpleNamespace(NUMBA_NUM_THREADS=8),
        get_num_threads=lambda: state["threads"],
        set_num_threads=set_num_threads,
    )
    monkeypatch.setitem(sys.modules, "numba", fake_numba)

    with native._temporary_numba_threads(3):
        assert state["threads"] == 3
    assert state["threads"] == 8


def test_serial_and_threaded_background_lag_paths_persist_same_progress(monkeypatch):
    _, serial_progress, _ = _run_fake_lag_loop(monkeypatch, workers=1)
    _, threaded_progress, threaded_numba_threads = _run_fake_lag_loop(monkeypatch, workers=2)

    def key(row):
        return row["job_id"], row["trial_idx"], row["lag_idx"]

    assert sorted(serial_progress, key=key) == sorted(threaded_progress, key=key)
    assert set(threaded_numba_threads) == {2}


def test_threaded_lag_path_does_not_submit_skipped_lags(monkeypatch):
    compute_calls, persisted, _ = _run_fake_lag_loop(
        monkeypatch,
        workers=2,
        n_lag=5,
        skip_lags={0: {1, 3}},
    )

    assert sorted(compute_calls) == [0, 2, 4]
    assert sorted(row["lag_idx"] for row in persisted) == [0, 2, 4]


def test_segthr_skip_returns_progress_without_running_heavy_compute():
    sub_job_seg = SimpleNamespace(
        index=5,
        ifos=["H1", "L1"],
        n_lag=1,
        lag_shifts=np.asarray([[0.0, 0.0]]),
        shift=None,
        duration=100.0,
        livetime=lambda lag: 50.0,
        circular_livetime=lambda lag, veto_windows=None: 2.0,
    )
    context = _analysis_context(
        config=SimpleNamespace(segTHR=10.0),
        sub_job_seg=sub_job_seg,
        veto_windows=[(0.0, 2.0)],
    )
    result = native._run_lag_analysis(
        context,
        lag=0,
    )

    assert result.events_data == []
    assert result.progress_record["status"] == "skipped_segTHR"
    assert result.progress_record["livetime"] == 0.0


def test_completed_lag_progress_uses_circular_livetime(monkeypatch):
    sub_job_seg = SimpleNamespace(
        index=5,
        ifos=["H1", "L1"],
        n_lag=1,
        lag_shifts=np.asarray([[0.0, 0.0]]),
        shift=None,
        duration=100.0,
        livetime=lambda lag: 3.0,
        circular_livetime=lambda lag, veto_windows=None: 77.0,
    )
    context = _analysis_context(
        config=SimpleNamespace(segTHR=0.0),
        sub_job_seg=sub_job_seg,
        veto_windows=[(0.0, 2.0)],
    )
    monkeypatch.setattr(native, "coherence_single_lag", lambda *args, **kwargs: [])
    monkeypatch.setattr(native, "supercluster_single_lag", lambda *args, **kwargs: None)

    result = native._run_lag_analysis(context, lag=0)

    assert result.progress_record["status"] == "completed"
    assert result.progress_record["livetime"] == 77.0


def test_zero_trigger_lag_progress_is_recorded_by_persist_helper():
    queue = ListQueue()
    result = native.LagResult(
        lag=7,
        lag_timer=0.0,
        time_lag=[0.0, 1.0],
        segment_lag=[0.0, 0.0],
        events_data=[],
        progress_record={
            "job_id": 3,
            "trial_idx": 0,
            "lag_idx": 7,
            "n_triggers": 0,
            "livetime": 100.0,
            "status": "completed",
        },
    )

    native._save_lag_outputs(
        _output_context(queue=queue),
        result,
    )

    assert queue.items == [{"type": "progress", **result.progress_record}]


def test_output_helper_writes_trigger_queue_and_preserves_lag_metadata(monkeypatch):
    from pycwb.workflow.subflow import job_segment_output

    queue = ListQueue()
    trigger_obj = SimpleNamespace(hash_id="trigger-1")
    monkeypatch.setattr(
        job_segment_output,
        "Trigger",
        SimpleNamespace(from_event=lambda event: trigger_obj),
    )

    event = SimpleNamespace(hash_id="event-1")
    result = native.LagResult(
        lag=2,
        lag_timer=0.0,
        time_lag=[1.0, -1.0],
        segment_lag=[0.0, 0.0],
        events_data=[(event, object(), object())],
        progress_record={},
    )

    convert_elapsed, write_elapsed = job_segment_output._write_trigger_records(
        _output_context(queue=queue),
        result,
    )

    assert convert_elapsed >= 0.0
    assert write_elapsed >= 0.0
    assert trigger_obj.time_lag == [1.0, -1.0]
    assert trigger_obj.segment_lag == [0.0, 0.0]
    assert queue.items == [{"type": "trigger", "trigger": trigger_obj}]


def test_output_helper_skips_failed_trigger_conversion(monkeypatch):
    from pycwb.workflow.subflow import job_segment_output

    queue = ListQueue()
    monkeypatch.setattr(
        job_segment_output,
        "Trigger",
        SimpleNamespace(from_event=lambda event: _raise(RuntimeError("boom"))),
    )
    result = native.LagResult(
        lag=2,
        lag_timer=0.0,
        time_lag=[],
        segment_lag=[],
        events_data=[(SimpleNamespace(hash_id="bad-event"), object(), object())],
        progress_record={},
    )

    job_segment_output._write_trigger_records(_output_context(queue=queue), result)

    assert queue.items == []


def test_output_helper_keeps_event_when_qveto_fails(monkeypatch):
    from pycwb.workflow.subflow import job_segment_output

    event = SimpleNamespace(hash_id="event-1")
    monkeypatch.setattr(
        job_segment_output,
        "get_qveto",
        lambda data: _raise(RuntimeError("qveto failed")),
    )

    elapsed = job_segment_output._compute_event_qveto(
        ["H1"],
        event,
        {
            "H1_wf_DAT_whiten": object(),
            "H1_wf_REC_whiten": object(),
        },
    )

    assert elapsed >= 0.0
    assert not hasattr(event, "qveto")


def test_flatten_job_segments_by_trial_preserves_trial_and_updates_job_id():
    seg = WaveSegment(
        index=42,
        ifos=["H1", "L1"],
        analyze_start=100.0,
        analyze_end=200.0,
        sample_rate=1024,
        seg_edge=4.0,
        injections=[
            {"name": "a", "trial_idx": 1, "job_id": 42},
            {"name": "b", "trial_idx": 0, "job_id": 42},
            {"name": "c", "trial_idx": 1, "job_id": 42},
        ],
    )

    flat = flatten_job_segments_by_trial([seg])

    assert [item.index for item in flat] == [1, 2]
    assert [item.trial_idx for item in flat] == [0, 1]
    assert [[inj["name"] for inj in item.injections] for item in flat] == [["b"], ["a", "c"]]
    assert [[inj["trial_idx"] for inj in item.injections] for item in flat] == [[0], [1, 1]]
    assert [[inj["job_id"] for inj in item.injections] for item in flat] == [[1], [2, 2]]
    assert [[inj["source_job_id"] for inj in item.injections] for item in flat] == [[42], [42, 42]]


def test_flatten_job_segments_by_trial_keeps_no_injection_segment():
    seg = WaveSegment(
        index=9,
        ifos=["H1", "L1"],
        analyze_start=100.0,
        analyze_end=200.0,
        sample_rate=1024,
        seg_edge=4.0,
        injections=[],
        trial_idx=5,
    )

    flat = flatten_job_segments_by_trial([seg])

    assert len(flat) == 1
    assert flat[0].index == 1
    assert flat[0].trial_idx == 0
    assert flat[0].injections == []
