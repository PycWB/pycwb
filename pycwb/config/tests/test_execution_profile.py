"""Configuration, provenance and isolation contracts for native execution."""

from dataclasses import FrozenInstanceError, asdict
from types import SimpleNamespace
import pickle

import numpy as np
import orjson
import pytest
from jsonschema import ValidationError

from pycwb.config import Config
from pycwb.config.processing import (
    ExecutionProfile,
    PROFILE_SCHEMA,
    execution_profile,
    resolve_execution_profile,
    wdm_options,
)


def test_yaml_defaults_validation_and_frozen_snapshot(tmp_path, monkeypatch):
    # Isolate configuration loading from external catalogs and detector geometry.
    for method in (
        "add_derived_key",
        "check_xtalk_file",
        "check_MRA_catalog",
        "check_lagStep",
        "check_analyze_injection_only",
    ):
        monkeypatch.setattr(Config, method, lambda *args: None)
    monkeypatch.setattr(Config, "MRAcatalog", "", raising=False)
    path = tmp_path / "user_parameters.yaml"
    path.write_text(
        "analysis: 2G\nifo: [H1, L1]\nrefIFO: L1\nexecution_profile:\n  sky_delay_reuse: false\n  gc_full_interval: 16\n"
    )
    config = Config()
    config.load_from_yaml(path)
    profile = config.execution_profile
    assert profile.sky_delay_reuse is False and profile.gc_full_interval == 16
    assert profile.scalar_dpf is False
    assert set(asdict(profile)) == set(PROFILE_SCHEMA["properties"])
    with pytest.raises(FrozenInstanceError):
        profile.sky_delay_reuse = True
    monkeypatch.setenv("PYCWB_SKY_DELAY_REUSE", "1")
    assert execution_profile(config) is profile
    path.write_text(path.read_text().replace("gc_full_interval: 16", "gc_full_interval: 0"))
    with pytest.raises(ValidationError):
        Config().load_from_yaml(path)


@pytest.mark.parametrize(
    "mapping",
    [
        {"typo": True},
        {"sky_delay_reuse": "false"},
        {"gc_full_interval": 0},
        {"regression_percentile_stride": 1.5},
        {"numba_max_energy_mode": "unknown"},
        {"regression_engine": "unknown"},
        {"band_td_cache": True},
    ],
)
def test_reject_invalid_profiles(mapping):
    with pytest.raises(ValidationError):
        resolve_execution_profile(mapping)


def test_profile_copy_roundtrip_and_catalog_provenance(tmp_path):
    from pycwb.modules.catalog.catalog import _build_schema_metadata

    source = {"scalar_dpf": True, "sky_delay_reuse": False}
    config = Config()
    config.load_from_dict({"execution_profile": source})
    source["scalar_dpf"] = False
    assert config.execution_profile.scalar_dpf
    dumped = orjson.loads(_build_schema_metadata(config)[b"config"])
    assert dumped["execution_profile"] == asdict(config.execution_profile)
    restored = Config()
    restored.load_from_dict(dumped)
    assert restored.execution_profile == config.execution_profile
    assert pickle.loads(pickle.dumps(restored)).execution_profile == config.execution_profile


def test_interleaved_profiles_ignore_environment_and_select_groups(monkeypatch):
    import importlib

    scan = importlib.import_module("pycwb.modules.likelihoodWP.sky_scan")
    observed = []

    def kernel(*args):
        observed.append(len(args[-1]) - 1)
        return observed[-1]

    monkeypatch.setattr(scan, "scan_sky_kernel", kernel)
    geometry = (np.ones((4, 2)), np.ones((4, 2)), np.zeros((2, 4), dtype=np.int64))
    cluster = (np.ones((1, 2)), None, None)
    grouped = Config()
    independent = Config()
    independent.load_from_dict({"execution_profile": {"sky_delay_reuse": False}})
    setups = [{}, {}]
    for value in ("0", "1", "nonsense"):
        monkeypatch.setenv("PYCWB_SKY_DELAY_REUSE", value)
        for config, setup in zip((grouped, independent), setups):
            scan.scan_sky(geometry, cluster, (), reuse_delays=execution_profile(config).sky_delay_reuse, setup=setup)
    assert observed == [1, 4, 1, 4, 1, 4]


def test_numba_backend_selection_ignores_environment(monkeypatch):
    from pycwb.modules.coherence_native.projection import _max_energy_backend

    monkeypatch.setenv("PYCWB_MAX_ENERGY_BACKEND", "numba")
    assert _max_energy_backend(SimpleNamespace(max_energy_backend="jax")) == "jax"


def test_wdm_options_and_reconstruction_cache_are_profile_specific(monkeypatch):
    from pycwb.modules.reconstruction.getMRAwaveform import _create_wdm_set_python

    a = SimpleNamespace(l_low=3, l_high=3, rateANA=4096, segEdge=20, TDSize=4, execution_profile=ExecutionProfile())
    b = SimpleNamespace(**{**vars(a), "execution_profile": ExecutionProfile(wdm_bounded_jax_forward=True)})
    monkeypatch.setenv("WDM_BOUNDED_JAX_FORWARD", "1")
    assert wdm_options(a)["bounded_jax_forward"] is False
    assert wdm_options(b)["bounded_jax_forward"] is True
    assert _create_wdm_set_python(a) is not _create_wdm_set_python(b)
    assert _create_wdm_set_python(a) is _create_wdm_set_python(a)


def test_regression_stride_is_an_explicit_jit_specialization():
    from pycwb.modules.data_conditioning.regression_jax import _jax_process_layers
    from pycwb.modules.data_conditioning.regression_numba import _numba_process_layers

    rng = np.random.default_rng(78)
    a, b = rng.normal(size=(2, 256)), rng.normal(size=(2, 256))
    args = (a, b, 2, 4, 10, 5, 0.95, 20, 0.0, 0.0, 10, 0, 0.0, 8.0, 2.0)
    outputs = []
    for stride in (1, 2, 1):
        actual, mask = _jax_process_layers(*args, percentile_stride=stride)
        expected, expected_mask = _numba_process_layers(*args, stride)
        np.testing.assert_array_equal(mask, expected_mask)
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-8)
        outputs.append(np.asarray(actual))
    np.testing.assert_array_equal(outputs[0], outputs[2])


def test_schema_and_runtime_defaults_match():
    assert {key: value["default"] for key, value in PROFILE_SCHEMA["properties"].items()} == asdict(ExecutionProfile())


def test_numba_fallback_receives_explicit_bounded_option(monkeypatch):
    from wdm_wavelet.wdm import WDM
    from pycwb.modules.coherence_native import time_delay_numba as module

    w = WDM(8, 8, 6, 10, backend="numba")
    signal = np.random.default_rng(99).normal(size=4096)
    calls = []
    original = module._wdm_t2w_numba

    def capture(*args, **kwargs):
        calls.append(kwargs["bounded"])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_HAS_T2W_NUMBA_CORE", False)
    monkeypatch.setattr(module, "_wdm_t2w_numba", capture)
    args = (signal, 8, w.m_H, w.filter, 1, 1, -1, 1, 0.1, 128, 0.0, 512.0, 64.0)
    a = module._time_delay_max_energy_pattern_nb(*args, bounded=False)
    count = len(calls)
    b = module._time_delay_max_energy_pattern_nb(*args, bounded=True)
    assert calls[:count] == [False] * count
    assert calls[count:] == [True] * count
    np.testing.assert_array_equal(a, b)


def test_explicit_jax_entrypoint_preserves_caller_profile(monkeypatch):
    import importlib

    module = importlib.import_module("pycwb.modules.data_conditioning.regression")
    from pycwb.modules.data_conditioning.regression import apply_regression_jax

    config = Config()
    original = config.execution_profile
    monkeypatch.setattr(module, "apply_regression", lambda selected, data: (selected.execution_profile, data))
    profile, result = apply_regression_jax(config, "input")
    assert profile.regression_engine == "jax"
    assert result == "input"
    assert config.execution_profile is original
    assert original.regression_engine == "numba"


def test_catalog_profile_guard_rejects_mixed_and_unrecorded_runs():
    from pycwb.config.processing import check_recorded_execution_profile

    config = Config()
    stored = {"execution_profile": asdict(config.execution_profile)}
    check_recorded_execution_profile(config, stored)
    stored["execution_profile"]["sky_delay_reuse"] = False
    with pytest.raises(ValueError, match="differs from the existing catalog"):
        check_recorded_execution_profile(config, stored)
    with pytest.raises(ValueError, match="predates recorded execution profiles"):
        check_recorded_execution_profile(config, {})
    with pytest.raises(ValueError, match="predates recorded execution profiles"):
        check_recorded_execution_profile(config, {"execution_profile": None})


@pytest.mark.parametrize("recorded", [{}, {"execution_profile": asdict(ExecutionProfile())}])
def test_run_setup_checks_profile_before_replacing_saved_yaml(tmp_path, monkeypatch, recorded):
    import importlib

    module = importlib.import_module("pycwb.workflow.subflow.prepare_job_runs")
    monkeypatch.chdir(tmp_path)
    catalog = tmp_path / "catalog" / module.Catalog.DEFAULT_FILENAME
    catalog.parent.mkdir()
    catalog.touch()

    def load(config, path):
        config.execution_profile = ExecutionProfile(sky_delay_reuse=False)
        config.catalog_dir = "catalog"
        config.outputDir = "output"

    monkeypatch.setattr(Config, "load_from_yaml", load)
    # This test isolates the effective-profile guard; YAML snapshot checks have
    # separate integration coverage with real configuration and Parquet files.
    monkeypatch.setattr(module, "validate_run_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "create_job_segment_from_config", lambda config: [])
    monkeypatch.setattr(module, "read_catalog_metadata", lambda path: {"config": recorded})

    def forbidden(*args, **kwargs):
        pytest.fail("profile mismatch must be rejected before changing the saved YAML")

    monkeypatch.setattr(module, "create_output_directory", forbidden)
    with pytest.raises(ValueError, match="profile"):
        module.prepare_job_runs(str(tmp_path), "user_parameters.yaml", overwrite=True)


def test_gpu_yaml_snapshot_catalog_roundtrip_and_resume_guard(tmp_path, monkeypatch):
    from pycwb.constants.gpu_options import GPUOptions
    from pycwb.config.processing import check_recorded_execution_profile

    for method in ("add_derived_key", "check_xtalk_file", "check_MRA_catalog",
                   "check_lagStep", "check_analyze_injection_only"):
        monkeypatch.setattr(Config, method, lambda *args: None)
    monkeypatch.setattr(Config, "MRAcatalog", "", raising=False)
    path = tmp_path / "user_parameters.yaml"
    path.write_text("analysis: 2G\nifo: [H1, L1]\nrefIFO: L1\ngpu:\n  likelihood: true\n  lag_workers: 6\n")
    config = Config()
    config.load_from_yaml(path)
    assert config.gpu == GPUOptions(likelihood=True, lag_workers=6)
    stored = orjson.loads(orjson.dumps(config.to_dict(), option=orjson.OPT_SERIALIZE_NUMPY))
    assert stored["gpu"]["lag_workers"] == 6
    check_recorded_execution_profile(config, stored)
    assert pickle.loads(pickle.dumps(config)).gpu == config.gpu
    stored["gpu"]["likelihood"] = False
    with pytest.raises(ValueError, match="gpu options differ"):
        check_recorded_execution_profile(config, stored)
