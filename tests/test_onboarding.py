"""User-facing onboarding contracts without an expensive scientific run."""

from types import SimpleNamespace

import pytest
import yaml
from jsonschema import ValidationError

from synthetic_recovery_helpers import check_recovery, copy_example
from pycwb.cli.main import create_parser, main
from pycwb.config import Config


@pytest.fixture
def without_wavelet_catalog(monkeypatch):
    """Exercise configuration loading without the unrelated binary catalog."""
    monkeypatch.setattr(Config, "check_xtalk_file", lambda *args: None)
    monkeypatch.setattr(Config, "check_MRA_catalog", lambda *args: None)


@pytest.fixture
def offline_working_directory(tmp_path, monkeypatch):
    import requests

    def offline(*args, **kwargs):
        pytest.fail("Configuration tests must not access the network")

    monkeypatch.setattr(requests, "get", offline)
    monkeypatch.chdir(tmp_path)


def test_example_configuration_loads(tmp_path, without_wavelet_catalog, offline_working_directory):
    path = copy_example(tmp_path / "demo")
    config = Config()
    config.load_from_yaml(path)
    assert config.ifo == ["L1", "H1"]
    assert config.injection["segment"]["noise"]["seeds"] == [150914, 150915]


@pytest.mark.parametrize(
    "content,error",
    [
        ("", TypeError),
        ("[]", TypeError),
        ("ifo: [H1]\n", ValidationError),
        ("ifos: [H1, L1]\nanalysis: 2G\nrefIFO: H1\n", ValidationError),
    ],
)
def test_run_rejects_invalid_configuration(tmp_path, content, error, offline_working_directory):
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    working_dir = tmp_path / "search"
    with pytest.raises(error):
        main(["run", str(path), "--work-dir", str(working_dir)])
    assert not (working_dir / "catalog").exists()
    assert not (working_dir / "wdmXTalk").exists()


def test_config_honors_external_schema_extension(tmp_path, without_wavelet_catalog, offline_working_directory):
    config = copy_example(tmp_path / "demo")
    params = yaml.safe_load(config.read_text())
    (config.parent / "extra.yaml").write_text(
        "properties:\n  label:\n    type: string\n"
    )
    params.update(pycwb_schema={"schema_file": "extra.yaml"}, label="my-analysis")
    config.write_text(yaml.safe_dump(params))
    loaded = Config()
    loaded.load_from_yaml(config)
    assert loaded.label == "my-analysis"


@pytest.mark.parametrize(
    "updates",
    [
        {"gpu": {"profile_lags": "5:1"}},
        {"gpu": {"wdm_prefilter": True, "setup_workers": 4}},
        {"execution_profile": {"band_td_cache": True, "compact_td_cache": False}},
        {"execution": {"worker_memory": "nonsense"}},
        {"detector_geometry": {"H1": "L1:cwb"}},
        {"detector_geometry": {"V1": "V1:lal"}},
        {"detector_definitions_file": "missing.json"},
    ],
)
def test_run_rejects_invalid_runtime_settings(tmp_path, updates, offline_working_directory):
    path = copy_example(tmp_path / "demo")
    params = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump({**params, **updates}))
    working_dir = tmp_path / "search"
    with pytest.raises((ValueError, OSError, ValidationError)):
        main(["run", str(path), "--work-dir", str(working_dir)])
    assert not (working_dir / "catalog").exists()
    assert not (working_dir / "wdmXTalk").exists()


def test_config_resolves_local_detector_definitions(tmp_path, without_wavelet_catalog, offline_working_directory):
    import json
    from copy import deepcopy

    from pycwb.constants.detectors import DETECTOR_GEOMETRIES

    path = copy_example(tmp_path / "demo")
    entry = deepcopy(DETECTOR_GEOMETRIES["H1:lal@pycwb-1"])
    definitions = {
        "schema_version": 1,
        "geometries": {
            "H1:custom": {
                "detector": "H1",
                "parameters": entry["parameters"],
            }
        },
    }
    (path.parent / "detectors.json").write_text(json.dumps(definitions))
    params = yaml.safe_load(path.read_text())
    params.update(
        detector_definitions_file="detectors.json",
        detector_geometry={"H1": "H1:custom"},
        gpu={"profile_lags": "0:3"},
        execution_profile={"sky_delay_reuse": False},
    )
    path.write_text(yaml.safe_dump(params))
    loaded = Config()
    loaded.load_from_yaml(path)
    assert loaded.detector_geometry["H1"] == "H1:custom"
    # A syntactically valid file can still contain physically invalid geometry.
    definitions["geometries"]["H1:custom"]["parameters"]["y"] = entry["parameters"]["x"]
    (path.parent / "detectors.json").write_text(json.dumps(definitions))
    with pytest.raises(ValueError, match="collinear"):
        Config().load_from_yaml(path)


@pytest.fixture
def completed_demo(tmp_path):
    from pycwb.modules.catalog import Catalog
    from pycwb.types.trigger import Trigger

    config_path = copy_example(tmp_path / "demo")
    params = yaml.safe_load(config_path.read_text())
    directory = config_path.parent / "catalog"
    directory.mkdir()
    catalog = Catalog.create(
        str(directory / "catalog.parquet"),
        SimpleNamespace(ifo=params["ifo"]),
        [{"index": 0}],
    )
    target = params["injection"]["parameters"]["gps_time"]
    trigger = Trigger(
        job_id=0,
        lag_idx=0,
        trial_idx=0,
        rho=8.0,
        ifo_list=params["ifo"],
        time=[target + 0.01, target - 0.01],
    )
    catalog.add_triggers(trigger)
    catalog.add_lag_progress(0, 0, 0, 1, 128.0)
    return config_path.parent


def test_demo_recovers_signal_and_checks_manifest(completed_demo):
    report = check_recovery(completed_demo, completed_demo / "user_parameters.yaml")
    assert report["ok"] and report["recovered_triggers"] == 1
    (completed_demo / "catalog/jobs.parquet").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        check_recovery(completed_demo, completed_demo / "user_parameters.yaml")


@pytest.mark.parametrize(
    "field,value",
    [
        ("rho", float("nan")),
        ("rho", 0.1),
        ("time_H1", 1.0),
        ("time_L1", None),
        ("trial_idx", 5),
        ("job_id", 20),
    ],
)
def test_demo_rejects_wrong_or_nonfinite_event(completed_demo, field, value):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = completed_demo / "catalog/catalog.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0][field] = value
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="No finite trigger"):
        check_recovery(completed_demo, completed_demo / "user_parameters.yaml")


def test_demo_rejects_unfinished_job(completed_demo):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = completed_demo / "catalog/progress.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["status"] = "failed"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="completed zero-lag"):
        check_recovery(completed_demo, completed_demo / "user_parameters.yaml")


def test_reference_commands_match_parser():
    parser = create_parser()
    assert parser.parse_args(["merge", "--work-dir", "run", "--wave"]).wave
    assert (
        parser.parse_args(
            ["xtalk", "catalog.bin", "--output_dir", "converted"]
        ).output_dir
        == "converted"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["prepare", "config.yaml"])
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "run"])
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["validate", "config.yaml"])
    assert error.value.code == 2
