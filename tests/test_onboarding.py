"""User-facing onboarding contracts without an expensive scientific run."""

from types import SimpleNamespace

import pytest
import yaml

from pycwb.cli.demo import check_demo, create_demo
from pycwb.cli.main import create_parser, main
from pycwb.cli.validate import validate_config


def test_packaged_demo_validates_without_downloading(tmp_path, monkeypatch):
    import requests

    def offline(*args, **kwargs):
        pytest.fail("Configuration checking must not access the network")

    monkeypatch.setattr(requests, "get", offline)
    config = create_demo(tmp_path / "demo")
    params = validate_config(config)
    assert params["ifo"] == ["L1", "H1"]
    assert params["injection"]["segment"]["noise"]["seeds"] == [150914, 150915]
    assert not (config.parent / "wdmXTalk").exists()


def test_demo_never_overwrites_existing_directory(tmp_path):
    existing = tmp_path / "analysis"
    existing.mkdir()
    data = existing / "user_parameters.yaml"
    data.write_text("important analysis settings")
    with pytest.raises(FileExistsError):
        create_demo(existing)
    assert data.read_text() == "important analysis settings"


def test_missing_catalog_does_not_create_project(tmp_path):
    destination = tmp_path / "new"
    with pytest.raises(FileNotFoundError):
        create_demo(destination, tmp_path / "missing.bin")
    assert not destination.exists()


@pytest.mark.parametrize(
    "content", ["", "[]", "ifo: [H1]\n", "ifos: [H1, L1]\nanalysis: 2G\nrefIFO: H1\n"]
)
def test_invalid_configuration_has_nonzero_exit(tmp_path, content, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    assert main(["validate", str(path)]) == 1
    assert "INVALID" in capsys.readouterr().err


def test_validate_honors_external_schema_extension(tmp_path):
    config = create_demo(tmp_path / "demo")
    params = yaml.safe_load(config.read_text())
    (config.parent / "extra.yaml").write_text(
        "properties:\n  label:\n    type: string\n"
    )
    params.update(pycwb_schema={"schema_file": "extra.yaml"}, label="my-analysis")
    config.write_text(yaml.safe_dump(params))
    assert validate_config(config)["label"] == "my-analysis"


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
def test_validate_rejects_invalid_runtime_settings(tmp_path, updates, capsys):
    path = create_demo(tmp_path / "demo")
    params = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump({**params, **updates}))
    assert main(["validate", str(path)]) == 1
    assert "INVALID" in capsys.readouterr().err


def test_validate_resolves_local_detector_definitions_offline(tmp_path, monkeypatch):
    import json
    from copy import deepcopy

    import requests

    from pycwb.constants.detectors import DETECTOR_GEOMETRIES

    def offline(*args, **kwargs):
        pytest.fail("Configuration checking must not access the network")

    monkeypatch.setattr(requests, "get", offline)
    path = create_demo(tmp_path / "demo")
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
    monkeypatch.chdir(tmp_path)
    assert validate_config(path)["detector_geometry"] == {"H1": "H1:custom"}
    # A syntactically valid file can still contain physically invalid geometry.
    definitions["geometries"]["H1:custom"]["parameters"]["y"] = entry["parameters"]["x"]
    (path.parent / "detectors.json").write_text(json.dumps(definitions))
    with pytest.raises(ValueError, match="collinear"):
        validate_config(path)


@pytest.fixture
def completed_demo(tmp_path):
    from pycwb.modules.catalog import Catalog
    from pycwb.types.trigger import Trigger

    config_path = create_demo(tmp_path / "demo")
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
    report = check_demo(completed_demo)
    assert report["ok"] and report["recovered_triggers"] == 1
    (completed_demo / "catalog/jobs.parquet").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        check_demo(completed_demo)


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
        check_demo(completed_demo)


def test_demo_rejects_unfinished_job(completed_demo):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = completed_demo / "catalog/progress.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["status"] = "failed"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="completed zero-lag"):
        check_demo(completed_demo)


def test_doctor_optional_failure_is_not_a_required_failure(monkeypatch, capsys):
    from pycwb.cli import doctor

    monkeypatch.setattr(doctor, "REQUIRED", {"json": "not-a-distribution"})
    monkeypatch.setattr(doctor, "OPTIONAL", {"missing_optional_pycwb_test": "missing"})
    assert main(["doctor", "--json"]) == 0
    import json

    assert json.loads(capsys.readouterr().out)["ok"]
    monkeypatch.setattr(doctor, "REQUIRED", {"missing_required_pycwb_test": "missing"})
    assert main(["doctor"]) == 1


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
        parser.parse_args(["demo", "run", "--run", "--check"])
