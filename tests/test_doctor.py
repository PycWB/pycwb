"""Doctor is a metadata inventory, not a backend readiness gate."""

import json
import sys

import pytest

from pycwb.cli import doctor
from pycwb.cli.main import main


@pytest.fixture
def installed_packages(tmp_path, monkeypatch):
    for name, version in [("z_backend", "2.0"), ("A-core", "1.3")]:
        directory = tmp_path / f"{name}-{version}.dist-info"
        directory.mkdir()
        (directory / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        )
    # A package can be inventoried even when importing its backend would fail.
    (tmp_path / "z_backend.py").write_text("raise RuntimeError('backend unavailable')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    distributions = doctor.metadata.distributions
    monkeypatch.setattr(
        doctor.metadata, "distributions", lambda: distributions(path=[str(tmp_path)])
    )


def test_json_reports_installed_metadata_without_import_probes(installed_packages, capsys):
    assert main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["packages"] == [
        {"name": "A-core", "version": "1.3"},
        {"name": "z_backend", "version": "2.0"},
    ]
    assert report["executable"] == sys.executable
    assert report["python"] and report["platform"] and report["pycwb"]
    assert "readiness are not checked" in report["scope"]
    assert "ok" not in report and "checks" not in report
    assert "z_backend" not in sys.modules


def test_text_report_explains_scope(installed_packages, capsys):
    assert main(["doctor"]) == 0
    text = capsys.readouterr().out
    assert sys.executable in text
    assert "A-core==1.3" in text and "z_backend==2.0" in text
    assert "readiness are not checked" in text


def test_empty_inventory_is_not_a_pipeline_failure(monkeypatch, capsys):
    monkeypatch.setattr(doctor.metadata, "distributions", lambda: [])
    assert main(["doctor", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["packages"] == []
