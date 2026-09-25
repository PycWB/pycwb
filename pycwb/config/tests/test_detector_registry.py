"""Versioned detector geometry survives YAML loading and catalog round trips."""

import orjson
import pytest
from pycwb.config import Config
from pycwb.modules.catalog.catalog import _build_schema_metadata


def test_yaml_pins_each_geometry_and_catalog_restores_it(tmp_path, monkeypatch):
    for method in ("add_derived_key", "check_xtalk_file", "check_MRA_catalog", "check_lagStep", "check_analyze_injection_only"):
        monkeypatch.setattr(Config, method, lambda *args: None)
    monkeypatch.setattr(Config, "MRAcatalog", "", raising=False)
    path = tmp_path / "user_parameters.yaml"
    path.write_text("analysis: 2G\nifo: [H1, L1, V1]\nrefIFO: L1\ndetector_geometry:\n  H1: H1:cwb\n  L1: L1:cwb\n")
    config = Config()
    config.load_from_yaml(path)
    expected = {"H1": "H1:cwb", "L1": "L1:cwb", "V1": "V1:lal@pycwb-1"}
    assert config.ifo == ["H1", "L1", "V1"]
    assert config.detector_geometry == expected
    metadata = orjson.loads(_build_schema_metadata(config)[b"config"])
    assert metadata["detector_geometry"] == expected
    restored = Config()
    restored.load_from_dict(metadata)
    assert restored.detector_geometry == expected
    path.write_text(path.read_text().replace("H1: H1:cwb", "H1: L1:cwb"))
    with pytest.raises(ValueError, match="does not belong"):
        Config().load_from_yaml(path)
