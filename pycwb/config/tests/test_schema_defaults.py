"""Tests for filling user parameters with defaults from the schema."""
import yaml

from pycwb.constants import user_parameters_schema
from pycwb.utils.yaml_helper import load_yaml


def _load_minimal(tmp_path, **params):
    config_file = tmp_path / "user_parameters.yaml"
    config_file.write_text(yaml.safe_dump({"analysis": "2G", "ifo": ["L1", "H1"], "refIFO": "L1", **params}))
    return load_yaml(str(config_file), user_parameters_schema)


def test_gwdatafind_defaults_to_empty_dict_meaning_not_used(tmp_path):
    params = _load_minimal(tmp_path)
    assert params["gwdatafind"] == {}
    assert not params["gwdatafind"]


def test_mutable_defaults_are_not_shared_with_schema(tmp_path):
    params = _load_minimal(tmp_path)
    params["gwdatafind"]["site"] = ["L", "H"]
    assert user_parameters_schema["properties"]["gwdatafind"]["default"] == {}


def test_gwdatafind_from_yaml_is_kept(tmp_path):
    gwdatafind = {"frametype": ["L1_HOFT_C00", "H1_HOFT_C00"], "host": "datafind.igwn.org", "urltype": "osdf"}
    params = _load_minimal(tmp_path, gwdatafind=gwdatafind)
    assert params["gwdatafind"] == gwdatafind
