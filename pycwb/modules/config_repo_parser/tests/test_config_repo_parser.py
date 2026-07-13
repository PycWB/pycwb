"""Tests for pycwb.modules.config_repo_parser — project name parsing and settings."""
import os
import tempfile
import pytest
import yaml
from pycwb.modules.config_repo_parser.config_repo_parser import (
    get_ifo_list,
    get_machine_settings,
)


class TestGetIfoList:
    """Tests for get_ifo_list — network short code to IFO list."""

    def _write_settings(self, base_dir: str, networks: dict) -> str:
        """Helper: write a settings.yaml with given networks."""
        path = os.path.join(base_dir, "settings.yaml")
        with open(path, "w") as f:
            yaml.dump({"networks": networks}, f)
        return base_dir

    def test_lh_returns_l1_h1(self):
        """LH network should map to ['L1', 'H1']."""
        with tempfile.TemporaryDirectory() as tmp:
            self._write_settings(tmp, {"LH": ["L1", "H1"]})
            result = get_ifo_list("LH", tmp)
            assert result == ["L1", "H1"]

    def test_single_ifo(self):
        """Single-IFO network like 'H'."""
        with tempfile.TemporaryDirectory() as tmp:
            self._write_settings(tmp, {"H": ["H1"]})
            result = get_ifo_list("H", tmp)
            assert result == ["H1"]

    def test_hlv_triple(self):
        """Three-IFO network."""
        with tempfile.TemporaryDirectory() as tmp:
            self._write_settings(tmp, {"HLV": ["H1", "L1", "V1"]})
            result = get_ifo_list("HLV", tmp)
            assert result == ["H1", "L1", "V1"]

    def test_string_value_is_wrapped_in_list(self):
        """A string value (instead of list) should be wrapped."""
        with tempfile.TemporaryDirectory() as tmp:
            self._write_settings(tmp, {"H": "H1"})
            result = get_ifo_list("H", tmp)
            assert result == ["H1"]

    def test_missing_settings_file_raises(self):
        """Missing settings.yaml should raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError):
                get_ifo_list("LH", tmp)

    def test_unknown_network_raises(self):
        """Unknown network code should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            self._write_settings(tmp, {"LH": ["L1", "H1"]})
            with pytest.raises(ValueError, match="not found"):
                get_ifo_list("ZZ", tmp)

    def test_empty_network_raises(self):
        """Empty network string should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            self._write_settings(tmp, {"LH": ["L1", "H1"]})
            with pytest.raises(ValueError, match="cannot be empty"):
                get_ifo_list("", tmp)


class TestGetMachineSettings:
    """Tests for get_machine_settings — machine profile loader."""

    def _write_settings(self, base_dir: str, machine_name: str) -> str:
        """Write settings.yaml with a machine key."""
        path = os.path.join(base_dir, "settings.yaml")
        with open(path, "w") as f:
            yaml.dump({"machine": machine_name}, f)
        return base_dir

    def _write_machine_file(self, base_dir: str, machine_name: str, config: dict) -> str:
        """Write machine/<name>.yaml."""
        machine_dir = os.path.join(base_dir, "machine")
        os.makedirs(machine_dir, exist_ok=True)
        path = os.path.join(machine_dir, f"{machine_name}.yaml")
        with open(path, "w") as f:
            yaml.dump(config, f)
        return base_dir

    def test_loads_machine_config(self):
        """Should load the specified machine profile."""
        with tempfile.TemporaryDirectory() as tmp:
            self._write_settings(tmp, "cit")
            self._write_machine_file(tmp, "cit", {
                "cluster": "condor",
                "container_image": "pycwb:latest",
                "job_memory": "8GB",
            })
            result = get_machine_settings(tmp)
            assert result["cluster"] == "condor"
            assert result["container_image"] == "pycwb:latest"
            assert result["job_memory"] == "8GB"

    def test_machine_override(self):
        """Explicit machine= arg should bypass settings.yaml."""
        with tempfile.TemporaryDirectory() as tmp:
            # settings.yaml points to 'cit' but we override
            self._write_settings(tmp, "cit")
            self._write_machine_file(tmp, "custom", {"cluster": "slurm"})
            result = get_machine_settings(tmp, machine="custom")
            assert result["cluster"] == "slurm"

    def test_missing_machine_file_raises(self):
        """Missing machine/<name>.yaml should raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp:
            self._write_settings(tmp, "nonexistent")
            with pytest.raises(FileNotFoundError):
                get_machine_settings(tmp)

    def test_no_machine_key_in_settings_raises(self):
        """settings.yaml without 'machine' key should raise."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "settings.yaml")
            with open(path, "w") as f:
                yaml.dump({"other": "data"}, f)
            with pytest.raises(ValueError, match="machine"):
                get_machine_settings(tmp)

    def test_empty_machine_config_returns_empty_dict(self):
        """Empty YAML should return {}."""
        with tempfile.TemporaryDirectory() as tmp:
            self._write_settings(tmp, "empty")
            self._write_machine_file(tmp, "empty", {})
            result = get_machine_settings(tmp)
            assert result == {}
