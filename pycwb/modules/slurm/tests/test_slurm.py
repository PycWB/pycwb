"""Tests for pycwb.modules.slurm — Slurm scheduler configuration."""
import os
import pytest
from pycwb.modules.slurm.slurm import Slurm


class TestSlurmInit:
    """Tests for Slurm constructor — argument handling and defaults."""

    def test_basic_construction(self):
        """Should construct with all defaults."""
        s = Slurm()
        assert s.working_dir == os.path.abspath(".")
        assert s.n_proc == 1
        assert s.memory == "6GB"
        assert s.disk == "4GB"
        assert s.time == "72:00:00"
        assert s.job_per_worker == 10
        assert s.n_retries == 5
        assert s.constraint is None
        assert s.partition is None
        assert s.conda_env is None

    def test_custom_working_dir(self):
        """Custom working directory stored as absolute path."""
        s = Slurm(working_dir="/tmp/slurm_test")
        assert s.working_dir == "/tmp/slurm_test"

    def test_custom_resources(self):
        """Custom n_proc, memory, disk, time."""
        s = Slurm(n_proc=8, memory="32GB", disk="100GB", time="24:00:00")
        assert s.n_proc == 8
        assert s.memory == "32GB"
        assert s.disk == "100GB"
        assert s.time == "24:00:00"

    def test_slurm_dir_path(self):
        """Slurm directory should be <working_dir>/slurm."""
        s = Slurm(working_dir="/my/work")
        assert s.slurm_dir == "/my/work/slurm"

    def test_constraint_and_partition(self):
        """Optional Slurm params constraint and partition."""
        s = Slurm(constraint="haswell", partition="large")
        assert s.constraint == "haswell"
        assert s.partition == "large"

    def test_conda_env_stored(self):
        """conda_env should be stored as provided."""
        s = Slurm(conda_env="pycwb")
        assert s.conda_env == "pycwb"

    def test_custom_job_per_worker(self):
        """job_per_worker should be customizable."""
        s = Slurm(job_per_worker=5)
        assert s.job_per_worker == 5

    def test_n_retries_default_and_custom(self):
        """n_retries default is 5, can be customized."""
        s1 = Slurm()
        assert s1.n_retries == 5
        s2 = Slurm(n_retries=10)
        assert s2.n_retries == 10

    def test_empty_time_defaults_to_72h(self):
        """Empty or None time should default to '72:00:00'."""
        s = Slurm(time=None)
        assert s.time == "72:00:00"
        s2 = Slurm(time="")
        assert s2.time == "72:00:00"

    def test_additional_init_stored(self):
        """additional_init string is stored."""
        s = Slurm(additional_init="module load gcc")
        assert s.additional_init == "module load gcc"
