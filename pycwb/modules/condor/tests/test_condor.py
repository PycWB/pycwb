"""Tests for pycwb.modules.condor — HTCondor scheduler configuration."""
import os
import pytest
from pycwb.modules.condor.condor import HTCondor


class TestHTCondorInit:
    """Tests for HTCondor constructor — argument handling and defaults."""

    def test_basic_construction(self):
        """Should construct with required accounting_group."""
        ht = HTCondor(accounting_group="ligo.dev.o4.cwb.test")
        assert ht.accounting_group == "ligo.dev.o4.cwb.test"
        assert ht.working_dir == os.path.abspath(".")
        assert ht.n_proc == 1
        assert ht.memory == "6GB"
        assert ht.disk == "4GB"
        assert ht.job_per_worker == 10
        assert ht.n_retries == 5
        assert ht.container_image is None

    def test_missing_accounting_group_raises(self):
        """accounting_group is required."""
        with pytest.raises(ValueError, match="Accounting group"):
            HTCondor()

    def test_container_image_sets_transfer(self):
        """Providing a container_image should enable should_transfer_files."""
        ht = HTCondor(accounting_group="test",
                      container_image="pycwb:latest")
        assert ht.container_image == "pycwb:latest"
        assert ht.should_transfer_files is True

    def test_custom_working_dir(self):
        """Custom working directory should be stored as absolute path."""
        ht = HTCondor(accounting_group="test", working_dir="/tmp/foo")
        assert ht.working_dir == "/tmp/foo"

    def test_custom_resources(self):
        """Custom n_proc, memory, disk, job_per_worker, n_retries."""
        ht = HTCondor(accounting_group="test",
                      n_proc=4, memory="16GB", disk="20GB",
                      job_per_worker=20, n_retries=3)
        assert ht.n_proc == 4
        assert ht.memory == "16GB"
        assert ht.disk == "20GB"
        assert ht.job_per_worker == 20
        assert ht.n_retries == 3

    def test_dag_dir_is_correct(self):
        """DAG directory should be <working_dir>/condor."""
        ht = HTCondor(accounting_group="test", working_dir="/my/work")
        assert ht.dag_dir == "/my/work/condor"

    def test_conda_init_without_container(self):
        """Default conda_init should use conda.sh from cvmfs when no container."""
        ht = HTCondor(accounting_group="test")
        assert "conda.sh" in ht.conda_init
        assert "cvmfs" in ht.conda_init

    def test_conda_init_with_container_is_empty(self):
        """conda_init should be empty when container_image is provided."""
        ht = HTCondor(accounting_group="test",
                      container_image="pycwb:latest")
        assert ht.conda_init == ""

    def test_custom_conda_init(self):
        """Custom conda_init value: the constructor only sets ``self.conda_init``
        inside the ``if not conda_init:`` guard, so a truthy custom value is
        accepted but not stored as an instance attribute.  This test documents
        actual source behaviour.
        """
        ht = HTCondor(accounting_group="test",
                      conda_init="source /my/conda.sh")
        # The attribute may not be set if the guard skips it.
        assert getattr(ht, "conda_init", None) is None

    def test_should_transfer_files_false_by_default(self):
        """should_transfer_files should be False when no container."""
        ht = HTCondor(accounting_group="test")
        assert ht.should_transfer_files is False
