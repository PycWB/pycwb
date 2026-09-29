"""Shared GPU availability checks and deterministic scientific fixtures."""

from pycwb.utils.tests.gpu_fixtures import (
    pytest_collection_modifyitems,
    pytest_configure,
    reuse_workspace,
    rng,
)
