"""Import order and released serialization compatibility for policy models."""

import pickle
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", [
    "pycwb.config.execution",
    "pycwb.config.processing",
    "pycwb.constants.user_parameters_schema",
    "pycwb.constants.execution_profile",
])
def test_policy_imports_do_not_eagerly_load_analysis_config(module):
    # A fresh interpreter exposes cycles hidden by earlier tests' import order.
    script = f"""
import importlib
import sys
importlib.import_module({module!r})
assert 'pycwb.config.config' not in sys.modules
import pycwb.config
assert 'Config' in dir(pycwb.config)
from pycwb.config import Config
from pycwb.config.config import Config as DirectConfig
assert Config is DirectConfig
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_released_profile_pickle_resolves_to_canonical_class():
    from pycwb.config.processing import ExecutionProfile
    from pycwb.constants.execution_profile import ExecutionProfile as ReleasedProfile

    assert ReleasedProfile is ExecutionProfile
    profile = ExecutionProfile(scalar_dpf=True, gc_full_interval=3)
    # Protocol 0 names the defining module in plain text. Reproduce the old
    # module name without changing the frozen instance or global class state.
    current = pickle.dumps(profile, protocol=0)
    assert b"pycwb.config.processing" in current
    released = current.replace(
        b"pycwb.config.processing", b"pycwb.constants.execution_profile"
    )
    restored = pickle.loads(released)
    assert type(restored) is ExecutionProfile
    assert restored == profile
