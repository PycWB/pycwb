"""Exercise the synthetic YAML through the installed production CLI."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from synthetic_recovery_helpers import check_recovery, copy_example


def test_installed_run_rejects_invalid_config_from_outside_checkout(tmp_path):
    """The normal run command checks YAML before accessing data or catalogs."""
    config = copy_example(tmp_path / "inputs")
    config.write_text("[]\n")
    working_dir = tmp_path / "search"
    result = subprocess.run(
        [sys.executable, "-I", "-m", "pycwb", "run", str(config),
         "--work-dir", str(working_dir)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "Expected a YAML mapping" in result.stderr
    assert not (working_dir / "catalog").exists()
    assert not (working_dir / "wdmXTalk").exists()


@pytest.mark.slow
def test_installed_synthetic_recovery(tmp_path):
    """Run the ordinary CLI outside the checkout and check persisted recovery."""
    import pycwb

    # Both pytest's recovery check and the CLI must exercise the installed wheel.
    checkout = Path(__file__).resolve().parents[1]
    assert not Path(pycwb.__file__).resolve().is_relative_to(checkout), (
        "Run this installed-package test with python -I -m pytest to avoid "
        "importing the source checkout."
    )
    config = copy_example(tmp_path / "inputs")
    if os.environ.get("PYCWB_DEMO_XTALK"):
        xtalk = Path(os.environ["PYCWB_DEMO_XTALK"]).resolve(strict=True)
        params = yaml.safe_load(config.read_text())
        params.update(filter_dir=str(xtalk.parent), wdmXTalk=xtalk.name)
        config.write_text(yaml.safe_dump(params, sort_keys=False))
    working_dir = tmp_path / "search"
    command = [
        sys.executable, "-I", "-m", "pycwb", "run", str(config),
        "--work-dir", str(working_dir),
    ]
    environment = dict(
        os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    result = subprocess.run(
        command, cwd=tmp_path, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600, check=False,
    )
    assert result.returncode == 0, result.stdout[-12000:]
    report = check_recovery(working_dir, config)
    assert report["ok"] and report["recovered_triggers"] >= 1
