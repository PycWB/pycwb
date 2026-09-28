"""Opt-in test of the installed package's complete beginner workflow."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_installed_demo(tmp_path):
    """Run outside the checkout; optionally reuse a verified local catalog."""
    command = [
        sys.executable,
        "-I",
        "-m",
        "pycwb",
        "demo",
        str(tmp_path / "demo"),
        "--run",
    ]
    if os.environ.get("PYCWB_DEMO_XTALK"):
        command += ["--xtalk", str(Path(os.environ["PYCWB_DEMO_XTALK"]).resolve())]
    environment = dict(
        os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-12000:]
    import json

    report = json.loads((tmp_path / "demo/demo-result.json").read_text())
    assert report["ok"] and report["recovered_triggers"] >= 1
