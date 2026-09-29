"""Opt-in recovery test using the standalone example and installed package."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_standalone_example_from_outside_checkout(tmp_path):
    """Creation uses the adjacent example YAML, independent of the caller's cwd."""
    script = Path(__file__).resolve().parents[1] / "examples/demo/run_demo.py"
    command = [sys.executable, "-I", str(script), str(tmp_path / "demo")]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "demo/user_parameters.yaml").read_text() == (
        script.with_name("user_parameters.yaml").read_text()
    )
    assert "pycwb demo" not in result.stdout
    assert str(script) in result.stdout
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0


@pytest.mark.slow
def test_installed_demo(tmp_path):
    """Run outside the checkout; optionally reuse a verified local catalog."""
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve().parents[1] / "examples/demo/run_demo.py"),
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
