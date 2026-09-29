import importlib
import sys
import warnings


def test_read_data_facade_does_not_import_deprecated_mdc():
    import pycwb.modules.read_data as read_data

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(read_data)

    assert not any(issubclass(item.category, DeprecationWarning) for item in caught)
    assert callable(read_data.generate_noise_for_job_seg)
    assert callable(read_data.generate_strain_from_injection)
    assert callable(read_data.save_to_gwf)


def test_mdc_shim_warns_and_does_not_expose_generate_noise():
    sys.modules.pop("pycwb.modules.read_data.mdc", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mdc = importlib.import_module("pycwb.modules.read_data.mdc")

    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    assert callable(mdc.generate_noise_for_job_seg)
    assert callable(mdc.generate_strain_from_injection)
    assert callable(mdc.project_to_detector)
    assert callable(mdc.save_to_gwf)
    assert callable(mdc.generate_injections)
    assert callable(mdc.generate_injection)
    assert not hasattr(mdc, "generate_noise")


def test_generate_noise_requires_explicit_gaussian_module():
    import pycwb.modules.noise as noise
    from pycwb.modules.noise.gaussian import generate_noise

    assert not hasattr(noise, "generate_noise")
    assert callable(generate_noise)


def test_job_setup_imports_in_a_fresh_interpreter():
    import os
    from pathlib import Path
    import subprocess

    root = Path(__file__).resolve().parents[4]
    env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for module in ("pycwb.modules.job_segment", "pycwb.workflow.subflow.prepare_job_runs"):
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            env=env, cwd=root, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
