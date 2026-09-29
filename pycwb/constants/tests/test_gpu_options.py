"""GPU choices are explicit, validated, serializable and job-local."""

from dataclasses import asdict
from types import SimpleNamespace
import pickle
import pytest
from jsonschema import ValidationError
from pycwb.constants.gpu_options import GPUOptions, gpu_options, resolve_gpu_options


def test_environment_cannot_change_options(monkeypatch):
    monkeypatch.setenv("PYCWB_GPU_DPF", "1")
    assert not gpu_options().dpf
    enabled = SimpleNamespace(gpu={"dpf": True, "lag_workers": 6})
    assert gpu_options(enabled).dpf
    assert not gpu_options(SimpleNamespace()).dpf
    assert pickle.loads(pickle.dumps(gpu_options(enabled))) == gpu_options(enabled)
    assert resolve_gpu_options(asdict(gpu_options(enabled))) == gpu_options(enabled)


@pytest.mark.parametrize(
    "options",
    [
        {"dpf": "1"},
        {"lag_workers": 0},
        {"lag_workers": 7},
        {"output_batch": -1},
        {"unknown": True},
    ],
)
def test_invalid_options_rejected(options):
    with pytest.raises(ValidationError):
        resolve_gpu_options(options)


def test_incompatible_prefilter_workers_rejected():
    with pytest.raises(ValueError, match="three"):
        resolve_gpu_options({"wdm_prefilter": True, "setup_workers": 4})


def test_integral_float_is_not_a_pool_worker_count():
    with pytest.raises(ValueError, match="integer worker/batch count"):
        resolve_gpu_options({"lag_workers": 2.0})
