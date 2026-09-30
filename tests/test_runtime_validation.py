"""Configuration and factory checks must agree before CUDA allocation or data IO."""
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml

from pycwb.config import Config
from pycwb.config.validation import validate_runtime_settings
from pycwb.modules.likelihood_gpu.likelihood import build_likelihood


@pytest.mark.parametrize('gpu, extra, message', [
    ({'dpf': True}, {'execution_profile': {'scalar_dpf': False}}, 'scalar_dpf'),
    ({'output_batch': 2}, {'save_waveform': True}, 'saved waveforms'),
    ({'q_reconstruction': True}, {'plot_waveform': True}, 'plots'),
    ({'worker_output': True}, {}, 'retired'),
])
def test_config_runtime_and_factory_reject_same_options(tmp_path, monkeypatch, gpu, extra, message):
    def unexpected_catalog_access(*args):
        pytest.fail("Invalid settings must be rejected before catalog access")

    monkeypatch.setattr(Config, "check_xtalk_file", unexpected_catalog_access)
    params = yaml.safe_load((Path(__file__).resolve().parents[1] / 'examples/demo/user_parameters.yaml').read_text())
    params.update(gpu=gpu, **extra)
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(params))
    for check in (lambda: Config().load_from_yaml(path), lambda: validate_runtime_settings(SimpleNamespace(**params)), lambda: build_likelihood(SimpleNamespace(**params))):
        with pytest.raises(ValueError, match=message):
            check()


def test_documented_default_and_catalog_settings_validate():
    for options in ({}, {'selection_cuda': True, 'output_batch': 32, 'q_reconstruction': True}):
        validate_runtime_settings(SimpleNamespace(gpu=options))
