"""Catch example failures before downloading strain or running a long search."""

from pathlib import Path
from unittest.mock import Mock
import importlib

import pytest
import yaml

from pycwb.config import Config
from pycwb.modules.injection.par_generator import get_injection_list_from_parameters
from pycwb.modules.job_segment import create_job_segment_from_config
from pycwb.modules.job_segment.frame import get_frame_meta
from pycwb.utils.module import import_function


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
CONFIGS = sorted(
    path for path in EXAMPLES.rglob("*.yaml")
    if isinstance(data := yaml.safe_load(path.read_text()), dict) and "ifo" in data
)


@pytest.fixture
def config_without_catalog(monkeypatch):
    monkeypatch.setattr(Config, "check_xtalk_file", lambda *args: None)
    monkeypatch.setattr(Config, "check_MRA_catalog", lambda *args: None)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: str(path.relative_to(EXAMPLES)))
def test_search_example_configuration(path, config_without_catalog, monkeypatch):
    directory = path.parent.parent if path.parent.name == "config" else path.parent
    monkeypatch.chdir(directory)
    config = Config()
    config.load_from_yaml(path)
    if config.injection:
        parameters = get_injection_list_from_parameters(config.injection)
        assert parameters and all(isinstance(p, dict) for p in parameters)
        generator = config.injection.get("generator")
        assert generator is None or isinstance(generator, str)
    if path.parent.name == "new_injection_infra_with_gaussian_noise":
        jobs = create_job_segment_from_config(config)
        assert jobs and jobs[0].noise["psds"] == [None, None]


@pytest.mark.parametrize("module_name", ["utils", "gwosc"])
def test_downloaded_frame_list_survives_working_directory_change(tmp_path, monkeypatch, module_name):
    module = importlib.import_module(f"pycwb.modules.gwosc.{module_name}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "get_urls", lambda **kwargs: [
        "https://example.test/H-H1_LOSC_4_V1-1126256640-4096.gwf"
    ])
    response = Mock(content=b"frame fixture")
    monkeypatch.setattr(module.requests, "get", Mock(return_value=response))
    if module_name == "utils":
        module.download_frames_files("inputs", ["H1"], 1126259000, 1126260000, 4096)
    else:
        monkeypatch.setattr(module, "event_info", lambda *args: (["H1"], 1126259462, 1126259000, 1126260000))
        module.download_frames_files("GW150914", "inputs", ["H1"])
    response.raise_for_status.assert_called_once()
    working_dir = tmp_path / "runs" / "public-data"
    working_dir.mkdir(parents=True)
    monkeypatch.chdir(working_dir)
    frames = get_frame_meta(tmp_path / "inputs/H1_frames.in", "H1")
    assert len(frames) == 1
    assert Path(frames[0].path).is_absolute()
    assert Path(frames[0].path).read_bytes() == b"frame fixture"


def test_tutorial_public_data_channels_match_o1_frame_release(tmp_path):
    prepare = import_function(str(EXAMPLES / "tutorials/prepare.py") + ".prepare")
    configs = prepare(tmp_path / "tutorial-work")
    for name in ("open_data", "background"):
        assert configs[name]["inRate"] == 4096
        assert configs[name]["channelNamesRaw"] == ["L1:LOSC-STRAIN", "H1:LOSC-STRAIN"]
        assert all(Path(path).is_absolute() for path in configs[name]["frFiles"])


def test_gwosc_template_matches_downloaded_frame_channels(tmp_path, monkeypatch):
    from pycwb.cli.gwosc import configure_downloaded_frames
    import gwpy.io.gwf

    params = tmp_path / 'user_parameters.yaml'
    params.write_text((ROOT / 'pycwb/vendor/template/gwosc/user_parameters.yaml').read_text())
    inputs = tmp_path / 'input'
    inputs.mkdir()
    for ifo in ['H1', 'L1']:
        (inputs / f'{ifo}_frames.in').write_text(f'/frames/{ifo}.gwf\n')
    monkeypatch.setattr(gwpy.io.gwf, 'get_channel_names', lambda path: [
        f'{Path(path).stem}:LOSC-STRAIN', f'{Path(path).stem}:LOSC-DQMASK'
    ])
    configure_downloaded_frames(params, inputs, ['H1', 'L1'])
    actual = yaml.safe_load(params.read_text())
    assert actual['ifo'] == ['H1', 'L1']
    assert actual['channelNamesRaw'] == ['H1:LOSC-STRAIN', 'L1:LOSC-STRAIN']
    assert actual['inRate'] / 2 ** actual['levelR'] == 2048
    assert all(Path(row[1]).is_absolute() for row in actual['DQF'])


def test_online_segment_preserves_analysis_and_padded_windows():
    from types import SimpleNamespace
    from pycwb.types.online import OnlineSegment
    from pycwb.workflow.subflow.process_online_segment import _online_seg_to_wave_seg

    online = OnlineSegment(index=2, ifos=['H1', 'L1'], segment_gps_start=1000.,
                           segment_gps_end=1060., seg_edge=8., sample_rate=4096.,
                           data_payload={}, wall_time_received=0., stride=8., overlap_frac=0.)
    job = _online_seg_to_wave_seg(online, SimpleNamespace(lagStep=1.))
    assert (job.analyze_start, job.analyze_end) == (1000., 1060.)
    assert (job.padded_start, job.padded_end) == (992., 1068.)
    assert job.lag_shifts.tolist() == [[0., 0.]]


def test_mesa_order_is_positive_integral(config_without_catalog, tmp_path):
    from jsonschema import ValidationError
    config = Config()
    base = yaml.safe_load((EXAMPLES / 'demo/user_parameters.yaml').read_text())
    path = tmp_path / 'mesa.yaml'
    path.write_text(yaml.safe_dump(base))
    config.load_from_yaml(path)
    assert isinstance(config.mesaOrder, int)
    for invalid in (0, 800.5):
        path.write_text(yaml.safe_dump({**base, 'mesaOrder': invalid}))
        with pytest.raises((ValidationError, ValueError)):
            Config().load_from_yaml(path)


def test_event_download_interval_includes_requested_window_and_edges(monkeypatch):
    module = importlib.import_module('pycwb.modules.gwosc.gwosc')
    monkeypatch.setattr(module, 'event_gps', lambda _: 1000.4)
    monkeypatch.setattr(module, 'event_detectors', lambda _: {'H1', 'L1'})
    assert module.event_info('event', ['L1', 'H1'], 100, 200) == (
        ['L1', 'H1'], 1000.4, 890, 1211)


@pytest.mark.parametrize('directory', [
    'new_injection_infra_with_LHV', 'new_injection_infra_with_gaussian_noise',
    'new_injection_infra_with_sky_patch',
])
def test_readme_injection_blocks_prepare_jobs(directory, config_without_catalog, monkeypatch):
    import re
    example = EXAMPLES / directory
    monkeypatch.chdir(example)
    config = Config()
    config.load_from_yaml(example / 'user_parameters.yaml')
    snippet = re.search(r'```yaml\n(.*?)```', (example / 'README.md').read_text(), re.S)[1]
    config.injection = yaml.safe_load(snippet)['injection']
    jobs = create_job_segment_from_config(config)
    assert jobs and any(job.injections for job in jobs)
