"""Regression coverage for simulation summary cli."""
import argparse
from pathlib import Path
import pytest


@pytest.mark.parametrize('explicit', [True,False])
@pytest.mark.parametrize('fail', [True,False])
def test_simulation_summary_resolves_workdir_and_restores_cwd(tmp_path, monkeypatch, explicit, fail):
    from pycwb.cli import simulation_summary as cli
    import pycwb.config
    import pycwb.modules.job_segment
    from pycwb.workflow.subflow import simulation_summary as summary
    work=tmp_path/'production'; (work/'config').mkdir(parents=True)
    (work/'input').mkdir(); (work/'input/dq.txt').write_text('fixture')
    (work/'config/user_parameters.yaml').write_text('injection: {}')
    monkeypatch.chdir(tmp_path)
    class Config:
        injection=True
        def load_from_yaml(self, path):
            assert Path(path).resolve()==work/'config/user_parameters.yaml'
    def build(config):
        assert Path('input/dq.txt').read_text()=='fixture'
        if fail: raise RuntimeError('job failure')
        return ['job']
    def summarize(config, jobs, output_file):
        assert jobs==['job']
        expected=tmp_path/'result.parquet' if explicit else work/'catalog/simulations.parquet'
        assert Path(output_file)==expected
        return [1]
    monkeypatch.setattr(pycwb.config,'Config',Config)
    monkeypatch.setattr(pycwb.modules.job_segment,'create_job_segment_from_config',build)
    monkeypatch.setattr(summary,'build_simulation_summary',summarize)
    parser=argparse.ArgumentParser();cli.init_parser(parser)
    argv=['--work-dir','production']
    if explicit: argv+=['production/config/user_parameters.yaml','--output','result.parquet']
    args=parser.parse_args(argv)
    if fail:
        with pytest.raises(RuntimeError, match='job failure'): cli.command(args)
    else: cli.command(args)
    assert Path.cwd()==tmp_path
