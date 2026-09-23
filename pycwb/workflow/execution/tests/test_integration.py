import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from pycwb.modules.catalog.catalog import Catalog
from pycwb.modules.condor.condor import HTCondor
from pycwb.modules.slurm.slurm import Slurm
from pycwb.workflow.execution.planner import prepare_plan, write_document
from pycwb.workflow.execution.scheduling import batch_job_ids
from pycwb.workflow.execution.settings import ExecutionSettings
from pycwb.workflow.execution.tests.test_execution import config, job
from pycwb.workflow.subflow.prepare_job_runs import load_batch_run


def setup_run(path):
    cfg = config(batch_size=2)
    from dataclasses import fields

    from pycwb.config import Config

    for field in fields(Config):
        if not field.init:
            setattr(cfg, field.name, None)
    cfg.outputDir, cfg.logDir, cfg.catalog_dir, cfg.trigger_dir = (
        "output",
        "log",
        "catalog",
        "trigger",
    )
    cfg.filter_dir, cfg.wdmXTalk = "", ""
    jobs = [job(10, "a.gwf"), job(20, "b.gwf"), job(30, "a.gwf")]
    (path / "catalog").mkdir(parents=True)
    for directory in ("config", "input", "wdmXTalk", "job_status", "log"):
        (path / directory).mkdir(exist_ok=True)
    source = path / "config" / "user_parameters.yaml"
    source.write_text("{}\n")
    Catalog.create(str(path / "catalog" / "catalog.parquet"), cfg, jobs)
    settings = ExecutionSettings.from_config(cfg)
    plan = prepare_plan(jobs, cfg, settings)
    write_document(path / "execution-plan.json", plan.document(jobs, settings))
    return cfg, jobs, plan, source


def test_slurm_uses_stable_batches_and_valid_shell(tmp_path):
    _, jobs, plan, _ = setup_run(tmp_path)
    scheduler = Slurm(
        working_dir=str(tmp_path),
        job_groups=[[jobs[i] for i in group] for group in plan.batches],
    )
    scheduler.generate_job_script(jobs)
    text = Path(scheduler.slurm_script).read_text()
    assert "job_groups=(10,30 20)" in text
    assert "--batch-id=$batch_id" in text
    assert "PYCWB_MEMORY_LIMIT_BYTES=" in text
    subprocess.run(["bash", "-n", scheduler.slurm_script], check=True)


def test_batch_plan_selection_and_identity(tmp_path, monkeypatch):
    _, _, _, source = setup_run(tmp_path)
    assert batch_job_ids(tmp_path, "b000000") == [10, 30]
    monkeypatch.chdir(tmp_path)
    selected, cfg, _, catalog = load_batch_run(
        str(tmp_path), str(source), None, batch_id="b000000"
    )
    assert [j.index for j in selected] == [10, 30]
    assert Path(catalog).name == "catalog_b000000.parquet"
    assert cfg.execution["profile"] == "scalable"
    with pytest.raises(ValueError, match="either"):
        load_batch_run(str(tmp_path), str(source), "10", batch_id="b000000")
    with pytest.raises(ValueError, match="form"):
        batch_job_ids(tmp_path, "../escape")
    path = tmp_path / "execution-plan.json"
    value = json.loads(path.read_text())
    value["plan"]["batches"][0] = [1]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identity"):
        batch_job_ids(tmp_path, "b000000")


def test_condor_transfer_fragments_are_self_contained(tmp_path, monkeypatch):
    pytest.importorskip("htcondor2")
    source_run = tmp_path / "source"
    _, jobs, plan, _ = setup_run(source_run)
    scheduler = HTCondor(
        working_dir=str(source_run),
        accounting_group="test",
        should_transfer_files=True,
        job_groups=[[jobs[i] for i in group] for group in plan.batches],
    )
    scheduler.create(jobs)
    dag = Path(scheduler.dag_file).read_text()
    assert 'batch_id="b000000"' in dag
    submit = (source_run / "condor" / "pycwb_batch.sub").read_text()
    assert "catalog_$(batch_id).parquet" in submit
    assert "catalog_$(jobs).parquet" not in submit
    subprocess.run(["bash", "-n", str(source_run / "condor" / "run.sh")], check=True)
    execute = tmp_path / "execute"
    (execute / "catalog" / "fragment").mkdir(parents=True)
    (execute / "config").mkdir()
    source = execute / "config" / "user_parameters.yaml"
    source.write_text("{}\n")
    shutil.copyfile(
        source_run / "catalog" / "fragment" / "catalog_b000000.parquet",
        execute / "catalog" / "fragment" / "catalog_b000000.parquet",
    )
    monkeypatch.chdir(execute)
    selected, _, _, _ = load_batch_run(
        str(execute), str(source), None, batch_id="b000000"
    )
    assert [j.index for j in selected] == [10, 30]
    assert not (execute / "execution-plan.json").exists()
    assert not (execute / "catalog" / "catalog.parquet").exists()


def test_condor_rejects_basename_collisions(tmp_path):
    jobs = [job(1, "/one/same.gwf"), job(2, "/two/same.gwf")]
    scheduler = HTCondor(
        working_dir=str(tmp_path),
        accounting_group="test",
        should_transfer_files=True,
        job_groups=[jobs],
    )
    with pytest.raises(ValueError, match="colliding"):
        scheduler.generate_condor_dag(jobs)


def test_cli_accepts_execution_batch():
    from pycwb.cli.batch_runner import init_parser

    parser = argparse.ArgumentParser()
    init_parser(parser)
    assert (
        parser.parse_args(["config.yaml", "--batch-id", "b000000"]).batch_id
        == "b000000"
    )
