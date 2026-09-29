import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from pycwb.config import Config
from pycwb.modules.catalog.catalog import Catalog
from pycwb.modules.condor.condor import HTCondor
from pycwb.modules.slurm.slurm import Slurm
from pycwb.workflow.execution.planner import prepare_plan
from pycwb.workflow.execution.scheduling import prepare_batch_fragments
from pycwb.workflow.execution.settings import ExecutionSettings
from pycwb.workflow.execution.tests.test_execution import config, job
from pycwb.workflow.subflow.prepare_job_runs import load_batch_run


@pytest.fixture(autouse=True)
def no_xtalk_download(monkeypatch):
    # Scheduling tests exercise YAML loading without a scientific data catalog.
    monkeypatch.setattr(Config, "check_xtalk_file", staticmethod(lambda path: None))
    monkeypatch.setattr(Config, "check_MRA_catalog", lambda self: None)


def setup_run(path):
    parameters = vars(config(batch_size=2))
    parameters.update(analysis="2G", ifo=["H1", "L1"], refIFO="H1")
    jobs = [job(10, "a.gwf"), job(20, "b.gwf"), job(30, "a.gwf")]
    (path / "catalog").mkdir(parents=True)
    for directory in ("config", "input", "wdmXTalk", "job_status", "log"):
        (path / directory).mkdir(exist_ok=True)
    source = path / "config" / "user_parameters.yaml"
    source.write_text(yaml.safe_dump(parameters))
    cfg = Config()
    cfg.load_from_yaml(source)
    Catalog.create(str(path / "catalog" / "catalog.parquet"), cfg, jobs)
    settings = ExecutionSettings.from_config(cfg)
    plan = prepare_plan(jobs, cfg, settings)
    return cfg, jobs, plan, source


def test_slurm_uses_stable_batches_and_valid_shell(tmp_path):
    _, jobs, plan, _ = setup_run(tmp_path)
    scheduler = Slurm(
        working_dir=str(tmp_path),
        job_groups=[[jobs[i] for i in group] for group in plan.batches],
    )
    scheduler.generate_job_script(jobs)
    text = Path(scheduler.slurm_script).read_text()
    assert "job_groups=" not in text
    assert "printf -v batch_id 'b%06d'" in text
    assert "--batch-id=$batch_id" in text
    assert "--memory-limit=" in text
    assert "--allocated-cores=" in text
    assert "PYCWB_MEMORY_LIMIT_BYTES" not in text
    subprocess.run(["bash", "-n", scheduler.slurm_script], check=True)


@pytest.mark.parametrize("cluster,transfer", [("slurm", False), ("condor", False), ("condor", True)])
def test_batch_setup_persists_selection_in_fragments(tmp_path, monkeypatch, cluster, transfer):
    if cluster == "condor":
        pytest.importorskip("htcondor2")
    from pycwb.workflow.batch import batch_setup
    import importlib

    preparation = importlib.import_module("pycwb.workflow.subflow.prepare_job_runs")
    _, jobs, _, source = setup_run(tmp_path)
    monkeypatch.setattr(preparation, "create_job_segment_from_config", lambda config: jobs)
    monkeypatch.chdir(tmp_path)
    batch_setup(str(source), working_dir=str(tmp_path), cluster=cluster,
                accounting_group="test", should_transfer_files=transfer)
    assert not (tmp_path / "execution-plan.json").exists()
    selected, cfg, _, catalog = load_batch_run(
        str(tmp_path), str(source), None, batch_id="b000000"
    )
    assert [j.index for j in selected] == [10, 30]
    assert Path(catalog).name == "catalog_b000000.parquet"
    assert cfg.execution["profile"] == "scalable"
    second, _, _, _ = load_batch_run(
        str(tmp_path), str(source), None, batch_id="b000001"
    )
    assert [j.index for j in second] == [20]
    # Reading again uses the same prepared membership on resume.
    resumed, _, _, _ = load_batch_run(str(tmp_path), str(source), None, batch_id="b000000")
    assert resumed == selected
    with pytest.raises(ValueError, match="either"):
        load_batch_run(str(tmp_path), str(source), "10", batch_id="b000000")
    with pytest.raises(ValueError, match="form"):
        load_batch_run(str(tmp_path), str(source), None, batch_id="../escape")


def test_missing_batch_fragment_never_falls_back_to_root(tmp_path, monkeypatch):
    _, _, _, source = setup_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Prepared batch fragment.*batch-setup"):
        load_batch_run(str(tmp_path), str(source), None, batch_id="b000000")
    assert not (tmp_path / "catalog/fragment").exists()


@pytest.mark.parametrize("change", ["regroup", "remove", "definition"])
def test_batch_fragments_preserve_existing_results_and_reject_changes(tmp_path, change):
    from dataclasses import replace

    cfg, jobs, _, _ = setup_run(tmp_path)
    groups = [[jobs[0], jobs[2]], [jobs[1]]]
    prepare_batch_fragments(tmp_path, cfg, groups)
    paths = list((tmp_path / "catalog/fragment").glob("*.parquet"))
    before = {path: path.read_bytes() for path in paths}
    prepare_batch_fragments(tmp_path, cfg, groups)
    assert {path: path.read_bytes() for path in paths} == before
    if change == "regroup":
        groups = [[jobs[0]], [jobs[1]], [jobs[2]]]
    elif change == "remove":
        groups = groups[:1]
    else:
        groups[0][0] = replace(jobs[0], analyze_end=jobs[0].analyze_end + 1)
    with pytest.raises(ValueError, match="Prepared batch jobs differ"):
        prepare_batch_fragments(tmp_path, cfg, groups)
    assert {path: path.read_bytes() for path in paths} == before
    assert not (tmp_path / "catalog/fragment/catalog_b000002.parquet").exists()


def test_condor_transfer_fragments_are_self_contained(tmp_path, monkeypatch):
    pytest.importorskip("htcondor2")
    source_run = tmp_path / "source"
    cfg, jobs, plan, _ = setup_run(source_run)
    groups = [[jobs[i] for i in group] for group in plan.batches]
    prepare_batch_fragments(source_run, cfg, groups)
    scheduler = HTCondor(
        working_dir=str(source_run),
        accounting_group="test",
        should_transfer_files=True,
        job_groups=groups,
    )
    scheduler.create(jobs)
    dag = Path(scheduler.dag_file).read_text()
    assert 'batch_id="b000000"' in dag
    submit = (source_run / "condor" / "pycwb_batch.sub").read_text()
    assert "catalog_$(batch_id).parquet" in submit
    assert "catalog_$(jobs).parquet" not in submit
    assert f"{source_run}/config" in submit
    subprocess.run(["bash", "-n", str(source_run / "condor" / "run.sh")], check=True)
    execute = tmp_path / "execute"
    (execute / "catalog" / "fragment").mkdir(parents=True)
    (execute / "config").mkdir()
    source = execute / "config" / "user_parameters.yaml"
    shutil.copyfile(source_run / "config" / "user_parameters.yaml", source)
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
    parameters = yaml.safe_load(source.read_text())
    parameters["fLow"] = 42
    source.write_text(yaml.safe_dump(parameters))
    with pytest.raises(ValueError, match="Changed settings: fLow"):
        load_batch_run(str(execute), str(source), None, batch_id="b000000")
    source.unlink()
    with pytest.raises(FileNotFoundError):
        load_batch_run(str(execute), str(source), None, batch_id="b000000")


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


@pytest.mark.parametrize("entry", ["prepare", "batch"])
def test_yaml_mismatch_fails_before_writing(tmp_path, monkeypatch, entry):
    from pycwb.workflow.subflow.prepare_job_runs import prepare_job_runs

    _, _, _, source = setup_run(tmp_path)
    original_catalog = (tmp_path / "catalog/catalog.parquet").read_bytes()
    parameters = yaml.safe_load(source.read_text())
    parameters["fLow"] = 42
    source.write_text(yaml.safe_dump(parameters))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Changed settings: fLow.*regenerate"):
        if entry == "prepare":
            prepare_job_runs(str(tmp_path), str(source), overwrite=True)
        else:
            load_batch_run(str(tmp_path), str(source), "10")
    assert (tmp_path / "catalog/catalog.parquet").read_bytes() == original_catalog
    assert not (tmp_path / "catalog/fragment").exists()
    assert not (tmp_path / "output").exists()


def test_batch_uses_yaml_and_allows_cli_overrides(tmp_path, monkeypatch):
    import orjson
    import pyarrow.parquet as pq

    _, _, _, source = setup_run(tmp_path)
    path = tmp_path / "catalog/catalog.parquet"
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata)
    stored = orjson.loads(metadata[b"config"])
    # Runtime metadata may contain overrides; only the YAML snapshot is checked.
    stored["fLow"] = -123
    metadata[b"config"] = orjson.dumps(stored)
    pq.write_table(table.replace_schema_metadata(metadata), path)
    source.write_text("# Formatting-only change\n" + source.read_text())
    monkeypatch.chdir(tmp_path)
    _, cfg, _, _ = load_batch_run(str(tmp_path), str(source), "10", n_proc=7)
    assert cfg.fLow == cfg._yaml_parameters["fLow"]
    assert cfg.nproc == 7
    assert cfg._yaml_parameters["nproc"] == 1
    # A fragment made with CLI overrides remains compatible on resume.
    _, cfg, _, _ = load_batch_run(str(tmp_path), str(source), "10", n_proc=3)
    assert cfg.nproc == 3


def test_changed_fragment_rejected_even_when_root_matches(tmp_path, monkeypatch):
    cfg, jobs, _, source = setup_run(tmp_path)
    cfg._yaml_parameters["fLow"] = 42
    fragment = tmp_path / "catalog/fragment/catalog_10.parquet"
    fragment.parent.mkdir()
    Catalog.create(str(fragment), cfg, jobs[:1], jobs_in_metadata=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="catalog_10.parquet.*Changed settings: fLow"):
        load_batch_run(str(tmp_path), str(source), "10")


@pytest.mark.parametrize("setting,value", [
    ("job_memory", "8GB"), ("job_disk", "12GB"),
    ("job_walltime", "96:00:00"), ("slurm_partition", "long"),
])
def test_submission_yaml_changes_preserve_catalog_and_allow_resume(
    tmp_path, monkeypatch, setting, value,
):
    from pycwb.workflow.subflow.prepare_job_runs import validate_run_config

    _, _, _, source = setup_run(tmp_path)
    path = tmp_path / "catalog/catalog.parquet"
    original = path.read_bytes()
    monkeypatch.chdir(tmp_path)
    # Prepare a fragment with the original submission settings as well.
    load_batch_run(str(tmp_path), str(source), "10")
    parameters = yaml.safe_load(source.read_text())
    parameters[setting] = value
    source.write_text(yaml.safe_dump(parameters))
    validate_run_config(source, tmp_path)
    _, cfg, _, _ = load_batch_run(str(tmp_path), str(source), "10")
    assert getattr(cfg, setting) == value
    assert path.read_bytes() == original


def test_legacy_catalog_requires_regeneration(tmp_path, monkeypatch):
    from pycwb.workflow.subflow.prepare_job_runs import validate_run_config

    cfg, jobs, _, source = setup_run(tmp_path)
    del cfg._yaml_parameters
    path = tmp_path / "catalog/catalog.parquet"
    path.unlink()
    Catalog.create(str(path), cfg, jobs)
    with pytest.raises(ValueError, match="no YAML snapshot.*regenerate"):
        validate_run_config(source, tmp_path)


def test_orphan_manifest_requires_regeneration(tmp_path):
    from pycwb.workflow.subflow.prepare_job_runs import validate_run_config

    _, _, _, source = setup_run(tmp_path)
    (tmp_path / "catalog/catalog.parquet").unlink()
    with pytest.raises(ValueError, match="orphaned Parquet.*regenerate"):
        validate_run_config(source, tmp_path)


def test_staging_replaces_yaml_and_embeds_external_schema(tmp_path, monkeypatch):
    from pycwb.modules.workflow_utils.job_setup import create_output_directory
    from pycwb.constants import user_parameters_schema
    from pycwb.utils.yaml_helper import load_yaml

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "parameters.yaml"
    source.write_text("analysis: 2G\nifo: [H1, L1]\nrefIFO: H1\n")
    work = tmp_path / "run"
    work.mkdir()
    monkeypatch.chdir(work)
    args = (str(work), "output", "log", "catalog", "trigger", str(source))
    create_output_directory(*args)
    original = source.read_text()
    (source_dir / "schema.yaml").write_text("tag: {type: string, default: example}\n")
    source.write_text(original + "pycwb_schema: {schema_file: schema.yaml}\n")
    expected = load_yaml(source, user_parameters_schema)
    create_output_directory(*args)
    staged = work / "config/user_parameters.yaml"
    shutil.rmtree(source_dir)
    assert load_yaml(staged, user_parameters_schema) == expected
    backups = list((work / "config").glob("user_parameters_old_*.yaml"))
    assert len(backups) == 1
    assert backups[0].read_text() == original


@pytest.mark.parametrize("reference", ["relative", "absolute", "parent"])
def test_custom_detector_files_survive_staging_and_transfer(tmp_path, monkeypatch, reference):
    import importlib

    preparation = importlib.import_module("pycwb.workflow.subflow.prepare_job_runs")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    definitions = inputs / "detectors.json"
    document = {"schema_version": 1, "geometries": {
        "X1:custom": {"detector": "X1", "parameters": {
            "name": "Example", "lat": 0.0, "lon": 0.0, "elevation": 0.0,
            "x": {"az": 0.0, "alt": 0.0, "midpoint": 2000.0},
            "y": {"az": 1.5707963267948966, "alt": 0.0, "midpoint": 2000.0},
        }}
    }}
    definitions.write_text(json.dumps(document))
    source = inputs / "parameters.yaml"
    definitions_ref = "detectors.json"
    if reference == "absolute":
        definitions_ref = str(definitions)
    elif reference == "parent":
        (inputs / "nested").mkdir()
        source = inputs / "nested/parameters.yaml"
        definitions_ref = "../detectors.json"
    source.write_text(yaml.safe_dump({
        "analysis": "2G", "ifo": ["H1", "X1"], "refIFO": "H1",
        "detector_definitions_file": definitions_ref,
        "detector_geometry": {"X1": "X1:custom"},
    }))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(preparation, "create_job_segment_from_config", lambda cfg: [job(10, "a.gwf")])
    work = tmp_path / "run"
    _, original_cfg, _ = preparation.prepare_job_runs(str(work), str(source))
    staged = work / "config/user_parameters.yaml"
    staged_params = yaml.safe_load(staged.read_text())
    assert not Path(staged_params["detector_definitions_file"]).is_absolute()
    assert (staged.parent / staged_params["detector_definitions_file"]).read_bytes() == definitions.read_bytes()
    # Preparing again from the original YAML compares the same content identity.
    preparation.prepare_job_runs(str(work), str(source), overwrite=True)
    assert not list(staged.parent.glob("user_parameters_old_*.yaml"))
    _, _, _, fragment = load_batch_run(str(work), str(staged), "10")

    execute = tmp_path / "execute"
    shutil.copytree(work / "config", execute / "config")
    (execute / "catalog/fragment").mkdir(parents=True)
    shutil.copyfile(fragment, execute / "catalog/fragment/catalog_10.parquet")
    shutil.rmtree(inputs)
    shutil.rmtree(work)
    transferred = execute / "config/user_parameters.yaml"
    selected, cfg, _, _ = load_batch_run(str(execute), str(transferred), "10")
    assert [segment.index for segment in selected] == [10]
    assert cfg.get_detector("X1").geometry_model == "custom"
    assert cfg._yaml_parameters == original_cfg._yaml_parameters
    assert cfg.detector_definitions_provenance["sha256"] == original_cfg.detector_definitions_provenance["sha256"]

    # Editing the transferred dependency must fail even when YAML is unchanged.
    document["geometries"]["X1:custom"]["parameters"]["lat"] = 0.1
    (transferred.parent / cfg.detector_definitions_file).write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Changed settings: detector_definitions_file"):
        load_batch_run(str(execute), str(transferred), "10")


@pytest.mark.parametrize("entry", ["prepare", "batch"])
@pytest.mark.parametrize("field", ["execution_profile", "gpu"])
def test_recorded_effective_profile_guard_is_preserved(tmp_path, monkeypatch, entry, field):
    import orjson
    import pyarrow.parquet as pq
    from pycwb.workflow.subflow.prepare_job_runs import prepare_job_runs

    _, _, _, source = setup_run(tmp_path)
    path = tmp_path / "catalog/catalog.parquet"
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata)
    stored = orjson.loads(metadata[b"config"])
    # YAML still matches its snapshot; the effective recorded settings do not.
    if field == "gpu":
        stored[field]["likelihood"] = True
    else:
        stored[field]["sky_delay_reuse"] = not stored[field]["sky_delay_reuse"]
    metadata[b"config"] = orjson.dumps(stored)
    pq.write_table(table.replace_schema_metadata(metadata), path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="differ.*existing catalog"):
        if entry == "prepare":
            # The effective-profile guard follows segment construction.
            import importlib
            module = importlib.import_module("pycwb.workflow.subflow.prepare_job_runs")
            monkeypatch.setattr(module, "create_job_segment_from_config", lambda cfg: [])
            prepare_job_runs(str(tmp_path), str(source), overwrite=True)
        else:
            load_batch_run(str(tmp_path), str(source), "10")
    assert not (tmp_path / "output").exists()
