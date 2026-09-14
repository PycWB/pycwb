"""Regression tests for master job manifests and inline batch fragments."""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pycwb.config import Config
from pycwb.modules.catalog.catalog import Catalog, JOB_MANIFEST_FILENAME
from pycwb.types.job import WaveSegment
from pycwb.workflow.merge import merge_catalog


def _job(index: int, *, sim_idx: int | None = None) -> WaveSegment:
    injections = (
        None
        if sim_idx is None
        else [
            {
                "sim_idx": sim_idx,
                "gps_time": 1_400_000_000.0 + sim_idx,
                "name": "WNB",
                "nested": {"seed": sim_idx},
            }
        ]
    )
    return WaveSegment(
        index=index,
        ifos=["H1", "L1"],
        analyze_start=1_400_000_000.0 + index,
        analyze_end=1_400_000_100.0 + index,
        sample_rate=4096.0,
        seg_edge=8.0,
        injections=injections,
    )


def _expected(jobs: list[WaveSegment]) -> list[dict]:
    return [dataclasses.asdict(job) for job in jobs]


def test_master_catalog_uses_run_level_job_manifest(tmp_path):
    jobs = [_job(1, sim_idx=11), _job(2, sim_idx=12)]
    catalog_path = tmp_path / "catalog.parquet"

    catalog = Catalog.create(str(catalog_path), Config(), jobs)

    metadata = pq.read_schema(catalog_path).metadata or {}
    assert b"jobs" not in metadata
    manifest_path = tmp_path / JOB_MANIFEST_FILENAME
    assert manifest_path.is_file()
    assert catalog.jobs == _expected(jobs)

    manifest = pq.read_table(manifest_path)
    assert manifest.schema.names == ["job_index", "job_json"]
    assert manifest["job_index"].to_pylist() == [1, 2]
    assert (manifest.schema.metadata or {})[b"pycwb_job_manifest_version"] == b"1"


def test_inline_jobs_are_read_without_a_manifest(tmp_path):
    jobs = [_job(7, sim_idx=21)]
    fragment = tmp_path / "catalog_7-7.parquet"

    catalog = Catalog.create(str(fragment), Config(), jobs, jobs_in_metadata=True)

    assert not (tmp_path / JOB_MANIFEST_FILENAME).exists()
    metadata = pq.read_schema(fragment).metadata or {}
    assert b"jobs" in metadata
    assert catalog.jobs == _expected(jobs)


def test_inline_jobs_take_precedence_over_run_manifest(tmp_path):
    inline_jobs = [_job(1, sim_idx=31)]
    manifest_jobs = [_job(2, sim_idx=32)]
    inline_path = tmp_path / "fragment.parquet"
    master_path = tmp_path / "catalog.parquet"

    inline_catalog = Catalog.create(
        str(inline_path), Config(), inline_jobs, jobs_in_metadata=True
    )
    Catalog.create(str(master_path), Config(), manifest_jobs)

    assert inline_catalog.jobs == _expected(inline_jobs)


def test_plain_parquet_does_not_require_or_guess_a_manifest(tmp_path):
    Catalog.create(str(tmp_path / "master.parquet"), Config(), [_job(1)])
    path = tmp_path / "plain.parquet"
    pq.write_table(pa.table({"job_id": [7]}), path)
    assert Catalog.open(str(path)).jobs == []


def test_explicit_missing_manifest_raises(tmp_path):
    path = tmp_path / "catalog.parquet"
    Catalog.create(str(path), Config(), [_job(1)])
    (tmp_path / JOB_MANIFEST_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match="job manifest is missing"):
        _ = Catalog.open(str(path)).jobs


def test_manifest_is_immutable_and_can_be_reused(tmp_path):
    jobs = [_job(1, sim_idx=41)]
    Catalog.create(str(tmp_path / "catalog.parquet"), Config(), jobs)
    Catalog.create(str(tmp_path / "catalog.M1.parquet"), Config(), jobs)

    with pytest.raises(ValueError, match="conflicts"):
        Catalog.create(
            str(tmp_path / "catalog.M2.parquet"), Config(), [_job(2, sim_idx=42)]
        )


def test_standard_merge_preserves_master_manifest(tmp_path):
    catalog_dir = tmp_path / "catalog"
    fragment_dir = catalog_dir / "fragment"
    fragment_dir.mkdir(parents=True)
    jobs = [_job(1, sim_idx=51), _job(2, sim_idx=52)]

    Catalog.create(str(catalog_dir / "catalog.parquet"), Config(), jobs)
    Catalog.create(
        str(fragment_dir / "catalog_1-1.parquet"),
        Config(),
        [jobs[0]],
        jobs_in_metadata=True,
    )
    Catalog.create(
        str(fragment_dir / "catalog_2-2.parquet"),
        Config(),
        [jobs[1]],
        jobs_in_metadata=True,
    )
    manifest_path = catalog_dir / JOB_MANIFEST_FILENAME
    before = manifest_path.read_bytes()

    merge_catalog(working_dir=str(tmp_path))

    assert manifest_path.read_bytes() == before
    assert Catalog.open(str(catalog_dir / "catalog.parquet")).jobs == _expected(jobs)


@pytest.mark.parametrize("index", [True, 1.5, "1", None])
def test_manifest_rejects_non_integer_indices(tmp_path, index):
    with pytest.raises(ValueError, match="integer index"):
        Catalog.create(str(tmp_path / "catalog.parquet"), Config(), [{"index": index}])


def test_manifest_reuse_ignores_dictionary_key_order(tmp_path):
    Catalog.create(
        str(tmp_path / "a.parquet"), Config(), [{"index": 1, "shift": [0, 0]}]
    )
    Catalog.create(
        str(tmp_path / "b.parquet"), Config(), [{"shift": [0, 0], "index": 1}]
    )


def test_manifest_rejects_wrong_identity(tmp_path):
    import shutil

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    Catalog.create(str(first / "catalog.parquet"), Config(), [_job(1)])
    Catalog.create(str(second / "catalog.parquet"), Config(), [_job(1)])
    shutil.copyfile(second / JOB_MANIFEST_FILENAME, first / JOB_MANIFEST_FILENAME)
    with pytest.raises(ValueError, match="identity"):
        _ = Catalog.open(str(first / "catalog.parquet")).jobs


def test_manifest_rejects_wrong_column_types(tmp_path):
    path = tmp_path / "catalog.parquet"
    Catalog.create(str(path), Config(), [_job(1)])
    manifest = tmp_path / JOB_MANIFEST_FILENAME
    table = pq.read_table(manifest)
    table = table.set_column(0, "job_index", pa.array([1.0]))
    pq.write_table(table, manifest)
    with pytest.raises(ValueError, match="schema"):
        _ = Catalog.open(str(path)).jobs


@pytest.mark.parametrize("keep_master", [False, True])
def test_partial_labeled_merge_reuses_full_run_manifest(tmp_path, keep_master):
    directory = tmp_path / "custom_catalog"
    fragments = directory / "fragment"
    fragments.mkdir(parents=True)
    jobs = [_job(1), _job(2)]
    Catalog.create(str(directory / "catalog.parquet"), Config(), jobs)
    Catalog.create(
        str(fragments / "catalog_1.parquet"), Config(), jobs[:1], jobs_in_metadata=True
    )
    before = (directory / JOB_MANIFEST_FILENAME).read_bytes()
    if not keep_master:
        (directory / "catalog.parquet").unlink()
    merge_catalog(str(tmp_path), catalog_dir="custom_catalog", merge_label="partial")
    assert Catalog.open(str(directory / "catalog.partial.parquet")).jobs == _expected(
        jobs
    )
    assert (directory / JOB_MANIFEST_FILENAME).read_bytes() == before


def test_conflicting_merge_preserves_existing_output(tmp_path, monkeypatch):
    directory = tmp_path / "catalog"
    fragments = directory / "fragment"
    fragments.mkdir(parents=True)
    Catalog.create(str(directory / "catalog.parquet"), Config(), [_job(1)])
    output = directory / "catalog.partial.parquet"
    Catalog.create(str(output), Config(), [_job(1)])
    before = output.read_bytes()
    Catalog.create(
        str(fragments / "catalog_1.parquet"),
        Config(),
        [_job(1, sim_idx=99)],
        jobs_in_metadata=True,
    )
    monkeypatch.setattr("pycwb.workflow.merge.click.confirm", lambda *a, **k: True)
    with pytest.raises(ValueError, match="conflicts"):
        merge_catalog(str(tmp_path), merge_label="partial")
    assert output.read_bytes() == before


def test_merge_without_master_deduplicates_identical_fragment_jobs(tmp_path):
    directory = tmp_path / "catalog"
    fragments = directory / "fragment"
    fragments.mkdir(parents=True)
    for name in ["catalog_1.parquet", "catalog_1_retry.parquet"]:
        Catalog.create(
            str(fragments / name), Config(), [_job(1)], jobs_in_metadata=True
        )
    merge_catalog(str(tmp_path))
    assert Catalog.open(str(directory / "catalog.parquet")).jobs == _expected([_job(1)])


@pytest.mark.parametrize(
    "payload",
    [b"not-json", b"{}", b'{"version":2,"path":"jobs.parquet","manifest_id":"x"}'],
)
def test_invalid_manifest_reference_is_not_treated_as_plain_parquet(tmp_path, payload):
    from pycwb.modules.catalog.provenance import MANIFEST_KEY

    path = tmp_path / "catalog.parquet"
    pq.write_table(
        pa.table({"job_id": [1]}).replace_schema_metadata({MANIFEST_KEY: payload}), path
    )
    with pytest.raises(ValueError, match="manifest reference"):
        _ = Catalog.open(str(path)).jobs


def test_conflicting_fragment_duplicates_are_rejected(tmp_path):
    fragments = tmp_path / "catalog/fragment"
    fragments.mkdir(parents=True)
    for name, job in [
        ("catalog_1.parquet", _job(1)),
        ("catalog_1_retry.parquet", _job(1, sim_idx=7)),
    ]:
        Catalog.create(str(fragments / name), Config(), [job], jobs_in_metadata=True)
    with pytest.raises(ValueError, match="Conflicting fragment"):
        merge_catalog(str(tmp_path))
    assert not (tmp_path / "catalog/catalog.parquet").exists()
