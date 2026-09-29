"""Persist stable batch membership in self-contained catalog fragments."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re


def validate_batch_id(batch_id: str) -> int:
    """Validate a stable selector and return its zero-based batch index."""
    if not re.fullmatch(r"b\d{6}", batch_id):
        raise ValueError("batch_id must have the form b000000")
    return int(batch_id[1:])


def prepare_batch_fragments(working_dir, config, job_groups) -> None:
    """Create batch catalogs, preserving existing results only for identical jobs.

    Check every existing batch before publishing new fragments so regrouping
    cannot silently reuse results belonging to a different selection.
    """
    import orjson

    from pycwb.modules.catalog import Catalog, read_catalog_metadata

    directory = Path(working_dir) / config.catalog_dir / "fragment"
    fragments = {
        directory / f"catalog_b{index:06d}.parquet": group
        for index, group in enumerate(job_groups)
    }
    for path in directory.glob("catalog_b*.parquet"):
        group = fragments.get(path)
        expected = (
            None if group is None else orjson.loads(orjson.dumps(
                [asdict(job) for job in group], option=orjson.OPT_SERIALIZE_NUMPY
            ))
        )
        if expected is None or read_catalog_metadata(str(path))["jobs"] != expected:
            raise ValueError(
                f"Prepared batch jobs differ from {path}. "
                "Use a new working directory to change batch membership or job definitions."
            )
    directory.mkdir(parents=True, exist_ok=True)
    for path, group in fragments.items():
        if not path.exists():
            Catalog.create(str(path), config, group, jobs_in_metadata=True)
