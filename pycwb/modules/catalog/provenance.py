"""Preserve run provenance when a single-run trigger table is written elsewhere.

Manifest paths are relative to the catalog containing the reference.  Derived
catalogs share the full run manifest; selection membership and exposure remain
in the selection's job IDs, progress, and interval products.
"""

from __future__ import annotations

import os

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

MANIFEST_KEY = b"pycwb_jobs_manifest"
MANIFEST_ID_KEY = b"pycwb_job_manifest_id"
PROVENANCE_KEYS = (b"pycwb_version", b"config", b"jobs", MANIFEST_KEY)


def resolve_job_manifest(filename: str, payload: bytes) -> tuple[str, str]:
    """Validate a reference and return its absolute path and expected identity."""
    try:
        reference = orjson.loads(payload)
    except orjson.JSONDecodeError as exc:
        raise ValueError(f"Invalid job manifest reference in {filename}") from exc
    if (
        not isinstance(reference, dict)
        or type(reference.get("version")) is not int
        or reference["version"] != 1
        or not isinstance(reference.get("path"), str)
        or not reference["path"]
        or os.path.isabs(reference["path"])
        or not isinstance(reference.get("manifest_id"), str)
        or not reference["manifest_id"]
    ):
        raise ValueError(f"Invalid or unsupported job manifest reference in {filename}")
    path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(filename)), reference["path"])
    )
    return path, reference["manifest_id"]


def manifest_reference(filename: str, manifest_path: str, manifest_id: str) -> bytes:
    """Encode a manifest reference relative to the destination catalog."""
    return orjson.dumps(
        {
            "version": 1,
            "path": os.path.relpath(
                manifest_path, os.path.dirname(os.path.abspath(filename))
            ),
            "manifest_id": manifest_id,
        }
    )


def catalog_provenance(source: str, destination: str) -> dict[bytes, bytes]:
    """Read only schema metadata and rebase its job reference for *destination*."""
    source_metadata = pq.read_schema(source).metadata or {}
    metadata = {
        key: source_metadata[key] for key in PROVENANCE_KEYS if key in source_metadata
    }
    if b"jobs" in metadata:
        # Inline metadata takes precedence for legacy catalogs and fragments.
        metadata.pop(MANIFEST_KEY, None)
    elif MANIFEST_KEY in metadata:
        path, manifest_id = resolve_job_manifest(source, metadata[MANIFEST_KEY])
        metadata[MANIFEST_KEY] = manifest_reference(destination, path, manifest_id)
    return metadata


def with_catalog_provenance(table: pa.Table, source: str, destination: str) -> pa.Table:
    """Attach source provenance without replacing the output's pandas metadata."""
    metadata = {
        key: value
        for key, value in (table.schema.metadata or {}).items()
        if key not in PROVENANCE_KEYS
    }
    metadata.update(catalog_provenance(source, destination))
    return table.replace_schema_metadata(metadata)


def write_catalog_dataframe(frame, destination: str, source: str) -> None:
    """Write a derived trigger DataFrame with its source catalog provenance."""
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = with_catalog_provenance(table, source, destination)
    pq.write_table(table, destination, compression="snappy")
