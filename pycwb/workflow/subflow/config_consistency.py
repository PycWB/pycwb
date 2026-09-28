"""Reject reuse of prepared jobs with a different YAML configuration."""

from pathlib import Path

import orjson
import pyarrow.parquet as pq

from pycwb.constants import user_parameters_schema
from pycwb.config.provenance import snapshot_yaml_parameters
from pycwb.utils.yaml_helper import load_yaml


def _validate_catalog_configs(config_file, catalog_files, parameters):
    """Compare parsed YAML (including defaults) before loading jobs or writing.

    Use only Parquet schema metadata: inspecting an existing run must not load
    its potentially large job manifest. CLI overrides and derived paths are
    intentionally absent from the stored YAML snapshot.
    """
    for path in catalog_files:
        metadata = pq.read_schema(path).metadata or {}
        stored = orjson.loads(metadata.get(b"config", b"{}"))
        snapshot = stored.get("_yaml_parameters")
        recovery = (
            "Use a new working directory, or clean the existing catalog, job "
            "manifest, progress and fragment Parquet files and regenerate the "
            "run with the YAML file. --overwrite does not bypass this check."
        )
        if snapshot is None:
            raise ValueError(
                f"Cannot verify YAML configuration {config_file} against {path}: "
                f"the Parquet metadata has no YAML snapshot. {recovery}"
            )
        if snapshot != parameters:
            changed = sorted(
                key for key in snapshot.keys() | parameters.keys()
                if key not in snapshot or key not in parameters
                or snapshot[key] != parameters[key]
            )
            raise ValueError(
                f"YAML configuration {config_file} does not match {path}. "
                f"Changed settings: {', '.join(changed)}. {recovery}"
            )


def validate_run_config(config_file, working_dir, *, fragment_id=None):
    """Check a run, or just its root and selected worker fragment.

    Preparation checks every catalog. Workers inspect only their own fragment
    and the root, keeping startup bounded for large batch submissions.
    """
    # The default directory is also checked when YAML changes catalog_dir.
    parameters = load_yaml(config_file, user_parameters_schema)
    parameters = snapshot_yaml_parameters(parameters, config_file)
    # Match the JSON representation used by the catalog serializer.
    parameters = orjson.loads(orjson.dumps(parameters))
    directories = {Path(working_dir) / "catalog",
                   Path(working_dir) / parameters.get("catalog_dir", "catalog")}
    for directory in sorted(directories):
        if fragment_id is None:
            parquet_files = sorted(directory.rglob("*.parquet"))
        else:
            parquet_files = [path for path in (
                directory / "catalog.parquet", directory / "jobs.parquet",
                directory / "progress.parquet",
                directory / "fragment" / f"catalog_{fragment_id}.parquet",
                directory / "fragment" / f"progress_{fragment_id}.parquet",
            ) if path.exists()]
        catalogs = [path for path in parquet_files if path.name.startswith("catalog")]
        if parquet_files and not catalogs:
            raise ValueError(
                f"Cannot verify YAML configuration: orphaned Parquet files in {directory}. "
                "Use a new working directory, or clean these files and regenerate the run."
            )
        _validate_catalog_configs(config_file, catalogs, parameters)
