"""Test-only fixtures and recovery assertions for the CLI synthetic example."""

from pathlib import Path
from typing import Any
import shutil


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/demo/user_parameters.yaml"


def copy_example(directory: Path) -> Path:
    """Copy the unmodified YAML into an isolated test directory."""
    directory.mkdir(parents=True)
    return Path(shutil.copyfile(EXAMPLE, directory / "user_parameters.yaml"))


def check_recovery(directory: str | Path, config_path: str | Path) -> dict[str, Any]:
    """Require a completed zero-lag trial and a finite event near injection time.

    The one-second tolerance covers detector delays and reconstruction timing.
    This smoke check is not a significance test or a numerical parity test.
    """
    import math

    import pyarrow.parquet as pq
    import yaml

    from pycwb.modules.catalog import Catalog

    directory = Path(directory).resolve()
    config = yaml.safe_load(Path(config_path).read_text())
    target = float(config["injection"]["parameters"]["gps_time"])
    catalog = Catalog.open(str(directory / "catalog/catalog.parquet"))
    # Resolve and verify the companion job manifest, even when no rows exist.
    jobs = catalog.jobs
    progress = pq.read_table(directory / "catalog/progress.parquet").to_pylist()
    completed = [
        row
        for row in progress
        if row["status"] == "completed"
        and row["lag_idx"] == 0
        and row["trial_idx"] == 0
    ]
    if (
        len(jobs) != 1
        or len(progress) != 1
        or len(completed) != 1
        or completed[0]["job_id"] != jobs[0]["index"]
    ):
        raise ValueError(
            "Expected one completed zero-lag synthetic job; inspect the CLI output and pycwb progress"
        )
    rows = pq.read_table(directory / "catalog/catalog.parquet").to_pylist()
    recovered = []
    for row in rows:
        times = [row.get(f"time_{ifo}") for ifo in config["ifo"]]
        rho = row.get("rho")
        if (
            row.get("job_id") == jobs[0]["index"]
            and row.get("trial_idx") == 0
            and row.get("lag_idx") == 0
            and rho is not None
            and math.isfinite(rho)
            and rho >= abs(config["netRHO"])
            and all(
                t is not None and math.isfinite(t) and abs(t - target) <= 1.0
                for t in times
            )
        ):
            recovered.append(row)
    if not recovered:
        raise ValueError(
            "No finite trigger recovered within 1 s of the synthetic injection; inspect the CLI output"
        )
    return {
        "ok": True,
        "injection_gps": target,
        "recovered_triggers": len(recovered),
        "total_triggers": len(rows),
        "timing_tolerance_seconds": 1.0,
        "catalog": str(directory / "catalog/catalog.parquet"),
    }
