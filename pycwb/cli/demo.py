"""A packaged injection example with an explicit recovery check."""

from argparse import ArgumentParser, Namespace
from typing import Any

import json
import os
import sys
import time
from pathlib import Path


def init_parser(parser: ArgumentParser) -> None:
    """Register creation, execution and result checking."""
    parser.add_argument(
        "directory",
        type=Path,
        help="New demo directory (existing directory for --check)",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--run", action="store_true", help="Create, run and check the demo"
    )
    action.add_argument(
        "--check",
        action="store_true",
        help="Check an existing demo's completed results",
    )
    parser.add_argument(
        "--xtalk", type=Path, help="Reuse a local OverlapCatalog16-1024.bin or .npz"
    )


def create_demo(directory: str | Path, xtalk: str | Path | None = None) -> Path:
    """Create a new project; never overwrite an existing directory."""
    from importlib.resources import files

    import yaml

    template = (
        files("pycwb").joinpath("vendor/template/demo/user_parameters.yaml").read_text()
    )
    if xtalk is not None:
        xtalk = Path(xtalk).resolve(strict=True)
        if not xtalk.is_file():
            raise ValueError(f"Cross-talk catalog is not a file: {xtalk}")
        config = yaml.safe_load(template)
        config.update(filter_dir=str(xtalk.parent), wdmXTalk=xtalk.name)
        template = yaml.safe_dump(config, sort_keys=False)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config_path = directory / "user_parameters.yaml"
    config_path.write_text(template)
    return config_path


def check_demo(directory: str | Path) -> dict[str, Any]:
    """Require a completed zero-lag trial and a finite event near injection time.

    The one-second tolerance covers detector delays and reconstruction timing.
    This smoke check is not a significance test or a numerical parity test.
    """
    import math

    import pyarrow.parquet as pq
    import yaml

    from pycwb.modules.catalog import Catalog

    directory = Path(directory).resolve()
    config = yaml.safe_load((directory / "user_parameters.yaml").read_text())
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
            "Expected one completed zero-lag demo job; inspect log/demo.log and pycwb progress"
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
            "No finite trigger recovered within 1 s of the demo injection; inspect log/demo.log"
        )
    return {
        "ok": True,
        "injection_gps": target,
        "recovered_triggers": len(recovered),
        "total_triggers": len(rows),
        "timing_tolerance_seconds": 1.0,
        "catalog": str(directory / "catalog/catalog.parquet"),
    }


def command(args: Namespace) -> int:
    """Create or check a demo, returning a failing exit status on failed recovery."""
    original_directory = Path.cwd()
    try:
        if args.check:
            print(json.dumps(check_demo(args.directory), indent=2))
            return 0
        config_path = create_demo(args.directory, args.xtalk)
        print(f"Created {config_path}")
        if not args.run:
            print("Next: cd into the demo directory, then run:")
            print("  pycwb validate user_parameters.yaml")
            print("  pycwb run user_parameters.yaml")
            print("  pycwb demo . --check")
            return 0
        from pycwb.workflow.run import search

        log_dir = config_path.parent / "log"
        log_dir.mkdir()
        started = time.monotonic()
        search(
            str(config_path),
            working_dir=str(config_path.parent),
            n_proc=1,
            log_file=str(log_dir / "demo.log"),
        )
        report = check_demo(config_path.parent)
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        (config_path.parent / "demo-result.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError) as error:
        print(f"Demo failed: {error}", file=sys.stderr)
        return 1
    finally:
        os.chdir(original_directory)
