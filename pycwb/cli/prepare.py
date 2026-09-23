"""Inspect and persist an execution plan without reading strain or running jobs."""

import argparse
import json
from pathlib import Path


def init_parser(parser: argparse.ArgumentParser) -> None:
    """Register metadata preparation and plan-output options."""
    parser.add_argument("user_parameter_file")
    parser.add_argument("--work-dir", "-d", default=".")
    parser.add_argument(
        "--output", help="Plan filename (default: WORK_DIR/execution-plan.json)"
    )
    parser.add_argument("--config-vars")
    parser.add_argument("--input-dir")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Prepare metadata only (always enabled)",
    )


def command(args: argparse.Namespace) -> None:
    """Prepare job metadata and publish a validated execution plan."""
    from pycwb.workflow.execution.planner import prepare_plan, write_document
    from pycwb.workflow.execution.settings import ExecutionSettings
    from pycwb.workflow.subflow.prepare_job_runs import prepare_job_runs

    output = Path(args.output).resolve() if args.output else None
    jobs, config, directory = prepare_job_runs(
        args.work_dir,
        args.user_parameter_file,
        dry_run=True,
        config_vars=args.config_vars,
        input_dir=args.input_dir,
    )
    settings = ExecutionSettings.from_config(config)
    plan = prepare_plan(jobs, config, settings)
    path = output or Path(directory) / "execution-plan.json"
    write_document(path, plan.document(jobs, settings))
    print(
        json.dumps(
            {
                "plan": str(path),
                "jobs": len(jobs),
                "batches": len(plan.batches),
                "frame_references": sum(map(len, plan.requests)),
                "unique_frame_channels": len(
                    {r.key for group in plan.requests for r in group}
                ),
                "profile": settings.profile,
            },
            indent=2,
        )
    )
