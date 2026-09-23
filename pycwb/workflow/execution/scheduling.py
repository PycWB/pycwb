"""Stable batch membership for shared-filesystem scheduler workers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def validate_batch_id(batch_id: str) -> int:
    """Validate a stable selector and return its zero-based batch index."""
    if not re.fullmatch(r"b\d{6}", batch_id):
        raise ValueError("batch_id must have the form b000000")
    return int(batch_id[1:])


def batch_job_ids(working_dir: str | Path, batch_id: str) -> list[int]:
    """Resolve explicit scientific IDs from an integrity-checked root plan."""
    index = validate_batch_id(batch_id)
    document = json.loads((Path(working_dir) / "execution-plan.json").read_text())
    identity = document.pop("identity")
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if hashlib.sha256(encoded.encode()).hexdigest() != identity:
        raise ValueError("Execution plan identity mismatch")
    plan = document["plan"]
    if plan["version"] != 1 or index >= len(plan["batches"]):
        raise ValueError(f"Unknown or unsupported execution batch {batch_id}")
    return [plan["job_ids"][task] for task in plan["batches"][index]]
