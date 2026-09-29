"""Scheduling and resource policy for the YAML ``execution`` block.

This module defines schema, defaults and validation only. Job planning and
worker admission live in ``pycwb.workflow.execution``; numerical processing
choices live in ``pycwb.config.processing``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from pycwb.utils.size import byte_size

EXECUTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "default": {},
    "properties": {
        "profile": {"enum": ["simple", "scalable"]},
        "planner": {"type": ["string", "null"]},
        "executor": {"type": ["string", "null"]},
        "preload": {"enum": ["off", "auto", "batch"]},
        **{
            name: {
                "type": ["string", "integer", "null"]
                if name == "memory_limit"
                else ["string", "integer"]
            }
            for name in (
                "memory_limit",
                "cache_limit",
                "worker_memory",
                "headroom",
                "message_limit",
            )
        },
        "batch_size": {"type": "integer", "minimum": 1},
        "cache_entries": {"type": "integer", "minimum": 1},
        "cores": {"type": ["integer", "null"], "minimum": 1},
    },
}


@dataclass(frozen=True)
class ExecutionSettings:
    """Validated execution policy, with legacy behavior as the default."""

    profile: str = "simple"
    planner: str | None = None
    executor: str | None = None
    preload: str = "auto"
    memory_limit: int | None = None
    cache_limit: int = 1024**3
    worker_memory: int = 6 * 1024**3
    headroom: int = 512 * 1024**2
    message_limit: int = 64 * 1024**2
    batch_size: int = 8
    cache_entries: int = 256
    cores: int | None = None

    @property
    def enabled(self) -> bool:
        """Whether an execution profile or explicit extension needs dispatch."""
        return self.profile != "simple" or self.planner is not None or self.executor is not None

    @classmethod
    def from_config(cls, config: Any) -> ExecutionSettings:
        """Normalize YAML/catalog settings and reject unsupported values."""
        raw = getattr(config, "execution", None)
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise TypeError("execution must be a mapping")
        unknown = raw.keys() - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown execution settings: {sorted(unknown)}")
        values = dict(raw)
        for key in (
            "memory_limit",
            "cache_limit",
            "worker_memory",
            "headroom",
            "message_limit",
        ):
            if key in values:
                if values[key] is None and key == "memory_limit":
                    continue
                values[key] = byte_size(values[key])
        result = cls(**values)
        if result.profile not in {"simple", "scalable"} or result.preload not in {
            "off",
            "auto",
            "batch",
        }:
            raise ValueError("Invalid execution profile or preload policy")
        for key in ("batch_size", "cache_entries", "cores"):
            value = getattr(result, key)
            if key == "cores" and value is None:
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"execution.{key} must be a positive integer")
        for key in ("worker_memory", "message_limit", "memory_limit"):
            value = getattr(result, key)
            if value is not None and value <= 0:
                raise ValueError(f"execution.{key} must be positive")
        for key in ("planner", "executor"):
            value = getattr(result, key)
            if value is not None and (not isinstance(value, str) or "." not in value):
                raise ValueError(f"execution.{key} must be a dotted factory path")
        return result
