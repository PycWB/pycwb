"""Helpers for post-production workflow configuration files.

The workflow runner originally accepted a compact YAML shape:

``global`` plus a list of steps with ``action``, ``args`` and optional
``output_alias``.  Newer workflows can also use ``vars``, explicit
``inputs`` / ``outputs`` blocks, ``@step.path`` references, and ``tmp://``
paths.  This module keeps those conveniences in one place so execution and
diagram generation agree on the same normalized view of the workflow.
"""

from __future__ import annotations

import copy
import os
import re
from typing import Any

import yaml


_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def load_workflow(workflow_file: str) -> dict:
    """Load a workflow YAML file."""
    with open(workflow_file, "r") as f:
        return yaml.safe_load(f) or {}


def workflow_base_context(workflow: dict) -> dict:
    """Return the initial workflow context.

    ``global`` remains the backwards-compatible top-level context.  ``vars``
    is merged on top so new workflows can place reusable values there while
    still exposing simple values such as ``work_dir`` to action kwargs.
    """
    context: dict[str, Any] = {}
    context.update(copy.deepcopy(workflow.get("global", {}) or {}))
    context = _deep_merge(context, copy.deepcopy(workflow.get("vars", {}) or {}))
    return context


def workflow_runtime(workflow: dict, context: dict) -> dict:
    """Return normalized runtime settings."""
    runtime = copy.deepcopy(workflow.get("runtime", {}) or {})
    work_dir = context.get("work_dir", ".")
    runtime.setdefault("tmp_dir", os.path.join(str(work_dir), "tmp", "postprod"))
    runtime.setdefault("cleanup_tmp", "never")
    runtime = resolve_value(runtime, context, runtime, resolve_refs=False)
    tmp_dir = str(runtime.get("tmp_dir", ""))
    if tmp_dir and not os.path.isabs(tmp_dir):
        runtime["tmp_dir"] = os.path.abspath(tmp_dir)
    return runtime


def prepare_step_args(step: dict, context: dict, runtime: dict) -> dict:
    """Build the kwargs passed to an action.

    The call kwargs contain the current context, then explicit ``inputs``,
    ``args`` and ``outputs`` from the step.  Step-level values override
    context values.  For nested ``outputs`` blocks, the full block is also
    available as the ``outputs`` kwarg.
    """
    call_args = copy.deepcopy(context)
    inputs = resolve_value(step.get("inputs", {}) or {}, context, runtime)
    args = resolve_value(step.get("args", {}) or {}, context, runtime)
    outputs = resolve_value(step.get("outputs", {}) or {}, context, runtime)

    call_args.update(inputs)
    call_args.update(args)
    if outputs:
        call_args["outputs"] = outputs
        if _is_flat_mapping(outputs):
            call_args.update(outputs)
    return call_args


def prepare_step_for_diagram(step: dict) -> dict:
    """Return the unexecuted data-flow view of a workflow step."""
    args: dict[str, Any] = {}
    args.update(copy.deepcopy(step.get("inputs", {}) or {}))
    args.update(copy.deepcopy(step.get("args", {}) or {}))
    outputs = copy.deepcopy(step.get("outputs", {}) or {})
    if outputs:
        args["outputs"] = outputs
        if _is_flat_mapping(outputs):
            args.update(outputs)
    return args


def resolve_value(value: Any, context: dict, runtime: dict | None = None, *, resolve_refs: bool = True) -> Any:
    """Resolve variables, refs and temporary paths in *value* recursively."""
    if isinstance(value, dict):
        return {
            k: resolve_value(v, context, runtime, resolve_refs=resolve_refs)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            resolve_value(v, context, runtime, resolve_refs=resolve_refs)
            for v in value
        ]
    if not isinstance(value, str):
        return value

    text = _expand_vars(value, context)
    if text.startswith("tmp://"):
        tmp_dir = (runtime or {}).get("tmp_dir", os.path.join(str(context.get("work_dir", ".")), "tmp", "postprod"))
        tmp_dir = _expand_vars(str(tmp_dir), context)
        rel = text[len("tmp://"):].lstrip("/")
        return os.path.join(tmp_dir, rel)
    if resolve_refs and text.startswith("@"):
        return resolve_reference(text, context)
    return text


def resolve_reference(ref: str, context: dict) -> Any:
    """Resolve an ``@step.path`` reference against the workflow context."""
    path = ref[1:]
    if not path:
        raise KeyError("Empty workflow reference '@'")
    return get_path(context, path)


def get_path(data: Any, path: str) -> Any:
    """Read a dotted path from nested dictionaries/lists."""
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur[part]
        elif isinstance(cur, (list, tuple)):
            cur = cur[int(part)]
        else:
            raise KeyError(f"Cannot resolve '{path}' through non-container value")
    return cur


def store_result(context: dict, step: dict, result: Any) -> None:
    """Store an action result in the context.

    Backwards compatibility:
    - ``output_alias: name`` stores the whole result under ``name``.
    - no alias and dict result spreads the dict into the context.

    New behavior:
    - ``id`` stores the whole structured result under that id.
    - ``output_alias`` may be a list when the action returns a list/tuple.
    """
    step_id = step.get("id")
    alias = step.get("output_alias")

    if step_id:
        context[str(step_id)] = result

    if alias is None:
        if not step_id and isinstance(result, dict):
            context.update(result)
        return

    if isinstance(alias, list):
        if not isinstance(result, (list, tuple)):
            raise TypeError("output_alias is a list, but action result is not a list/tuple")
        if len(alias) != len(result):
            raise ValueError(
                f"output_alias has {len(alias)} names but result has {len(result)} values"
            )
        for name, value in zip(alias, result):
            context[str(name)] = value
        return

    context[str(alias)] = result


def iter_strings(value: Any):
    """Yield all string leaves from a nested value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from iter_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from iter_strings(v)


def workflow_refs(value: Any) -> list[str]:
    """Return ``@step.path`` references found in a nested value."""
    refs: list[str] = []
    for text in iter_strings(value):
        if text.startswith("@") and len(text) > 1:
            refs.append(text[1:])
    return refs


def _expand_vars(text: str, context: dict) -> str:
    def repl(match: re.Match) -> str:
        value = get_path(context, match.group(1))
        return str(value)
    return _VAR_PATTERN.sub(repl, text)


def _deep_merge(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _is_flat_mapping(value: Any) -> bool:
    return isinstance(value, dict) and all(
        not isinstance(v, (dict, list, tuple)) for v in value.values()
    )
