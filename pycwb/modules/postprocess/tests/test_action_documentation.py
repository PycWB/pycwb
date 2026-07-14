"""Keep the post-production action catalog synchronized with registrations."""

from __future__ import annotations

import ast
import re
from pathlib import Path


POSTPROCESS_DIR = Path(__file__).resolve().parents[1]
ACTION_REFERENCE = Path(__file__).resolve().parents[4] / "docs/source/postproduction_actions.rst"


def _registered_action_paths() -> set[str]:
    """Return short workflow paths for functions decorated with action_spec."""
    actions: set[str] = set()
    for module_path in POSTPROCESS_DIR.glob("*.py"):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if isinstance(target, ast.Name) and target.id == "action_spec":
                    actions.add(
                        f"postprocess.{module_path.stem}.{node.name}"
                    )
                    break
    return actions


def test_action_reference_has_every_registered_signature_once():
    documented = ACTION_REFERENCE.read_text(encoding="utf-8")
    documented_signatures = re.findall(
        r"^\.\. autofunction:: pycwb\.modules\.(postprocess\.[\w.]+)$",
        documented,
        flags=re.MULTILINE,
    )
    registered = _registered_action_paths()

    assert len(documented_signatures) == len(set(documented_signatures)), (
        "The post-production action reference contains duplicate signatures"
    )
    assert set(documented_signatures) == registered, (
        "Action reference mismatch: "
        f"missing={sorted(registered - set(documented_signatures))}, "
        f"stale={sorted(set(documented_signatures) - registered)}"
    )
