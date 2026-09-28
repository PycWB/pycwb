"""Reject new lint, import-boundary and public-contract debt against a reviewed baseline.

Run from the repository root. Baselines record diagnostic identity and occurrence
count, not line numbers, so moving an existing function does not hide duplicates.
Use --write-baseline only when reviewing an intentional policy/backlog change.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tools/quality/baseline.json"


def production_files(root: Path) -> list[Path]:
    """Collect owned Python code, excluding tests, vendor files and SCM output."""
    return sorted(path for path in (root / "pycwb").rglob("*.py")
                  if not {"tests", "vendor"}.intersection(path.parts)
                  and path.name != "_version.py")


def audit(root: Path) -> Counter[str]:
    """Count diagnostics without importing scientific libraries or loading devices."""
    result: Counter[str] = Counter()
    paths = production_files(root)
    process = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--config", str(ROOT / "pyproject.toml"),
         "--output-format", "json", *(str(p) for p in paths)],
        text=True, capture_output=True, check=False,
    )
    if process.returncode not in (0, 1):
        raise RuntimeError(process.stderr)
    for item in json.loads(process.stdout):
        path = Path(item["filename"]).relative_to(root)
        result[f"lint|{path}|{item['code']}|{item['message']}"] += 1
    for path in paths:
        relative = path.relative_to(root)
        tree = ast.parse(path.read_text())

        def visit(nodes: list[ast.stmt], prefix: str = "") -> None:
            for node in nodes:
                if isinstance(node, ast.ClassDef):
                    visit(node.body, prefix + node.name + ".")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                    # Protocol/overload stubs describe their contract on the class.
                    stub = all(isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                               and n.value.value is Ellipsis for n in node.body)
                    identity = f"{relative}|{prefix}{node.name}"
                    if not stub and ast.get_docstring(node) is None:
                        result[f"docstring|{identity}"] += 1
                    args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                    args += [a for a in (node.args.vararg, node.args.kwarg) if a is not None]
                    missing = [a.arg for a in args if a.arg not in {"self", "cls"} and a.annotation is None]
                    if node.returns is None:
                        missing.append("return")
                    if missing:
                        result[f"typing|{identity}|{','.join(sorted(missing))}"] += 1
        visit(tree.body)
        for node in ast.walk(tree):
            modules = ([n.name for n in node.names] if isinstance(node, ast.Import)
                       else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if isinstance(node, ast.ImportFrom) and node.level:
                package = list(relative.with_suffix("").parts[:-1])
                prefix = package[:len(package) - node.level + 1]
                modules = [".".join(prefix + ([node.module] if node.module else []))]
            for module in modules:
                lower_layer = relative.parts[1] in {"constants", "types", "utils", "modules"}
                scientific = relative.parts[1] == "modules" and "background_cuda" not in relative.parts
                if lower_layer and module.startswith("pycwb.workflow") or scientific and module.startswith("pycwb.modules.background_cuda"):
                    result[f"boundary|{relative}|{module}"] += 1
    return result


def main() -> int:
    """Report introduced debt; existing allowances never excuse extra occurrences."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    actual = audit(args.source.resolve())
    if args.write_baseline:
        BASELINE.write_text(json.dumps(dict(sorted(actual.items())), indent=2) + "\n")
        print(f"Recorded {sum(actual.values())} existing diagnostics")
        return 0
    baseline = Counter(json.loads(BASELINE.read_text()))
    introduced = actual - baseline
    if introduced:
        for key, count in sorted(introduced.items()):
            print(f"+{count} {key}")
        return 1
    print(f"No new quality debt; {sum(actual.values())} existing diagnostics "
          f"({sum((baseline - actual).values())} baseline occurrences resolved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
