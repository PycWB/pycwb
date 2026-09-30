"""Shell-visible contracts shared by both CLI entry points."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# The installed ``pycwb`` executable calls this [project.scripts] target.
SCRIPT_TARGET = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]["pycwb"]


def run_entrypoint(entrypoint, prelude, arguments):
    code = prelude + f"\nimport runpy, sys\nsys.argv = {['pycwb', *arguments]!r}\n"
    if entrypoint == "module":
        code += "runpy.run_module('pycwb', run_name='__main__')\n"
    else:
        module, function = SCRIPT_TARGET.split(":")
        code += (f"import importlib\n"
                 f"sys.exit(getattr(importlib.import_module({module!r}), {function!r})())\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT), env.get("PYTHONPATH")])
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("entrypoint", ["module", "script"])
@pytest.mark.parametrize("status,expected", [(None, 0), (0, 0), (7, 7)])
def test_shell_receives_command_status(entrypoint, status, expected):
    result = run_entrypoint(
        entrypoint,
        f"import pycwb.cli.run\npycwb.cli.run.command = lambda args: {status!r}",
        ["run", "unused.yaml"],
    )
    assert result.returncode == expected, result.stderr


@pytest.mark.parametrize("entrypoint", ["module", "script"])
@pytest.mark.parametrize("argument", ["--help", "--version"])
def test_information_commands_do_not_load_scientific_implementations(
    entrypoint, argument
):
    result = run_entrypoint(
        entrypoint,
        """
import importlib.abc
import sys

class BlockScientificImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('pycwb.modules.xtalk', 'pycwb.modules.gwosc', 'pycwb.config')):
            raise ImportError(f'Scientific implementation unavailable: {fullname}')

sys.meta_path.insert(0, BlockScientificImports())
""",
        [argument],
    )
    assert result.returncode == 0, result.stderr
    assert "pycwb" in result.stdout
