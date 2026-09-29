"""Report the local environment using installed distribution metadata."""

from argparse import ArgumentParser, Namespace
from importlib import metadata
import json
import platform
import sys
from typing import Any


def init_parser(parser: ArgumentParser) -> None:
    """Register machine-readable output for bug reports and provenance."""
    parser.add_argument(
        "--json", action="store_true", help="Print a JSON environment inventory"
    )


def environment_report() -> dict[str, Any]:
    """Read package versions without probing backend imports or hardware.

    Distribution metadata is the source of truth for the inventory. There is
    deliberately no second dependency list or configuration-readiness verdict.
    """
    from pycwb import __version__

    packages = [
        {"name": distribution.metadata["Name"] or "<unknown>",
         "version": distribution.version}
        for distribution in metadata.distributions()
    ]
    packages.sort(key=lambda package: (package["name"].casefold(), package["version"]))
    return {
        "pycwb": __version__,
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "scope": (
            "Installed package inventory; backend imports, hardware and pipeline "
            "readiness are not checked."
        ),
        "packages": packages,
    }


def command(args: Namespace) -> int:
    """Print the inventory; zero means the report was generated successfully."""
    report = environment_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"PycWB {report['pycwb']} | Python {report['python']} | {report['platform']}")
        print(f"Interpreter: {report['executable']}")
        print(report["scope"])
        for package in report["packages"]:
            print(f"{package['name']}=={package['version']}")
    return 0
