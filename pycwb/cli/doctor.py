"""Local environment diagnostics suitable for attaching to a bug report."""

from argparse import ArgumentParser, Namespace
from typing import Any

import importlib
import json
import platform
import sys
from importlib import metadata

REQUIRED = {
    "numpy": "numpy",
    "scipy": "scipy",
    "numba": "numba",
    "jax": "jax",
    "lal": "lalsuite",
    "lalsimulation": "lalsuite",
    "wdm_wavelet": "wdm-wavelet",
    "pyarrow": "pyarrow",
    "gwpy": "gwpy",
    "astropy": "astropy",
}
OPTIONAL = {
    "ROOT": "ROOT",
    "pycbc": "pycbc",
    "htcondor2": "htcondor",
    "xgboost": "xgboost",
}


def init_parser(parser: ArgumentParser) -> None:
    """Register machine-readable output."""
    parser.add_argument(
        "--json", action="store_true", help="Print a JSON diagnostic report"
    )


def environment_report() -> dict[str, Any]:
    """Probe imports without network access or running an analysis."""
    from pycwb import __version__

    report = {
        "pycwb": __version__,
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "checks": [],
    }
    for required, modules in ((True, REQUIRED), (False, OPTIONAL)):
        for module_name, distribution in modules.items():
            check = {"module": module_name, "required": required}
            try:
                module = importlib.import_module(module_name)
                try:
                    check["version"] = metadata.version(distribution)
                except metadata.PackageNotFoundError:
                    check["version"] = str(getattr(module, "__version__", "unknown"))
                if module_name == "jax":
                    check["devices"] = [str(device) for device in module.devices()]
                check["ok"] = True
            except Exception as error:  # noqa: BLE001 -- report broken dependency imports
                check.update(ok=False, error=f"{type(error).__name__}: {error}")
            report["checks"].append(check)
    report["ok"] = all(check["ok"] for check in report["checks"] if check["required"])
    return report


def command(args: Namespace) -> int:
    """Return failure only for required runtime probes."""
    report = environment_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"PycWB {report['pycwb']} | Python {report['python']} | {report['platform']}"
        )
        for check in report["checks"]:
            status = (
                "OK" if check["ok"] else ("FAIL" if check["required"] else "OPTIONAL")
            )
            print(
                f"{status:8} {check['module']}: {check.get('version', check.get('error'))}"
            )
        print("Environment checks only; use the demo to test a complete search.")
    return 0 if report["ok"] else 1
