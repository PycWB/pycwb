"""Shared command registry for the executable, tests and documentation."""

import argparse
from importlib import import_module

from pycwb import __version__

COMMANDS = (
    ("doctor", "doctor", "Report interpreter, platform and installed package versions"),
    ("validate", "validate", "Check configuration syntax without downloading data"),
    ("run", "run", "Run search"),
    ("flow", "flow", "Run search through the Prefect wrapper"),
    ("batch-setup", "batch_setup", "Set up batch run"),
    ("config-setup", "config_setup", "Set up project configuration and batch jobs"),
    ("clone-dir", "clone_dir", "Clone a directory to a new location"),
    ("batch-runner", "batch_runner", "Run one batch payload"),
    ("xtalk", "xtalk", "Convert an xtalk file"),
    ("merge", "merge_catalog", "Merge catalog or wave files"),
    ("post-process", "post_process", "Post-process results"),
    ("gwosc", "gwosc", "Download data and configuration for a GW event"),
    ("gwosc-data", "gwosc_data", "Download data for a configuration"),
    (
        "get-external-modules",
        "get_external_modules",
        "Fetch configured external modules",
    ),
    ("online", "online", "Run an online search"),
    ("progress", "progress", "Show progress from run catalogs"),
    ("simulation-summary", "simulation_summary", "Build a per-simulation summary"),
    ("match-simulations", "match_simulations", "Match triggers to simulations"),
)


def create_parser():
    """Build the same parser for command execution and reference generation."""
    parser = argparse.ArgumentParser(prog="pycwb")
    parser.add_argument(
        "-v", "--version", action="version", version=f"pycwb {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command_name", help="commands")
    for name, module_name, description in COMMANDS:
        module = import_module(f"pycwb.cli.{module_name}")
        command_parser = subparsers.add_parser(
            name, help=description, description=description
        )
        module.init_parser(command_parser)
        command_parser.set_defaults(func=module.command)
    return parser


def main(argv=None):
    """Parse arguments and propagate command exit codes to the shell."""
    parser = create_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    result = args.func(args)
    return result if isinstance(result, int) else 0
