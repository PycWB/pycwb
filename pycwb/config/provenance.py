"""Portable snapshots of YAML settings and configuration dependencies."""

from copy import deepcopy
import hashlib
from pathlib import Path


def snapshot_yaml_parameters(parameters, yaml_file):
    """Freeze settings before derivation or CLI overrides.

    A detector-definition file is identified by its bytes, allowing staging to
    relocate it while still detecting edits to the definitions themselves.
    """
    snapshot = deepcopy(parameters)
    definitions = parameters.get("detector_definitions_file")
    if definitions:
        path = Path(definitions)
        if not path.is_absolute():
            path = Path(yaml_file).resolve().parent / path
        snapshot["detector_definitions_file"] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        }
    return snapshot
