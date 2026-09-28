"""Offline YAML/schema checks; full data and wavelet checks happen at preparation."""

import sys
from pathlib import Path


def init_parser(parser):
    """Register the configuration to check."""
    parser.add_argument("user_parameter_file", help="Rendered user-parameter YAML file")


def validate_config(path):
    """Validate YAML and runtime settings without loading a wavelet catalog.

    Returns the configuration with schema defaults. This deliberately does not
    create a Config object: doing that can download cross-talk data.
    """
    from types import SimpleNamespace

    import yaml

    from pycwb.config.detector_definitions import load_detector_definitions
    from pycwb.constants import user_parameters_schema
    from pycwb.constants.detectors import resolve_detector_geometries
    from pycwb.constants.execution_profile import resolve_execution_profile
    from pycwb.constants.gpu_options import resolve_gpu_options
    from pycwb.utils.skymap_coord import validate_user_sky_config
    from pycwb.utils.yaml_helper import load_yaml
    from pycwb.workflow.execution.settings import ExecutionSettings

    path = Path(path)
    with path.open() as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("Expected a YAML mapping of parameter names to values")
    params = load_yaml(str(path), user_parameters_schema)
    validate_user_sky_config(
        params.get("sky_mask"), context="sky_mask", default_coordsys="geo"
    )
    validate_user_sky_config(
        (params.get("injection") or {}).get("sky_distribution"),
        context="injection.sky_distribution",
        default_coordsys="icrs",
    )
    ExecutionSettings.from_config(SimpleNamespace(**params))
    resolve_execution_profile(params.get("execution_profile"))
    resolve_gpu_options(params.get("gpu"))
    registry, _ = load_detector_definitions(
        params.get("detector_definitions_file"), path
    )
    resolve_detector_geometries(
        params.get("ifo", []), params.get("detector_geometry", {}), registry=registry
    )
    return params


def command(args):
    """Print a bounded diagnostic and return a nonzero status on invalid input."""
    import yaml
    from jsonschema import ValidationError

    try:
        validate_config(args.user_parameter_file)
    except ValidationError as error:
        field = ".".join(map(str, error.absolute_path)) or "configuration"
        print(f"INVALID {field}: {error.message}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 1
    print(f"VALID: {args.user_parameter_file}")
    print(
        "Checked YAML, schema, sky units, execution/GPU settings and detector definitions. Data availability, "
        "wavelet compatibility and scientific suitability are checked separately."
    )
    return 0
