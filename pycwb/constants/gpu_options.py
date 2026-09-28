"""Immutable GPU execution options, recorded in YAML/catalogs and passed to workers."""

from dataclasses import dataclass, asdict, fields


@dataclass(frozen=True)
class GPUOptions:
    """Job-owned CUDA switches; no ambient environment is consulted."""

    selection_cuda: bool = False
    dpf: bool = False
    likelihood: bool = False
    chirp: bool = False
    subnet: bool = False
    subnet_batch: bool = False
    td: bool = False
    reuse_workspace: bool = False
    reuse_td_workspace: bool = False
    q_reconstruction: bool = False
    worker_output: bool = False
    read_processes: bool = False
    overlap_setup: bool = False
    wdm_prefilter: bool = False
    quiet_driver: bool = False
    validate_stages: bool = False
    validate_td: bool = False
    validate_setup: bool = False
    validate_td_setup: bool = False
    validate_read: bool = False
    validate_conditioning: bool = False
    validate_reconstruction: bool = False
    lag_workers: int = 1
    read_workers: int = 1
    condition_workers: int = 1
    setup_workers: int = 1
    td_setup_workers: int = 1
    output_batch: int = 1
    profile_lags: str | None = None
    stage_failure_dir: str | None = None

    def __post_init__(self):
        from jsonschema import validate

        validate(asdict(self), GPU_SCHEMA)
        for name in _WORKER_LIMITS:
            if type(getattr(self, name)) is not int:
                raise ValueError(f"gpu.{name} must be an integer worker/batch count")
        if self.profile_lags is not None:
            parts = self.profile_lags.split(":")
            if (
                len(parts) != 2
                or not all(p.isdecimal() for p in parts)
                or not 0 <= int(parts[0]) < int(parts[1])
            ):
                raise ValueError(
                    "gpu.profile_lags must be start:stop with 0 <= start < stop"
                )
        if self.wdm_prefilter and self.setup_workers > 3:
            raise ValueError("gpu.wdm_prefilter supports at most three setup workers")


_WORKER_LIMITS = {
    "lag_workers": 6,
    "read_workers": 2,
    "condition_workers": 3,
    "setup_workers": 8,
    "td_setup_workers": 3,
    "output_batch": 256,
}
GPU_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "default": {},
    "cwb": False,
    "properties": {
        field.name: (
            {"type": "boolean", "default": field.default}
            if type(field.default) is bool
            else {
                "type": "integer",
                "minimum": 1,
                "maximum": _WORKER_LIMITS[field.name],
                "default": field.default,
            }
            if field.name in _WORKER_LIMITS
            else {"type": ["string", "null"], "default": field.default}
        )
        for field in fields(GPUOptions)
    },
}


def resolve_gpu_options(value=None):
    """Validate and fill defaults for a mapping or an immutable snapshot."""
    from jsonschema import validate

    if isinstance(value, GPUOptions):
        return value
    value = {} if value is None else value
    # Validate before construction to report unknown YAML keys consistently.
    validate(value, GPU_SCHEMA)
    return GPUOptions(**value)


def gpu_options(config=None):
    """Return the explicit options belonging to a config, or standalone defaults."""
    if isinstance(config, GPUOptions):
        return config
    if isinstance(config, dict):
        return resolve_gpu_options(config)
    return resolve_gpu_options(getattr(config, "gpu", None))
