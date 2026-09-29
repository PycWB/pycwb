"""Immutable GPU execution options, recorded in YAML/catalogs and passed to workers."""

from typing import Any

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

    def __post_init__(self) -> None:
        from jsonschema import validate

        validate(asdict(self), GPU_SCHEMA)
        for name in _WORKER_LIMITS:
            if type(getattr(self, name)) is not int:
                raise ValueError(f"gpu.{name} must be an integer worker/batch count")
        if self.worker_output:
            raise ValueError(
                "gpu.worker_output was retired after slower LF measurements; remove it to use parent output"
            )
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
GPU_DESCRIPTIONS = {
    "selection_cuda": "Use the ordered CUDA selector; false uses JAX alignment and selection.",
    "dpf": "Use CUDA DPF regulation. Requires execution_profile.scalar_dpf=true.",
    "likelihood": "Use ordered CUDA likelihood sky scoring with native host orchestration.",
    "chirp": "Use CUDA trial scoring with shared native-order chirp sampling and finalization.",
    "subnet": "Use CUDA subnet sky scoring.",
    "subnet_batch": "Batch CUDA subnet sky scans; takes precedence over subnet.",
    "td": "Extract time-delay vectors with CUDA.",
    "reuse_workspace": "Retain per-stage device allocation slots between calls.",
    "reuse_td_workspace": "Reuse bounded time-delay device scratch.",
    "q_reconstruction": "Reconstruct only whitened REC/DAT products for Q-veto. Requires catalog-only background without saved products or plots.",
    "worker_output": "Retired experiment. False is retained for old default snapshots; true fails with migration guidance. Use parent output.",
    "read_processes": "Spawn frame readers instead of using threads when read_workers exceeds one.",
    "overlap_setup": "Overlap coherence and time-delay setup; reserve the sum of their worker counts.",
    "wdm_prefilter": "CUDA WDM prefilter with native CPU FFT. Requires at most three setup workers; M=4096 is rejected at runtime.",
    "quiet_driver": "Suppress verbose Numba CUDA allocation logs; warnings and errors remain.",
    "lag_workers": "Spawned CUDA lag workers, capped by pending lags. Injection jobs run serially.",
    "read_workers": "Concurrent frame readers. Input providers own read admission when supplied.",
    "condition_workers": "Concurrent CPU detector-conditioning tasks.",
    "setup_workers": "Concurrent coherence resolution preparations.",
    "td_setup_workers": "Concurrent time-delay resolution preparations.",
    "output_batch": "Parent catalog commit cadence in lags; batches above one reject injections and saved waveforms. Failed uncommitted batches are recomputed on resume.",
    "profile_lags": "Optional half-open lag range start:stop with 0 <= start < stop for timing records.",
    "stage_failure_dir": "Optional directory for serialized stage mismatch evidence when paired checks fail.",
    "validate_stages": "Also run the native reference for selection, subnet and likelihood and require exact results. Exclude validation runs from speed measurements.",
    "validate_td": "Also run the native reference for time-delay extraction and require exact results. Exclude validation runs from speed measurements.",
    "validate_setup": "Also run the native reference for coherence preparation and require exact results. Exclude validation runs from speed measurements.",
    "validate_td_setup": "Also run the native reference for time-delay preparation and require exact results. Exclude validation runs from speed measurements.",
    "validate_read": "Also run the native reference for frame reads and require exact results. Exclude validation runs from speed measurements.",
    "validate_conditioning": "Also run the native reference for conditioning and require exact results. Exclude validation runs from speed measurements.",
    "validate_reconstruction": "Also run the native reference for Q-veto reconstruction and require exact results. Exclude validation runs from speed measurements.",
}

GPU_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "default": {},
    "cwb": False,
    "properties": {
        field.name: {
            "description": GPU_DESCRIPTIONS[field.name],
            **(
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
            ),
        }
        for field in fields(GPUOptions)
    },
}


def resolve_gpu_options(value: GPUOptions | dict[str, Any] | None = None) -> GPUOptions:
    """Validate and fill defaults for a mapping or an immutable snapshot."""
    from jsonschema import validate

    if isinstance(value, GPUOptions):
        return value
    value = {} if value is None else value
    # Validate before construction to report unknown YAML keys consistently.
    validate(value, GPU_SCHEMA)
    return GPUOptions(**value)


def gpu_options(config: object | None = None) -> GPUOptions:
    """Return the explicit options belonging to a config, or standalone defaults."""
    if isinstance(config, GPUOptions):
        return config
    if isinstance(config, dict):
        return resolve_gpu_options(config)
    return resolve_gpu_options(getattr(config, "gpu", None))
