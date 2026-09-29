"""Native processing policy for the persisted YAML ``execution_profile`` block.

Numerical choices, implementation optimizations, WDM options and runtime
instrumentation are grouped separately below. The public dataclass and YAML
remain flat: changing their names or nesting would require a catalog migration.
Scheduling/resource policy is defined in ``pycwb.config.execution``.
"""

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ExecutionProfile:
    """Frozen processing snapshot passed explicitly into scientific helpers.

    Field order is retained for existing positional callers. Schema definitions
    below group responsibilities without changing the serialized representation.
    """

    scalar_dpf: bool = False
    sky_delay_reuse: bool = True
    native_chirp: bool = False
    release_waveform_stats: bool = False
    coherence_early_cuts: bool = False
    preindex_shifts: bool = False
    cluster_runs: bool = False
    compact_coherence: bool = False
    direct_max_energy_input: bool = False
    tiled_wdm: bool = False
    compact_td_cache: bool = False
    band_td_cache: bool = False
    staged_td: bool = False
    bounded_jax_max_energy: bool = False
    numba_max_energy_mode: str = "parallel"
    regression_cap: bool = False
    regression_engine: str = "numba"
    regression_percentile_stride: int = 1
    gc_full_interval: int = 1
    perf_diagnostics: bool = False
    require_gpu: bool = False
    wdm_bounded_jax_forward: bool = False
    wdm_bounded_jax_inverse: bool = False
    wdm_deterministic_jax_inverse: bool = False
    wdm_bounded_numba: bool = False
    wdm_compact_complex: bool = False


# Choices that can change numerical methods, conventions or sampled statistics.
_NUMERICAL_OPTIONS: dict[str, dict[str, Any]] = {
    "scalar_dpf": {"type": "boolean", "description": "Use the scalar DPF regulator kernel."},
    "native_chirp": {
        "type": "boolean",
        "description": "Use the native micropixel chirp estimator.",
    },
    "release_waveform_stats": {
        "type": "boolean",
        "description": "Use release-compatible waveform statistics and sky error regions.",
    },
    "direct_max_energy_input": {
        "type": "boolean",
        "description": "Experimental: use conditioned strain directly for max energy.",
    },
    "regression_cap": {
        "type": "boolean",
        "description": "Cap regression witnesses with release-compatible amplitudes.",
    },
    "regression_engine": {
        "type": "string",
        "description": "Regression numerical backend.",
        "enum": ["numba", "jax"],
    },
    "regression_percentile_stride": {
        "type": "integer",
        "description": "Sampling stride for regression percentile statistics.",
        "minimum": 1,
    },
}

# Alternative traversal, storage and reuse strategies; parity needs validation.
_OPTIMIZATION_OPTIONS: dict[str, dict[str, Any]] = {
    "sky_delay_reuse": {
        "type": "boolean",
        "description": "Group identical sky delay tuples; false uses singleton groups.",
    },
    "coherence_early_cuts": {
        "type": "boolean",
        "description": "Apply coherence cuts before materializing clusters.",
    },
    "preindex_shifts": {
        "type": "boolean",
        "description": "Precompute shifted time indices for pixel selection.",
    },
    "cluster_runs": {"type": "boolean", "description": "Use run-based pixel connectivity."},
    "compact_coherence": {
        "type": "boolean",
        "description": "Share immutable coherence energy storage.",
    },
    "tiled_wdm": {
        "type": "boolean",
        "description": "Use bounded tiled forward WDM for supported CPU shapes.",
    },
    "compact_td_cache": {"type": "boolean", "description": "Store compact time-delay inputs."},
    "band_td_cache": {
        "type": "boolean",
        "description": "Experimental: restrict compact TD storage to selected frequency bands.",
    },
    "staged_td": {
        "type": "boolean",
        "description": "Populate coarse TD vectors before fine likelihood vectors.",
    },
    "bounded_jax_max_energy": {
        "type": "boolean",
        "description": "Use bounded JAX transforms inside max-energy calculation.",
    },
    "numba_max_energy_mode": {
        "type": "string",
        "description": "Numba max-energy traversal: parallel, time-major, or streaming.",
        "enum": ["parallel", "time-major", "streaming"],
    },
}

# Instance options passed explicitly to the external WDM implementation.
_WDM_OPTIONS: dict[str, dict[str, Any]] = {
    "wdm_bounded_jax_forward": {
        "type": "boolean",
        "description": "Use bounded forward JAX WDM transforms.",
    },
    "wdm_bounded_jax_inverse": {
        "type": "boolean",
        "description": "Use bounded inverse JAX WDM transforms.",
    },
    "wdm_deterministic_jax_inverse": {
        "type": "boolean",
        "description": "Use deterministic inverse JAX accumulation.",
    },
    "wdm_bounded_numba": {
        "type": "boolean",
        "description": "Use bounded Numba forward WDM transforms.",
    },
    "wdm_compact_complex": {
        "type": "boolean",
        "description": "Construct complex WDM maps without quadrature temporaries.",
    },
}

# Diagnostics, cleanup cadence and device requirements within a job.
_RUNTIME_OPTIONS: dict[str, dict[str, Any]] = {
    "gc_full_interval": {
        "type": "integer",
        "description": "Number of lag cleanup calls between full garbage collections.",
        "minimum": 1,
    },
    "perf_diagnostics": {
        "type": "boolean",
        "description": "Log detailed native performance diagnostics.",
    },
    "require_gpu": {
        "type": "boolean",
        "description": "Require a JAX GPU device for batch jobs; fail on CPU fallback.",
    },
}

# Defaults have one source of truth. Retain dataclass field order in the schema
# so generated reference tables and serialized configuration stay stable.
_DEFAULTS = asdict(ExecutionProfile())
_OPTION_SCHEMAS: dict[str, dict[str, Any]] = {
    **_NUMERICAL_OPTIONS,
    **_OPTIMIZATION_OPTIONS,
    **_WDM_OPTIONS,
    **_RUNTIME_OPTIONS,
}
PROFILE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "description": "Explicit native execution settings, resolved and recorded with the run.",
    "default": dict(_DEFAULTS),
    "properties": {
        name: {**_OPTION_SCHEMAS[name], "default": default} for name, default in _DEFAULTS.items()
    },
    "allOf": [
        {
            "if": {"properties": {"band_td_cache": {"const": True}}, "required": ["band_td_cache"]},
            "then": {
                "properties": {"compact_td_cache": {"const": True}},
                "required": ["compact_td_cache"],
            },
        }
    ],
    "cwb": False,
}


def resolve_execution_profile(value: Any = None) -> ExecutionProfile:
    """Validate a partial mapping and fill defaults without reading the environment."""
    if isinstance(value, ExecutionProfile):
        return value
    if value is None:
        value = {}
    from jsonschema import validate

    validate(value, PROFILE_SCHEMA)
    return ExecutionProfile(**value)


DEFAULT_EXECUTION_PROFILE = ExecutionProfile()


def execution_profile(config: Any = None) -> ExecutionProfile:
    """Return a job-owned frozen snapshot (also supports lightweight direct callers)."""
    if config is None:
        return DEFAULT_EXECUTION_PROFILE
    if isinstance(config, ExecutionProfile):
        return config
    value = getattr(config, "execution_profile", None)
    if not isinstance(value, ExecutionProfile):
        value = resolve_execution_profile(value)
        config.execution_profile = value
    return value


def wdm_options(config: Any = None) -> dict[str, bool]:
    """Explicit WDM instance options, including false values (no ambient defaults)."""
    profile = execution_profile(config)
    return {
        "bounded_jax_forward": profile.wdm_bounded_jax_forward,
        "bounded_jax_inverse": profile.wdm_bounded_jax_inverse,
        "deterministic_jax_inverse": profile.wdm_deterministic_jax_inverse,
        "bounded_numba": profile.wdm_bounded_numba,
        "compact_complex": profile.wdm_compact_complex,
    }


def recorded_execution_profile(recorded_config: dict[str, Any]) -> ExecutionProfile:
    """Read a recorded profile without inventing historic environment settings."""
    if recorded_config.get("execution_profile") is None:
        raise ValueError(
            "The catalog predates recorded execution profiles; its former environment "
            "settings cannot be recovered. Create a new run with explicit YAML settings."
        )
    return resolve_execution_profile(recorded_config["execution_profile"])


def check_recorded_execution_profile(config: Any, recorded_config: dict[str, Any]) -> None:
    """Prevent mixing different execution profiles under one catalog's metadata."""
    from pycwb.constants.gpu_options import gpu_options, resolve_gpu_options

    if gpu_options(config) != resolve_gpu_options(recorded_config.get("gpu")):
        raise ValueError(
            "gpu options differ from the existing catalog; use a new working directory"
        )
    if execution_profile(config) != recorded_execution_profile(recorded_config):
        raise ValueError(
            "execution_profile differs from the existing catalog. Restore its settings "
            "or use a new working directory for the changed profile."
        )
