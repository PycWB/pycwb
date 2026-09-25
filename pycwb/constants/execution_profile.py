"""Immutable, serializable native execution settings and their JSON schema."""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ExecutionProfile:
    """Resolved once per Config; passed explicitly into setup and numerical helpers."""

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


PROFILE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "description": "Explicit native execution settings, resolved and recorded with the run.",
    "default": asdict(ExecutionProfile()),
    "properties": {
        "scalar_dpf": {"type": "boolean", "default": False, "description": "Use the scalar DPF regulator kernel."},
        "sky_delay_reuse": {
            "type": "boolean",
            "default": True,
            "description": "Group identical sky delay tuples; false uses singleton groups.",
        },
        "native_chirp": {
            "type": "boolean",
            "default": False,
            "description": "Use the native micropixel chirp estimator.",
        },
        "release_waveform_stats": {
            "type": "boolean",
            "default": False,
            "description": "Use release-compatible waveform statistics and sky error regions.",
        },
        "coherence_early_cuts": {
            "type": "boolean",
            "default": False,
            "description": "Apply coherence cuts before materializing clusters.",
        },
        "preindex_shifts": {
            "type": "boolean",
            "default": False,
            "description": "Precompute shifted time indices for pixel selection.",
        },
        "cluster_runs": {"type": "boolean", "default": False, "description": "Use run-based pixel connectivity."},
        "compact_coherence": {
            "type": "boolean",
            "default": False,
            "description": "Share immutable coherence energy storage.",
        },
        "direct_max_energy_input": {
            "type": "boolean",
            "default": False,
            "description": "Experimental: use conditioned strain directly for max energy.",
        },
        "tiled_wdm": {
            "type": "boolean",
            "default": False,
            "description": "Use bounded tiled forward WDM for supported CPU shapes.",
        },
        "compact_td_cache": {"type": "boolean", "default": False, "description": "Store compact time-delay inputs."},
        "band_td_cache": {
            "type": "boolean",
            "default": False,
            "description": "Experimental: restrict compact TD storage to selected frequency bands.",
        },
        "staged_td": {
            "type": "boolean",
            "default": False,
            "description": "Populate coarse TD vectors before fine likelihood vectors.",
        },
        "bounded_jax_max_energy": {
            "type": "boolean",
            "default": False,
            "description": "Use bounded JAX transforms inside max-energy calculation.",
        },
        "numba_max_energy_mode": {
            "type": "string",
            "default": "parallel",
            "description": "Numba max-energy traversal: parallel, time-major, or streaming.",
            "enum": ["parallel", "time-major", "streaming"],
        },
        "regression_cap": {
            "type": "boolean",
            "default": False,
            "description": "Cap regression witnesses with release-compatible amplitudes.",
        },
        "regression_engine": {
            "type": "string",
            "default": "numba",
            "description": "Regression numerical backend.",
            "enum": ["numba", "jax"],
        },
        "regression_percentile_stride": {
            "type": "integer",
            "default": 1,
            "description": "Sampling stride for regression percentile statistics.",
            "minimum": 1,
        },
        "gc_full_interval": {
            "type": "integer",
            "default": 1,
            "description": "Number of lag cleanup calls between full garbage collections.",
            "minimum": 1,
        },
        "perf_diagnostics": {
            "type": "boolean",
            "default": False,
            "description": "Log detailed native performance diagnostics.",
        },
        "require_gpu": {
            "type": "boolean",
            "default": False,
            "description": "Require a JAX GPU device for batch jobs; fail on CPU fallback.",
        },
        "wdm_bounded_jax_forward": {
            "type": "boolean",
            "default": False,
            "description": "Use bounded forward JAX WDM transforms.",
        },
        "wdm_bounded_jax_inverse": {
            "type": "boolean",
            "default": False,
            "description": "Use bounded inverse JAX WDM transforms.",
        },
        "wdm_deterministic_jax_inverse": {
            "type": "boolean",
            "default": False,
            "description": "Use deterministic inverse JAX accumulation.",
        },
        "wdm_bounded_numba": {
            "type": "boolean",
            "default": False,
            "description": "Use bounded Numba forward WDM transforms.",
        },
        "wdm_compact_complex": {
            "type": "boolean",
            "default": False,
            "description": "Construct complex WDM maps without quadrature temporaries.",
        },
    },
    "allOf": [
        {
            "if": {"properties": {"band_td_cache": {"const": True}}, "required": ["band_td_cache"]},
            "then": {"properties": {"compact_td_cache": {"const": True}}, "required": ["compact_td_cache"]},
        }
    ],
    "cwb": False,
}


def resolve_execution_profile(value=None):
    """Validate a partial mapping and fill defaults without reading the environment."""
    if isinstance(value, ExecutionProfile):
        return value
    if value is None:
        value = {}
    from jsonschema import validate

    validate(value, PROFILE_SCHEMA)
    return ExecutionProfile(**value)


DEFAULT_EXECUTION_PROFILE = ExecutionProfile()


def execution_profile(config=None):
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


def wdm_options(config=None):
    """Explicit WDM instance options, including false values (no ambient defaults)."""
    profile = execution_profile(config)
    return {
        "bounded_jax_forward": profile.wdm_bounded_jax_forward,
        "bounded_jax_inverse": profile.wdm_bounded_jax_inverse,
        "deterministic_jax_inverse": profile.wdm_deterministic_jax_inverse,
        "bounded_numba": profile.wdm_bounded_numba,
        "compact_complex": profile.wdm_compact_complex,
    }


def recorded_execution_profile(recorded_config):
    """Read a recorded profile without inventing historic environment settings."""
    if recorded_config.get("execution_profile") is None:
        raise ValueError(
            "The catalog predates recorded execution profiles; its former environment "
            "settings cannot be recovered. Create a new run with explicit YAML settings."
        )
    return resolve_execution_profile(recorded_config["execution_profile"])


def check_recorded_execution_profile(config, recorded_config):
    """Prevent mixing different execution profiles under one catalog's metadata."""
    if execution_profile(config) != recorded_execution_profile(recorded_config):
        raise ValueError(
            "execution_profile differs from the existing catalog. Restore its settings "
            "or use a new working directory for the changed profile."
        )
