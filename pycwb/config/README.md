# Configuration ownership

This package owns configuration models, validation and persisted run settings.
Importing a policy module does not load the full analysis `Config`; the public
`from pycwb.config import Config` entry point loads that class on demand.

| Module | YAML block | Responsibility |
| --- | --- | --- |
| `execution.py` | `execution` | Scheduling profile, planner/executor selection, CPU and memory budgets, input-cache policy |
| `processing.py` | `execution_profile` | Native numerical choices, implementation optimizations, WDM options and runtime diagnostics |
| `validation.py` | Multiple blocks | Cross-field checks shared by configuration loading and runtime restoration |
| `config.py` | Full configuration | Load settings, resolve model snapshots and derive analysis parameters |

`execution.profile: scalable` selects scheduling infrastructure in
`pycwb.workflow.execution`; it does not select a scientific backend. The
`segment_processer` setting selects the job processor. `execution_profile`
controls processing inside each job and is independent of scheduler choice.

## Processing groups

The schema in `processing.py` separates four responsibilities:

- Numerical methods and conventions, including DPF, chirp and regression choices.
- Traversal, storage and reuse optimizations, including sky-delay grouping and TD caches.
- Explicit constructor options for the external WDM implementation.
- Diagnostics, cleanup cadence and device requirements.

These are documentation/code groups, not nested YAML blocks. Keep the existing
flat `execution_profile` keys and `ExecutionProfile` field order: configurations
are recorded in catalogs and callers may construct profiles positionally.
Schema defaults come from the frozen dataclass, so there is one default per option.
Changing an optimization does not establish numerical parity; validate the
relevant scientific workload separately.

## Compatibility and resume

New code imports `ExecutionSettings` from `pycwb.config.execution` and
`ExecutionProfile` and its helpers from `pycwb.config.processing`.
The released `pycwb.constants.execution_profile` path only re-exports those
objects, preserving existing imports and class lookup for older pickles.

The YAML keys, defaults, schema constraints and catalog representation are
unchanged. Resume guards still reject incompatible processing/GPU settings
and catalogs without a recorded processing profile. Renaming persisted fields
or changing their nesting requires an explicit migration.
