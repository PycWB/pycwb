# Explicit native execution profile

Put native execution choices in `user_parameters.yaml`. The default schema
validates the `execution_profile` mapping; unknown keys, invalid backend names,
string booleans and nonpositive integer settings are rejected. Unspecified
settings receive documented defaults.

```yaml
max_energy_backend: jax
execution_profile:
  sky_delay_reuse: true
  scalar_dpf: true
  compact_td_cache: true
  staged_td: true
  gc_full_interval: 16
  perf_diagnostics: false
```

`Config.load_from_yaml` resolves this mapping into an immutable
`ExecutionProfile`. Dictionary restoration resolves it too. Setup retains the
resolved profile and numerical helpers receive explicit options. Configure a new
Config/setup for a different profile; do not mutate a prepared job's configuration.
Two jobs in one Python process can use different profiles without changing module
globals, environment variables, or another job's compiled-kernel selection.

The full resolved profile, including omitted defaults, is serialized in the run
catalog's `config.execution_profile` metadata and survives worker serialization
and catalog restoration. The original user YAML is also retained by normal job
setup. The schema is the source for parameter defaults and descriptions. Reusing an
existing catalog with a different profile is rejected before replacing the saved
user YAML. Batch workers restore the recorded profile. Catalogs from before this
change lack the historical environment settings and require a new run with an
explicit profile; the loader does not guess those settings.

## Defaults and options

Defaults preserve the previous behavior with execution environment variables
unset. Sky delay grouping defaults to true; other boolean profile options default
to false. `gc_full_interval` and `regression_percentile_stride` default to 1.
`regression_engine` defaults to `numba`; `numba_max_energy_mode` defaults to
`parallel` (also accepts `time-major` and `streaming`).

| Area | Profile settings |
| --- | --- |
| Sky likelihood | `sky_delay_reuse`, `scalar_dpf`, `native_chirp`, `release_waveform_stats` |
| Coherence | `coherence_early_cuts`, `preindex_shifts`, `cluster_runs`, `compact_coherence`, `direct_max_energy_input`, `tiled_wdm` |
| Time-delay inputs | `compact_td_cache`, `band_td_cache`, `staged_td` |
| Max energy | `bounded_jax_max_energy`, `numba_max_energy_mode` |
| Regression | `regression_engine`, `regression_cap`, `regression_percentile_stride` |
| Cleanup and diagnostics | `gc_full_interval`, `perf_diagnostics`, `require_gpu` |
| WDM | `wdm_bounded_jax_forward`, `wdm_bounded_jax_inverse`, `wdm_deterministic_jax_inverse`, `wdm_bounded_numba`, `wdm_compact_complex` |

`band_td_cache: true` requires `compact_td_cache: true`. Band-restricted TD storage,
direct max-energy input and alternate traversal modes retain their experimental
status. Scientific release options remain explicit choices; this migration does
not change their defaults or claim new scientific validation.

The existing top-level `max_energy_backend` remains the backend selector
(`jax`, `numba`, `auto`, and existing aliases); `coherence_timing` controls setup
timing logs. Neither has an environment override. Auto selection records the
resolved backend in each coherence setup.

## Migration

The former `PYCWB_*` execution switches and `WDM_*` transform switches are no
longer read. Setting them has no effect on the native profile. Replace the old
shell-based performance recipe with the mapping in
[`examples/performance/bounded_cpu.yaml`](../../examples/performance/bounded_cpu.yaml),
merged into the analysis YAML. This is a parameter fragment, not a standalone
analysis configuration.

The WDM companion library now accepts explicit instance options matching the
`wdm_*` profile fields without their prefix. PycWB passes all values, including
false values, into transforms for conditioning, injections, coherence, TD inputs
and reconstruction. Reconstruction caches include these options in their keys.
Use the companion WDM changes together with this PycWB change.

Installation paths (`HOME_WAT_FILTERS`), environment discovery and third-party
process controls such as thread counts/device visibility are not numerical
execution switches and are outside this profile. CPU/GPU placement remains a
launcher responsibility; `require_gpu` explicitly rejects CPU fallback in batch
jobs. The bounded CPU recipe's timing evidence applies to CPU workers.

## Validation

The native and companion-WDM regression run passed 973 tests with one optional
backend-parity test skipped. Focused configuration/provenance checks cover YAML
validation, frozen settings, schema/runtime default agreement, catalog and pickle
round trips, interleaved profiles, JIT percentile-stride specialization, explicit
Numba fallback and JAX-wrapper selection, and catalog reuse protection. WDM
forward/inverse dispatch tests change old environment variables between calls
and verify the selected options remain instance-specific.

A saved HF likelihood replay returned identical output digests for reference,
grouped and singleton scans with conflicting legacy environment switches set.
These checks validate configuration isolation and the tested numerical contracts;
a new full-job throughput or GPU campaign was not run for this migration.
