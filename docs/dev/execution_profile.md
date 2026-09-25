# Execution-profile implementation and validation

User-facing settings, defaults, dependencies, examples and migration guidance
are maintained in the published [Performance Guide](../source/dev_performance.rst).
Scientific calculation choices are explained in
[Units and Conventions](../source/units_conventions.rst).
Start from [User Parameters](../source/schema.rst) for the parameter index.

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
