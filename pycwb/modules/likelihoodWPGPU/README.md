> [!WARNING]
> This module is experimental only. It is not a validated production backend.
> Validate results against the native implementation before scientific use.

# Experimental JAX likelihood implementation

This package is a separate JAX implementation of coherent likelihood evaluation.
It shares preparation and selected scientific helpers with `likelihoodWP`, but
maintains its own evaluation path, JIT-compiled kernels and vectorized sky scan.
Its kernels primarily use FP32 and layouts chosen for JAX device execution.

The public entry points are `prepare_likelihood_inputs`, `likelihood`, and
`likelihood_wrapper`. Current in-repository callers include likelihood benchmarks
and selected kernel tests. The modular GPU job workflow does **not** select this
package. It is retained for JAX experimentation and existing API users.

## Difference from likelihood_gpu

[`likelihood_gpu`](../likelihood_gpu/README.md) supplies CUDA callbacks to the
native `likelihoodWP` evaluator. It is the likelihood package used by
`workflow.subflow.process_job_segment_gpu`. Enabling the `gpu` YAML options in
that workflow does not enable this JAX implementation.

These packages have different APIs and numerical contracts. Do not replace one
with the other based on the GPU naming, or claim equivalent thresholds, precision,
outputs or performance without a representative numerical comparison.

## Status and maintenance

This is an experimental backend, not a fully validated replacement for the native
or CUDA job workflow. Benchmark and kernel checks do not establish whole-job
resume, injection or output-product support. See the
[backend support matrix](../../../docs/source/backends.rst).

If JAX is developed further, prefer adapting its kernels to the shared likelihood
stage/callback contracts before adding more duplicate orchestration. Existing
public entry points should be deliberately migrated or deprecated rather than
silently redirected to the CUDA implementation.

Relevant callers and checks:

- `benchmark/likelihood/benchmark_cpu_vs_gpu.py`
- `benchmark/likelihood/data_generator_native.py`
- `pycwb/modules/likelihoodWP/tests/test_gpu_sky_mask_contract.py`
