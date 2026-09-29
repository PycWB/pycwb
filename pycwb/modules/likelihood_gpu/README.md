> [!WARNING]
> This module is experimental only. It is not a validated production backend.
> Validate results against the native implementation before scientific use.

# CUDA callbacks for the native likelihood

This package supplies optional CUDA implementations to the native
`pycwb.modules.likelihoodWP.likelihood.evaluate_cluster_likelihood` evaluator.
It does not implement a second complete likelihood pipeline.

`build_likelihood(config)` returns a process-owned likelihood stage. With no
accelerated options enabled it returns the native evaluator unchanged. With
options enabled it binds selected callbacks:

| Option under `gpu` | Callback |
| --- | --- |
| `dpf` | DPF regulator; requires `execution_profile.scalar_dpf: true` |
| `likelihood` | Full likelihood sky scan |
| `chirp` | Scoring for native seeded chirp bootstrap trials |

The GPU job workflow imports this factory. Select that workflow with
`segment_processer: pycwb.workflow.subflow.process_job_segment_gpu.process_job_segment`.
The workflow requires a visible NVIDIA GPU and JAX x64; runtime compilation uses
NVRTC and Numba CUDA. CUDA sources are stored beside their Python wrappers.
Importing this package namespace creates no CUDA context.

## Difference from likelihoodWPGPU

[`likelihoodWPGPU`](../likelihoodWPGPU/README.md) is a separate experimental
JAX likelihood implementation, primarily using FP32 kernels. This package uses
CUDA callbacks within the native evaluator and preserves its mixed-precision
contracts. The two APIs are not interchangeable, and neither numerical parity
nor performance can be inferred from their names.

## Validation and maintenance

The CUDA workflow remains experimental. Historical whole-job parity/performance
evidence covers catalog-only LF/HF/LD background searches, not every injection
or output configuration. Use `gpu.validate_stages: true` for paired native stage
checks on representative inputs; disable validation when timing performance.
Keep scientific decisions and orchestration in the native evaluator, and isolate
accelerated kernels behind its callback contracts.

Tests live in `tests/`; CUDA tests skip without the required device/runtime. A
CPU-only test pass does not validate CUDA numerics. See the
[backend support matrix](../../../docs/source/backends.rst) and
[GPU workflow guide](../../../docs/dev/gpu_workflow.md) for configuration and limits.
