> [!WARNING]
> These GPU utilities are experimental only. They support experimental backends
> and do not establish production readiness or scientific validity.

# Shared GPU utilities

This package provides runtime and memory helpers used by the GPU modules:

- `cuda_runtime.py`: NVRTC compilation, cached CUDA modules, kernel launches,
  and device-buffer ownership.
- `workspace.py`: bounded reusable device storage and active-size host transfers.
- `geometry_cache.py`: cached device copies of immutable trial geometry.

Importing the package namespace does not load CUDA dependencies or create a CUDA
context. Using the runtime helpers requires the CUDA dependencies and a compatible
device. Device resources belong to the process using them.

Tests live in `tests/`, including runtime, workspace, geometry-cache, and ownership
checks. CUDA checks skip when the device/runtime is unavailable; passing CPU-only
checks does not validate the accelerated scientific stages.

See the [GPU workflow guide](../../../docs/dev/gpu_workflow.md) for runtime
requirements, configuration, and test commands.
