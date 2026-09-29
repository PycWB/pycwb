> [!WARNING]
> This module is experimental only. It is not a validated production backend.
> Validate results against the native implementation before scientific use.

# GPU superclustering

This package supplies optional CUDA subnet sky scans and time-delay (TD) vector
extraction within native superclustering. `build_supercluster(config)` returns a
single-lag stage with process-owned device resources. With no accelerated options
enabled, it returns the native superclustering stage.

The `gpu.subnet` option enables per-cluster CUDA subnet scans; `gpu.subnet_batch`
enables batched scans and takes precedence when both are set. `gpu.td` enables
resident-cache CUDA TD extraction, and `gpu.reuse_td_workspace` enables reusable
TD scratch buffers. `build_td_inputs_cache` provides bounded native preparation
controlled by `gpu.td_setup_workers`.

Use `gpu.validate_stages`, `gpu.validate_td`, and `gpu.validate_td_setup` for
paired native comparisons of the corresponding stages. Validation adds work
and should be disabled for performance timing.

Tests in `tests/` cover subnet scans and TD vectors; CUDA checks skip without the
required device/runtime. A CPU-only test pass does not validate GPU numerics.
See the [GPU workflow guide](../../../docs/dev/gpu_workflow.md) for processor
selection, runtime requirements, memory limits, and test commands, and the
[backend support matrix](../../../docs/source/backends.rst) for supported scope.
