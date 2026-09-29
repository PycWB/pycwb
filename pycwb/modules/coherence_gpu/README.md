> [!WARNING]
> This module is experimental only. It is not a validated production backend.
> Validate results against the native implementation before scientific use.

# GPU coherence

This package supplies GPU pixel selection and optional accelerated map preparation
for the native coherence stage. `build_coherence(config)` returns a single-lag
coherence callable and its `GPUSelector`, which owns resident device maps.
Native coherence orchestration and clustering remain in `coherence_native`.

The package contains JAX alignment and selection kernels, direct CUDA selection
kernels, bounded parallel preparation, and an optional CUDA WDM prefilter and
packet-energy path with CPU FFT processing.

Relevant options in the `gpu` configuration block include `selection_cuda`,
`setup_workers`, and `wdm_prefilter`. Use `validate_stages` and `validate_setup`
for paired checks against native selection and preparation on representative
inputs. These checks add work and should be disabled for performance timing.

Tests live in `tests/`; checks requiring CUDA skip without the device/runtime.
A CPU-only test pass does not validate GPU numerics. See the
[GPU workflow guide](../../../docs/dev/gpu_workflow.md) for processor selection,
runtime requirements, option limits, and test commands, and the
[backend support matrix](../../../docs/source/backends.rst) for supported scope.
