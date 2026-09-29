# Modular GPU background workflow

Opt-in GPU implementations of the per-lag background stages plus a complete
opt-in job processor. Production numerical modules and defaults are unchanged;
the GPU workflow is activated by selecting the processor below in
`user_parameters.yaml`. Historical feature-branch validation showed each accelerated stage reproduces the native CPU
arithmetic bit for bit (mixed FP32/FP64 and reduction order preserved) and passed paired persisted-output gates; see the evidence directories in the
enclosing workspace under `runs/gpu_lf_exploration`, `runs/gpu_three_searches`
and `runs/gpu_review_20260921`.

## Selecting the processor

```yaml
segment_processer: pycwb.workflow.subflow.process_job_segment_gpu.process_job_segment
parallel_lag_workers: 1
execution_profile:
  scalar_dpf: true
gpu:
  selection_cuda: true
  dpf: true
  likelihood: true
  chirp: true
  subnet_batch: true
  td: true
  lag_workers: 1
  reuse_workspace: true
  reuse_td_workspace: true
```

Environment required by the processor:

```bash
conda activate pycwb-gpu-validation
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The processor raises before preparation if JAX x64 is off or no GPU is visible;
it never falls back to the CPU silently. Previous timing presets used environment switches; migrate them to the explicit
`gpu` YAML block before reuse. Options are validated, serialized in the catalog,
and passed to spawned workers. Resuming with different options is rejected.

## How functions are composed

The GPU entry point chooses ordinary preparation functions directly in
`process_job_segment_gpu.process_job_segment`: `read_data`, `condition_data`,
`prepare_coherence` and `prepare_td`. It passes these by name to the supplied
native job recipe, which retains trial handling, vetoes, resume and output.

For each worker, `coherence_gpu.coherence.build_coherence`,
`super_cluster_gpu.super_cluster.build_supercluster`, and
`likelihood_gpu.likelihood.build_likelihood` return process-owned callables.
`_build_analyzer` passes them to `native._run_lag_analysis` using
`functools.partial`. Optional paired validation wraps those functions explicitly.
No native module globals are mutated, and no fixed stage bundle is needed.

These are choices within the supplied native/GPU recipes, not a required sequence
for all workflows. Users select their own `segment_processer` and compose normal
function calls. They can reuse the native recipe through its optional callables,
replace the whole lag loop with `lag_processor`, or implement a different segment
recipe directly. The [custom processor example](../../examples/custom_workflow/README.md)
shows an additional operation selected through YAML and run by `pycwb run`.

Detailed contracts for optional numerical callbacks live in `callbacks.py` beside
`coherence_native`, `super_cluster_native` and `likelihoodWP`. They describe
individual operations, not workflow order. Scientific modules do not import the
workflow. Import the GPU packages directly; the development-only `background_cuda`
compatibility namespace has been removed.

Processing remains lag-major: shared preparation, then complete per-lag
selection, clustering, likelihood and output. `gpu.lag_workers>1` spawns
independent workers, each owning its CUDA context, compiled kernels, resident
maps and caches; prepared inputs are shared through the native memory-mapped
context and only the parent writes catalog files. Worker count is bounded by
device memory: on the 12 GB reference card six workers fit LF, four fit LD and
three fit HF, and the six-worker LF preset fails with `CUDA_ERROR_OUT_OF_MEMORY`
when other display clients hold more than ~300 MiB.

## Switches

All options belong to the `gpu` YAML mapping. Booleans use YAML `true`/`false`.
Unknown keys and out-of-range counts are rejected. Validation options duplicate
native CPU work and must not be enabled in speed measurements. The old
`PYCWB_GPU_*` environment switches are no longer read.

| YAML key under `gpu` | Effect | Bounds / notes |
|---|---|---|
| `lag_workers` | Spawned GPU lag workers | 1–6; 1 runs lags serially in the parent |
| `selection_cuda` | Direct CUDA selection kernels instead of the JAX alignment/selection pair | With this on, workers keep JAX on the CPU (no per-worker CUDA client) |
| `dpf` | CUDA DPF regulator (`likelihoodWP.compute_dpf_regulator_scalar`) | Requires `execution_profile.scalar_dpf: true` |
| `likelihood` | CUDA full sky likelihood scan (binds the unified `_scan_sky` interface) | |
| `chirp` | CUDA scoring of the native seeded micropixel bootstrap trials | Bootstrap sampling and final metadata stay native |
| `subnet` | CUDA subnet sky scan per cluster | |
| `subnet_batch` | Batched subnet sky scan over all clusters of a lag | Supersedes `subnet` when both are set |
| `td` | Resident-cache CUDA TD extraction | 2 GiB resident input budget per worker |
| `reuse_workspace` | Reusable bounded DPF/likelihood device buffers, active-byte transfers | 64 MiB per workspace |
| `reuse_td_workspace` | Reusable TD scratch shared by detector/layer sessions | 384 MiB |
| `q_reconstruction` | Reconstruct only the whitened DAT/REC waveforms Q-veto needs | Rejects saved waveforms, plots, injections |
| `output_batch` | Parent catalog batch size in lags | 1–256; background without saved waveforms only |
| `worker_output` | Retired experimental option | `true` is rejected; remove it to use parent output |
| `read_workers` / `read_processes` | Parallel native frame decoding, threads or spawned processes | 1–2 workers |
| `condition_workers` | Parallel native detector conditioning threads | 1–3 |
| `setup_workers` | Coherence preparation threads by resolution | 1–8 (≤3 with the prefilter) |
| `td_setup_workers` | TD-cache preparation threads by resolution | 1–3 |
| `overlap_setup` | Overlap TD preparation with coherence preparation | |
| `wdm_prefilter` | CUDA six-block WDM prefilter and packet maximum, native CPU FFT | Requires `setup_workers` 1–3; rejects M=4096 |
| `quiet_driver` | Silence Numba driver allocation tracing in workers | Logging only |
| `profile_lags` | `start:stop` lag range for per-process cProfile dumps | Diagnostic; distorts timing |
| `stage_failure_dir` | Directory receiving pickled mismatches from paired stage validation | Validation |
| `validate_stages` | Paired CPU/GPU coherence, supercluster, likelihood checks | Validation |
| `validate_td` | Compare every TD vector with the CPU extraction | Validation |
| `validate_setup` | Repeat serial preparation and compare maps/thresholds/caches | Validation |
| `validate_td_setup` | Compare parallel TD-cache preparation with serial | Validation |
| `validate_read` | Repeat serial frame decoding and compare | Validation |
| `validate_conditioning` | Repeat serial conditioning and compare | Validation |
| `validate_reconstruction` | Compare Q-veto waveforms with complete native reconstruction | Validation |

## Module map

| Package / workflow | Responsibility |
|---|---|
| `modules/coherence_gpu` | Resident pixel selection, map preparation, WDM prefilter and packet energy kernels |
| `modules/super_cluster_gpu` | Subnet scan/batching, TD extraction and parallel TD preparation |
| `modules/likelihood_gpu` | DPF, likelihood sky scan, chirp bootstrap |
| `utils/gpu` | CUDA compiler/runtime cache, bounded workspaces and geometry cache |
| `modules/stage_validation.py` | Paired native/accelerated scientific result checks |
| `workflow/profiling.py` | Optional lag analysis and output profiling |
| `modules/read_data/parallel.py` | Bounded parallel frame decoding |
| `modules/data_conditioning/parallel.py` | Bounded parallel native conditioning |
| `workflow/subflow/process_job_segment_gpu.py` | Function selection and job pipeline assembly |
| `workflow/subflow/process_job_segment_gpu_parallel.py` | Spawned GPU lag workers with native shared inputs |
| `workflow/subflow/gpu_setup_overlap.py` | Join overlapping preparation before worker creation |
| `workflow/subflow/gpu_output.py` | Parent-only buffered output and durable progress |
| `workflow/subflow/gpu_reconstruction.py` | Parent-owned catalog-only Q-veto flow |

CUDA `.cu` sources live beside their Python wrappers and are included in package
builds. The new scientific packages include `module.yaml` metadata. Importing
these package namespaces does not initialize CUDA; factories and sessions own
runtime resources.

## Tests

```bash
# CPU-only environment: GPU tests skip
python -m pytest pycwb/modules/coherence_gpu/tests pycwb/modules/super_cluster_gpu/tests pycwb/modules/likelihood_gpu/tests pycwb/utils/gpu/tests pycwb/modules/tests/test_validation.py pycwb/workflow/subflow/tests pycwb/workflow/tests pycwb/workflow/execution/tests -q
# GPU environment: bit-exact parity against the CPU kernels
JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m pytest pycwb/modules/coherence_gpu/tests pycwb/modules/super_cluster_gpu/tests pycwb/modules/likelihood_gpu/tests pycwb/utils/gpu/tests pycwb/modules/tests/test_validation.py pycwb/workflow/subflow/tests pycwb/workflow/tests pycwb/workflow/execution/tests -q
```

## Measured results and limits

The complete LF job (H1/L1, 1,200 s, 2,400 lags, sky order 5) ran in 171.9 s
whole-launcher time with a populated Numba cache and 206.6 s with an empty one,
against 1,153 s for the six-process CPU path; all 5,542 events were bitwise
identical. Across the 15-pair LF/HF/LD campaign every persisted event matched
the CPU reference; speedups were 6.3× (LF), 4.8× (HF, three workers) and 4.5×
(LD, four workers). Full details, hardware and cache conditions are in
`runs/gpu_lf_exploration/FINAL_REPORT.md` and `runs/gpu_three_searches`.

Validated performance scope: catalog-only background jobs. Product/injection
restrictions are conditional on batching and Q-only reconstruction; default output
uses native save logic and injections select serial lag dispatch. See the authored
[backend guide](../source/backends.rst) for the support matrix. FP64 GPU FFT preparation remains rejected; the
LD chirp bootstrap is bound by FP64 `pow` throughput on consumer GPUs; the LF and
HF lag phases are bound by per-lag Python work in the workers, not by kernels.

## Release integration

The release detector registry supplies event geometry; no AST rewriting or second
detector cache is needed. The rejected GPU FFT experiment has been removed.
Standard dependency/runtime variables (CUDA visibility, JAX precision/platform,
BLAS threads, Slurm allocation) remain external runtime contracts. PycWB stage
choices and worker counts are explicit YAML settings. Batch scheduler ceilings
are passed as `--allocated-cores` and `--memory-limit` arguments.
