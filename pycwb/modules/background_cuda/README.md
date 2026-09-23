# Experimental CUDA background stages

Opt-in GPU implementations of the per-lag background stages plus a complete
opt-in job processor. Production numerical modules and defaults are unchanged;
nothing in this package is imported unless the processor below is selected in
`user_parameters.yaml`. Every accelerated stage reproduces the native CPU
arithmetic bit for bit (mixed FP32/FP64 and reduction order preserved) and has
passed paired persisted-output gates; see the evidence directories in the
enclosing workspace under `runs/gpu_lf_exploration`, `runs/gpu_three_searches`
and `runs/gpu_review_20260921`.

## Selecting the processor

```yaml
segment_processer: pycwb.modules.background_cuda.processor.process_job_segment
parallel_lag_workers: 1          # lag workers are controlled by PYCWB_GPU_LAG_WORKERS
```

Environment required by the processor:

```bash
conda activate pycwb-gpu-validation
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The processor raises before preparation if JAX x64 is off or no GPU is visible;
it never falls back to the CPU silently. The measured LF preset is reproduced by
`runs/gpu_lf_exploration/run_lf_gpu.py --name NAME --overlap-setup`; the
switches it sets are listed in that script's `GPU_OPTIONS` and in the table
below.

## How stages are composed

`processor._build_analyzer` clones the native `_run_lag_analysis` with private
globals through `binding.specialize`, replacing only the collaborators whose
switch is enabled. No production module is mutated; the serial CPU path and
other callers keep the native bindings. The private production names this
package binds are pinned by `tests/test_bindings.py`.

Processing remains lag-major: shared preparation, then complete per-lag
selection, clustering, likelihood and output. `PYCWB_GPU_LAG_WORKERS>1` spawns
independent workers, each owning its CUDA context, compiled kernels, resident
maps and caches; prepared inputs are shared through the native memory-mapped
context and only the parent writes catalog files. Worker count is bounded by
device memory: on the 12 GB reference card six workers fit LF, four fit LD and
three fit HF, and the six-worker LF preset fails with `CUDA_ERROR_OUT_OF_MEMORY`
when other display clients hold more than ~300 MiB.

## Switches

All switches are read at call time through `flags.py`. Values are the literal
string `"1"` for booleans. "Validation" switches duplicate native CPU work for
exact comparison and must not be enabled in speed measurements.

| Switch (`PYCWB_GPU_` prefix) | Effect | Bounds / notes |
|---|---|---|
| `LAG_WORKERS` | Spawned GPU lag workers | 1–6; 1 runs lags serially in the parent |
| `SELECTION_CUDA` | Direct CUDA selection kernels instead of the JAX alignment/selection pair | With this on, workers keep JAX on the CPU (no per-worker CUDA client) |
| `DPF` | CUDA DPF regulator (`likelihoodWP.calculate_dpf_scalar`) | Forces scalar DPF mode |
| `LIKELIHOOD` | CUDA full sky likelihood scan (binds all three native sky-scan hooks) | |
| `CHIRP` | CUDA scoring of the native seeded micropixel bootstrap trials | Bootstrap sampling and final metadata stay native |
| `SUBNET` | CUDA subnet sky scan per cluster | |
| `SUBNET_BATCH` | Batched subnet sky scan over all clusters of a lag | Supersedes `SUBNET` when both are set |
| `TD` | Resident-cache CUDA TD extraction | 2 GiB resident input budget per worker |
| `REUSE_WORKSPACE` | Reusable bounded DPF/likelihood device buffers, active-byte transfers | 64 MiB per workspace |
| `REUSE_TD_WORKSPACE` | Reusable TD scratch shared by detector/layer sessions | 384 MiB |
| `EVENT_GEOMETRY_CACHE` | Cache native `Detector` geometry objects during event output | Deep copies preserve constructor ownership |
| `Q_RECONSTRUCTION` | Reconstruct only the whitened DAT/REC waveforms Q-veto needs | Rejects saved waveforms, plots, injections |
| `OUTPUT_BATCH` | Parent catalog batch size in lags | 1–256; background without saved waveforms only |
| `WORKER_OUTPUT` | Move waveform/Q-veto computation into workers | Catalog-only background; slower than parent output in the measured LF job |
| `READ_WORKERS` / `READ_PROCESSES` | Parallel native frame decoding, threads or spawned processes | 1–2 workers |
| `CONDITION_WORKERS` | Parallel native detector conditioning threads | 1–3 |
| `SETUP_WORKERS` | Coherence preparation threads by resolution | 1–8 (≤3 with the prefilter) |
| `TD_SETUP_WORKERS` | TD-cache preparation threads by resolution | 1–3 |
| `OVERLAP_SETUP` | Overlap TD preparation with coherence preparation | |
| `WDM_PREFILTER` | CUDA six-block WDM prefilter and packet maximum, native CPU FFT | Requires `SETUP_WORKERS` 1–3; rejects M=4096 |
| `MAX_ENERGY` | Rejected experiment: GPU FFT max-energy | Raw maps differ from CPU; not interchangeable |
| `COMPARE_MAX_ENERGY` | Log CPU/GPU map differences of the rejected experiment | Diagnostic |
| `GC_INTERVAL` | Launcher-side value copied into `PYCWB_GC_FULL_INTERVAL` | Read by the launcher, not by this package |
| `QUIET_DRIVER` | Silence Numba driver allocation tracing in workers | Logging only |
| `PROFILE_LAGS` | `start:stop` lag range for per-process cProfile dumps | Diagnostic; distorts timing |
| `STAGE_FAILURE_DIR` | Directory receiving pickled mismatches from paired stage validation | Validation |
| `VALIDATE_STAGES` | Paired CPU/GPU coherence, supercluster, likelihood checks | Validation |
| `VALIDATE_TD` | Compare every TD vector with the CPU extraction | Validation |
| `VALIDATE_SETUP` | Repeat serial preparation and compare maps/thresholds/caches | Validation |
| `VALIDATE_TD_SETUP` | Compare parallel TD-cache preparation with serial | Validation |
| `VALIDATE_READ` | Repeat serial frame decoding and compare | Validation |
| `VALIDATE_CONDITIONING` | Repeat serial conditioning and compare | Validation |
| `VALIDATE_RECONSTRUCTION` | Compare Q-veto waveforms with complete native reconstruction | Validation |
| `VALIDATE_EVENT_GEOMETRY` | Compare cached-geometry events with native events | Validation |
| `VALIDATE_MAX_ENERGY_SELECTION` | Retain a CPU setup for coherence comparison in the rejected experiment | Validation |

## Module map

| File | Role |
|---|---|
| `processor.py` | Entry point, stage composition, resident-map selector |
| `process_parallel.py` | Spawned GPU lag workers on the native shared-input pool |
| `flags.py`, `binding.py` | Switch accessors; private function binding |
| `cuda_runtime.py`, `workspace.py`, `geometry_cache.py` | NVRTC compile cache, bounded device buffers, resident geometry |
| `selection_cuda.py/.cu`, `alignment_jax.py`, `selection_jax.py` | Pixel selection backends |
| `dpf_regulator.py/.cu`, `likelihood_scan.py/.cu` | Likelihood stage kernels |
| `subnet_scan.py/.cu`, `subnet_batch.py` | Subnet sky scan and per-lag batching |
| `td_vectors.py/.cu` | Resident TD filter extraction |
| `chirp_bootstrap.py/.cu`, `chirp_bootstrap_plan.py` | Chirp bootstrap trial scoring |
| `packet_energy.py/.cu`, `wdm_prefilter.py/.cu`, `wdm_hybrid.py`, `max_energy_hybrid.py` | Setup-phase prefilter path |
| `max_energy_jax.py` | Rejected GPU FFT experiment, kept for the recorded disposition |
| `read_parallel.py`, `conditioning_parallel.py`, `setup_parallel.py`, `td_setup_parallel.py`, `setup_overlap.py` | Bounded parallel preparation (CPU) |
| `output_buffer.py`, `reconstruction.py`, `worker_output.py`, `event_geometry.py` | Output path |
| `validation.py`, `profiling.py` | Paired exact validation; opt-in profiles |
| `tests/` | pytest suite; GPU tests skip without a device |

## Tests

```bash
# CPU-only environment: GPU tests skip
python -m pytest pycwb/modules/background_cuda/tests -q
# GPU environment: bit-exact parity against the CPU kernels
JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m pytest pycwb/modules/background_cuda/tests -q
```

## Measured results and limits

The complete LF job (H1/L1, 1,200 s, 2,400 lags, sky order 5) ran in 171.9 s
whole-launcher time with a populated Numba cache and 206.6 s with an empty one,
against 1,153 s for the six-process CPU path; all 5,542 events were bitwise
identical. Across the 15-pair LF/HF/LD campaign every persisted event matched
the CPU reference; speedups were 6.3× (LF), 4.8× (HF, three workers) and 4.5×
(LD, four workers). Full details, hardware and cache conditions are in
`runs/gpu_lf_exploration/FINAL_REPORT.md` and `runs/gpu_three_searches`.

Limits: catalog-only background jobs (saved waveforms, plots and injections are
rejected rather than skipped); FP64 GPU FFT preparation remains rejected; the
LD chirp bootstrap is bound by FP64 `pow` throughput on consumer GPUs; the LF and
HF lag phases are bound by per-lag Python work in the workers, not by kernels.
