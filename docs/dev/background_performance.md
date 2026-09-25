# Native background performance profile

The bounded CPU profile preserves lag-major processing: shared setup is followed by complete per-lag selection, clustering, likelihood, and output. It requires the companion WDM package with bounded JAX forward/inverse support. There is no ROOT or C++ numerical runtime dependency in this path.

Merge `examples/performance/bounded_cpu.yaml` into your `user_parameters.yaml` before the ordinary native job command. See [execution-profile configuration and provenance](execution_profile.md). It does not modify analysis thresholds or the run template. Use the native segment processor:

```yaml
segment_processer: pycwb.workflow.subflow.process_job_segment_native.process_job_segment
# Optional, explicit H1/L1 cWB 6.4.6.9 physical input compatibility:
detector_geometry: cwb_6.4.6.9
```

The geometry default remains `lal`. The release model supports H1/L1 only; it changes literal arm/vertex constants, injection projection, sky delays, and antenna evaluation. It also reproduces release angle narrowing before antenna export. Do not mix geometry models when comparing runs.

For reproducing the single-worker pilot measurements, use one worker and `NUMBA_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`. These library settings are not a hard process CPU affinity; use the scheduler's CPU allocation for production. Budget node memory using process-tree PSS plus operating-system margin, not summed RSS, which double-counts shared pages.

## What changes

- Bounded forward/inverse WDM workspaces reduce transient setup allocation while retaining transform precision and accumulation order for validated shapes. Unsupported shapes retain the existing path.
- Staged TD extraction and compact storage avoid unnecessary copies and delay-vector work. The final profile retains a dense coefficient cache.
- Indexed lag selection and run-based connectivity reduce repeated work. Connectivity compression never discards the original pixels used by likelihood.
- Sky delay/scratch reuse and scalar DPF reduce temporary allocation in likelihood. The [shared CPU sky scan](unified_sky_scan.md) now enables grouping by default; `execution_profile.sky_delay_reuse: false` selects singleton groups. Scratch reuse is internal.
- GC interval 16 collects young generations between outputs and periodically collects older objects, with a memory-growth safeguard. Automatic GC remains enabled.

The profile also explicitly enables regression witness capping, native micropixel chirp, and release waveform-summary arithmetic. These correct scientific discrepancies and must be enabled on both sides of an optimization-only comparison. `Search: OLD` retains the release's uncapped regression behavior.

## Evidence and limits

The original 100-lag Chunk 16a job-1 workspace comparison reduced individual-process peak RSS from 6.386 to 3.967 GiB for HF, and from 3.716 to 2.784 GiB for LF. All saved fields and 1,400 coherence count tuples matched. Elapsed time increased about 1–2%; job-2 HF measured about 3%. GC interval 16 subsequently removed about 22.5 s HF / 13.4 s LF of explicit cleanup in job-1 pilots. Do not add these separate measurements into an inferred total speedup.

Final paired jobs 3/4 and held-out job 5 preserved all 854 saved events and 4,200 coherence tuples. Peak RSS decreased 15–38%, depending on transform shape; all six candidate runs were about 2–4% faster. Short sky/injection runs measured about 1.4–2.1% elapsed overhead. The full 2,399-lag HF/LF jobs completed at 4.002/2.865 GiB peak RSS, with exact first-100-lag output prefixes and early-to-late median RSS increases of 5.6/19.5 MiB. These are measured workloads, not universal performance guarantees.

A release oracle audit matched all 387,902 selected pixel occurrences and connected components across 1,400 lag/resolution pairs. Small likelihood floating-point differences remain; matching counts is not a claim of bitwise cWB likelihood equivalence or statistical detection-efficiency certification. Detailed combined and operational acceptance results are recorded in the parent workspace's `runs/final_validation` and `runs/operational_validation` campaigns.

Direct conditioned-strain max-energy input stays disabled because an intermediate LF cluster changed. Band-limited and sparse TD storage are not part of this profile. Sparse tiles saved additional memory in pilots, but strict segment-boundary coefficient exactness was not established. CPU JAX is not assumed faster than Numba: the backend choice follows full-workflow measurements.

The optional GPU implementation uses a deterministic ordered inverse. Full CPU/GPU pilot outputs matched, but host memory increased and kernel speedups did not translate into comparable whole-pipeline gains. GPU is not selected by this CPU profile.

Short-segment caveat: final job-4 HF uses a 1,210-second padded segment, giving 9,680 time bins at M=1024. That unaligned shape retains the original forward transform. Its measured peak reduction is 6.315→5.264 GiB (16.6%), with exact outputs and shorter runtime, rather than the approximately 38% reduction seen for aligned HF segments. Do not assume every HF worker fits within 4 GiB; inspect segment geometry and production sky settings.
