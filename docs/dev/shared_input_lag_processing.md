# Shared-input lag processing

`process_job_segment_parallel.process_job_segment` uses the native segment
preparation and output path, and distributes background lags across spawned
processes. The scientific lag analysis is unchanged. Each process has its own
Python interpreter and GIL; this option needs neither a free-threaded Python
build nor GIL-releasing Numba wrappers.

## Configuration

```yaml
segment_processer: pycwb.workflow.subflow.process_job_segment_parallel.process_job_segment
parallel_lag_workers: 6
parallel_lag_inner_threads: 1
```

Allocate six physical CPU cores to the segment. Before starting Python, set
`NUMBA_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1` to avoid nested CPU parallelism in lag workers. These settings
do not necessarily limit JAX preparation to one core. The scheduler allocation
or CPU affinity should bound the entire job, including preparation and outputs.
Multiple simultaneously processed segments each create their own lag pool;
account for this when configuring batch-level concurrency.

Run from the updated checkout, or install it in the environment used by both
the parent and its workers. Custom Python launchers must protect execution with
`if __name__ == "__main__":` so spawned workers can safely import the launcher.
The native segment processor remains the default unless explicitly selected.

## Execution and ownership

1. The segment parent loads, conditions and prepares the detector data once.
2. Completed lags are removed using the native resume records. No remaining
   lags means no serialization or worker startup. One effective worker runs
   serially; injection trials also use serial analysis.
3. The parent writes the prepared `LagAnalysisContext` to an uncompressed
   joblib file in a temporary directory inside the job working directory.
4. Each spawned worker loads that file once with `mmap_mode="c"`. Eligible
   numerical arrays use copy-on-write maps: input pages are physically shared,
   while writes are private to a worker. Python metadata and non-mappable
   objects are deserialized separately in each process. The analysis must treat
   prepared inputs as read-only: a private write can persist across that worker's
   later lags. Copy-on-write is isolation between workers, not a per-lag reset.
5. Workers receive lag indices, call the native `_run_lag_analysis`, and return
   `LagResult` objects. Results are serialized back to the parent. Each worker
   also performs its own bounded garbage collection without initializing JAX
   for cleanup.
6. The parent verifies the returned lag index and saves each result through the
   native output path. Workers never receive the output context and never write
   catalogs, trigger products or progress records. Each worker has its own log.

At most twice the effective worker count is submitted but not yet saved. The
parent drains completed results before submitting replacements, then releases
its references to those results. Completion order may differ from lag order;
there is no promise of ordered output arrival.

The process pool uses `spawn`, avoiding unsafe inheritance of initialized JAX
or OpenMP runtimes through `fork`. On an analysis or output exception, cancellation is requested for
outstanding futures and the exception propagates. Work already dispatched to
the pool may no longer be cancellable; it is joined before the shared file is
deleted. Already committed lags retain the native resume
semantics; this executor does not add transactions around partial output writes.

## Memory and scratch

The parent retains its original prepared state for output handling. Workers
need private interpreters, scratch arrays and results. Sharing inputs therefore
reduces duplication but does not imply constant total memory as worker count
increases. Monitor total job memory as well as memory per allocated core.

Use sufficient local disk scratch for the job directory. The temporary
`.lag-inputs-*` directory is removed after workers exit, including ordinary
exceptions. A tmpfs job directory consumes RAM for the input file itself.
Filesystem page cache can count against a job's memory limit even when it is not
fully represented by process-tree proportional set size (PSS). Do not use PSS
alone as a memory reservation limit. Abrupt termination may leave scratch;
remove it only after confirming the associated job has stopped. Network-backed
scratch has not been benchmarked.

## Code map and validation

- `process_job_segment_parallel.py`: process selection, bounded scheduling,
  input-file lifetime, worker initialization and native analysis calls.
- `process_job_segment_native.py`: shared preparation, analysis, output and
  resume logic.
- `process_job_segment_nogil.py`: separate experimental thread wrappers, loaded
  only when one of the legacy thread entry points is selected.
- `test_experimental_lag_parallel.py`: copy-on-write isolation under real spawn,
  resume and serial fallbacks, result identity, exception propagation, cleanup
  and compatibility of the experimental thread wrappers.

Measure both complete-job walltime and the prepared-input lag phase. The latter
still includes input serialization, pool startup, runtime cache/JIT startup,
result transfer and output handling. Actual CPU-hours measure executed CPU time;
allocated core-hours equal the reserved cores multiplied by walltime, including
idle preparation and the final wait for slow lags.

The LF/HF 400-lag and LD 200-lag comparisons at sky order 5 validated persisted
events, progress excluding timestamps, and per-resolution coherence counts
against native serial execution. These checks do not establish equality of
every intermediate array, nor performance for other sky orders, segments or
GPU configurations. Full-run measurements and the original projections are
recorded in the enclosing benchmark workspace under
`runs/lag_parallel_full_2400` and `runs/lag_parallel_scaling_400_200`.

The full 2,400-lag LF check on six physical cores completed in 19.22 minutes
against a 19.97-minute projection from the 400-lag sample. The lag phase took
16.88 minutes, actual CPU time was 1.750 CPU-hours, and peak process-tree PSS
was 6.45 GiB. All lags completed once; the first 400 matched the earlier serial
reference exactly. The other 2,000 lags were checked for completeness and output
consistency, without a new full serial reference. These are measurements for
one segment at sky order 5, not general resource guarantees.
