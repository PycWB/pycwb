# Systematic job preparation and execution

Status: design proposal, September 20, 2026. An initial opt-in implementation
is now available; see [scalable execution](scalable_execution.md) for the actual
settings, behavior, limitations and validation commands. The proposal below also
contains future work; it is not a claim that every acceptance gate is complete.
The default profile remains simple.

Revision: select simple/scalable execution through configuration, using the
existing commands. Separate command families are no longer the recommended
user interface.

## Recommendation

Build an execution layer spanning preparation, local runs, and cluster batches.
Keep scientific jobs and their identities intact, but schedule them in bounded
groups that reuse input files. A supervisor owns the input cache and resource
budget; replaceable analysis processes consume it. Reuse the existing native
preparation, lag analysis, and output semantics through explicit interfaces.

Start with raw frame reuse and memory admission. Add persistent compute workers
or reusable conditioned data only after separate correctness and resource gates.
Changing only `segment_processer` cannot coordinate reuse across scheduler jobs.

## Configuration and extension points

Use **job planner** and **job executor** as the public component names.
The planner chooses grouping, ordering, input reuse, and resource reservations;
the executor runs that plan and owns supervision, cache lifetime, admission,
retries, and output coordination. A supervisor is an internal component of the
scalable executor. Reserve **job generator** for the distinct operation of
constructing scientific segments from configuration/DQ/injections; the existing
generator can initially be shared by both execution modes.

Expose a simple profile switch for ordinary users and importable component
overrides for advanced users, following the existing processor convention.
All examples below are proposed configuration, not implemented settings.

```yaml
# Default for existing configurations, including when execution is absent.
execution:
  profile: simple
```

```yaml
execution:
  profile: scalable
  memory_limit: 32GiB       # illustrative allocation cap, not a required size
  cache_limit: 4GiB         # maximum; actual allowance may be smaller
  preload: auto
```

```yaml
# Advanced extension: override either component with a compatible factory.
execution:
  profile: scalable
  planner: my_package.execution.create_planner
  executor: my_package.execution.create_executor
```

| Profile | Default planner | Default executor |
| --- | --- | --- |
| `simple` | Existing ordering and fixed-count batch grouping | Compatibility adapter preserving current local/batch behavior |
| `scalable` | Bounded grouping by shared inputs and resource estimates | Cache-owning supervisor with memory admission and bounded dispatch |

Keep `segment_processer` as the existing, independent science-processing
selection, including its current spelling for backward compatibility. Switching
execution profiles must not silently change the selected numerical processor.
Both profiles use the same scientific job generation and stable identities.
Add a configurable job generator only when there is an actual alternative
scientific segmentation use case; grouping alone belongs in the planner.

Resolve profile defaults first, then explicit component/resource overrides.
Resolve CLI overrides before persisting the plan, so submitted workers receive
the same effective settings. Add the nested execution schema, validate resource
units and bounds, and report the resolved factories in plan inspection/logs.
Catalog-loaded configuration must also be normalized and validated at the
execution boundary, including legacy catalogs with no execution block.

Define small, versioned contracts:

- `JobPlanner.plan(jobs, config, resources) -> ExecutionPlan`: deterministic,
  serializable metadata; no strain decoding, numerical runtime initialization,
  or analysis-output writes. Shared preparation owns scientific job generation.
- `JobExecutor.execute(plan, processor, options) -> ExecutionSummary`: owns
  process lifetimes, input-provider services, resource admission, cancellation,
  and completion/failure reporting. Support a selected batch of the same plan.
- `SegmentProcessor`: preserve the current callable via an adapter. A declared
  optional input-provider capability enables caching without changing the
  numerical implementation. Legacy custom callables receive no new unsupported
  keyword arguments; direct-read fallback must be explicit in the plan report.

Validate plan schema and required capabilities when combining components. A
custom planner and executor may be mixed when their contracts are compatible;
reject incompatible combinations before submission or input loading. Require
the resolved plugins to be importable on execute nodes and record their version
identities. Cluster backend (`slurm`/`condor`) remains independent of the profile.

The common flow becomes: configuration resolution -> job generation -> selected
planner -> selected executor -> selected segment processor. Add new internal
implementations while keeping existing run and batch commands as dispatchers.

## Findings in the current checkout

| Area | Current behavior | Implication |
| --- | --- | --- |
| Preparation | `workflow/subflow/prepare_job_runs.py` builds segments, creates directories, changes cwd, and initializes the catalog | Separate plan construction from execution and output initialization |
| Local run | `workflow/run.py` runs segments in sequence; its preparation call supplies `n_proc=1` | Introduce a common runtime with explicit resource settings |
| Batch run | `workflow/batch.py` submits all pending segments and uses `max_tasks_per_child=1` | A cache inside an analysis worker disappears after each segment |
| Scheduler grouping | Condor and Slurm partition consecutive job IDs by `job_per_worker` | Membership should be an explicit list chosen using input reuse and resource estimates |
| Condor transfer | Frame paths are already deduplicated within each current batch | New gains must distinguish transfer reuse from decoded-data reuse |
| Frame reading | `read_from_job_segment` creates a read pool when configured, reads each frame intersection, then merges | Repeated jobs can reopen/decode the same files; reads and temporary copies need bounds |
| Mutation | Frame loading relabels timestamps; calibration/rescaling and injections modify buffers | Cache unmodified physical samples and give each job owned mutable data |
| Lag parallelism | `process_job_segment_parallel.py` shares prepared arrays using a temporary joblib mapping within one segment | Preserve this path; raw-frame caching solves a different lifetime problem |
| Memory | Explicit reference releases, GC, heap trimming, and worker replacement already exist | Add allocation-level accounting and admission rather than assuming GC controls peaks |
| Output | The collector orders trigger flushing before progress, but logs and suppresses write failures | The new runtime must propagate failures and acknowledge durable completion |

Relevant implementation: `modules/job_segment/job_segment.py`,
`modules/read_data/read_data.py`, `modules/read_data/data_check.py`,
`modules/catalog/catalog.py`, and the workflow files above.

### Concrete opportunity and limits

Read-only inspection of
`runs/lag_parallel_full_2400/BurstLF_processes_w6_n2400/catalog/jobs.parquet`
in the enclosing workspace found five job definitions, 12 frame references,
and four distinct 4,096-second frame files. Each file is referenced three times.
The full benchmark executed job 1, not all five jobs.

Across those five definitions, requested channel-time totals 12,160 seconds;
its union is 12,000 seconds. Only 1.32% of the requested samples repeat. Under
a float64 decoded-storage assumption at 16,384 Hz, the union occupies about
1.465 GiB, while decoding all four full channels occupies 2 GiB. These are
payload estimates, excluding decoding buffers, copies, and analysis state.

The recorded job-1 log reports 21.51 seconds for reading, 19.08 seconds for
conditioning, and 81.54 seconds for coherence setup. Complete time was 19.217
minutes on six cores. Even eliminating that job's reading entirely would remove
only about 1.9% of its measured walltime. This is one warm-cache LF observation,
not a forecast for simulations, superlags, short jobs, or other storage systems.

Distinguish three opportunities in measurements:

1. Repeated file transfer/open/decompression, even for different sample windows.
2. Repeated sample windows, particularly across trials and superlags.
3. Repeated scientific preparation, reusable only when its complete inputs match.

## Proposed architecture

```text
configuration + existing scientific segment construction
                       |
              prepare_execution_plan
                       |
      immutable jobs + input requests + execution batches
                       |
            local runner / cluster batch worker
                       |
        supervisor: admission, raw cache, prefetch, writer
                       |
           replaceable segment analysis processes
                       |
       existing native preparation and lag executor
                       |
          existing products + committed progress
```

Use a small `workflow/execution/` package for contracts, planner, frame store,
resource accounting, runtime, and scheduler adapters. Extract shared functions
from existing workflows where appropriate. Avoid a second copy of numerical
processing or independent local/cluster orchestration implementations.

### 1. Prepare an explicit execution plan

Implement `prepare_execution_plan(...)` with these records:

- **Job specification:** existing job ID, trial identity, exact detector order,
  physical padded windows, channels, rates, lag vectors, vetoes, and injections.
  Apply job/trial/lag selection before calculating reuse. Preserve existing
  catalog IDs, including IDs assigned by the current trial-flattening logic.
- **Input request:** logical source identity, source fingerprint, detector and
  channel, physical sample interval, rate/dtype metadata, and consumer job IDs.
- **Execution batch:** stable batch ID, explicit ordered job-ID list, required
  inputs, estimated decoded bytes and walltime, and CPU/RAM/scratch requirements.
- **Plan metadata:** schema and planner versions, immutable job-manifest identity,
  resolved scientific configuration identity, resource policy, and input mapping.

Retain `catalog/jobs.parquet` as the authoritative scientific job manifest.
Store scheduling and deduplicated input metadata in versioned sidecars linked
to it. Do not duplicate large frame metadata into each execution record.
Serialize only descriptors, never decoded arrays, into the plan.

Separate configuration resolution, metadata validation, planning, and catalog
initialization. A plan-only operation may inspect local metadata but should not
start analysis, decode strain, submit jobs, or initialize analysis runtimes.
Resolve relative paths explicitly rather than relying on process-wide cwd.

Materialize stochastic choices before execution, or retain explicit seeds keyed
to stable scientific identities. Changing schedule must not change noise,
injections, or generated lag vectors. Verify this against current seeded behavior.

### 2. Group jobs by reuse without merging analysis windows

Build an inverted index from `(source identity, channel)` to jobs. Within each
candidate group, use physical padded intervals, including superlag shifts, to
estimate sample overlap and incremental decoded bytes. Preserve detector/channel
identity when only one detector input is shared.

Use deterministic greedy grouping initially: prefer the next job offering the
most estimated avoided input cost per additional retained byte, subject to a
batch walltime limit, cache budget, job-count cap, and CPU/memory requirements.
Use stable job-ID ordering to break ties. Estimate preparation/lag cost for load
balancing so one high-reuse batch does not dominate campaign completion time.

Do not use connected components as final batches: adjacent padded windows can
connect an entire campaign. Bounded groups may overlap in their input sources.
Use the inverted index and interval sweeps rather than comparing every job pair.
For disjoint or low-reuse jobs, use ordinary balanced scheduling with no preload.

Grouping changes execution and transfer membership only. Each job retains its
analysis window, conditioning boundaries, vetoes, lag identities, and outputs.
Keep a batch's membership fixed for retries; schedule only its pending work.

### 3. Introduce an input-provider interface

Add an explicit optional input provider to the shared segment preparation path.
The current direct reader remains the fallback. A new optimized runtime supplies
the cached provider; it calls the existing scientific stages afterward.

Cache raw channel samples in their physical time coordinate, before timestamp
shifts, DC calibration, resampling, synthetic noise, injection, or conditioning.
Use immutable cached arrays with independent job metadata. Materialize an owned
job buffer before the first mutating operation. Never expose a writable alias
of the shared raw cache to a trial or an analysis worker.

Cache keys include source identity/fingerprint, channel, sample interval,
sample rate, dtype, and reader/schema version. Use integer sample offsets after
validating alignment. Keep logical source identity separate from staged local
paths, and avoid basename collisions during cluster file transfer. Validate
source changes; use supplied checksums when available, with documented stat
fingerprints for run-scoped local caches rather than hashing all files repeatedly.

Two candidate read granularities need measurement:

- Full frame/channel decoding when reuse is high and the payload fits.
- Bounded requested intervals or tiles when whole files are large or sparsely used.

Overlapping requests should reuse covered samples; avoid rereading the entire
union whenever a request extends it. Coalesce nearby requests only when measured
reader costs justify the extra decoded data. A backend may decode larger blocks
internally, so requested bytes alone are not evidence of physical I/O savings.

Construct the exact original padded job window and preserve existing frame
ordering, gap/error handling, dtype, sample rate, and epoch behavior. Stream or
copy frame slices into a preallocated destination when validated against the
current merge behavior, bounding transient memory.

### 4. Keep cache lifetime independent of compute-worker lifetime

For the first version, run one supervisor per local allocation or cluster batch.
It owns cache metadata, byte reservations, and bounded read/decode workers.
Concurrent requests for the same missing entry share one in-progress read.
Use read-only local array mappings and small IPC descriptors to share cached
inputs with spawned analysis processes, avoiding serialized multi-hundred-MiB
array transfers. The source cache has a bounded disk quota and a separate
resident-memory target; disk backing does not make resident pages free.

Retain replaceable segment processes initially. Their termination can reclaim
JAX/Numba/allocator state without discarding the supervisor's raw cache. Pin cache
entries with leases while readers consume them, and release leases on completion
or worker death. Remove backing files only after users have exited or unmapped.

Continue using the current per-segment prepared-input lag sharing. Account for
the parent's originals, serialization peak, mapped pages, private worker state,
and result queues. Sharing raw frames does not eliminate these allocations.

Longer-lived compute workers and reusable lag pools are a later experiment with
explicit context reset and measured memory-growth limits. Do not simply remove
`max_tasks_per_child=1` from the current executor.

### 5. Add memory admission and bounded prefetch

Choose the effective allocation ceiling from explicit user limits, scheduler /
cgroup constraints, and host availability. Missing telemetry means conservative
limits and disabled speculative prefetch, not unlimited cache allocation.

For each proposed stage transition, require a conservative estimate satisfying:

```text
supervisor and writer overhead
+ raw-cache resident budget and decode-in-flight reservations
+ shared prepared state, including construction/serialization overlap
+ sum of active workers' private peak requirements
+ job-owned raw buffers and queued-result allowances
+ safety headroom
<= effective allocation memory budget
```

Estimate private peaks by search configuration and stage, including segment
length, sample rate, resolutions, sky order, lag concurrency, and injections.
Calibrate using measured peaks with margins. Account for shared storage once,
and private copies separately. Treat GPU memory as a separate budget if enabled.
Reserve CPU capacity for preparation, I/O, lag workers, and inner numerical
threads jointly; do not multiply independent concurrency settings unchecked.

Support `preload=off|auto|batch` and an explicit maximum cache size:

- `off`: direct reads, useful as a correctness and performance baseline.
- `auto`: demand caching plus a small look-ahead of scheduled inputs, bounded by
  both bytes and concurrent reads. Prefetch only inputs with likely near reuse.
- `batch`: preload the selected batch's union only when the entire reservation,
  including decode scratch and active analysis peaks, fits. Otherwise fall back
  to bounded loading and report that decision.

Evict unpinned entries with no remaining consumers first, then prefer eviction
of large entries with distant next use. Use high/low watermarks to avoid repeated
eviction and rereading. Under pressure, stop prefetch, evict idle cache, and stop
new stage/job admission. Change compute concurrency only at safe boundaries.
If even one job cannot fit the configured budget, fail before launching it with
the estimate and required resources. Forecasts reduce OOM risk but cannot promise
that an unmeasured numerical peak will fit; retain the external allocation limit
and report reservation overruns for subsequent planning.

Track allocation/cgroup memory including file cache where available, process
RSS/PSS for diagnosis, cache resident/pinned bytes, decoder reservations, queue
bytes, page faults, and swap growth. GC and heap trimming remain cleanup tools;
they do not replace these limits. Disk-backed mappings on tmpfs consume RAM.

### 6. Route preparation, local, and batch entry points through configuration

Retain the existing user-facing command family. These commands dispatch using
the resolved execution profile and components:

```text
pycwb prepare CONFIG --work-dir RUN --plan-only
pycwb run CONFIG --work-dir RUN
pycwb batch-setup CONFIG --work-dir RUN --cluster slurm|condor
pycwb batch-runner CONFIG --work-dir RUN
```

`prepare` is an optional new plan-inspection/persistence command. Extend existing
commands with plan/batch selection as needed; there is no requirement to learn
`run-plan`, `batch-plan`, or a separate optimized command family. Existing
`--jobs` selection remains supported. A new `--batch-id` selects a persisted
execution batch in the scalable runner; ambiguous simultaneous selectors should
be rejected. Old configurations use the simple compatibility adapters.

Preparation reports predicted reuse, chosen grouping, input and scratch bytes,
memory reservations, and estimated batch costs. Expose total allocated cores,
segment concurrency, lag concurrency, inner threads, and I/O concurrency with
validation and sensible derived values. Optional one-shot local execution can
compose prepare and run through these same APIs.

Cluster adapters submit explicit batch IDs instead of deriving consecutive
scientific job ranges. Transfer only each batch's immutable metadata and unique
source files. Name catalog/progress/wave fragments by stable batch identity,
while preserving original job/trial/lag identities inside them. Keep merging
and simulation-summary behavior compatible. Test shared-filesystem and Condor
file-transfer modes separately. Request resources from the plan's estimates.

Keep existing commands available during rollout. Unsupported custom processors
use direct input loading until they adopt the provider contract. Online reading,
cross-node cache services, and cross-run scientific-state reuse are later scope.

### 7. Preserve reliable outputs and resume

Resolve completed jobs and trials before reading or prefetching their inputs.
Keep one writer per output fragment, with bounded byte-aware transport, failure
propagation, and acknowledgements after required products are durable. A progress
record must not acknowledge a lag whose outputs failed to save. Block new work
when the writer cannot keep up and surface writer death promptly.

Retry with stable job/trial/lag IDs and existing stale-output cleanup semantics.
Exercise crashes between output and progress writes. Batch failure retries only
uncommitted work; cache eviction or supervisor restart affects performance only.
Reject incompatible manifest/config identities. Regrouping a partially completed
campaign requires explicit progress reconciliation across old fragments and
exclusive output ownership; it must not silently duplicate completed science.

## Implementation sequence and acceptance gates

| Phase | Deliverable | Gate before proceeding |
| --- | --- | --- |
| 0. Baseline | Metadata overlap report and stage/resource instrumentation for multi-job workloads | Distinguish file references, sample reuse, actual reads/decodes, startup cost, and end-to-end time |
| 1. Planning | Execution schema/profile resolution, component contracts, simple adapters, typed plan, input index, bounded grouping, plan inspection CLI | Old configurations retain behavior; deterministic plans; unchanged scientific identities; valid plugin combinations; correct superlag requests and nonconsecutive membership |
| 2. Cached local execution | Input-provider seam, supervisor cache, owned job buffers, memory admission, scalable `run` dispatch | Raw inputs and persisted outputs match; low-memory and disabled-cache fallback work; mutation cannot contaminate another job |
| 3. Prefetch and throughput | Bounded look-ahead, stage reservations, coordinated segment/lag concurrency | Benefit measured at fixed CPU/RAM; allocation memory bounded; no swap growth or long-run cache/worker growth |
| 4. Cluster integration | Profile dispatch in `batch-setup`/`batch-runner`, Slurm/Condor adapters, stable fragments and writer acknowledgements | Shared-filesystem and transfer-mode retries, writer failure, worker death, and resume pass |
| 5. Optional reuse | Persistent compute contexts or exact-window conditioning cache | Independent numerical-identity and sustained-memory gates; retain only demonstrated improvements |

Phase 5 conditioned-data keys must include the exact physical window, processing
configuration, calibration, boundaries, source fingerprint, and noise/injection
identity. Two segments using the same frame do not have interchangeable whitening
or WDM products. Start with exact identical background inputs; trial-dependent
data and differing LF/HF/LD processing require separate proof.

## Validation campaign

Compare the new runtime against the existing native/shared-input paths using
the same source/config/input identities, seeds, hardware allocation, and outputs.

- **Input correctness:** multiple files and channels, partial overlap, nonzero
  superlags, detector-specific reuse, boundaries, gaps, rate mismatch, stale
  sources, staged-path collisions, and concurrent requests for one cache miss.
  Compare raw arrays and timestamps exactly before scientific processing.
- **Scientific preservation:** LF/HF/LD, adjacent and short segments, seeded
  injections and multiple trials. Compare persisted fields, waveform datasets,
  progress excluding timestamps, and existing intermediate selection/coherence
  checks. Preserve lag-major processing and usable output after each committed lag.
- **Resource behavior:** cache disabled, demand-only, and bounded prefetch;
  generous and restrictive allocation limits; decode/prepare peak overlap;
  slow writer, many large results, worker recycling, and long sequences of jobs.
- **Failure behavior:** missing/corrupt input, cache eviction while entries are
  pinned, worker/supervisor death, writer failure, partial output, restart, and
  batch retry. Verify clear failure and absence of false completion records.
- **Performance:** adjacent jobs sharing files, superlags sharing detector data,
  repeated-input trials, and disjoint jobs. Separate cold and warm-cache runs
  where controlled measurements are possible without disturbing other work.
  Repeat representative runs and hold out at least one segment/configuration.

Report complete campaign walltime, time to first result, jobs/lags per second,
actual CPU-hours, allocated core-hours, read/decode calls and bytes, cache hits
weighted by bytes, prefetch wastage, scratch use, and total allocation peak
memory. A drop from 12 frame references to four reads is not a threefold I/O or
end-to-end speedup claim. Use paired measurements to decide promotion; define
acceptable regression thresholds after baseline variability is measured.

The first useful milestone is phases 0–2: a reviewable execution plan and a local
runner that reuses raw frames across jobs under an explicit memory budget. This
establishes the contracts needed for batch integration without changing the
scientific algorithms.

CPU-only high-core execution is part of phase 3, not an optional GPU follow-up.
See [CPU/HPC pipeline experiments](cpu_hpc_pipeline_experiments.md) for the
112-lags-per-worker local workload and the separate 128-core acceptance plan.

The follow-up [job-management decision](job_management_decision.md) narrows the
recommended next step to a bounded opt-in pipeline. New CPU measurements against
an optimized simple loop show a 14% wall-time gain with substantial extra RAM;
they do not justify replacing the simple default or claim 128-core performance.
