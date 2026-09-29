# Configurable job execution

The default remains `execution.profile: simple`. The opt-in `scalable` profile
adds shared-frame planning, a bounded raw-input cache, and supervised segment
processes to the existing local and cluster commands. `segment_processer`
continues to select the scientific implementation independently.

```yaml
execution:
  profile: scalable
  memory_limit: 32GiB
  worker_memory: 8GiB
  cache_limit: 2GiB
  headroom: 1GiB
  preload: auto
  batch_size: 8
  cores: 8
```

These sizes are an example, not a recommended limit for every search.
`worker_memory` must cover a whole segment process tree, including its lag
workers, conditioned arrays, selected pixels, numerical scratch and libraries.
Measure representative high-occupancy jobs for each search configuration and
leave margin. Increasing sky resolution, pixel selection, segment duration or
lag concurrency can require a larger reservation even with identical input files.

## Commands and configuration

Use the normal `pycwb run`, `pycwb batch-setup`, and `pycwb batch-runner` commands.
Cluster setup persists explicit bounded groups in self-contained catalog
fragments for both shared-filesystem and file-transfer runs. Generated
Slurm/Condor scripts select them using `--batch-id b000000`; scientific job IDs
remain unchanged. `--jobs` remains available and cannot be combined with
`--batch-id`. Each fragment contains its own job metadata, so execute nodes do
not require the submit-side catalog. A missing batch fragment requires running
`batch-setup`; there is no fallback to a separate scheduling file. Staged frame
basename collisions are rejected.

| Setting | Default | Meaning |
| --- | --- | --- |
| `profile` | `simple` | `simple` or `scalable` |
| `planner`, `executor` | unset | Optional dotted, zero-argument factories |
| `memory_limit` | detected | Allocation ceiling; clipped to available resources |
| `worker_memory` | `6GiB` | Conservative private/analysis reservation per segment tree |
| `cache_limit` | `1GiB` | Maximum raw sample payload; actual allowance can be smaller |
| `headroom` | `512MiB` | Unallocated safety margin |
| `message_limit` | `64MiB` | Maximum serialized output message |
| `worker_shutdown_timeout` | `60` seconds | Deadline to exit after reporting completion; a timeout fails and cleans up the allocation |
| `preload` | `auto` | `off`, `auto`, or `batch` |
| `batch_size` | `8` | Maximum tasks in a planned group |
| `cache_entries` | `256` | Maximum live mapped cache entries |
| `cores` | allocation affinity | Optional total logical-CPU cap; SMT siblings are separate affinity IDs |

Memory sizes accept nonnegative integer bytes or explicit SI/IEC units, such as
`500MB` or `2GiB`. Worker/message/explicit allocation limits must be positive.
Unknown settings and invalid values are rejected, including catalog-loaded
configuration. Existing catalogs without an execution block use the simple path.

## Planning and raw input reuse

The planner groups jobs sharing frame/channel sources using a deterministic,
bounded greedy search. It does not combine scientific windows. Physical request
intervals include padding and superlag offsets. Metadata plans contain task
indices as well as scientific IDs, so repeated trial selections remain distinct.
Batch selection reads membership from the prepared catalog fragment. Setup
rejects changes to existing batch job definitions or membership; use a new
working directory when changing the grouping.

Only sources reused within a group are candidates for caching. Overlapping or
adjacent requested intervals are coalesced; large unused gaps are not decoded
into the cache. The supervisor writes immutable arrays into temporary local
`.npy` files and supplies read-only mapping descriptors to spawned workers.
Each job receives an owned copy before calibration, timestamp shifts or other
mutation. The cache validates file stat fingerprints and sample alignment.

`off` uses direct input reads. `auto` loads a reusable source's planned interval
when needed. `batch` attempts to preload a group's union when it fits; otherwise
it falls back to bounded demand loading. This is synchronous bounded loading,
not asynchronous background prefetch. Unsupported custom processors execute
with direct reads and a warning.

Leases pin active entries. Unpinned entries are evicted in least-recently-used
order, subject to both payload bytes and entry count. An entry that cannot fit
is bypassed. Cache files are removed on normal completion and handled failures;
an abrupt supervisor kill can leave a temporary directory to remove manually.
Place the work directory on storage suitable for local memory mapping. A tmpfs
cache also consumes allocation RAM.

## Memory and CPU accounting

Admission uses the minimum of the explicit ceiling, host availability, visible
cgroup-v2 headroom and the explicit `--memory-limit` allocation ceiling. Scheduler
adapters pass `--allocated-cores` and `--memory-limit` to the batch runner. CPU affinity and scheduler CPU counts bound segment
slots, with each slot reserving the configured lag workers and inner threads.

For each segment, the reservation is at least `worker_memory` and at least an
input estimate derived from frame duration, channel rate, read concurrency and
owned input buffers. In particular, a short requested slice can require decoding
a whole large GWF frame. Cache decode admission reserves that larger temporary
buffer too. Incorrect or missing frame metadata can still make an estimate low.

Cache decoding runs in a disposable subprocess, so its allocator does not grow
the supervisor across many inputs. Decoder admission and periodic sampling
cover the decode phase as well as analysis. A monitor records process-tree
RSS/PSS, logical cache payload, available RAM and host swap growth. It stops
workers when observed accounted memory reaches the safety margin. PSS plus cache
payload deliberately overcounts mapped pages rather than treating them as free.

These are conservative reservations and sampled checks, not a hard OS memory
limiter or a model predicting pixel counts. An allocation can jump between
samples. Use scheduler/cgroup enforcement for a hard ceiling, and increase
`worker_memory` for larger analysis peaks. The runtime fails rather than silently
changing numerical selection settings to fit memory. It does not budget GPU RAM.

## GPU processors

The experimental `workflow.subflow.process_job_segment_gpu` accepts the same input provider.
When supplied, it uses the supervisor's bounded read path rather than spawning
its own independent frame readers. Its CUDA numerical stages remain unchanged.

The execution budget is host RAM only. For the single-device GPU configuration,
use one segment at a time (`batch-runner --n-workers 1`); multiple lag workers
inside that segment can already occupy most VRAM. The processor declares the
maximum concurrency from `gpu.lag_workers` and the preparation worker counts
for CPU-slot accounting. Device admission is not inferred from available host RAM. The existing optional GPU
catalog batcher still owns its writes; enabling raw-input reuse does not replace
that output path or establish new GPU-product support.

## Output, failures, and resume

One supervisor writer owns each catalog fragment. Worker output transport is
bounded to one acknowledged message per worker. Trigger products are flushed
before corresponding progress is saved; write errors fail execution. This uses
the existing catalog atomic-write semantics and does not add power-loss fsync
guarantees. Failed workers and their nested process groups are stopped, then
cache leases and temporary files are released.

Completed lags are resolved before input access. Resume skips committed work and
uses the existing stale-trigger cleanup. Per-fragment and per-job locks prevent
simultaneous execution into the same output namespace on a shared filesystem.
Keep batch membership and scientific configuration stable when resuming. This
version does not reconcile progress across regrouped fragments or validate a
complete scientific configuration identity against a previous run.

Plans and metrics for executed fragments are written under `RUN/execution/`.
The runtime plan describes pending work, so a resume can replace it with a
smaller plan; the root cluster plan retains submission membership. Metrics record
completed tasks, failure details, actual cache/direct reads, payload bytes,
reservations, elapsed time and sampled memory. They do not measure physical disk
bytes or distinguish operating-system warm and cold caches.

## Extensions

See `pycwb/workflow/execution/contracts.py` for version-one protocols:

- A planner factory returns an object with `plan(jobs, config, settings)` returning
  an `ExecutionPlan`. Its task permutation and input requests are validated.
- An executor factory returns an object with `execute(plan, context)`.
- A processor opting into cached input sets `supports_input_provider = True`
  and accepts `input_provider`; its `read` returns an owned GWPy TimeSeries.

Factories must be importable on every execute node. The simple executor adapter
is available for a custom planner with legacy processing. There is no separate
job creator setting: the existing scientific segment generator is shared by both
profiles. Configuration-specific analysis cost prediction, persistent compute
workers, conditioned-data caching and distributed caches are future work.

## Verification and measurements

The execution tests cover settings, deterministic grouping, explicit scheduler
membership, transferred fragments, raw sample identity, mutation isolation,
pinning/eviction, source changes, resume, writer failure, worker death, and memory
pressure. A parameter matrix varies frame duration, rate and cache capacity;
resource tests separately vary analysis and full-frame decode reservations.

Run these alongside scheduler and lag processor regressions:

```sh
python -m pytest pycwb/workflow/execution/tests \
  pycwb/modules/slurm/tests pycwb/modules/condor/tests \
  pycwb/modules/workflow_utils/tests pycwb/modules/catalog/tests \
  pycwb/modules/read_data/tests \
  pycwb/modules/job_segment/tests/test_experimental_lag_parallel.py
```

`python -m benchmark.execution --help` describes the input-reuse benchmark.
Use both generated inputs and an independent existing job manifest, compare all
input digests, and record timing and memory. `--scratch-mib` adds variable private
working buffers to exercise accounting; it is not a substitute for real
high-occupancy scientific workloads. Fresh worker startup and cache decoding can
make small jobs slower. Keep this profile opt-in and measure end-to-end searches,
not just the reduction in frame read calls.
