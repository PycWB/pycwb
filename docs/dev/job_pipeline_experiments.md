# Whole-job pipeline experiment plan

Status: executable and measured experiment in `benchmark/job_pipeline.py`, not a
production profile change. Measurements and the decision are recorded below.

## Objective

Improve complete multi-job throughput by overlapping upcoming input/preparation
with current analysis. Cache hit rate alone is not the optimization target.
Keep numerical settings, scientific windows and required outputs unchanged.

## Candidates

1. **Sequential baseline:** existing GPU processor, all six physical cores / twelve
   logical CPUs available, six GPU lag workers, one GPU segment at a time.
2. **Read-ahead:** a bounded CPU producer reads the next job while GPU analysis
   runs; ready raw arrays are passed by file mapping rather than through IPC.
3. **Condition-ahead:** the producer also resamples and conditions the next job.
4. **Full CPU preparation ahead:** produce the existing lag analysis context on
   CPU, including coherence/TD/sky preparation, then hand it to the GPU consumer.
   This avoids a second simultaneous CUDA setup context; exact comparisons must
   establish that the existing CPU preparation preserves the GPU preset's outputs.
5. **Persistent preparation/consumption processes:** retain process and compilation
   state across jobs at the conditioning boundary, keeping the same queue limits.
6. **Prepare during analysis:** with persistent processes, release the next
   preparation only after the current consumer enters its lag-analysis phase.
   This tests whether avoiding competition with consumer setup helps.

Start with queue depth one, one producer and one GPU consumer. Do not run two
full GPU segment trees concurrently on the 12-GiB device. Prefer shared CPU
availability initially; test a physical-core partition only if contention makes
it worthwhile. CPU affinity must distinguish physical cores from SMT siblings.

## Bounds and ownership

The producer reserves RAM before reading/preparing. Limits cover an active
consumer, producer peak, ready payload, serialization/file-backed pages and
headroom. Queue count alone is insufficient. Publish complete artifacts atomically,
release them only after consumer shutdown, and stop both process trees on errors.
A producer uses CPU-only JAX/CUDA visibility. Only the consumer writes analysis
outputs; preparing a job must never create completion records. Record per-stage
start/end times and queue waits to prove real overlap and identify starvation.

Initially use fresh catalog-only background runs. Reject injections, stochastic
noise and unsupported products at the experimental adapter rather than pretend
that those paths have been validated. Existing numerical functions/context types
are reused. Prototype adapters stay separate from the default runtime until an
experiment demonstrates an improvement.

## Experiment sequence

- Use two adjacent LF jobs, all 2,400 lags each, for an all-machine baseline and
  each viable pipeline depth. Keep GPU flags and compilation-cache conditions
  identical; include initial fill, final drain, artifact writes and shutdown.
- Compare every persisted event field bitwise, complete unique progress excluding
  timestamps, and coherence-count records. Compare job 1 to the independent CPU
  reference as well as comparing profiles.
- Rank by full wall time and throughput, while recording CPU use, host PSS/RSS,
  accounted file payload, VRAM, swap growth, queue delay and preparation overlap.
- Confirm the chosen candidate against baseline in reversed order on a longer
  three-job holdout (jobs 3–5, including a frame-boundary crossing) for different
  input/pixel occupancy and more than one queue refill. Freeze the candidate
  before that holdout. These are confirmation measurements, not a statistical
  claim from many identical repetitions. Do not adjust
  the numerical search or reservations merely to improve the measured ranking.
- Test restrictive memory, oversized prepared artifacts, decoder/preparer and
  consumer failures, cancellation, and cleanup. A sampled monitor is a guard,
  not a replacement for an external hard cgroup/scheduler memory limit.
- Exercise CPU consumption of the same staged contexts to determine whether
  overlap helps or simply competes for already-busy CPUs.

## Decision

Keep a stage boundary only if it produces correct results and improves complete
throughput under the same resources. Report slower variants and first-fill cost.
If CPU preparation competes with GPU feeding/output or cannot keep up, stop at
the earlier boundary. If no variant improves throughput, retain the baseline and
report the bottleneck rather than promote the architecture on intuition.

## Prototype interface and safeguards

`run_pipeline(directory, config, jobs, catalog, stage, ...)` accepts `read`,
`conditioned`, or `full` preparation, plus CPU/GPU consumption and a serial mode
for measuring the same stage implementation without overlap. `persistent=True`
reuses one producer and one consumer process. `defer_prepare=True` starts upcoming
preparation after the consumer has entered analysis. The caller creates
a fresh output catalog and log directory. This experimental entry point does not
replace the public run/batch API, resume handling, or allocation ownership locks.

The supervisor launches isolated preparation/consumption subprocesses. Preparation
uses CPU-only JAX and hides CUDA devices before Python imports; consumption owns
the existing lag processors and output routines. The checkpoint adapters use
private function bindings, leaving numerical functions and module globals intact.
The conditioned adapter bypasses already-completed native stages and is deliberately
restricted to deterministic, catalog-only background. It is not a general staged
science API. A production implementation should expose explicit preparation and
analysis functions instead of promoting this adapter unchanged.

The queue has one upcoming job and one active consumer. Both retained artifacts
are included in admission. `payload_limit` is an upper bound: spare available RAM
after the producer, consumer, and headroom reservations is divided between the two
artifacts. Serialization rejects a payload before writing beyond its assigned
quota. Publication is atomic and includes source fingerprints and a task token.
Consumer mappings use copy-on-write; artifacts are removed after the consumer
exits. Errors and Python interrupts terminate child process groups and remove
temporary artifacts. Forced supervisor termination is not a durable recovery API.

Producer/consumer reservations remain explicit estimates. Full-frame decode sizes
are considered, and a live process-tree PSS plus retained-file-payload monitor
aborts on unexpected growth or external RAM pressure. This intentionally
overcounts mapped pages. Different pixel occupancy can exceed an estimate; the
prototype fails rather than reducing scientific work. Sampled monitoring cannot
guarantee protection against a sudden allocation or GPU OOM. Use a scheduler or
cgroup for a hard host limit. There is one GPU segment tree, but no new VRAM
admission controller.

These experiments isolate stage overlap. They do not yet combine it with the
existing shared-frame cache. Once input/preparation is hidden completely, cache
reuse may reduce CPU/I/O work without reducing the critical path further.

Tests: `python -m pytest benchmark/tests/test_job_pipeline.py
pycwb/workflow/execution/tests -q`. These cover byte quotas, variable payloads,
queue bounds, CPU-only preparation, memory admission/pressure, failure and
interrupt cleanup, plus the existing frame-size/rate/cache-pressure matrix.

## Measured decision, 2026-09-20

Hardware: RTX 4070 SUPER, six physical / twelve logical CPU cores, about 30.4 GiB
host RAM. All twelve affinity IDs were available. Initial GPU workload: two LF
jobs, 2,400 lags each, six GPU lag workers, one segment consumer. These are complete
command times, including preparation, serialization, startup, output and teardown.

| Variant | Seconds | Reduction versus baseline |
| --- | ---: | ---: |
| Sequential baseline | 341.11 | — |
| Read ahead | 340.81 | 0.09% |
| Read and condition ahead | 332.74 | 2.45% |
| Full CPU preparation ahead | 365.90 | -7.27% |
| Persistent read/condition processes | 323.70 | 5.10% |
| Persistent, preparation during analysis | 323.22 | 5.24% |

The two persistent policies are effectively tied; their sub-percent difference
does not justify making phase gating mandatory. Persistence here applies to the
producer and consumer coordinator. Existing reader pools and lag workers still
start per job. Full CPU preparation increases contention with GPU feeding/output
and incurs a much larger initial fill. Keep GPU-assisted setup in the consumer.

The deferred persistent policy was frozen for a longer, reverse-order holdout:
jobs 3–5, including a frame-boundary crossing and shorter windows. Geometry limits
the shorter jobs to 2,380 lags, giving **7,160 total**, not 7,200. The pipeline took
**469.56 s versus 517.35 s**, a **9.24% wall-time reduction**. All **17,550 triggers**,
all progress rows, and all **50,120 coherence records** matched exactly. Host PSS
increased from **9.45 to 12.23 GiB**. Sampled CPU use increased from 5.26 to 6.37
logical-CPU equivalents; total sampled CPU work also increased. The gain trades
additional CPU/RAM for less waiting, rather than reducing every resource cost.

A CPU-only pilot used identical full-context preparation/consumption stages with
and without overlap, persistent processes, six lag workers, and 128 lags per job.
Overlap took **284.98 s versus 328.24 s**, a **13.18% reduction**. Both runs used the
same populated CPU compilation cache (233 files before and after); all 625
triggers, 256 progress rows and 1,792 coherence records matched. This smaller CPU
workload is not directly comparable to the longer GPU workload or every other CPU
execution strategy.

All completed initial GPU candidates also preserved 11,086 triggers, 4,800
progress rows and 33,600 coherence records, including the independent native CPU
reference for job 1. Progress comparisons exclude only timestamps. **97 tests
passed** across the pipeline and existing execution suites. Search thresholds and
pixel cuts were not changed to obtain the timings or satisfy memory limits.

A separate HF preparation probe used 8,192-Hz analysis data, versus 4,096 Hz in
the LF cases. Its 153.45-MiB conditioned artifact preserved all 41 checked leaves
through serialization. This validates a different real search/payload size, not
full HF GPU throughput or VRAM admission. The CPU pilot also matched the same
128-lag prefix of the GPU run exactly.

Recommendation: use a **JobPlanner**, **JobPreparer**, and **PipelineExecutor**.
For GPU execution, begin with persistent read/resample/conditioning preparation,
one byte-bounded ready job, and GPU-assisted setup/analysis in the consumer. The
CPU pilot supports experimenting with full CPU preparation ahead for CPU execution.
Choose the boundary by backend and workload; do not assume full preparation is
always faster. Add shared-frame reuse inside the preparer after measuring its
initial-fill and contention costs. Queue depth beyond one did not address any
observed GPU consumer starvation here.

Before production promotion, extract explicit stage APIs and integrate with the
existing run/batch manifest, resume, allocation-lock and output-ownership paths.
Persistent compiler/allocator caches also need budget-aware recycling between
jobs; a bounded queue alone does not bound retained process state. The existing
simple/scalable profiles were not changed to call this experimental adapter.

These are sequential measurements with warm Numba caches and normal OS caching,
not statistical confidence intervals or guarantees across GPUs/search settings.
The earlier 410.98-second result used six logical CPUs (three physical cores), so
its allocation difference must not be counted as a pipeline gain. Raw measurements,
source snapshots, comparisons and the detailed report are in the local
`../runs/job_pipeline_20260920/` bundle relative to the repository root.

## CPU and HPC extension

The follow-up [CPU/HPC experiment plan](cpu_hpc_pipeline_experiments.md) treats
CPU-only execution as a first-class target. It uses 672 lags per job on six
workers (112 per worker on average), separately measures private worker memory,
and defines the checks needed on an actual 128-core allocation. The small CPU
pilot above is not evidence of 128-core performance.

A stronger CPU baseline and explicit core-partition experiment are now recorded
in [the job-management decision](job_management_decision.md). They support a
narrow opt-in pipeline, with moderate gains and significant RAM tradeoffs, rather
than replacing the simple default. The preparer now honors
`gpu.overlap_setup: false` for sequential coherence/TD setup inside its process;
cross-job preparation ahead remains independent of this switch.
