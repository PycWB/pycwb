# CPU pipeline experiments and 128-core planning

This extends the whole-job pipeline experiment to CPU execution. GPU feeding is
only one application of the staged executor. The same planner/preparer/consumer
architecture must support a CPU-only allocation.

## Local experiment

Use six lag processes on the six-physical-core host (all twelve SMT affinity IDs
available), one Numba/BLAS thread per lag process, two existing LF jobs and **672
lags per job**. This represents **112 lags per worker on average**. Lag dispatch
is dynamic; workers need not finish exactly 112 each. Keep input windows, pixel
selection, thresholds, output products and compilation cache unchanged.

Compare identical persistent full-preparation and CPU-consumption stages:

- Sequential: prepare job 1, consume job 1, prepare job 2, consume job 2.
- Overlapped: prepare job 2 while consuming job 1, with at most one upcoming job.

Include startup, first fill, serialization, output and final drain in wall time.
Measure whole-process-tree PSS and CPU time, plus per-process USS to distinguish
private worker memory from shared mappings. Compare every persisted scientific
field and complete unique lag progress (timestamps excluded), plus coherence
records. This comparison isolates overlap; it is not a comparison against every
CPU processor or the public simple profile. No GPU computation is used.

The original CPU pilot had only 128 lags *total per job*, not per worker. Its
13.18% gain must not be assumed to hold for this larger workload.

## What 128 cores change

Do not launch 128 CPU processes on six cores to claim HPC performance. Matching
112 lags per worker keeps the nominal amount of analysis work per worker fixed,
but cannot reproduce memory bandwidth, NUMA, filesystem traffic or coordination
costs of a 128-core node. Twelve SMT affinity IDs are not twelve physical cores.

A nominal 128-worker workload at 112 lags each has 14,336 lag tasks. The current
1200-second LF segment only provides 2,400 valid lags. Reaching that total on real
inputs therefore requires multiple scientific jobs, or a separately validated
segment configuration; merely requesting 14,336 lags from this job is invalid.
Scientific job identities and valid lag ranges must remain unchanged.

The production CPU scheduler should:

1. Share immutable prepared arrays across lag workers, as the current mapped
   input processor already does. Group nearby jobs by frame reuse in the
   preparer; matching frame files does not make conditioned products identical.
2. Admit analysis concurrency using private per-worker scratch, shared contexts,
   preparer peak, retained ready artifacts, writer buffers and headroom. Read
   memory limits and CPU affinity from the scheduler/cgroup allocation.
3. Coordinate preparation and lag concurrency from one core budget. Benchmark
   shared cores against reserving a few preparation cores on real hardware.
   Do not stack full-size segment, lag and inner-thread pools independently.
4. Maintain a byte-bounded ready queue, beginning with one upcoming context.
   Increase preparation capacity only when measured consumer starvation warrants
   it and memory permits it. A count limit alone does not bound RAM.
5. Measure writer backpressure and serialization as well as preparation. The
   current lag executor limits outstanding futures to twice worker count; that
   is not a byte bound for variable-size lag results. Production high-core work
   needs a result-byte budget or equivalent backpressure before claiming bounded
   total memory at 128 workers.
6. Consider NUMA locality and node-local scratch. Shared file mappings avoid
   duplicate copies, but do not remove remote-memory access or bandwidth limits.

For a candidate worker count W, the memory reservation should account for:

    shared live contexts + W * private worker reservation
      + preparation peak + ready payloads + writer/IPC bound + headroom

Use a conservative reservation validated across search settings and pixel loads;
a single LF peak USS sample is evidence, not a universal bound. USS can omit
pages that become private later through copy-on-write. A live pressure monitor
and scheduler hard memory limit remain necessary. The prototype's fixed consumer
reservation is not a 128-worker auto-sizing policy.

## HPC acceptance experiment

On an actual allocated node, sweep analysis worker counts and preparation core
reservations at fixed scientific work. Also run a separate weak-scaling series
with approximately 112 valid lags per analysis worker, distributed over real job
IDs when necessary. Report these two experiments separately. Include jobs with
shared and distinct frame files, different segment/frame sizes, LF/HF settings,
and varying pixel occupancy. Use node-local and shared scratch where available.

Require exact outputs, bounded memory, no sustained swap growth, and measured
complete throughput improvements. Record time spent preparing, waiting for a
ready context, analyzing and writing; total CPU work; PSS and private memory;
read/decode counts; and allocation topology. Do not extrapolate a local speedup
percentage into a promised 128-core result.

Local raw measurements are in `../runs/cpu_pipeline_112_per_core/` relative to the
repository root, including `REPORT.md` and reproduction commands in `COMMANDS.md`.

A useful diagnostic is preparation time P versus consumption time A at the chosen
worker count. For N jobs with constant stages and no contention, a single-stage
producer/consumer model takes P + A + (N-1)*max(P, A), compared with N*(P+A)
sequentially. This is an ideal scheduling model, not a hardware prediction. If
A shrinks below P as lag concurrency increases, deeper queues only delay eventual
starvation: preparation throughput, shared-frame reuse or additional bounded
preparers must improve. At fixed 112 lags per worker A may remain similar, but
having several contexts and more total lag tasks changes RAM and writer demand.

## Local result, 2026-09-20

| Mode | Complete wall time | Peak tree PSS | Sampled CPU seconds |
| --- | ---: | ---: | ---: |
| Sequential preparation | 804.04 s | 9.04 GiB | 5195.56 |
| Preparation ahead | 748.15 s | 11.28 GiB | 5206.04 |

Overlapping preparation saved **55.89 seconds (6.95%)**, giving 7.47% more
throughput in this paired run. The between-job gap fell from 92.45 to 0.051
seconds. However, first-job consumption increased from 301.81 to 338.16 seconds
while the producer competed for CPU; second-job consumption remained similar
(307.83 versus 308.41 seconds). Second-job preparation increased from 92.39 to
130.02 seconds under overlap and still finished 208.19 seconds before consumption.
One prepared job ahead was sufficient for this workload.

All 3,112 trigger rows, 1,344 progress rows and 9,408 coherence records matched
exactly, excluding progress timestamps. Complete unique 0–671 lag coverage was
verified for each job. The CPU result also matched the same lag prefix of the
previous public GPU baseline. Both runs retained the same 233-file Numba cache.
Scientific settings were not adjusted in response to intermediate results.

Individual lag-worker peak USS was approximately 0.55–0.56 GiB; this is not a
safe universal reservation. The overlap monitor also observed approximately
130 MiB of **system-wide swap growth**. It did not attribute swap to individual
processes, so the no-swap acceptance gate is not established by this experiment.
Use per-cgroup swap accounting in a dedicated allocation to settle that question.

This larger workload confirms a CPU throughput benefit locally, smaller than
the earlier 128-total-lag pilot. It does not determine the best core partition
or preparer count for 128-core hardware. Keep the experimental staged adapter
separate from production until the shared core/RAM budget, result-byte
backpressure and lifecycle integration described above are implemented and tested.

## Follow-up decision

A three-job, preparation-heavy comparison against an optimized in-process loop
measured **13.97% less wall time** with shared-core preparation ahead, at 59% more
peak RAM and 8% more sampled CPU work. A fixed one-core preparer with five analysis
cores was 33.34% slower than the loop. Exact outputs and 100 tests passed.
See [the job-management decision](job_management_decision.md) for methods,
resource costs, a constrained-runtime stall/workaround, and the recommended scope.
