# Decision: staged job management

The current evidence supports a small **opt-in preparation-ahead executor**, not
replacing the simple default or adding a general scheduling framework. CPU gains
are useful but moderate and depend on available RAM and the preparation/analysis
balance. Actual 128-core performance remains unmeasured.

## Stronger CPU comparison, 2026-09-20

Three different LF jobs (3–5), including a frame-boundary crossing and shorter
windows, each processed 112 valid lags. All cases had the same six physical cores
and twelve SMT affinity IDs, scientific settings, outputs and warm Numba cache.
The baseline already parallelized preparation and shared inputs among lag workers;
it was a sequential in-process loop without cross-job checkpoints or supervision.

| Execution | Complete wall time | Peak tree PSS | Sampled CPU seconds |
| --- | ---: | ---: | ---: |
| Simple in-process loop | 428.95 s | 8.22 GiB | 2640.10 |
| Prepare ahead, shared cores | 369.02 s | 13.05 GiB | 2850.56 |
| One preparation core + five analysis cores | 571.95 s | 9.89 GiB | 1906.53 |

Shared-core preparation ahead saved **13.97% wall time**, at approximately **59%
more peak RAM** and **8% more sampled CPU work**. The earlier analysis-heavy CPU
experiment (672 lags per job, 112 per worker on average) saved **6.95%** against
serialized staged execution; that was a different baseline. The GPU holdout saved
9.24% using an earlier preparation boundary. These gains are not additive.

The fixed partition was **33.34% slower** than the simple loop. It reduced actual
sampled CPU work by approximately 28%, but increased time on the same whole-node
allocation. Its preparer could not supply jobs quickly enough: analysis waited
92 and 96 seconds between jobs. Shared-core execution reduced those waits to
17 and 27 seconds, but also slowed the overlapping analysis stages. Queue depth
alone cannot cure inadequate preparation throughput.

The fixed partition used sequential coherence/TD setup internally. Its original
concurrent setup stalled on the one-core allocation; repeated diagnostic stacks
showed both stages waiting for JAX transforms with no CPU progress. The prototype
now honors `gpu.overlap_setup: false` to permit sequential internal setup while
retaining cross-job overlap. The default behavior is unchanged. This is a tested
workaround for the observed configuration, not a diagnosis or fix of an upstream
runtime bug. Failed and interrupted attempts remain in the evidence bundle and
are excluded from completed timings.

## Correctness and limits

All completed variants matched 819 triggers, 336 progress rows and 2,352 coherence
records exactly, excluding progress timestamps. Each job had unique completed
lags 0–111. All also matched the corresponding prefix of the earlier public GPU
holdout. The experiment did not tune thresholds, pixel cuts, windows or output
requirements. **100 tests passed** across pipeline and execution suites, including
serial/concurrent setup selection. No completed case had sampled process swap;
system-wide swap increased 6.27 MiB in the shared case and can include other work.

This preparation-heavy workload puts about 18.7 lags on each of six workers,
matching the task-count ratio of 2,400 lags across 128 workers. It cannot simulate
128-core NUMA, bandwidth, writer load, process startup or filesystem contention.
A single completed run per variant does not establish a confidence interval,
universal speedup or optimal core partition. The shared-frame cache was not
combined with this pipeline experiment. Scientific settings and pixel occupancy
outside these cases can require different memory reservations.

## Recommended scope

Keep the existing run/batch interface and simple profile. The optional optimized
path should have an explicit preparer, initially one byte-bounded upcoming
context, backend-appropriate stage boundaries and shared CPU/RAM admission.
A separate job-creation abstraction is unnecessary when scientific job generation
is unchanged. Do not make a fixed one-core preparation reservation the default.

Before production promotion, replace private adapters with explicit stage APIs,
add result-byte backpressure, integrate resume/output ownership, and validate a
real HPC allocation. Measure starvation and resource pressure to choose
preparation capacity; do not derive universal reservations from these LF peaks.
Prioritize reducing actual preparation work (including repeated frame decoding)
over adding deeper queues or a general scheduler. The experimental adapter is
still separate from the public execution profiles.

Detailed plan, timings, source/environment fingerprints, exact comparisons,
reproduction scripts, diagnostics and timeline plots are in
`../runs/cpu_pipeline_decision/` relative to the repository root. See also
[CPU/HPC experiments](cpu_hpc_pipeline_experiments.md) and
[pipeline experiments](job_pipeline_experiments.md).
