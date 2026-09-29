# Experimental workflow execution

This package contains pycWB's **experimental, opt-in resource-aware execution
layer**. It provides job planning, bounded raw-input reuse, worker supervision,
and coordinated output writing. Its interfaces and behavior may change; validate
representative workloads before relying on the scalable profile for a campaign.
The default remains `execution.profile: simple`.

This is the infrastructure for load balancing and adaptive execution within an
allocation. It groups jobs that share input frames and limits concurrent segment
workers using CPU availability and memory reservations. Cache loading adapts to
capacity, and sampled memory monitoring can stop execution under pressure.
It does not predict scientific processing cost or dynamically redistribute jobs
between cluster nodes. Slurm/Condor batch membership is fixed when planned.

## Configuration

The `execution` block controls job scheduling and resource management. The
separate `execution_profile` block controls native processing options within
each job. Both can appear in the same analysis configuration; see the
[Performance Guide](../../../docs/source/dev_performance.rst) for the latter.

Yes: this package is connected to the normal analysis YAML configuration. Add a
top-level `execution` block to an existing valid pycWB configuration:

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

These values illustrate the syntax, not suitable limits for every search.
`worker_memory` reserves host RAM for an entire segment process tree, including
nested lag workers. Size it from representative measurements. `batch_size` limits
jobs per planned group; it is not the number of concurrent workers. CPU and
memory budgets, together with the runner's worker limit, constrain concurrency.

The configuration schema accepts this block, and `ExecutionSettings.from_config`
validates it both when loading YAML and when restoring configuration from a
catalog. Unknown execution keys and invalid values are rejected. The normal
`pycwb run`, `pycwb batch-setup`, and `pycwb batch-runner` paths use these settings.
Cluster setup persists planned groups for batch runners.

Omit the block, or use `profile: simple`, for the existing execution path.
Resource/cache settings alone do not enable scalable execution. Explicit
`planner` or `executor` factory paths also enable dispatch through this package;
they are advanced extension points.

The scientific implementation is selected separately with `segment_processer`.
Enabling scalable execution does not change scientific windows or numerical
selection settings. Processors that support `input_provider` can reuse cached raw
inputs; unsupported processors use direct reads with a warning.

`preload` controls raw-input loading:

- `"off"`: direct reads without the shared input cache. Quote this value in YAML
  to avoid parsers interpreting it as a boolean.
- `auto`: load reusable planned input intervals on demand.
- `batch`: preload a group's reusable inputs if they fit, otherwise fall back to
  bounded demand loading.

`batch-setup` persists each planned batch in a self-contained catalog fragment
before generating scheduler submissions. `batch-runner --batch-id` reads job
membership only from that fragment, on shared filesystems and transferred
execute nodes alike. Setup rejects changes to existing batch jobs; use a new
working directory when regrouping. Executed fragments write diagnostic plans
and metrics under `RUN/execution/`.

## Package layout

| Module | Responsibility |
| --- | --- |
| `settings.py` | Configuration schema, defaults, and validation |
| `planner.py` | Deterministic job ordering and shared-frame grouping |
| `executor.py` | Execution dispatch, segment supervision, resume, and cleanup |
| `resources.py` | CPU/RAM accounting and sampled memory monitoring |
| `cache.py` | Bounded raw-input cache and worker input providers |
| `decoder.py` | Isolated frame decoding into memory-mapped array files |
| `writer.py` | Acknowledged output transport and coordinated catalog writes |
| `scheduling.py` | Persisted batch membership lookup |
| `contracts.py` | Custom planner, executor, and input-provider interfaces |

## Experimental limitations

- Memory reservations and sampled monitoring are not a hard OS memory limit.
  Scientific memory peaks can exceed estimates between samples. GPU VRAM is not
  budgeted by this layer.
- Input reuse is local raw-data caching, not conditioned-data caching or a
  distributed cache. Preloading is synchronous.
- Fresh worker startup and decoding overhead can make small jobs slower. Measure
  end-to-end performance before assuming a speedup.
- Keep batch membership and scientific configuration stable when resuming;
  progress is not reconciled across regrouped catalog fragments.

See [Configurable job execution](../../../docs/dev/scalable_execution.md) for
all settings, resource accounting, scheduler integration, and verification notes.
