# Experimental subflows

These modules are opt-in research experiments. Their interfaces may change and
they are not imported by ordinary native, ROOT or GPU workflows.

## GIL-releasing lag threads

`process_job_segment_nogil.py` wraps selected native Numba kernels with GIL release
at their call boundaries. The arithmetic is unchanged; the complete Python lag
loop is not GIL-free. `full=False` wraps coherence; `full=True` also wraps subnet
and likelihood operations. This is separate from the shared-input process
workflow under the neighboring `subflow` package.

The module moved from `pycwb.workflow.subflow.process_job_segment_nogil` to
`pycwb.workflow.subflow_experimental.process_job_segment_nogil`. Update experimental imports; the
old module is not retained as a compatibility wrapper.

A custom processor can adapt the native lag hook as follows:

```python
from dataclasses import replace

from pycwb.workflow.subflow_experimental.process_job_segment_nogil import process_lags
from pycwb.workflow.subflow import process_job_segment_native as native


def experimental_lags(context, output_context, skip_lags):
    # Keep injection processing on the established native path.
    if context.sub_job_seg.injections:
        return native._process_lags(context, output_context, skip_lags)
    pending = list(native._iter_pending_lags(context, skip_lags))
    if not pending:
        return
    workers = min(len(pending), max(1, context.config.parallel_lag_workers))
    context = replace(context, numba_threads=native._parallel_inner_threads(context.config, workers))
    process_lags(context, output_context, pending, workers, full=False)


def process_job_segment(working_dir, config, job_seg, **kwargs):
    return native.process_job_segment(
        working_dir, config, job_seg, **kwargs, lag_processor=experimental_lags,
    )


process_job_segment.supports_input_provider = True
```

Save a custom processor module, select its `process_job_segment` function through
`segment_processer` in YAML, and run with `pycwb run`. The example reuses private
native helpers and must track changes to the native recipe. Set BLAS/OpenMP
thread limits before starting the run. Qualification of one experiment does not
establish scaling or scientific equivalence for every workload.

See [the experiment notes](../../../docs/dev/experimental_lag_parallelism.md) and
[shared-input lag processing](../../../docs/dev/shared_input_lag_processing.md) for
ownership, resource limits and retained numerical-comparison evidence.
