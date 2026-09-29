# Experimental lag parallelism

For the shared-input process implementation, configuration, resource ownership
and failure handling, see [Shared-input lag processing](shared_input_lag_processing.md).
The native processor remains the default. All entry points reuse its segment
preparation, scientific stages and output finalization.

## Optional GIL-releasing thread experiments

The thread experiments have no `segment_processer` entry point. They live in
`pycwb/workflow/subflow_experimental/process_job_segment_nogil.py`, which exposes one function:

```python
from pycwb.workflow.subflow_experimental.process_job_segment_nogil import process_lags

process_lags(context, output_context, pending_lags, workers, full=False)
```

A custom segment processor can supply a `lag_processor` adapter that turns
`skip_lags` into pending lag indices and chooses the worker count. See the
[experimental package README](../../pycwb/workflow/subflow_experimental/README.md) for the adapter;
`process_lags` itself accepts pending indices rather than the native hook signature. `full=False` selects the
coherence-only wrappers; `full=True` adds the subnet and likelihood wrappers.
The module is imported only when requested. The wrappers call existing compiled
functions with `nogil=True` at selected Python-to-Numba boundaries, preserving
the original arithmetic and precision. Private copies of orchestration
functions bind these wrappers; the native module globals and existing Numba
kernel files are unchanged. The whole lag loop is not GIL-free, and these
wrappers do not require free-threaded Python.

The coherence-only variant wraps selection, connectivity and subnet
statistics. The full variant also wraps subnet sky/MRA kernels and the
three likelihood sky-scan variants. Other Python operations still hold the GIL,
so CPU utilization and scaling can differ from independent processes.

Threads share Python input objects without serializing them. Both thread
variants use the same bounded scheduler and parent output writer as the process
path. Injection trials fall back to serial native analysis. Use the inner-thread
limits described in the shared-input documentation when comparing resources.

The earlier full LF comparison remains in the enclosing workspace at
`runs/lag_parallel_scaling_400_200/LF_REPORT.md`. Its retained HF/LD study was
narrowed to processes at one and six physical cores. Short-run projections are
not a substitute for measurements at the intended lag count.
