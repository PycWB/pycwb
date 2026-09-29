# Compose a custom processor through the CLI

This small example adds a diagnostic between superclustering and likelihood:

```text
native preparation
  → coherence
  → superclustering
  → report_candidates             # extra user-defined operation
  → likelihood and event creation
  → native output and progress
```

The ordinary calls in [`processor.py`](processor.py) determine the composition.
There is no stage class or registry. The diagnostic only logs the number of
candidates; it does not change scientific products. This example runs lags
serially and honors `skip_lags` when resuming.

## Run it

With PycWB installed, copy [`../demo/user_parameters.yaml`](../demo/user_parameters.yaml)
to `custom_parameters.yaml` and add this entry, replacing the absolute path:

```yaml
# Custom workflow, loaded by the ordinary run command.
segment_processer: /absolute/path/to/checkout/examples/custom_workflow/processor.py.process_job_segment
```

Use an absolute path because the CLI changes into the output working directory
before importing the processor. Then run:

```bash
pycwb validate custom_parameters.yaml
pycwb run custom_parameters.yaml --work-dir custom_search
pycwb progress --work-dir custom_search
```

The Python file is a plugin loaded by PycWB, not a separate runner to execute.
Use a fresh working directory. The synthetic example needs the same cross-talk
catalog and scientific dependencies as the ordinary native run; see its README.
The log will include a line like `Custom workflow: lag 0 has ... candidates before likelihood`.

## Change the composition

`supercluster_with_diagnostics` is ordinary Python: insert further operations,
replace its constituent functions, or remove the diagnostic. To change the whole
per-lag sequence, implement it in `process_lags` by calling scientific modules
directly. To change trial handling or preparation order as well, implement those
calls in your `process_job_segment` entry point instead of delegating to native.
Data dependencies still apply: likelihood requires prepared cluster amplitudes.

The example deliberately reuses private native helpers for lag metadata, vetoes,
event creation and persistence. They implement the supplied native recipe and
can change with that recipe; they are not a universal workflow contract. A fully
independent processor owns those responsibilities, including resume records and
resource cleanup. No change to `types`, a stage bundle, or the workflow loader is
needed to select a different composition.
