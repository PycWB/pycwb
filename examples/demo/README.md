# Synthetic injection through the PycWB CLI

From the repository root, with PycWB installed:

```bash
pycwb validate examples/demo/user_parameters.yaml
pycwb run examples/demo/user_parameters.yaml --work-dir my_first_search
pycwb progress --work-dir my_first_search
```

The YAML configures one seeded 128-second H1/L1 segment containing a deliberately
loud sine-Gaussian injection at GPS 1126259526 and 150 Hz. The ordinary pipeline
generates the noise and signal, processes the segment, and writes its catalog and
waveform products. No example-specific Python runner is needed. Use a fresh
working directory for each run.

After the run, make the input-polarization and injection/reconstruction plots:

```bash
python examples/demo/plot_results.py my_first_search
```

The helper reads the run's saved YAML, catalog and `output/wave.h5`. It writes
`injected_signal.png`, `reconstruction.png` and a result summary under
`my_first_search/plots/`. The source plot regenerates the configured polarizations;
the detector plot uses the saved `INJ` and `REC` samples with their original
epochs, sample rates and strain amplitudes. It selects the loudest candidate
within one second of the injected epoch in both detectors and reports an error
if no such candidate exists.

The [Your First Search](../../docs/source/start_here.rst) page explains the YAML,
catalog fields and output files using figures produced by this helper.

No detector strain download or collaboration account is needed. The first run
may download the cross-talk catalog and compile numerical kernels. To reuse a
local catalog, copy the YAML and set `filter_dir` to its directory and `wdmXTalk`
to its filename before validating and running that copy.

Inspect `my_first_search/catalog/catalog.parquet` and the products under
`my_first_search/trigger/`. Completion reported by `pycwb progress` establishes
that the job finished; inspect the trigger times and ranking statistic to assess
recovery. A loud injection is a smoke test, not sensitivity or significance
validation. The automated recovery assertion belongs to `tests/test_demo_e2e.py`.

## Difference from the Colab notebooks

This example demonstrates configuration-driven use of the production CLI with
synthetic input. The notebooks in [`examples/colab`](../colab/) analyse real
GW150914 open data by calling individual Python stages and inspecting intermediate
results. Some notebook APIs and installation cells target older releases. They
serve a different educational purpose and are not a replacement for this CLI
example or its recovery test.
