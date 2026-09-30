# Changes

## Unreleased

### Breaking and result changes

- The native conditioning and likelihood Python APIs were renamed, and detector
  helpers now require configured `Detector` instances. Update imports and calls
  using [the migration guide](docs/source/migration.rst); retired function names
  are not compatibility aliases. Current tutorials and the native Colab notebook
  now use the supported entry points. ROOT-era examples are historical references.
- Master/merged catalogs use a referenced `jobs.parquet` manifest. Archive it
  with the catalog and progress, and use the current `Catalog.jobs` reader.
  Older inline-metadata readers can silently lose the job list and exposure.
- Time-delay rounding, Hz-based defragmentation, noise RMS, packet normalization,
  regression, injection scaling/placement/whitening, ranking features and
  efficiency/FAR conventions can change results for unchanged YAML inputs.
  `BATCH` is now enforced (`0` disables the cap). Revalidate fresh runs and
  retrain native XGBoost models; see the migration guide for required inputs,
  supported IFAR labels and opt-in numerical settings.
- Runtime dependencies now include `wdm-wavelet>=0.4.0`,
  `burst-waveform>=0.5.0` and `joblib>=1.3`, with Python 3.11 or newer.
  Package versions are generated from Git tags by `setuptools_scm`; a new
  release version is assigned by the release-tag workflow.
- The experimental `pycwb flow` command and `pycwb.prefect_flow` (Prefect and
  Dask wrapper) were removed; use `pycwb run` or `batch-setup`.

### Review fixes

- Size shared time-delay buffers for rounded subnet indices as well as the
  fine likelihood grid. At 2048 Hz, an H1/L1 subnet could previously address
  delays outside the buffer with `upTDF: 4` or `8`.
- Keep zero-lag selection in `trigger_selection`. `process_background` now uses
  its triggers and exposure as given, ignores `exclude_zero_lag` and
  `unshifted_job_ids` with a warning, and warns about unshifted triggers.
  Whole-job selections and splits write the matching `progress_file`, and a
  run whose jobs are all superlag-shifted keeps every regular lag 0 as
  background instead of dropping its exposure.

### Other changes

- Preserve Parquet list types across prediction-cut batches, including scored
  catalogs with no surviving rows. Keep zero-lag separation in the upstream
  selection stage; training consumes its selected background unchanged and
  warns when a background input still contains unshifted triggers. The
  standard example now selects and cleans every training chunk before
  training and disables redundant FAR lag filtering.
- Reuse configured detector instances for injection arrival times, including
  external geometries. Already projected strains without sky coordinates keep
  their measured detector centroids instead of aborting reconstruction.
- Allow submission settings such as YAML `job_memory`, `job_disk` and walltime
  to change on resubmission. Analysis settings and prepared batch membership
  remain checked. Legacy catalogs without a YAML snapshot cannot be verified:
  continue them with their original software or regenerate in a new working
  directory. `--force-overwrite` does not bypass this requirement.
- Bound completed scalable workers' shutdown with
  `execution.worker_shutdown_timeout` (60 seconds by default). A timeout fails
  the allocation and cleans up worker processes and reservations.
- Record `cwb-compatible-v1` catalog preprocessing in newly trained models and
  trusted cWB model imports. Scoring rejects incompatible or unversioned models
  by default. Older native-catalog models need retraining because `norm`,
  `sSNR` and detector-indexed inputs changed. Independently verified compatible
  unversioned models can declare `ML_options['model_preprocessing']` in their
  scoring config; this declaration does not convert old feature definitions.

- Default omitted `lagOff` and `lagMax` to zero, so the default single lag is
  unshifted. Earlier defaults were `lagOff: 6` and `lagMax: 150`. Catalogs
  prepared with v1.1.0a3 or earlier have no YAML snapshot, and the resume check
  compares every schema key including defaults, so earlier runs cannot be
  resumed after upgrading. Finish them with their original software, or
  regenerate them in a new working directory with explicit lag settings.
- Apply native supercluster size and statistics cuts to isolated clusters even
  when no clusters link. Previously those candidates bypassed the cuts. Trigger
  selection and background counts can change; production impact has not been
  quantified against a full cWB reference run.
- Accept bare elementwise `max`/`min` in prediction cuts, matching training cuts.
- Reject FAR-table attachment when its recorded ranking statistic differs from
  the requested one. Unlabeled legacy tables retain the `rho` convention;
  tables for custom statistics must declare `ranking_par`.
- Correct the configuration-mismatch message to name `--force-overwrite`.

- Import the frame-reader job type directly from its defining module, avoiding a cold-import cycle through job segmentation and injection SNR setup.

- Demean only the regression self-witness, preserving the original target transform and separate target/witness normalization in Numba and JAX. This corrects nonzero-mean conditioning differences that can change downstream chirp estimates.

- Require Python 3.11 or newer, matching the worker-recycling API used by batch and online searches. Remove the older-Python exception-group backport.
- Run lint/type checks in a separate Python 3.11 environment with compatible NumPy stubs; runtime CI remains on Python 3.13.
- Run the strict Sphinx check directly in CI so documentation builds do not require `make` in the runtime image.
- Install the documentation checkout before running Sphinx so fresh CI checkouts include the generated package version.

- Propagate command exit statuses to the shell and defer scientific implementation imports until the selected command runs.

- Run the synthetic injection YAML in `examples/demo/` through the ordinary `pycwb` CLI; keep recovery assertions in the test suite.
- Support `python -m pycwb`.
- Generate CLI help and parameter summaries from the implementation. Correct detector-key and injection-window guidance.
- Document installation channels, output interpretation, troubleshooting, analysis archiving, support access and release validation scope.
- Build generated references consistently on local builds and Read the Docs; add documentation and onboarding checks to CI.

These CLI features are new in this checkout and are not available in older published releases. The demo tests a small native H1/L1 CPU search; it is not a production sensitivity or significance validation.

Earlier releases are listed in the [GitLab releases](https://git.ligo.org/yumeng.xu/pycwb/-/releases) and [PyPI history](https://pypi.org/project/PycWB/#history). Their notes have not been reconstructed here.
