# Changes

## Unreleased

- Default omitted `lagOff` and `lagMax` to zero, so the default single lag is
  unshifted. Earlier defaults were `lagOff: 6` and `lagMax: 150`. Runs whose saved
  YAML snapshot used those implicit defaults will fail the resume consistency
  check after upgrading. To continue such a run, explicitly restore its recorded
  lag settings; use a new working directory to change the run to zero-lag.
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
- Add offline configuration checks, including execution/GPU settings and detector definitions (`pycwb validate`), a metadata-based environment inventory (`pycwb doctor`), and `python -m pycwb`.
- Generate CLI help and parameter summaries from the implementation. Correct detector-key and injection-window guidance.
- Document installation channels, output interpretation, troubleshooting, analysis archiving, support access and release validation scope.
- Build generated references consistently on local builds and Read the Docs; add documentation and onboarding checks to CI.

`doctor` reports installed package versions without a hard-coded dependency list or backend-readiness verdict.

These commands are new in this checkout and are not available in older published releases. The demo tests a small native H1/L1 CPU search; it is not a production sensitivity or significance validation.

Earlier releases are listed in the [GitLab releases](https://git.ligo.org/yumeng.xu/pycwb/-/releases) and [PyPI history](https://pypi.org/project/PycWB/#history). Their notes have not been reconstructed here.
