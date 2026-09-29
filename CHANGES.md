# Changes

## Unreleased

- Require Python 3.11 or newer, matching the worker-recycling API used by batch and online searches. Remove the older-Python exception-group backport.
- Run lint/type checks in a separate Python 3.11 environment with compatible NumPy stubs; runtime CI remains on Python 3.13.
- Run the strict Sphinx check directly in CI so documentation builds do not require `make` in the runtime image.
- Install the documentation checkout before running Sphinx so fresh CI checkouts include the generated package version.

- Propagate command exit statuses to the shell and defer scientific implementation imports until the selected command runs.

- Keep the synthetic injection example and recovery check in `examples/demo/`, outside the production CLI.
- Add offline configuration checks, including execution/GPU settings and detector definitions (`pycwb validate`), a metadata-based environment inventory (`pycwb doctor`), and `python -m pycwb`.
- Generate CLI help and parameter summaries from the implementation. Correct detector-key and injection-window guidance.
- Document installation channels, output interpretation, troubleshooting, analysis archiving, support access and release validation scope.
- Build generated references consistently on local builds and Read the Docs; add documentation and onboarding checks to CI.

`doctor` reports installed package versions without a hard-coded dependency list or backend-readiness verdict.

These commands are new in this checkout and are not available in older published releases. The demo tests a small native H1/L1 CPU search; it is not a production sensitivity or significance validation.

Earlier releases are listed in the [GitLab releases](https://git.ligo.org/yumeng.xu/pycwb/-/releases) and [PyPI history](https://pypi.org/project/PycWB/#history). Their notes have not been reconstructed here.
