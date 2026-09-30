# External-frame benchmark

Run this directory's configuration with `pycwb run user_parameters_mdc.yaml` or
`python pycwb_mdc.py`. First supply the frame data referenced by
`input/OPBM_H1.frames` and `input/OPBM_L1.frames` and the configured data-quality
files. Frame-list entries must be absolute paths visible on the worker. The
listed external OPBM files are not included in this repository.

This is a production-sized benchmark, not a self-contained smoke test. Use
`--list-jobs`, then select a job and lag with the CLI before running the full
workload. For a portable benchmark input, start from `examples/demo` and record
its execution profile, dependencies and hardware with the timing results.
