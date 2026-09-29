# Synthetic recovery example

Run from the repository root with PycWB installed:

```bash
python examples/demo/run_demo.py my_first_search --run
python examples/demo/run_demo.py my_first_search --check
```

The script copies the adjacent `user_parameters.yaml` into a new directory,
runs one seeded H1/L1 injection, and checks completion and recovery. Omit
`--run` to create the configuration only. Existing directories are never
overwritten. Use `--xtalk /path/to/OverlapCatalog16-1024.bin` to reuse a local
cross-talk catalog; otherwise the first search may download it.

You can also run the created configuration with the ordinary pipeline:

```bash
cd my_first_search
pycwb validate user_parameters.yaml
pycwb run user_parameters.yaml
python /path/to/pycwb/examples/demo/run_demo.py . --check
```

The recovery check requires one completed zero-lag job and a finite trigger
above the configured threshold within one second of the injection in both
detectors. It is a smoke test, not sensitivity or significance validation.
The installed-package integration test is `tests/test_demo_e2e.py`.
