# MESA whitening with the native pipeline

Run from this directory in a PycWB environment with `memspectrum` installed:

```bash
pycwb gwosc-data config/user_parameters.yaml
pycwb run config/user_parameters.yaml --list-jobs
pycwb run config/user_parameters.yaml --jobs 1 --trial-idx 0 --lags '0,0'
```

The configuration selects `whiteMethod: mesa`. The small processor module
delegates to the native pipeline, which preserves normal injection, catalog,
waveform and restart behavior. It replaces the former copied ROOT pipeline and
removed `fake_conditioning` function. The schema extension adds an example label
to the saved metadata; residuals use the standard saved `NUL` waveform products.

The configured interval requires public detector data. The commands above select
one job, one trial and zero lag for an initial check. Running without selectors
processes the full configured injection/time-slide workload and costs much more.
Use a separate copy of the directory for comparisons, and keep frame paths,
noise seeds and all other scientific settings fixed when comparing whiteners.
