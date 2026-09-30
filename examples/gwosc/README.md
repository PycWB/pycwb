# Download and search a GWOSC event

Start in a fresh directory with PycWB installed:

```bash
mkdir GW170104
cd GW170104
pycwb gwosc GW170104
pycwb run user_parameters.yaml --list-jobs
pycwb run user_parameters.yaml --work-dir search
```

The download command writes frame lists with absolute paths, data-quality files,
an analysis interval and `user_parameters.yaml`. The generated default configuration
uses the downloaded 4096 Hz data and discovers strain-channel names from the
frames, since GWOSC releases use different channel names. The YAML in this example
directory illustrates the settings; use the configuration generated for your event.
The default event download supports H1/L1. Custom templates remain the caller's
responsibility, including network, channels and sampling rate.

For a compact GW150914 exercise with a shorter interval and coarse sky grid, use
`examples/tutorials/prepare.py` and the public-data tutorial instead.
