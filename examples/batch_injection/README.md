# A batch of CBC injections

Copy this directory to a fresh working directory and run there:

```bash
pycwb run user_parameters_injection.yaml --list-jobs
pycwb run user_parameters_injection.yaml --jobs 0
pycwb progress --work-dir .
```

`generate_parameters.py` defines ten IMRPhenomXPHM binaries with varying
aligned spin. Each waveform has its own approximant and polarization parameters,
and is analyzed in an independent segment with seeded Gaussian noise. The
default generator uses the standard LALSuite/GWSignal installation. Omit
`--jobs 0` to process all ten jobs. The Python runner invokes the same current
`pycwb.workflow.run.search` entry point.

## Optional hyperbolic-waveform extension

The retained `pycbc_inject/hyperbolicTD` adapter targets a **custom LALSuite
build**, not the LALSuite package installed with PycWB. In particular it needs
`SimInspiralWaveformParamsInsertHyperbolicEccentricity` and
`SimInspiralWaveformParamsInsertImpactParameter`, plus a compatible waveform
approximant supplied by that build. PyCBC is also required.

To study those waveforms, create a separate configuration and explicitly choose
the model installed in your custom LAL build:

```yaml
injection:
  parameters_from_python:
    function: ./hyperbolic_parameters.get_injection_parameters
    args:
      approximant: MODEL_NAME_FROM_YOUR_LAL_BUILD
  generator: ./pycbc_inject/hyperbolicTD/waveform.get_td_waveform
```

Retain the segment/noise settings from the complete configuration. The model
name above is a placeholder to replace; an ordinary CBC approximant does not
turn into a hyperbolic model by adding eccentricity/impact parameters. The
custom extension is not exercised by the portable CBC example or its tests.
