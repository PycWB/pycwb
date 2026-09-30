# Example for injecting into data/frames

This example demonstrates how to inject into the real data

## Set up frame and data-quality files

Copy this example directory to a fresh working directory. The frames are not included; download the configured O1 LOSC V1 interval at 4096 Hz and select one job for the first run:

```bash
pycwb gwosc-data user_parameters.yaml
pycwb run user_parameters.yaml --list-jobs
pycwb run user_parameters.yaml --jobs 1 --trial-idx 0 --lags 0,0
```

## Explanation

The addition of the following lines in the configuration file, together with the `injection_parameters.py`,  will allow the injection of the signal into the data

```yaml
injection:
  allow_reuse_data: True
  repeat_injection: 1
  parameters_from_python:
    function: "./injection_parameters.get_injection_parameters"
  sky_distribution:
    type: UniformAllSky
  time_distribution:
    type: 'rate'
    rate: 1/200
    jitter: 50
  generator: pycwb.modules.injection.gwsignal_waveform.get_td_waveform
```

The `injection_parameters.py` file contains the function `get_injection_parameters` to generate a list of parameters for the injection. Each record supplies its own `approximant`, `f_lower` and `pol`; the enclosing injection block selects the generator.

The injection process contains the following steps:
1. Generate the parameters for the injection using the function `get_injection_parameters` in `injection_parameters.py`
2. Place the parameters on the sky with given sky distribution: 
    - `UniformAllSky`
    - `Patch`: for a circle on the sky
    - `Fixed`: one coordinate
    - `existing`: a coordinate table with explicit units
    - `Custom`: use a healpix map with the probability distribution on each pixel
    - If the sky distribution is not given, this step will be skipped, user has to set the ra, dec in the first step
3. Place the signal in the data with given time distribution: 
    - `rate`: evenly distributed in gps time with `rate` and `jitter`
    - `poisson`: next event will be placed `t + np.random.exponential(1/rate)` with given `rate`
    - Omit `time_distribution` to use explicit per-injection `gps_time` values

