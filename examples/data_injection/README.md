# Example for injecting into data/frames

This example demonstrates how to inject into the real data

## Seting up the framefiles and DQ files

The framefiles are not included in this repository, but you can set them up through command

```bash
pycwb gwosc GW150914
```

## Explanation

The addition of the following lines in the configuration file, together with the `injection_parameters.py`,  will allow the injection of the signal into the data

```yaml
injection:
  allow_reuse_data: True
  repeat_injection: 1
  parameters_from_python:
    file: "injection_parameters.py"
    function: "get_injection_parameters"
  sky_distribution:
    type: UniformAllSky
  time_distribution:
    type: 'rate'
    rate: 1/200
    jitter: 50
  approximant: "IMRPhenomXPHM"
```

The `injection_parameters.py` file contains the function `get_injection_parameters` to generate a list of parameters for the injection.

The injection process contains the following steps:
1. Generate the parameters for the injection using the function `get_injection_parameters` in `injection_parameters.py`
2. Place the parameters on the sky with given sky distribution: 
    - `UniformAllSky`
    - `Patch`: for a circle on the sky
    - `existing`: use a given ra, dec list
    - `Custom`: use a healpix map with the probability distribution on each pixel
    - If the sky distribution is not given, this step will be skipped, user has to set the ra, dec in the first step
3. Place the signal in the data with given time distribution: 
    - `rate`: evenly distributed in gps time with `rate` and `jitter`
    - `poisson`: next event will be placed `t + np.random.exponential(1/rate)` with given `rate`
    - `custom`: this step will be skipped, user has to set the gps time in the first step



