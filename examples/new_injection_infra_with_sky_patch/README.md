# Example to run injection with a circular patch sky distribution

This example demonstrates how to use the new injection infrastructure to inject a gravitational wave signal into generated Gaussian noise to the signal.

## Sky Patch Distribution

PycWB support multiple sky distribution types to place the injected signal on the sky. In this example, we use a circular patch distribution. The patch is defined by its center coordinates (RA, Dec) and a radius. The unit of the coordinates can be either degrees `deg` or radians `rad`. It must be specified in the configuration file to avoid ambiguity.

```yaml
injection:
  seed: 150914
  parameters_from_python:
    function: "./injection_parameters.get_injection_parameters"
  repeat_injection: 20
  sky_distribution:
    type: Patch
    patch:
      unit: 'deg'
      center:
        ra: 0.0
        dec: 90.0
      radius: 5.0
  time_distribution:
    type: 'rate'
    rate: 1/100
    jitter: 30
  noise:
    type: "GaussianNoise"
    delta_seeds: 
      H1: 10
      L1: 20
```

## Injection

The injected signal is 36-29 solar mass binary black hole similar to GW150914, with a distance of 2000 Mpc.

```python
{
    "mass1": 36.0,
    "mass2": 29.0,
    "spin1z": 0.0,
    "spin2z": 0.0,
    "distance": 2000,
    "pol": 0,
    "t_start": -2.0, 
    "t_end": 1.0,
    "polarization": 0.0,
    "approximant": "IMRPhenomTPHM",
    "f_lower": 20.0,
}
```

The `t_start` and `t_end` are the estimated start time and end time relative to the `t0` given by the waveform generator (`pycbc.get_td_waveform` by default). These are used to estimate the start gps time and end gps time of the injection, to place the injection correctly in the job segments. It is ok to overestimate the `t_start` and `t_end`, but it should not be underestimated, otherwise some injections may be missed.


## Run the search

```bash
pycwb run user_parameters.yaml
```
