# Example to run injection with a circular patch sky distribution

This example demonstrates how to use the new injection infrastructure to inject a gravitational wave signal into generated Gaussian noise to the signal.

## LHV network

To use detectors other than LH with aLIGO Zero Det High Power PSD, you need to provide the PSD files for the detectors in the `user_parameters.yaml` file. The example below uses LHV (LIGO Hanford, LIGO Livingston, Virgo) network.

```yaml
injection:
  seed: 150914
  parameters_from_python:
    function: "./injection_parameters.get_injection_parameters"
  repeat_injection: 100
  sky_distribution:
    type: UniformAllSky
  time_distribution:
    type: 'rate'
    rate: 1/30
    jitter: 10
  noise:
    type: "GaussianNoise"
    psds:
      H1: "input/aligo_O4high.txt"
      L1: "input/aligo_O4high.txt"
      V1: "input/avirgo_O4high_NEW.txt"
    delta_seeds: 
      H1: 10
      L1: 20
      V1: 30
```

The PSD files used here are downloaded from the LIGO DCC. You can find the files in the following links:

https://dcc.ligo.org/LIGO-T2000012/public

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
