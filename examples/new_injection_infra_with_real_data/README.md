# Example to run injection with real data and the new injection infrastructure.

The time period of the example is 2-hour data before the event GW190521, which does not contain any known gravitational wave signal.

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


## Fetch the required data

```bash
pycwb gwosc-data user_parameters.yaml
```

This command will read the `ifo`, `gps_start` and `gps_end` from the `user_parameters.yaml` file and download the frame files and data quality files from GWOSC.


## Run the search

```bash
pycwb run user_parameters.yaml
```
