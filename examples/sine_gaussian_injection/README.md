# Example: Sine Gaussian Injection

## Install the required packages

```bash
pip install burst-waveform
```

## Fetch the required data

```bash
pycwb gwosc-data user_parameters.yaml
```

This command will read the `ifo`, `gps_start` and `gps_end` from the `user_parameters.yaml` file and download the frame files and data quality files from GWOSC.


## Run the search

```bash
pycwb run user_parameters.yaml
```
