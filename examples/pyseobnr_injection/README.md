# Optional pySEOBNR injection generator

This example generates ten SEOBNRv5HM binaries using the current
`GenerateWaveform` API. The adapter maps `coa_phase` to `phi_ref`, preserves LAL
waveform epochs and sampling rates in native time series, and leaves sky
polarization to PycWB's detector projection.

Use a compatible pySEOBNR environment with PycWB installed. For example:

```bash
conda create -n pycwb-seobnr -c conda-forge python=3.11 pyseobnr
conda activate pycwb-seobnr
# From the matching PycWB source checkout:
python -m pip install .
# Then copy this example directory to a fresh working directory and run there:
pycwb run user_parameters_injection.yaml --list-jobs
pycwb run user_parameters_injection.yaml --jobs 0
```

Omit `--jobs 0` to process the full batch. This is an optional dependency: a
standard LALSuite installation alone does not provide pySEOBNR. Python 3.11 with
Conda's pySEOBNR was used for waveform checks; building its `pygsl_lite`
dependency from PyPI failed in the Python 3.13 test environment. Do not assume
that installing the core PycWB package also installs this integration.
