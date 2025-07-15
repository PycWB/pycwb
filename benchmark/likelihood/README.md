# Likelihood Benchmark

This benchmark is designed to test the performance of the likelihood in Python

## Prepare the test data for the script

Run the user parameter file with the customized processor `data_generator.py`

```bash
pycwb run user_parameters_injection.yaml
```

A `test_data.pkl` file will be generated for all the Python variables required for likelihood benchmark.

## Run the benchmark

Run the script `performance_test_opt_sky.py` to run the benchmark.

```bash
python performance_test_opt_sky.py
```