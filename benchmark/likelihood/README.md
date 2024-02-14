# Likelihood Benchmark

This benchmark is designed to test the performance of the likelihood in Python

## Prepare the test data for the script

Run the script `generate_data_for_likelihood.py` to generate the cluster data for the likelihood benchmark.

```bash
python generate_data_for_likelihood.py
```

A `test_data.pkl` file will be generated for all the Python variables required for likelihood benchmark.

## Run the benchmark

Run the script `performance_test_opt_sky.py` to run the benchmark.

```bash
python performance_test_opt_sky.py
```