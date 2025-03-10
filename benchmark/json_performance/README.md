# Code for testing the orjson performance

This is a simple benchmark to test the performance of the orjson library.

To generate the data, run the following command:

```bash
python data_gen.py
```

To run the benchmark, run the following command:

```bash
python read_perf.py
```


Results for 1M events on CIT

```
Time taken to read from file: 46.73175929673016
Time taken to read from file: 36.13206199463457
Time taken to read from file: 36.519760328345
```