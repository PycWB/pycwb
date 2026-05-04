# tests/parquet_perf

Performance benchmark suite for `InjectionParams` Parquet I/O and query patterns.

## Scripts

| Script | Purpose |
|---|---|
| `generate_params.py` | Synthesise 10k / 50k / 100k mixed WNB, SG, BBH injection dicts → `data/params_*.json` |
| `write_parquet.py` | Build `InjectionParams` objects and write Arrow struct columns to `parquet/params_*.parquet` |
| `benchmark_queries.py` | Read and query the Parquet files with PyArrow and DuckDB; print timing table |

## Quickstart

```bash
cd tests/parquet_perf

# 1. Generate synthetic injection parameters
python generate_params.py

# 2. Write InjectionParams struct columns to Parquet
python write_parquet.py

# 3. Run queries and print benchmark table
python benchmark_queries.py

# Or do all three in one command:
python benchmark_queries.py --generate
```

## Waveform types

| Type | `approximant` | Key JSON params |
|---|---|---|
| WNB | `WNB` | `frequency`, `bandwidth`, `duration`, `hrss` |
| Sine-Gaussian | `SGE` | `frequency`, `Q`, `hrss` |
| BBH | `IMRPhenomTPHM` etc. | `mass1`, `mass2`, `spin1z`, `distance` |

## Queries benchmarked

**PyArrow (typed struct columns)**
- WNB only / SG only / BBH only  
- `hrss > 1e-22` (float32 typed → predicate pushdown)  
- GPS time window (float64 typed → predicate pushdown)  
- BBH heavy total mass > 60 M☉ (JSON blob parse)  
- BBH |spin1z| > 0.5 (JSON blob)  
- SG with Q = 100 (JSON blob)  
- WNB with frequency < 100 Hz (JSON blob)  
- Full round-trip deserialisation to `InjectionParams`

**DuckDB (SQL over Parquet)**  
Same filters plus aggregations, using `json_extract` for blob fields.
