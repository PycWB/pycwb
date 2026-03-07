# Fix: MIN_SKYRES_HEALPIX incorrectly applied to likelihood sky scan

## Problem

`setup_supercluster` (in `super_cluster.py`) capped the HEALPix sky resolution to
`MIN_SKYRES_HEALPIX` (default 4, i.e. nside=16, **3,072 pixels**) before computing
`ml / FP / FX`, and those reduced-resolution arrays were passed all the way through to
the likelihood sky scan.

The ROOT/CWB pipeline (`supercluster.py`) uses `MIN_SKYRES_HEALPIX` **only** during the
fast subnet-cut step (via `network.update_sky_map`), then restores the full
`config.healpix` resolution (e.g. order 7 → nside=128, **196,608 pixels**) before
running likelihood. The native Python pipeline was never restoring full resolution,
causing the likelihood to scan a coarse grid and report the wrong sky direction.

### Symptom

End-to-end comparison (`tests/sample/run_mix_e2e.py`) with `healpix=7` and
`MIN_SKYRES_HEALPIX=4` showed:

| Pipeline | phi (°) | theta (°) | n_sky |
|----------|---------|-----------|-------|
| ROOT/CWB (hybrid) | 58.01   | 73.67     | 196,608 |
| Native Python (before fix) | 187.50  | 6.60      | **3,072** |
| Native Python (after fix)  | 236.25  | 52.46     | 196,608 |

The remaining angular offset between hybrid and native (58° vs 236°) is a known
sky-ring degeneracy of a 2-detector network — the `nSkyStat` maximum values differ by
less than 1 unit.

## Root cause

In `setup_supercluster` the original code was:

```python
healpix_order = int(config.healpix)        # e.g. 7
min_skyres = int(getattr(config, "MIN_SKYRES_HEALPIX", healpix_order))  # e.g. 4
if healpix_order > min_skyres:
    healpix_order = min_skyres             # overwrites to 4 — BUG

ml, FP, FX = compute_sky_delay_and_patterns(..., healpix_order=healpix_order, ...)
# returns (n_ifo, 3072) arrays

return {"ml": ml, "FP": FP, "FX": FX, "n_sky": 3072, ...}
```

`process_job_segment_native.py` then passed these 3,072-pixel arrays to
`setup_likelihood`, which performed the likelihood sky scan at the wrong resolution.

## Fix (applied 2026-03-07)

Three files were changed:

### 1. `super_cluster.py` — `setup_supercluster`

Compute **two** sets of sky arrays:

- `ml / FP / FX` at `MIN_SKYRES_HEALPIX` resolution → subnet cut only (fast)
- `ml_likelihood / FP_likelihood / FX_likelihood` at full `config.healpix` resolution → likelihood sky scan

The return dict now includes both sets:

```python
return {
    "ml":   ml_subnet,           # reduced resolution for apply_subnet_cut
    "FP":   FP_subnet,
    "FX":   FX_subnet,
    "n_sky": int(ml_subnet.shape[1]),
    "ml_likelihood":   ml,       # full resolution for likelihood sky scan
    "FP_likelihood":   FP,
    "FX_likelihood":   FX,
    "n_sky_likelihood": int(ml.shape[1]),
    ...
}
```

### 2. `process_job_segment_native.py` — `lh_setup` call

```python
# Before:
lh_setup = setup_likelihood(config, strains, config.nIFO,
                            ml=sc_setup["ml"],
                            FP=sc_setup["FP"],
                            FX=sc_setup["FX"])

# After:
lh_setup = setup_likelihood(config, strains, config.nIFO,
                            ml=sc_setup.get("ml_likelihood", sc_setup["ml"]),
                            FP=sc_setup.get("FP_likelihood", sc_setup["FP"]),
                            FX=sc_setup.get("FX_likelihood", sc_setup["FX"]))
```

### 3. `tests/sample/run_mix_e2e.py`

Same `.get("ml_likelihood", ...)` pattern applied to both `lh_setup` and
`lh_setup_cross`.

## Notes

- `supercluster_single_lag` (inside `super_cluster.py`) uses `setup["ml"]`,
  `setup["FP"]`, `setup["FX"]`, and `setup["n_sky"]` for `apply_subnet_cut` — these
  correctly remain at the reduced resolution.
- `supercluster_wrapper` has the same original bug but is not used in the streaming
  native pipeline; it may need a similar fix if activated in the future.
- The `.get()` fallback ensures backward compatibility with any callers that already
  produce a setup dict without the `*_likelihood` keys.
