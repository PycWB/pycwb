# Fix: HEALPix Sky Coordinate Convention (FP / FX antenna patterns and ml time delays)

## Problem

`compute_sky_delay_and_patterns` in `detector.py` is called with HEALPix pixel
angles `(phi, theta)` where:

- `phi`   = **geographic longitude** (Earth-fixed azimuth, 0–2π)
- `theta` = geographic colatitude (0 at North Pole)

The time-delay calculation is correct: the dot product
`n_hat · (ECEF_det − ECEF_ref) / c` requires `n_hat` expressed in the ECEF
frame, which these geographic angles provide directly.

However, computing the **antenna patterns** uses `det.atenna_pattern(ra, dec, psi,
gps)`, which expects an _equatorial_ right ascension `ra` (not geographic longitude).
The conversion from geographic longitude `phi_geo` to equatorial RA is:

```
RA = phi_geo − GMST(t)   [or equivalently,  GHA = GMST − RA = −phi_geo]
```

The original code passed `ra = phi` (raw HEALPix longitude) directly to
`atenna_pattern`, which treated it as an equatorial RA.  This effectively made
`gha = GMST − phi_geo` instead of the correct `gha = −phi_geo`, rotating every
antenna-pattern vector by `+GMST ≈ 138°` in azimuth.

### Symptom

End-to-end comparison (GW150914-like injection, `tests/sample/run_mix_e2e.py`):

| Quantity | Before fix | After fix |
|----------|-----------|-----------|
| FP max-abs diff vs CWB | **0.432** | **3.0 × 10⁻⁶** |
| FX max-abs diff vs CWB | **0.432** | **2.9 × 10⁻⁶** |

The sub-µ residual after the fix is floating-point round-off from the GMST
computation and is scientifically negligible.

## Root cause

```python
# Before — WRONG: phi is geographic longitude, not equatorial RA
f_plus, f_cross = det.atenna_pattern(ra, dec, 0.0, float(gps_time))
#   where ra  = phi  (HEALPix pixel azimuth in geographic frame)
#         dec = pi/2 - theta
```

`atenna_pattern` internally computes `gha = GMST(gps_time) − ra_arg`, so:

```
gha = GMST − phi_geo          ← wrong sign / wrong offset
```

CWB's `setAntenna` uses `gha = −phi` (geographic, no GMST), so the required
substitution is:

```
atenna_pattern(ra_arg, …)  where  GMST − ra_arg = −phi_geo
               ra_arg = GMST + phi_geo
```

## Fix (applied 2026-03-07)

File: `pycwb/types/detector.py`, function `compute_sky_delay_and_patterns`.

```python
# After
gmst_rad = gmst_accurate(float(gps_time))
ra_eff = ra + gmst_rad   # gives  gha = GMST − ra_eff = GMST − (GMST + phi_geo) = −phi_geo

f_plus, f_cross = det.atenna_pattern(ra_eff, dec, 0.0, float(gps_time))
```

The GMST value cancels internally and the result is independent of the actual GPS
time chosen.  The time-delay `ml` computation (using `n_hat` in ECEF) is unchanged.
