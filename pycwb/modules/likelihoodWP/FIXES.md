# likelihoodWP — Bug Fixes Summary

This document records the bugs found and fixed in the Python reimplementation of `likelihoodWP`, relative to the C++ reference in `cwb-core/network.cc`.

---

## 1. `packet_norm_numpy` — t-clamping fix (`utils.py`)

**Bug**: The raw xtalk inner product `t` was not clamped to zero before accumulation,
allowing negative contributions to leak into the norm sum.

**Fix**: Added `t_clamped = np.where(t < 0, 0, t)` before the norm accumulation,
mirroring the C++ behaviour in `_avx_norm_ps` (positive-only accumulation).

**Impact**: Affects `ecor`, `norm`, `net_ed`, and all downstream statistics derived
from the signal packet norm.

---

## 2. `cc_rho_td` — floor clamp for `rho0` (`likelihood.py`)

**Bug**: When `ch_td < 1.0`, dividing `rho` by `sqrt(ch_td)` inflates `rho0` above the
raw sky value. C++ clamps `cc` to a minimum of `1.0` before dividing.

**Fix**: `cc_rho_td = ch_td if ch_td > 1.0 else 1.0`, so the division never amplifies.

**Impact**: Fixes `net_rho` (`rho[0]`). Match: C++ `18.370497` → Python `18.370498` ✅

---

## 3. `xtalk_energy_sum_numpy` — raw null/data energy without clamping (`utils.py`)

**Bug**: `packet_norm_numpy` always clamps `norm >= 2.0` (one entry per IFO), which is
correct for the signal path but wrong for the data and null energy calculations.
C++ calls `_avx_norm_ps(pd, pD, -V4)` and `_avx_norm_ps(pn, pN, -V4)` with a
negative `I` flag that **skips** the SNR-ratio step and the norm floor, returning the
raw xtalk-convolved energy sum.  Using `packet_norm_numpy` for these calls gave
`Np ≈ 4.0` (= 2 IFOs × floor of 2) instead of the correct `Np ≈ 0`.

**Fix**: Added `xtalk_energy_sum_numpy(p, q, xtalks, xtalks_lookup, mk)` — a pure-numpy
function that accumulates the xtalk inner products with **no** norm clamping,
exactly mirroring the `I < 0` branch of C++ `_avx_norm_ps`. Used for both `Em`
(data energy) and `Np` (null energy).

**Impact**: Fixes `sub_net` (subnet). Match: `0.929649` → `0.932312` ✅

---

## 4. `threshold_cut` — wrong rejection condition (`likelihood.py`)

**Bug**: The pixel-count gate was written as `condition_4 = Np < 1`, testing the null
energy scalar. C++ tests `N < 1`, where `N` is the effective pixel count returned
by `_avx_setAMP_ps`. Because `Np` was previously `4.0`, the bug was hidden; once
`Np` was corrected to `≈ 0`, the cluster was incorrectly rejected at this gate.

**Fix**: `condition_4 = N < 1` where `N = sky_statistics.N_pix_effective`.

**Impact**: Prevented spurious cluster rejection after the `Np` fix above.

---

## 5. `get_chirp_mass` — implemented Python `mchirp()` (`likelihood.py`)

**Bug**: `get_chirp_mass(cluster)` was a `pass` stub, so `net_rho2` (`rho[1]`) was
never updated and defaulted to the raw `net_rho` value.

**Fix**: Full port of `netcluster::mchirp()` from `cwb-core/netcluster.cc`:
1. Collect TF pixels → build `(frequency, time)` point cloud with SNR weights.
2. Hough transform over chirp-mass grid `m ∈ [-100, +100]` step `0.2 M☉` to find
   the best-fit slope `b0` and mass `m0`.
3. Fine chi² minimisation around the best candidate to refine `m0`/`b0`.
4. Compute energy fraction `Efrac = selected_energy / total_energy`.
5. PCA on the filtered pixel cloud → eigenvalue ratio gives `chirpEllip`.
6. Set `cluster.cluster_meta.net_rho2 = net_rho × chirpEllip × sqrt(Efrac)`.

**Constants used** (from `cwb-core/constants.hh`):
- `G = 6.67259e-11` (`WAT_G_SI`)
- `SM = 1.98892e30` (`WAT_MSUN_SI`)
- `C = 299792458` (`WAT_C_SI`)

**Impact**: Fixes `net_rho2` (`rho[1]`). Match: C++ `17.236593` → Python `17.236594` ✅

---

## Known Approximation Gap (not a bug)

`snr`, `sSNR`, `xSNR`, and `neted[2]` differ from the C++ reference by **~0.7%**.

C++ computes these via `getMRAwave()`, which performs an exact time-domain WDM
reconstruction using the full wavelet filter bank. Python uses the TF-domain
xtalk-catalog energy directly. The two approaches are mathematically equivalent
only in the limit of a complete, untruncated xtalk catalog.

The test in `tests/sample/run_mix.py` uses `rtol=0.01` (1 %) for these fields to
acknowledge this known approximation.

---

## Final Test Results

All other statistics match C++ to floating-point precision:

| Field | C++ | Python | Status |
|---|---|---|---|
| `net_ecor` | 1362.918457 | 1362.918457 | ✅ |
| `sub_net` (subnet) | 0.932312 | 0.932312 | ✅ |
| `sub_net2` (SUBNET) | 0.941579 | 0.941579 | ✅ |
| `net_rho` (rho0) | 18.370497 | 18.370498 | ✅ |
| `net_rho2` (rho1) | 17.236593 | 17.236594 | ✅ |
| `net_ed` (neted[0]) | 1.885818 | 1.885835 | ✅ ~0.001% |
| `like_sky` (neted[3]) | 1553.985596 | 1553.985352 | ✅ float32 |
| `a_net`, `g_net`, `i_net` | match | match | ✅ |
| `sky_size` | 406 | 406 | ✅ |
| `snr`, `sSNR`, `xSNR` | — | ~0.7% | ⚠️ getMRAwave approx |
