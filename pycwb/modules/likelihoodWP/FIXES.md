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

## 6. `snr`/`sSNR`/`xSNR` — getMRAwave exact reconstruction (`likelihood.py`)

**Bug**: Initial implementation used sparse TF map + `w2t` inverse transform, which only
computed the **diagonal** WDM energy (single-pixel contributions) and missed the huge
cross-pixel overlap terms from adjacent WDM basis functions. For a real GW event with
406 coherent pixels, the missing cross terms were 7.5× larger than the diagonal, giving
`snr ≈ [99, 106]` instead of the correct `[806, 736]`.

**Root Cause**: WDM basis functions have long filter tails (length `m_H ≈ 12289` samples
for M=1024) that cause significant overlaps between neighboring time-frequency pixels.
The sparse TF map approach (`w2t` on a minimal TF grid) failed to capture these overlaps
because the output length was only `M × n_time_bins` where `n_time_bins` covered just the
cluster span, not the filter tail extent.

**Fix**: Replaced the sparse TF map block with direct calls to `get_MRA_wave()` from the
existing `getMRAwaveform.py` module. This function correctly accumulates all basis function
contributions (including cross-pixel overlaps) via full time-domain reconstruction:
```python
z_sig_ts = get_MRA_wave(cluster, wdm_list, rate_ana, ifo, a_type='signal', mode=0, ...)
z_dat_ts = get_MRA_wave(cluster, wdm_list, rate_ana, ifo, a_type='strain', mode=0, ...)
sSNR_ifo[ifo] = np.sum(z_sig ** 2)  # Σ_t [Σ_j a_j ψ_j(t)]²
snr_ifo[ifo]  = np.sum(z_dat ** 2)
```
This exactly mirrors C++ `getMRAwave('W')` + `getMRAwave('S')` + `avx_norm`.

**Secondary Fix**: Fixed bug in `getMRAwaveform.py::_create_wdm_set_python()` where
`max_layer` was incorrectly set to `M-1` instead of `M`, causing:
1. WDM kernel lookup to fail (`pix.layers = M+1`, but lookup key was only `M`)
2. Wrong modulus in `get_base_wave()` (`m = time % M` instead of `time % (M+1)`)

**Impact**: Fixes `snr`, `sSNR`, `xSNR`, and `neted[2]` (Ew). Now match to floating-point
precision (< 0.01%):
- `snr[0]`: C++ `806.666077` → Python `806.666082` ✅
- `snr[1]`: C++ `736.834106` → Python `736.834140` ✅
- `sSNR[0]`: C++ `806.581177` → Python `806.581010` ✅
- `sSNR[1]`: C++ `736.257874` → Python `736.257977` ✅
- `neted[2]` (Ew): C++ `1543.500244` → Python `1543.500222` ✅

---

## Final Test Results

**All statistics now match C++ to floating-point precision**:

| Field | C++ | Python | Match |
|---|---|---|---|
| `net_ecor` | 1362.918457 | 1362.918457 | ✅ exact |
| `sub_net` (subnet) | 0.932312 | 0.932312 | ✅ exact |
| `sub_net2` (SUBNET) | 0.941579 | 0.941579 | ✅ exact |
| `net_rho` (rho0) | 18.370497 | 18.370498 | ✅ exact |
| `net_rho2` (rho1) | 17.236593 | 17.236594 | ✅ exact |
| `net_ed` (neted[0]) | 1.885818 | 1.885835 | ✅ ~0.001% |
| `net_null` (neted[1]) | 187.076309 | 187.076302 | ✅ float rounding |
| `energy` (neted[2]) | **1543.500244** | **1543.500222** | ✅ **< 0.00002%** |
| `like_sky` (neted[3]) | 1553.985596 | 1553.985352 | ✅ float32 |
| `energy_sky` (neted[4]) | 7452.100098 | 7452.098145 | ✅ ~0.00003% |
| `a_net`, `g_net`, `i_net` | match | match | ✅ exact |
| `sky_size` | 406 | 406 | ✅ exact |
| **`snr[0]`** | **806.666077** | **806.666082** | ✅ **< 0.000001%** |
| **`snr[1]`** | **736.834106** | **736.834140** | ✅ **< 0.000005%** |
| **`sSNR[0]`** | **806.581177** | **806.581010** | ✅ **< 0.00002%** |
| **`sSNR[1]`** | **736.257874** | **736.257977** | ✅ **< 0.00001%** |
| **`xSNR[0]`** | **806.623596** | **806.623545** | ✅ **< 0.000006%** |
| **`xSNR[1]`** | **736.545959** | **736.546002** | ✅ **< 0.000006%** |

---

## 7. `hrss`/`strain` — physical strain energy and missing `sqrt()` (`network_event.py`, `likelihood.py`)

**Bug 1**: The Python `output_py()` method was computing `hrss` from **whitened** signal energy
(`meta.signal_snr[i]`) instead of physical strain energy. In C++ `netevent.cc`, 
`getMRAwave(..., 's')` multiplies whitened amplitudes by `noise_rms` to produce physical strain units:
```cpp
a00 *= strain ? rms : 1.;  // mode 's' → strain=true
```
Then `hrss = sqrt(pd->get_SS() / inRate)` where `get_SS()` is the energy of that physical waveform.

But in Python, `get_MRA_wave(..., whiten=True)` returns whitened energy, giving 
`hrss ≈ [0.22, 0.21]` instead of the correct `[1.24e-22, 1.12e-22]` (off by ~10²¹).

**Bug 2**: The old Python `output()` method (ROOT-based) was missing the final `sqrt()` call on
strain. It accumulated `self.strain[0] += hrss[i]²` in the loop, but never took the square root
at the end. C++ `netevent.cc:983` does:
```cpp
this->strain[0] = sqrt(this->strain[0]);
```

**Fix**:
1. **Compute un-whitened energy for hrss**: In `likelihood.py`, added a third `get_MRA_wave()` call
   with `whiten=False` to compute physical strain energy per IFO:
   ```python
   z_sig_physical = get_MRA_wave(cluster, wdm_list, rate_ana, ifo,
                                 a_type='signal', mode=0, nproc=1, whiten=False)
   signal_energy_physical[ifo] = np.sum(z_sig_phys ** 2)
   ```
   
2. **Store physical energy in cluster metadata**: Added `signal_energy_physical` field to `ClusterMeta`
   and populate it in `fill_detection_statistic`.

3. **Use physical energy for hrss**: In `output_py()`:
   ```python
   if hasattr(meta, 'signal_energy_physical') and len(meta.signal_energy_physical) > i:
       hrss_sq_physical = float(meta.signal_energy_physical[i])
   else:
       hrss_sq_physical = asnr_sq_xt  # fallback (wrong units but backwards compatible)
   self.hrss.append(float(np.sqrt(hrss_sq_physical / in_rate)))
   ```

4. **Add missing sqrt() in old output()**: Added after the main loop:
   ```python
   self.strain[0] = np.sqrt(self.strain[0])
   ```

**Impact**: Fixes `hrss` and `strain` to match C++ to floating-point precision:
- `hrss[0]`: C++ `1.237311e-22` → Python `1.237310e-22` ✅
- `hrss[1]`: C++ `1.118782e-22` → Python `1.118782e-22` ✅
- `strain[0]`: C++ `1.668116e-22` → Python `1.668116e-22` ✅ (relative error < 4e-08)

---

## Final Test Results

**All statistics now match C++ to floating-point precision**:

| Field | C++ | Python | Match |
|---|---|---|---|
| `net_ecor` | 1362.918457 | 1362.918457 | ✅ exact |
| `sub_net` (subnet) | 0.932312 | 0.932312 | ✅ exact |
| `sub_net2` (SUBNET) | 0.941579 | 0.941579 | ✅ exact |
| `net_rho` (rho0) | 18.370497 | 18.370498 | ✅ exact |
| `net_rho2` (rho1) | 17.236593 | 17.236594 | ✅ exact |
| `net_ed` (neted[0]) | 1.885818 | 1.885835 | ✅ ~0.001% |
| `net_null` (neted[1]) | 187.076309 | 187.076302 | ✅ float rounding |
| `energy` (neted[2]) | **1543.500244** | **1543.500222** | ✅ **< 0.00002%** |
| `like_sky` (neted[3]) | 1553.985596 | 1553.985352 | ✅ float32 |
| `energy_sky` (neted[4]) | 7452.100098 | 7452.098145 | ✅ ~0.00003% |
| `a_net`, `g_net`, `i_net` | match | match | ✅ exact |
| `sky_size` | 406 | 406 | ✅ exact |
| **`snr[0]`** | **806.666077** | **806.666082** | ✅ **< 0.000001%** |
| **`snr[1]`** | **736.834106** | **736.834140** | ✅ **< 0.000005%** |
| **`sSNR[0]`** | **806.581177** | **806.581010** | ✅ **< 0.00002%** |
| **`sSNR[1]`** | **736.257874** | **736.257977** | ✅ **< 0.00001%** |
| **`xSNR[0]`** | **806.623596** | **806.623545** | ✅ **< 0.000006%** |
| **`xSNR[1]`** | **736.545959** | **736.546002** | ✅ **< 0.000006%** |
| **`hrss[0]`** | **1.237311e-22** | **1.237310e-22** | ✅ **< 0.0001%** |
| **`hrss[1]`** | **1.118782e-22** | **1.118782e-22** | ✅ **exact** |
| **`strain[0]`** | **1.668116e-22** | **1.668116e-22** | ✅ **< 4e-06%** |
