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

---

## 8. Per-IFO fields: `null`, `nill`, `bp`, `bx`, `time`, `duration`, `frequency`, `bandwidth`

**Context**: After adding comprehensive field comparison with scientific notation logging (for fields that can be as small as 1e-44), additional discrepancies were discovered in per-IFO output fields.

### 8a. `nill` (null stream energy) — wrong formula

**Bug**: The Python `output_py()` was computing `nill = wave_snr - signal_snr`, but C++ `netevent.cc` uses:
```cpp
this->nill[i] = pd.xSNR - pd.sSNR;  // cross SNR - signal SNR
```
This caused `nill` to be off by a factor of ~2.

**Fix**: Changed formula in `output_py()` (`network_event.py`):
```python
# Before: self.nill.append(float(wave_sq_xt - asnr_sq_xt))
# After:  self.nill.append(float(xsnr_sq_xt - asnr_sq_xt))
```

**Impact**: Fixes `nill` to match C++ (< 0.3% error):
- `nill[0]`: C++ `4.241943e-02` → Python `4.253490e-02` ✅
- `nill[1]`: C++ `2.880859e-01` → Python `2.880252e-01` ✅

### 8b. `null` (null energy per IFO) — not using per-IFO null from getMRAwave

**Bug**: The Python code was splitting the total pixel null energy evenly across IFOs:
```python
null_val = sum(p.null for p in all_pixels) / n_ifo
```
But C++ `netevent.cc` uses the per-detector null energy from `getMRAwave()`:
```cpp
this->null[i] = pd.null;  // from getMRAwave reconstruction
```
The C++ `network::getMRAwave()` computes null energy per IFO by reconstructing the null stream wavelet and computing its energy.

**Fix**: 
1. Added `null_energy: list` field to `ClusterMeta` in `network_cluster.py`
2. In `likelihood.py`, store per-IFO null from `getMRAwave`:
   ```python
   cluster.cluster_meta.null_energy = null_ifo.tolist()
   ```
3. In `output_py()` (`network_event.py`), use per-IFO null:
   ```python
   null_val = float(meta.null_energy[i])
   ```

**Impact**: Fixes `null` to match C++ (< 0.03% error):
- `null[0]`: C++ `4.290383e-05` → Python `4.291500e-05` ✅
- `null[1]`: C++ `2.361072e-03` → Python `2.361077e-03` ✅

### 8c. `bp`, `bx` (antenna patterns) — wrong polarization angle and scaling

**Bug 1**: The Python code was using `psi=0.0` (default) instead of the cluster's polarization angle:
```python
fp, fx = det.atenna_pattern(ra_rad, dec_rad, 0.0, gps_t)  # WRONG: psi=0.0
```

**Bug 2**: The C++ `detector::antenna()` returns `wavecomplex(fp/2, fx/2)` — the antenna patterns divided by 2:
```cpp
wavecomplex z(fp/2., fx/2.);
return z;
```
But the Python code was not applying this scaling.

**Fix** (`network_event.py`): 
```python
psi_rad = float(np.radians(meta.psi))  # Use cluster psi
fp, fx = det.atenna_pattern(ra_rad, dec_rad, psi_rad, gps_t)
self.bp.append(float(fp / 2.0))  # Divide by 2 to match C++ convention
self.bx.append(float(fx / 2.0))
```

**Impact**: Antenna patterns are now closer after applying psi and /2 scaling, but still show some differences:
- `bp[0]`: C++ `0.398082` → Python `0.427151` (~7% diff)
- `bp[1]`: C++ `-0.167759` → Python `-0.317525` 
- `bx[0]`: C++ `0.420414` → Python `-0.212679` 
- `bx[1]`: C++ `-0.435123` → Python `0.178662` 

**Note**: The remaining differences may be due to coordinate transformation differences between C++ detector-frame calculations and Python equatorial-frame transformations, or additional implementation details in the C++ `detector::antenna()` method that need further investigation.

### 8d. `time`, `duration`, `frequency`, `bandwidth` — multiple root-cause bugs

#### 8d-i. `c_time` / `c_freq` wrong normalization (`likelihood.py`)

**Bug**: `To` (centroid time) and `Fo` (centroid frequency) were divided by `Lw` — the MRA
waveform likelihood which includes WDM cross-pixel overlap terms (~7.6× larger than the
diagonal pixel energy). This gave `c_time ≈ 38s` instead of the correct `~289s`.

**Original broken code**:
```python
To /= Lw   # Lw ≈ 1543 (includes cross-pixel overlaps)
Fo /= Lw
```

**Fix**: C++ computes `To` and `Fo` as `getWFtime()` / `getWFfreq()` — energy-weighted
centroids of the time-domain MRA-reconstructed waveform, then normalises by `Lw` (sum of
per-IFO `sSNR`). Replaced the pixel-based centroid with exact C++ equivalents using FFT:

```python
# For each IFO, after get_MRA_wave():
e_sig = z_sig ** 2
wf_time_ifo = t_start + np.dot(e_sig, np.arange(n)) / (E_sig * rate_wf)  # getWFtime()
Z_fft = np.fft.rfft(z_sig)
power = Z_fft.real**2 + Z_fft.imag**2
wf_freq_ifo = np.dot(power, np.arange(len(power))) * rate_wf / n / E_fft  # getWFfreq()
To += sSNR_ifo[ifo_i] * wf_time_ifo
Fo += sSNR_ifo[ifo_i] * wf_freq_ifo
...
To /= Lw; Fo /= Lw  # Lw = sum(sSNR_ifo), mirrors C++ netevent.cc line 931
```

**Impact**: `c_time` corrected from `38.07s` to `289.327s` (~0.006s remaining diff vs C++).

#### 8d-ii. `time` per IFO — sky-delay correction (`network_event.py`)

**Bug**: The sky time-delay offset (`tau_i - tau_0`) was not applied per IFO, so all IFOs
shared the same `c_time + gps[i]`.

**Fix**: 
```python
sky_td = list(cluster.sky_time_delay)
td_rate = float(config.TDRate)
tau_ref = float(sky_td[0]) / td_rate
tau_ifo = [float(sky_td[i]) / td_rate for i in range(n_ifo)]
self.time = [c_time + float(self.gps[i]) + (tau_ifo[i] - tau_ref if i > 0 else 0.0)
             for i in range(n_ifo)]
```

**Impact**: `time[0]`: C++ `1126259162.326904` → Python `1126259162.326891` ✅ (<15μs)

#### 8d-iii. `duration` per IFO — wrong data field and rounding (`network_event.py`)

**Bug 1**: Used energy-weighted mean from `p.data[i].asnr`; C++ uses min/max of per-IFO
pixel time bins from `p.data[i].index // p.layers * (1/rate)`.

**Bug 2**: Python used float division `p.time / mm` where C++ uses integer division
`pList[M].time / mm` (both `size_t`).

**Bug 3**: WDM time offset `dT` was always `0.5`; C++ uses `dT = (mm == mp) ? 0 : 0.5`.

**Fix** (per-IFO range, then WDM sub-bin RMS for slot [0]):
```python
# Per-IFO range (slots 1+):
time_bin = int(p.data[ifo_idx].index) // int(p.layers)  # integer division
t_starts.append(dt * time_bin); t_stops.append(dt * (time_bin + 1))
per_ifo_duration.append(max(t_stops) - min(t_starts))

# WDM sub-bin energy-weighted RMS for slot [0]:
dT = 0.0 if mm == mp else 0.5          # correct WDM offset flag
time_bin = int(p.time) // mm           # integer division (matches C++ size_t/size_t)
iT = (float(time_bin) - dT) * dt
```

**Impact**:
- `duration[0]`: C++ `0.076357` → Python `0.076357` ✅ exact
- `duration[1]`: C++ `1.750000` → Python `1.750000` ✅ exact

#### 8d-iv. `frequency` per IFO — two different source values (`network_event.py`)

**Bug**: All IFOs received `c_freq` (pixel-centroid Fo, ~121.75 Hz). C++ uses two different
values:
- `frequency[i>0]` = `cFreq_net.data[kid]` = WDM sub-bin likelihood-weighted mean (case `'f'`/`'L'` in `netcluster::get()`)
- `frequency[0]` = `pcd->cFreq` = `Fo` from likelihood (MRA spectral centroid = `meta.c_freq`)

**Fix**: The WDM sub-bin accumulators `a_f, b_f` are already computed in the duration/bandwidth
block. `a_f/b_f` is the likelihood-weighted sub-bin mean frequency = C++ `cFreq_net.data[kid]`.
```python
lh_freq_net = a_f / b_f          # C++ cFreq_net (slots 1+)
self.frequency = [lh_freq_net] * n_ifo
self.frequency[0] = c_freq        # C++ pcd->cFreq = meta.c_freq (MRA spectral centroid)
```

**Impact**:
- `frequency[0]`: C++ `110.805565` → Python `110.805571` ✅ (<6e-6 rel err)
- `frequency[1]`: C++ `77.210429` → Python `77.210429` ✅ exact

#### 8d-v. `bandwidth` per IFO — correct after WDM sub-bin fix

The bandwidth sub-bin formula was already correct in structure; fixing the integer division and
`dT` flag (see 8d-iii) brought it into agreement.

**Impact**:
- `bandwidth[0]`: C++ `60.234792` → Python `60.234790` ✅ (<3e-8 rel err)
- `bandwidth[1]`: C++ `336.000000` → Python `336.000000` ✅ exact

### 8e. `noise` (noise RMS) — different values

**Bug**: The Python code samples noise RMS from the TF map at pixel locations, but shows ~12% difference from C++:
- `noise[0]`: C++ `4.471662e-24` → Python `3.913027e-24` (~12% diff)
- `noise[1]`: C++ `4.307999e-24` → Python `3.829385e-24` (~11% diff)

**Status**: This may be due to different TF map sampling methods or noise estimation procedures. Requires further investigation of C++ `netevent.cc` noise computation details.

---

## Current Test Status Summary

### ✅ Fields Matching Well (< 0.5% error):
- **Network statistics**: `net_ecor`, `sub_net`, `sub_net2`, `net_rho`, `net_rho2`, `net_ed`, `net_null`, `energy`, `like_sky`, `energy_sky`
- **Per-IFO energies**: `snr`, `sSNR`, `xSNR` (< 0.00002% error)
- **Strain fields**: `hrss` (< 0.0001% error), `strain` (< 4e-06% error)
- **Null statistics**: `null` (< 0.03% error), `nill` (< 0.3% error)
- **`time`**: < 15 μs error per IFO ✅
- **`duration`**: exact match for both IFOs ✅
- **`frequency`**: < 6e-6 relative error for slot [0]; exact for slot [1] ✅
- **`bandwidth`**: < 3e-8 relative error for slot [0]; exact for slot [1] ✅

### ⚠️ Fields with Remaining Differences:
- **`bp`, `bx`** (antenna patterns): Differences remain; likely due to coordinate transformation implementation details or GPS-epoch corrections in C++ `detector::antenna()`
- **`noise`** (RMS): ~12% difference, requires investigation of C++ noise computation

