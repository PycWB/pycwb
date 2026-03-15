# Bug Fixes — `cwb_coherence` module

This file records algorithm fixes applied to the `cwb_coherence` module to match the
behaviour of the reference CWB C++ implementation.

---

## 1. `_time_delay_max_energy_pattern_jit` — wrong layer zeroing before `Gamma2Gauss` (`time_delay_max_energy.py`)

### Background

`maxEnergy` accumulates the element-wise maximum WDM packet energy over all time-delay shifts
`k = N, 2N, … K` (left and right).  Before passing the accumulated map to `Gamma2Gauss`, C++
zeros certain boundary frequency layers to suppress edge artefacts.  The relevant C++ code
(from `wseries.cc`) is:

```cpp
int M  = tmp.maxLayer() + 1;         // ← key: see below
this->getLayer(xx, 0.1);  xx = 0;  this->putLayer(xx, 0.1);   // zeros layer 0
this->getLayer(xx, M-1);  xx = 0;  this->putLayer(xx, M-1);   // zeros layer M-1
if (m == 5 || m == 6 || m == 9) {
    this->getLayer(xx, 1);    xx = 0;  this->putLayer(xx, 1);       // zeros layer 1
    this->getLayer(xx, M-2);  xx = 0;  this->putLayer(xx, M-2);     // zeros layer M-2
}
```

### Root Cause

`tmp` (type `WSeries<float>`) is populated by `wdmPacket(pattern, 'E')`, which internally
calls `this->resize(J)`.  `resize` calls `pWavelet->reset()`, which sets `m_Level = 0`
(from `WDM.hh`: `inline virtual void reset() { m_Level = 0; }`).

Therefore:
```
tmp.maxLayer() = 0   →   M = tmp.maxLayer() + 1 = 1
```

Consequently:
- `getLayer(xx, 0.1)` → zeros frequency layer **0**
- `getLayer(xx, M-1 = 0)` → zeros frequency layer **0** again *(same layer, no-op)*
- For patterns 5, 6, 9: `getLayer(xx, 1)` → zeros layer **1**;
  `getLayer(xx, M-2 = -1)` → accesses the 90°-phase half of layer 1, which does not exist
  after `resize(J)` (only the 0°-phase band is present), so this is effectively a no-op.

**The actual C++ behaviour is: only layer 0 is zeroed** (and layer 1 additionally for
patterns 5, 6, 9).

### Bug

The original Python code used the *nominal* number of layers `M = current_max.shape[0]`
(e.g., 17 for resolution level 6) instead of the post-`resize`/`reset` value of 1:

```python
# Before — WRONG: zeros layer 0 AND the actual last layer (e.g. layer 16)
current_max = current_max.at[0, :].set(0.0)
current_max = current_max.at[current_max.shape[0] - 1, :].set(0.0)   # ← incorrect
if pattern in (5, 6, 9) and current_max.shape[0] > 3:
    current_max = current_max.at[1, :].set(0.0)
    current_max = current_max.at[current_max.shape[0] - 2, :].set(0.0)  # ← incorrect
```

For resolution level 6 (17 frequency layers), layer 16 contained **146,693** nonzero pixels.
Incorrectly zeroing those pixels changed the fill fraction `fff`, the order statistics `val`
and `med`, and consequently the fitted Gamma shape `ALP` — shifting it from the correct value
**1.94817** to **2.07174**, a ~6.3% error.

### Fix

Remove the incorrect last-layer zeroing; zero only layer 0 (and layer 1 for patterns 5, 6, 9):

```python
# After — matches C++ (M=1 after wdmPacket's resize+reset → only layer 0 is zeroed)
current_max = current_max.at[0, :].set(0.0)
if pattern in (5, 6, 9) and current_max.shape[0] > 2:
    current_max = current_max.at[1, :].set(0.0)
```

This fix was applied to **both** JIT variants of the function (the three-state and two-state
`while_loop` overloads).

### Applicability

This bug affects every call to `_time_delay_max_energy_pattern_jit` regardless of pattern,
because the last layer is always non-trivially populated.  The impact is largest at higher
WDM resolution levels with more frequency layers.

---

## 2. `Gamma2Gauss` — `waveSplit` inclusive range off-by-one (`time_frequency_map.py`)

### Background

`Gamma2Gauss` fits a Gamma distribution to the whitened pixel energies and maps them to a
Gaussian statistic.  The fit requires two order statistics computed by C++
`wavearray::waveSplit(nL, nR, m)`, which selects the `(nR-m)`-th largest element in the
**inclusive** range `[nL, nR]` using a quickselect algorithm.

### Bug

The Python implementation used `flat[nL:nR]` (Python exclusive slice, i.e. the range
`[nL, nR)`) as the input to `np.partition`, silently omitting the element at index `nR`:

```python
# Before — WRONG: exclusive slice misses element at nR
region = flat[nL:nR]
med = float(np.partition(region, rel_med)[rel_med])
...
ws_region2 = transformed[nL:nR]
qv  = float(np.partition(ws_region2, rel_rms)[rel_rms])
```

Because the element at `nR` participates in the sorted rank used by `waveSplit`, omitting it
shifted the median by the gap between adjacent sorted values (~5 × 10⁻⁶), which was then
amplified through the `data /= avr` step to ~5 × 10⁻⁵ in the post-G2G pixel values.

### Fix

Use `flat[nL:nR + 1]` for the partition inputs to match C++'s inclusive range, while keeping
the statistics loop as `flat[nL:nR]` to match C++'s `for (i = nL; i < nR; i++)`:

```python
# After — inclusive slice matches C++ waveSplit [nL, nR]
ws_region = flat[nL:nR + 1]
med = float(np.partition(ws_region, rel_med)[rel_med])
...
ws_region2 = transformed[nL:nR + 1]
qv  = float(np.partition(ws_region2, rel_rms)[rel_rms])
```

The split index clamp was also updated from `min(..., nR - 1)` to `min(..., nR)` to allow
selecting the element at `nR`.

### Verification

Before fix, Eo diffs at levels 10, 8, 6 (the most affected levels):

| Level | Before | After |
|---|---|---|
| 10 (M=1025) | 9.99 × 10⁻⁶ | 6.63 × 10⁻¹¹ |
| 8  (M=257)  | 2.73 × 10⁻⁵ | 1.87 × 10⁻⁶  |
| 6  (M=65)   | 1.66 × 10⁻⁵ | 9.43 × 10⁻⁷  |

---

## 3. `_threshold_python` — shape-branch pixel set mismatch (`coherence.py`)

### Background

`_threshold_python` implements `network::THRESHOLD(double p, double shape)` (the shape
branch, active when `config.pattern != 0`).  C++ computes the combined detector energy on the
raw 1-D time-major flat array using explicit flat indices:

```cpp
size_t M  = pw->maxLayer() + 1;
size_t nL = size_t(Edge * pw->wrate() * M);
size_t nR = pw->size() - nL - 1;           // note the -1
wavearray<double> w = *pw;
for (int i = 1; i < N; i++) w += getifo(i)->TFmap;
// statistics loop uses [nL, nR):
for (int i = nL; i < nR; i++) { ... }
```

### Bug

The Python shape branch obtained its pixel set by 2D edge-cropping then ravelling:

```python
# Before — WRONG: 2D crop gives M*(T - 2*e) pixels; C++ uses M*(T - 2*e) - 1
energies = [_get_tf_energy_array(tfm, edge=edge) for tfm in tf_maps]
combined = np.sum(energies, axis=0)
work = combined.ravel()          # size = M*(T - 2*e)
positive = work[work > 1.0e-3]
avr = float(np.mean(positive))
```

Because C++ sets `nR = size - nL - 1` (the trailing `-1`), the C++ pixel count is
`size - 2*nL - 1 = M*(T - 2*e) - 1` — **one fewer** than Python.  For levels where the
extra pixel had energy > 0.001, the positive count `nn` differed by 1, shifting `avr` and
`alp` enough to change the final threshold Eo by up to ~4 × 10⁻⁶.

### Fix

Replace 2D edge-cropping with the same flat-array logic as C++:

```python
# After — flat-array indexing matches C++ exactly
pw0 = tf_maps[0]
arr0 = np.asarray(pw0.data, dtype=np.float64)
M = int(arr0.shape[0]) if arr0.ndim == 2 else 1
# C++ stores time-major: flat[t*M + m].  Python (M, T) -> transpose then ravel.
w = arr0.T.ravel().copy()
for tfm in tf_maps[1:]:
    w += np.asarray(tfm.data, dtype=np.float64).T.ravel()

nL = int(float(edge or 0.0) * pw0.wavelet_rate * M)
nR = int(w.size) - nL - 1          # mirrors C++: nR = pw->size() - nL - 1

region  = np.clip(w[nL:nR], 0.0, n_ifo * 100.0)
positive = region[region > 1.0e-3]
```

### Verification

Before and after fix, Eo diffs for all 7 WDM resolution levels (pattern=10):

| Level | M    | Before fix | After fix |
|-------|------|------------|-----------|
| 10    | 1025 | 6.63 × 10⁻¹¹ | 6.63 × 10⁻¹¹ |
| 9     | 513  | 1.27 × 10⁻⁶  | **6.47 × 10⁻¹¹** |
| 8     | 257  | 1.87 × 10⁻⁶  | **6.68 × 10⁻¹¹** |
| 7     | 129  | 2.59 × 10⁻⁶  | **7.11 × 10⁻¹¹** |
| 6     | 65   | 9.43 × 10⁻⁷  | **8.05 × 10⁻¹¹** |
| 5     | 33   | 3.87 × 10⁻⁶  | **9.32 × 10⁻¹¹** |
| 4     | 17   | 6.43 × 10⁻⁶  | **1.09 × 10⁻¹⁰** |

All levels now agree with C++ at ≲ 1 × 10⁻¹⁰, well below the 1 × 10⁻⁶ target.

### Applicability

| `config.pattern` | Branch | Fix applies? |
|---|---|---|
| `0` | No-shape branch (unchanged) | ❌ No |
| `1` – `10` (incl. default `10`) | Shape branch | ✅ Yes |

---

## 4. `_threshold_python` — THRESHOLD(p) exact algorithm, no-shape branch (`coherence.py`)

### Background

When `config.pattern == 0`, CWB calls `network::THRESHOLD(double p)` (no shape argument),
which uses two order statistics and an iterative Gamma shape search.

### Bug

The original Python no-shape branch used a simple `np.quantile` approximation for the median
and a linear formula for `m`:

```python
# Before — approximate
med = float(np.quantile(positive, 0.8))
m = max(float(n_ifo), med / 2.0)
result = _igamma_inv_upper(n_ifo * m, float(bpp)) / 2.0 + n_ifo * np.log(m)
return result
```

### Fix

Replaced with the exact CWB algorithm from `network.cc`:

1. **Fill fraction `fff`** — `wavecount(1e-4) / size`: counts pixels with energy > 1 × 10⁻⁴
   divided by the **total** array size.
2. **`val`** — the `(k_val+1)`-th largest element (`k_val = int(bpp × fff × N)`) via
   `np.sort` + reverse indexing.
3. **`med`** — the `(k_med+1)`-th largest element (`k_med = int(0.2 × fff × N)`).
4. **Iterative `m` search** — increment by 0.01 starting from 1.0 until
   `P(Gamma(n_ifo × m) ≥ med) ≥ 0.2`, step back once if `m > 1.01`.
5. **Final threshold** — `0.3 × (iGamma(n_ifo×m, bpp) + val) + n_ifo × log(m)`.

```python
# After — exact CWB algorithm
fff = float(np.sum(work > 1.0e-4) / work.size)
sorted_work = np.sort(work)
k_val = int(float(bpp) * fff * n_total)
k_med = int(0.2 * fff * n_total)
val = float(sorted_work[max(0, n_total - k_val - 1)])
med = float(sorted_work[max(0, n_total - k_med - 1)])

m = 1.0; p00 = 0.0
while p00 < 0.2:
    p00 = float(gammaincc(n_ifo * m, med))
    m += 0.01
if m > 1.01:
    m -= 0.01

result = 0.3 * (_igamma_inv_upper(n_ifo * m, float(bpp)) + val) + n_ifo * np.log(m)
```

### Applicability

| `config.pattern` | Branch | Fix applies? |
|---|---|---|
| `0` | No-shape branch | ✅ Yes |
| `1` – `10` | Shape branch | ❌ No |

---
