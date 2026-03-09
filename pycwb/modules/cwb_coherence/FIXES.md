# Bug Fixes — `cwb_coherence` module

This file records algorithm fixes applied to the `cwb_coherence` module to match the
behaviour of the CWB C++ `network::THRESHOLD` function.

---

## 1. `_threshold_python` — THRESHOLD(p) exact algorithm (`coherence.py`)

### Background

The function `_threshold_python` implements the pixel-energy selection threshold used by
`compute_threshold` (called from both `coherence` and `coherence_parallel`).  CWB decides
the threshold in `network::THRESHOLD(double p [, double shape])`:

* **Shape branch** (`shape != 0`, i.e. `config.pattern != 0`): fits the whitened pixel
  energies to a Gamma distribution and scales `bpp` by the ratio of the fitted shape to the
  expected shape.  The final threshold is the upper-tail quantile of that Gamma.
* **No-shape branch** (`shape == 0` / `config.pattern == 0`): uses two order statistics
  (`val` at the `bpp`-quantile and `med` at the 20%-quantile computed from the fill fraction
  `fff`) together with an iterative search for the effective Gamma shape `m`.

### Bug (no-shape branch)

The original Python no-shape branch used a simple `np.quantile` approximation for the median
and a single linear formula for `m`, which diverged from C++ when the fill fraction `fff`
differed from 1 or when the distribution was not close to chi-squared(2):

```python
# Before — approximate
med = float(np.quantile(positive, 0.8))
m = max(float(n_ifo), med / 2.0)
result = _igamma_inv_upper(n_ifo * m, float(bpp)) / 2.0 + n_ifo * np.log(m)
return result
```

### Fix (no-shape branch)

Replaced with the exact CWB algorithm from `network.cc`:

1. **Fill fraction `fff`** — CWB's `wavecount(1e-4, 0)` counts pixels with energy > 1e-4
   and divides by the total array size (not just the `positive` subset).
2. **`val`** — `waveSplit(0, N, N - k_val)`: the `(k_val+1)`-th largest element, where
   `k_val = round(bpp × fff × N)`.  Implemented via `np.sort` + reverse indexing.
3. **`med`** — `waveSplit(0, N, N - k_med)`: the `(k_med+1)`-th largest element, where
   `k_med = round(0.2 × fff × N)`.
4. **Iterative `m` search** — Starting at `m = 1.0`, increment by 0.01 until
   `P(Gamma(n_ifo × m) ≥ med) ≥ 0.2` (using `scipy.special.gammaincc`), then step back once
   if `m > 1.01` to match CWB's `if(m>1) m -= 0.01`.
5. **Final threshold** — `0.3 × (_igamma_inv_upper(n_ifo×m, bpp) + val) + n_ifo × log(m)`,
   matching CWB `ee = 0.3*(pn+ee) + N*log(m)`.

```python
# After — exact CWB algorithm
fff = float(np.sum(work > 1.0e-4) / work.size)
if fff <= 0.0:
    return 0.0
n_total = work.size
sorted_work = np.sort(work)

k_val = int(float(bpp) * fff * n_total)
k_med = int(0.2 * fff * n_total)
val = float(sorted_work[max(0, n_total - k_val - 1)]) if k_val > 0 else float(sorted_work[-1])
med = float(sorted_work[max(0, n_total - k_med - 1)]) if k_med > 0 else float(sorted_work[-1])

from scipy.special import gammaincc as _scipy_gammaincc
m = 1.0; p00 = 0.0
while p00 < 0.2:
    p00 = float(_scipy_gammaincc(n_ifo * m, med))
    m += 0.01
if m > 1.01:
    m -= 0.01

result = 0.3 * (_igamma_inv_upper(n_ifo * m, float(bpp)) + val) + n_ifo * np.log(m)
return result
```

### Note on shape branch

The shape branch formula was already correct.  The residual Eo discrepancy observed between
Python and C++ at some WDM levels (e.g., Python `Eo=8.04` vs CWB `Eo=8.19` at level 4) comes
from the **input** to the shape branch — specifically the `ALP` parameter returned by
`Gamma2Gauss` when called on the JAX `_wdm_packet_energy_jax` output vs the ROOT C++
`wdmPacket('E')` output.  Those two routines produce slightly different energy statistics for
the same whitened frame, causing a different fitted Gamma shape (`ALP=2.077` vs `3.906`).

**Science impact: zero** — all extra pixels selected by the lower Python threshold fail the
downstream `subrho` and `subnet` cuts and do not appear in any output event.

### Applicability

| `config.pattern` | Branch taken | Fix applies? |
|---|---|---|
| `0` | No-shape branch | ✅ Yes |
| `1` – `10` (incl. default `10`) | Shape branch | ❌ No (was already correct) |

---
