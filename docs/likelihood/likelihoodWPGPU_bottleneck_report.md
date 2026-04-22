# likelihoodWPGPU — CPU Bottleneck Report

This report identifies every section of the `likelihoodWPGPU` module that
remains on the CPU (NumPy / Numba / plain Python) and estimates its
computational cost relative to the GPU-accelerated sky scan.

---

## 1. Pipeline Overview

The per-cluster likelihood pipeline has three phases:

| Phase | Location | Engine | Complexity |
|-------|----------|--------|------------|
| **A. Sky scan** | `sky_scan.py` | JAX `vmap` | O(n_sky × n_ifo × n_pix) |
| **B. Best-sky post-processing** | `likelihood.py` → `calculate_sky_statistics_gpu()` | Mixed (JAX + NumPy/Numba) | O(n_pix × n_neighbors × n_ifo) |
| **C. Detection statistic + waveform** | `likelihood.py` → `fill_detection_statistic()` | Pure Python + NumPy | O(n_core² × n_ifo) + WDM synthesis |

Phases B and C run **once** (at the best sky direction only), while Phase A
runs over all 49k–196k sky directions.  For most clusters the sky scan
dominates, so the GPU pay-off is in Phase A.  However:

* Very large clusters (n_pix > 500) will spend significant wall-clock time in
  Phase B's xtalk loops.
* `fill_detection_statistic` (Phase C) has quadratic worst-case scaling in
  the number of core pixels.

---

## 2. GPU-Accelerated Components (Phase A)

All functions below run entirely inside JAX `jit` / `vmap`:

| Function | File | Complexity per sky direction |
|----------|------|-----------------------------|
| `_gather_delayed_data` | `sky_scan.py` | O(n_ifo × n_pix) — advanced indexing |
| `compute_pixel_energy` | `sky_stat.py` | O(n_ifo × n_pix) |
| `compute_dpf` | `dpf.py` | O(n_ifo × n_pix) |
| `project_gw_packet` | `sky_stat.py` | O(n_ifo × n_pix) |
| `orthogonalise_polarisations` | `sky_stat.py` | O(n_ifo × n_pix) |
| `compute_coherent_statistics` | `sky_stat.py` | O(n_ifo × n_pix) |
| `calculate_dpf_regulator` | `dpf.py` | O(n_sky × n_ifo × n_pix) — called once |

**Total GPU work per cluster:** O(n_sky × n_ifo × n_pix)

**Memory note:** `_gather_delayed_data` materialises `all_v00`, `all_v90` as
`(n_sky, n_ifo, n_pix)` JAX arrays.  For n_sky = 196608, n_ifo = 3,
n_pix = 500 this is ~1.1 GB of FP32 — may cause OOM on small GPUs.

---

## 3. CPU-Bound Components (Phases B + C)

### 3.1 `l_max` Selection Loop — `sky_scan.py` L237–241

```
Complexity: O(n_sky)
Engine:     Plain Python for-loop over NumPy array
```

A Python `for` loop that scans all sky directions to find the *last* index
with the maximum AA statistic (for C++ tie-breaking parity).  With
n_sky ≈ 196k this takes ~1–5 ms.

**Cost: LOW.** Single scalar comparison per iteration; negligible compared to
  the vmap kernel time.

**GPU-izable?** Yes — a reverse `jnp.argmax` or custom scan would replace it,
but the pay-off is negligible.

---

### 3.2 Time-Delay Application — `calculate_sky_statistics_gpu()` L98–106

```
Complexity: O(n_ifo × n_pix)
Engine:     Python for-loop + NumPy fancy indexing
```

Applies the best-sky time delay to extract `v00`, `v90` from the delay
tensors.  Two nested Python loops (outer: n_ifo, inner: n_pix via slicing).

**Cost: LOW.** n_ifo ≤ 5; the per-IFO slice is a single NumPy copy.

---

### 3.3 `_numpy_packet_rotation` ← `avx_packet_ps` — `likelihood.py` L256–258

```
Complexity: O(n_ifo × n_pix)   ×2 calls (data + signal)
Engine:     Plain NumPy (Python for-loops over n_ifo)
```

Decomposes each IFO's data into principal-component amplitudes (a, A) and
applies rotation + normalisation.  **Not Numba-compiled** — uses Python
for-loops over IFOs with vectorised NumPy over pixels.

**Cost: MEDIUM.** For typical clusters (n_pix < 200) the cost is low, but for
  large clusters (n_pix > 1000) the two calls together become noticeable
  (~5–20 ms).

**Note:** A JAX `jit`-compiled version (`compute_packet_rotation`) already
exists in `utils.py` but is **not called** — `likelihood.py` uses the CPU
`avx_packet_ps` wrapper instead.  Switching to the JAX version would
eliminate this cost.

---

### 3.4 `packet_norm_numpy` — `xtalk_ops.py` (from likelihoodWP)

```
Complexity: O(n_pix × n_neighbors × n_ifo)   ×2 calls
Engine:     Numba @njit
```

For each pixel, iterates over its xtalk neighbors (~10–50 per pixel in a
typical MRA catalog) and accumulates cross-talk-weighted dot-products.  Called
twice: once before noise correction, once after amplitude setting.

**Cost: MEDIUM–HIGH.** Numba JIT-compiled, so each call is fast for small
  clusters.  For large clusters (n_pix > 500, n_neighbors ~ 30) the xtalk
  neighbor iteration dominates — ~10–50 ms per call.

**GPU-izable?** Difficult — the neighbor list is ragged (each pixel has a
  different number of xtalk neighbors), requiring either padding + masking or
  a sparse matrix representation.  Not impossible with JAX `vmap` over a
  padded neighbor array, but requires refactoring the xtalk data structure.

---

### 3.5 `gw_norm_numpy` — `xtalk_ops.py` (from likelihoodWP)

```
Complexity: O(n_ifo × n_pix)
Engine:     Numba @njit
```

Simple post-processing of signal norms: per-IFO SNR ratio computation, then
conditional norm assignment per pixel.

**Cost: LOW.** Numba-compiled, O(n_ifo × n_pix), fast even for large clusters.

---

### 3.6 `compute_noise_correction` — `utils.py`

```
Complexity: O(n_ifo × n_pix)
Engine:     NumPy (vectorised, float64)
```

Single pass over pixels computing Gaussian noise correction terms (Gn, Ec, Dc,
Rc, Eh, Es).  Fully vectorised NumPy with no Python loops.

**Cost: LOW.** Vectorised float64 ops; < 1 ms for any realistic cluster size.

---

### 3.7 `set_packet_amplitudes` — `utils.py`

```
Complexity: O(n_ifo × n_pix)   ×2 calls
Engine:     NumPy (vectorised)
```

Broadcasts amplitude + rotation factors to all pixels.

**Cost: LOW.** Pure vectorised NumPy; negligible.

---

### 3.8 `compute_null_packet` — `utils.py`

```
Complexity: O(n_ifo × n_pix)
Engine:     NumPy subtraction
```

**Cost: NEGLIGIBLE.** One array subtraction.

---

### 3.9 `xtalk_energy_sum` — `utils.py`

```
Complexity: O(n_pix × n_neighbors × n_ifo)   ×2 calls  (Em + Np)
Engine:     Plain Python for-loop + NumPy dot-products
```

For each pixel, looks up xtalk neighbors, computes cross-talk-weighted energy
sums.  **Not Numba-compiled** — pure Python for-loop with NumPy inner products.

**Cost: HIGH.** This is the **most expensive CPU function** in the post-
  processing phase.  The Python for-loop over n_pix dominates; for n_pix = 500,
  n_neighbors = 30, each call takes ~50–200 ms.  Two calls (Em, Np) sum to
  ~100–400 ms.

**GPU-izable?** Same ragged-neighbor challenge as `packet_norm_numpy`.
  Alternatively, wrapping with Numba `@njit` would give a 10–50× speed-up
  immediately without any GPU work.

---

### 3.10 `project_polarisation` — `utils.py`

```
Complexity: O(n_ifo × n_pix)   ×2 calls
Engine:     NumPy (float64) + Python for-loop (last pixel only)
```

Projects data onto the network polarisation plane.  The bulk dot-product is
vectorised NumPy; a small Python for-loop handles the last-pixel DSP
projection (n_ifo iterations only).

**Cost: LOW.** Vectorised bulk; tiny Python loop at end.

---

### 3.11 `load_data_from_pixels` — from likelihoodWP

```
Complexity: O(n_pix × n_ifo × tsize)
Engine:     Bulk NumPy (vectorised)
```

Extracts time-delay amplitudes and noise RMS from pixel objects into contiguous
arrays.

**Cost: LOW–MEDIUM.** Bound by Python object attribute access (Pixel objects),
  not by arithmetic.  Typically < 10 ms.

---

### 3.12 `fill_detection_statistic` — from likelihoodWP

```
Complexity: O(n_core × n_eligible × n_ifo)  +  WDM synthesis per IFO
Engine:     Pure Python + NumPy dot-products (no Numba)
```

The most complex post-processing function.  For each pixel, iterates over an
eligible-pixel set (filtered by xtalk lookup) and accumulates weighted inner
products.  Then calls `get_MRA_wave()` for time-domain waveform reconstruction
(involves WDM synthesis + FFT).

**Cost: HIGH** for large clusters.  The nested pixel loops have worst-case
  O(n_pix²) scaling.  For small clusters (n_pix < 100) this is fast (< 50 ms)
  but for large clusters (n_pix > 500) it can reach seconds.

**GPU-izable?** The xtalk-dependent loops have the same ragged-neighbor
  challenge.  The WDM synthesis calls an external pure-Python routine that
  would need a separate JAX port.

---

### 3.13 `get_chirp_mass` / `get_error_region` — from likelihoodWP

```
Complexity: O(n_pix) each
Engine:     Pure Python
```

**Cost: NEGLIGIBLE.** Single-pass metadata computation.

---

### 3.14 `_populate_pixel_noise_rms` — from likelihoodWP

```
Complexity: O(n_pix × n_ifo)
Engine:     Plain Python + NumPy indexing
```

Maps noise RMS from whitening grid to each pixel.

**Cost: LOW.** Linear scan; Python loop with array lookups.

---

## 4. Cost Summary

| # | Function | Calls | Complexity | Engine | Cost Rating |
|---|----------|-------|------------|--------|-------------|
| A | **Sky scan (vmap)** | 1 | O(n_sky × n_ifo × n_pix) | JAX | **GPU** ✓ |
| B1 | `l_max` selection | 1 | O(n_sky) | Python | ■□□□□ Negligible |
| B2 | Time-delay application | 1 | O(n_ifo × n_pix) | NumPy | ■□□□□ Negligible |
| B3 | `_numpy_packet_rotation` | 2 | O(n_ifo × n_pix) | NumPy | ■■□□□ Low–Med |
| B4 | `packet_norm_numpy` | 2 | O(n_pix × n_nbr × n_ifo) | Numba | ■■■□□ Medium |
| B5 | `gw_norm_numpy` | 1 | O(n_ifo × n_pix) | Numba | ■□□□□ Negligible |
| B6 | `compute_noise_correction` | 1 | O(n_ifo × n_pix) | NumPy | ■□□□□ Negligible |
| B7 | `set_packet_amplitudes` | 2 | O(n_ifo × n_pix) | NumPy | ■□□□□ Negligible |
| B8 | `compute_null_packet` | 1 | O(n_ifo × n_pix) | NumPy | ■□□□□ Negligible |
| **B9** | **`xtalk_energy_sum`** | **2** | **O(n_pix × n_nbr × n_ifo)** | **Python** | **■■■■□ HIGH** |
| B10 | `project_polarisation` | 2 | O(n_ifo × n_pix) | NumPy | ■□□□□ Negligible |
| C1 | `load_data_from_pixels` | 1 | O(n_pix × n_ifo × tsize) | NumPy | ■■□□□ Low–Med |
| **C2** | **`fill_detection_statistic`** | **1** | **O(n_core² × n_ifo) + WDM** | **Python** | **■■■■■ HIGHEST** |
| C3 | `get_chirp_mass` | 1 | O(n_pix) | Python | ■□□□□ Negligible |
| C4 | `_populate_pixel_noise_rms` | 1 | O(n_pix × n_ifo) | Python | ■□□□□ Negligible |

---

## 5. Recommendations (Priority Order)

### Immediate (no GPU needed)

1. **Numba-ify `xtalk_energy_sum`** — add `@njit` to the existing Python
   for-loop.  Expected 10–50× speed-up for zero API change.  This is the
   single highest-impact change.

2. **Switch to the JAX `compute_packet_rotation`** — a JAX jit version already
   exists in `utils.py` but is unused.  Replace the `_numpy_packet_rotation`
   call in `likelihood.py` with the JAX version to eliminate the CPU round-trip.

### Medium-term

3. **Port `packet_norm_numpy` to JAX** — requires padding the ragged xtalk
   neighbor lists into a fixed-width 2D array (n_pix × max_neighbors) with
   a mask.  This enables `vmap` over pixels and eliminates the Numba
   dependency.

4. **Port `fill_detection_statistic` inner loops to Numba** — the O(n_core²)
   Python loops are the asymptotic bottleneck for large clusters.  A Numba
   `@njit` + `prange` version would give 10–100× speed-up.

### Long-term

5. **GPU-native xtalk operations** — represent the xtalk catalog as a sparse
   CSR/COO matrix and use JAX sparse matmul for all xtalk-convolved sums
   (`packet_norm`, `xtalk_energy_sum`, `fill_detection_statistic`).  This
   would move the entire pipeline to the GPU but requires rethinking the
   xtalk data structure.

6. **GPU WDM waveform synthesis** — `get_MRA_wave()` inside
   `fill_detection_statistic` performs per-IFO WDM→time-domain synthesis.
   Porting this to JAX would require a JAX-native WDM inverse transform
   (already partially available in the `wdm-wavelet` package).

---

## 6. Unused JAX Code

`compute_packet_rotation` in `utils.py` is a fully-functional JAX `jit`-
compiled equivalent of `avx_packet_ps`, but `likelihood.py` calls the CPU
version via `_numpy_packet_rotation` instead.  This should be investigated —
if the outputs match numerically, the CPU wrapper can be replaced.
