# Plans to Review

Optimisation ideas that were identified during benchmark analysis but **not implemented**
(or reverted) because they are either physically unsafe or have insufficient benefit to
justify the risk.  Each section documents the idea, the expected gain, and the reason
for deferral.

---

## Priority 1 — Phase-shift max energy in `time_delay_max_energy` (REVERTED)

### Idea

`_time_delay_max_energy_pattern_jit` computes, for each candidate delay step `k`, the
sequence:

1. Convert TF coefficients → time domain (`w2t`)
2. Shift time series by `k` samples
3. Convert back to TF domain (`t2w`)
4. Evaluate the WDM energy map

The `w2t + t2w` round-trip is expensive.  The idea was to avoid it by applying a
per-band phase rotation directly to the complex WDM TF coefficients:

```python
# For WDM layer m, delay τ = k / sample_rate:
phase = exp(-i · 2π · (m · df) · τ)
shifted_data = base_data * phase[:, None]   # broadcast over time
```

### Expected gain

~8.5 s reduction in coherence (timing: ~17.3 s → ~8.9 s observed in benchmark, so the
JAX phase-shift is faster in walltime by ~2× per level).

### Why it was reverted (BREAKS PHYSICS)

The phase-rotation approximation is only exact for the **continuous Fourier transform**.
WDM (Wilson-Daubechies-Meyer) uses a filter bank with finite-length, overlapping basis
functions.  Shifting the time domain by `k` samples **mixes** coefficients across
neighbouring time bins in the WDM domain; it is NOT equivalent to a scalar phase
rotation of each independent `(m, t)` coefficient.

**Empirical evidence from benchmark run (`run_optmize2.log` vs `run_optimize1.log`):**

| Level | `alp` (reference) | `alp` (phase-shift) | Pixel count (ref) | Pixel count (phase-shift) |
|------:|------------------:|--------------------:|------------------:|--------------------------:|
| 10    | 1.52727           | 1.49712             | 51                | 50                        |
|  9    | 1.5472            | 1.4849              | 71                | 69                        |
|  8    | 1.62344           | 1.48312             | 77                | 63                        |
|  4    | (reference)       | (phase-shift)       | 89                | **34** (−62 %)            |

Overall: 477 → 373 coherence pixels (−22 %), `like_sky`: 1558 → 1528 (−2 %),
`energy_sky`: 7489 → 7236 (−3.4 %).  Although the test event was still detected in
both cases, the intermediate statistics are no longer equivalent to the C++ reference.

The function `_time_delay_max_energy_phase_jit` is still present in
`pycwb/types/time_frequency_map.py` for reference but is **not called**.  Its docstring
carries an explicit `.. warning::` block.

### What would make this approach valid

One would need to derive the **exact** WDM-domain transfer function for a time shift of
`k` samples.  For a Meyer wavelet of order `β`, the filter kernel `h_m[n]` has compact
support; the WDM-domain equivalent of a time shift is a **convolution** in the time
index of the TF map, not a scalar multiplication.  Deriving and implementing this
correctly requires careful comparison against the C++ `WSeries::maxEnergy` on a suite of
known waveforms before any production use.

---

## Priority 4 — Sky pre-filtering in `find_optimal_sky_localization`

### Idea

`find_optimal_sky_localization` iterates over all `n_sky = 196 608` HEALPix sky
points in a `prange` (Numba parallel) loop.  A significant fraction of sky points
carry negligible antenna power and will always produce low correlation statistics.
The idea is to **skip sky points** whose summed antenna power falls below a threshold
computed from `FP[l]` and `FX[l]` before entering the expensive per-pixel DPF and
GW-statistics computation:

```python
@njit(parallel=True, cache=True)
def find_optimal_sky_localization(...):
    for l in prange(n_sky):
        # Pre-filter: skip low-power sky directions
        antenna_power = 0.0
        for i in range(n_ifo):
            antenna_power += FP[l, i] ** 2 + FX[l, i] ** 2
        if antenna_power < antenna_power_threshold:
            continue   # <-- skip entire DPF + avx_GW_ps block
        ...
```

### Expected gain

`find_optimal_sky_localization` accounts for ~6.8 s (single-core) in the benchmark.
Depending on the detector network and time of year (antenna null zones), 20–50 % of
sky points may fall below a reasonable power threshold, potentially saving 1–3 s.

### Why it is deferred (DANGEROUS)

1. **Changes the detection set.**  By skipping sky locations in a serial order before
   computing the DPF, we may miss the _true_ maximum-likelihood direction when the
   optimal sky point passes through a region of low _antenna power_ but high
   _coherent energy_ due to favourable polarisation angles.  This could reduce the
   detection efficiency for certain source configurations (e.g., edge-on binaries near
   detector null zones, unusual polarisation states).

2. **Threshold choice is non-trivial.**  Any fixed threshold introduces a bias that
   depends on the detector network geometry, the noise floor, and the signal morphology.
   There is no parameter-free way to set `antenna_power_threshold` without validating
   against a large injection campaign.

3. **Breaks the `prange` reduction.**  Once a `continue` inside `prange` is used as a
   selection mechanism, the resulting `l_max` is necessarily drawn from a subset of sky
   points; any follow-up sky statistic (error region, sky maps) derived from the full
   `n_sky` array will be inconsistent with the reduced search set.

4. **Inconsistency with `calculate_dpf`.**  The regulator `REG[1]` is computed from
   _all_ sky points; filtering out some points in `find_optimal_sky_localization` without
   also adjusting `REG[1]` would introduce a systematic bias in the network regulator.

**Before revisiting this idea**, a full injection study across all detector configurations
and source inclination angles must be performed to quantify the efficiency loss.  The
implementation must also provide a configuration knob to disable the filter and be gated
behind a clearly documented `use_sky_prefilter: bool = False` option in the run config.

---

## Priority 3 — Fuse `calculate_dpf` into `find_optimal_sky_localization`

### Idea

Both `calculate_dpf` (~1.5 s) and `find_optimal_sky_localization` (~6.8 s) call
`dpf_np_loops_vec(FP[l], FX[l], rms)` for **all 196 608 sky points** independently.
The idea is to fuse them into a single two-pass Numba function so the DPF geometry is
only computed once:

```
Pass 1  (parallel, lightweight):  aa[l] = dpf_np_loops_vec(...)[0]  for all l
         → compute REG[1] from aa
Pass 2  (parallel, full):         run full per-sky statistics with REG[1] now known
```

### Expected gain

Eliminates one full parallel DPF sweep (~1.5 s on 1 core), reducing total likelihood
wall-time from ~8.3 s (1.5 s + 6.8 s) to ~7.3 s.

### Why the gain is near-zero in practice

`dpf_np_loops_vec` computes **all** output arrays `(f, F, fp, fx, si, co, ni)` in
every call regardless of whether the caller uses all return values.  Numba's `@njit`
does not dead-code-eliminate unused return values from JIT-compiled functions.  Hence:

* **Pass 1** (which only uses `[0]`, the NI scalar) still executes the full DPF
  computation — the same work as the current `calculate_dpf` call.
* **Pass 2** also executes the full DPF computation to obtain `f, F, fp, fx, si, co, ni`.

There is **no reduction in the total number of FIR/DPF operations**; fusion only saves
the Python-level function-call overhead between the two passes (~0.01–0.05 s).

### What would make this worthwhile

A dedicated `dpf_ni_only` Numba function that **returns only the NI scalar** (skipping
allocation and population of the `f`, `F`, `fp`, `fx`, `si`, `co`, `ni` arrays) could
make Pass 1 meaningfully faster.  Profiling suggests ~30–40 % of `dpf_np_loops_vec`
runtime is memory-bound array writes, so a scalar-only variant could reduce Pass 1 from
~1.5 s to ~0.9 s, yielding a net saving of ~0.6 s.

Implementation sketch:

```python
@njit(cache=True)
def dpf_ni_only(Fp0, Fx0, rms):
    """Return only the NI scalar (index 0 of dpf_np_loops_vec) without allocating
    per-pixel output arrays."""
    NPIX, NIFO = rms.shape
    NI = np.float32(0.0)
    NN = np.uint32(0)
    for i in range(NPIX):
        fp_i = np.float32(0.0)
        fx_i = np.float32(0.0)
        ni_i = np.float32(0.0)
        ...  # accumulate pixel sums directly without storing arrays
    return sqrt(NI / (NN + np.float32(0.01)))
```

This is a safe, backward-compatible refactoring with no physical implications and should
be revisited once the higher-impact optimisations (Phase-shift max_energy, WDM TF cache)
have been benchmarked and validated.
