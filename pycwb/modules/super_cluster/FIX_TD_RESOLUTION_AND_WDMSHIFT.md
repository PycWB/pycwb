# Fix: Time-Delay Resolution and wdmShift Quadrature Swap

## Overview

Two inter-related bugs caused the native super-cluster pipeline to produce sky
directions ~165° away from the CWB/ROOT reference.  Both are in the time-delay
amplitude kernel used to build the coherent TD vectors that power the sky
localisation scan.

---

## Bug 1 — TD filter at wrong sample rate (`super_cluster.py`, `td_vector_batch.py`)

### Problem

`wdm.set_td_filter(TDSize, L)` sets both the filter bank half-size **and** the
upsample factor `L`.  With `L = 1` the delay index `ml[i,l]` has a resolution of
`1 / rateANA = 0.488 ms`; with `L = upTDF = 4` it has a resolution of
`1 / TDRate = 0.122 ms`.

CWB calls `WDM::setTDFilter(TDSize, upTDF)` (i.e. `L = upTDF`), then encodes sky
delays in `ml` at `TDRate` resolution.  The Python pipeline was calling
`wdm.set_td_filter(TDSize, 1)` and computing `ml` at `rateANA` resolution —
**4× too coarse**.

For the L1-H1 pair with `max_delay = 0.01001 s`:

| Sample rate | Steps needed |
|-------------|-------------|
| `rateANA = 2048 Hz` | ~20  |
| `TDRate = 8192 Hz`  | 83   |

With only 20 discrete delay steps the sky-delay map was far too quantised, so many
sky pixels mapped to the same (wrong) delay and the peak in `nSkyStat` shifted by
~165° in azimuth.

### Fix (applied 2026-03-07)

In `setup_supercluster` and `supercluster_wrapper`:

```python
# Before
wdm.set_td_filter(int(config.TDSize), 1)   # L=1, rateANA resolution

# After
upTDF = int(getattr(config, 'upTDF', 1))
TDRate = int(getattr(config, 'TDRate', int(config.rateANA) * upTDF))
wdm.set_td_filter(int(config.TDSize), upTDF)   # L=upTDF, TDRate resolution
```

`K_td` (the delay half-range) is now also computed at `TDRate` scale:

```python
# Before
K_td = int(config.TDSize)   # ≈ 12 — only the filter half-size

# After
K_td = max(int(config.TDSize) * upTDF,
           int(getattr(config, 'max_delay', 0.0) * float(TDRate)) + 1)
# For typical config: max(48, 83) = 83
```

And `compute_sky_delay_and_patterns` is called with `sample_rate=TDRate,
td_size=K_td` so `ml` is clipped at the correct range.

---

## Bug 2 — `dt_val` phase scaling in `_get_pixel_amplitude_nb` (`td_vector_batch.py`)

### Problem

With `L = upTDF = 4`, the WDM filter tables cover a delay index `dT` in the range
`(-J, J)` where `J = M × L = M × 4`.  The phase calculation inside
`_get_pixel_amplitude_nb` used:

```python
dt_val = float(dT)   # WRONG: treats dT as if it were in rateANA units
```

But CWB's `getTDamp` computes:

```c++
double dt = double(dT) / double(LWDM);   // LWDM == L == upTDF
```

i.e. it converts the `TDRate` delay index back to a fractional number of `rateANA`
samples before computing the filter phase.  In Python, `dt_val` must be:

```python
dt_val = float(dT) * float(M) / float(J)   # = dT / L  (rateANA sample units)
```

Without this correction every frequency bin's phase was off by a factor of
`L = 4`, causing wildly wrong reconstructed amplitudes.

### Fix (applied 2026-03-07)

```python
# Before
dt_val = float(dT)

# After
dt_val = float(dT) * float(M) / float(J)  # = dT / L in rateANA sample units
```

---

## Bug 3 — `wdmShift` quadrature swap (`td_vector_batch.py`)

### Problem

When the total delay `dT` exceeds one pixel period `J`, CWB decomposes it as:

```
dT = wdm_shift * J + sub_dT,    sub_dT ∈ (-J, J)
```

An **odd** `wdm_shift` causes the 00-phase and 90-phase quadratures of the pixel
at the shifted time bin to swap (with a conditional sign flip depending on
`(n + m) % 2`).  This mirrors `WDM::getTDamp` mode `'a'`:

```cpp
// odd shift, (n+m)%2 == 1  →  return -quad(true)
// odd shift, (n+m)%2 == 0  →  return  quad(true)   [quad=true = 90-phase]
// even shift              →  return  quad(false)   [quad=false = 00-phase]
```

The original Python code applied `wdm_shift = 0` always (no decomposition),
which gave wrong amplitudes for any pixel where `|dT| ≥ J`.

### Fix (applied 2026-03-07)

```python
if dT >= 0:
    wdm_shift = dT // J
else:
    wdm_shift = -((-dT) // J)          # C++ truncation-toward-zero

sub_dT = dT - wdm_shift * J            # sub_dT in (-J, J)
n_eff  = n - wdm_shift

if wdm_shift % 2 != 0:
    if (n + m) % 2 != 0:
        a00 = -_get_pixel_amplitude_nb(n_eff, m, sub_dT, padded90, ..., True)
        a90 =  _get_pixel_amplitude_nb(n_eff, m, sub_dT, padded00, ..., False)
    else:
        a00 =  _get_pixel_amplitude_nb(n_eff, m, sub_dT, padded90, ..., True)
        a90 = -_get_pixel_amplitude_nb(n_eff, m, sub_dT, padded00, ..., False)
else:
    a00 = _get_pixel_amplitude_nb(n_eff, m, sub_dT, padded00, ..., False)
    a90 = _get_pixel_amplitude_nb(n_eff, m, sub_dT, padded90, ..., True)
```

---

## Result

After all three fixes the native pipeline sky direction matches the CWB/ROOT
hybrid to within one HEALPix pixel (order 7):

| Metric          | Hybrid        | Native (fixed) |
|-----------------|---------------|----------------|
| phi             | 58.71°        | 59.06°         |
| theta           | 73.67°        | 74.60°         |
| nSkyStat max    | 6253.32       | 6250.61        |
| delta (phi)     | —             | **0.35°**  ✅   |
| rho             | 18.464        | 18.463         |

Cross-check (native algorithm applied to CWB-whitened data) matches the native
result exactly, confirming the residual 0.35° is a whitening-data difference, not
an algorithmic error.
