# Shared CPU sky scan

The CPU likelihood uses one compiled group-based scan in `sky_scan_scratch.py`.
Delay grouping is enabled by default. Set `PYCWB_SKY_DELAY_REUSE=0` before
importing PycWB to use one sky direction per group. Scratch reuse is internal;
`PYCWB_SKY_SCRATCH_REUSE` is obsolete and ignored, including when set to `0`.

Both modes evaluate the same direction-statistic arithmetic. Each parallel
group owns its delayed data and scratch buffers; no mutable scratch crosses
group boundaries. Final best-direction selection walks the original
`sky_valid_indices`, retaining last-wins ties and repeated-index semantics.
The geometry cache separates grouped/singleton and main/coarse grids. Replacing
the delay array invalidates the corresponding entry; in-place mutation remains
unsupported.

`sky_scan.scan_sky(geometry, cluster, settings, reuse_delays=True, setup=None,
big_cluster=False)` provides a Python interface:

- `geometry`: `(FP, FX, ml)`.
- `cluster`: `(rms, td00, td90)`.
- `settings`: `(REG, netCC, delta_regulator, network_energy_threshold, sky_valid_indices)`.

The compiled kernel continues to receive explicit arrays. The original
`scan_sky_for_best_fit` signature remains as a compiled singleton wrapper;
`scan_sky_grouped_delays` aliases the shared kernel. Existing no-GIL and CUDA
binding names remain available. Production likelihood dispatches through
`_scan_sky_scratch` for both organization modes, so the existing backend override
continues to intercept the scan. CUDA retains its own implementation.

The allocating DPF, projection, orthogonalization and coherent-statistic helpers
now allocate buffers and call their corresponding `*_into` kernels. Their
returned arrays own their storage. `sky_scratch.py` retains compatibility exports.

## Validation, 25 September 2026

397 likelihood and CUDA-binding tests passed with two Numba threads. These
include both scan modes, masks, ties, duplicate indices, input immutability,
poisoned/reused scratch buffers, allocating wrappers, cache separation and all
three compiled no-GIL compatibility hooks. The independent allocating reference
is frozen under `tests/sky_scan_reference`; the real HF input and its expected
full output tuple are saved in `tests/data/sky_scan_preconsolidation.npz`.
CUDA binding checks do not constitute a new GPU numerical benchmark.

The consolidation was also validated on `consistency-bugfix-optimization`:
344 likelihood tests passed with two Numba threads; two no-GIL hook checks
were skipped because that branch predates the no-GIL processor. The native
golden-fixture checks ran in both modes. CUDA and no-GIL integration results
above were obtained on the newer `fix-injection-postproduction` checkout.

An isolated warm scan benchmark used one saved HF fixture with 8 pixels,
12,288 directions and 657 distinct delay groups. Seven measured repetitions
followed warm-up, alternating implementation order. Group preparation was
outside the timer. Median full-grid times were:

| Numba threads | Frozen original | Shared grouped | Shared singleton |
| --- | ---: | ---: | ---: |
| 1 | 14.848 ms | 9.751 ms | 16.320 ms |
| 2 | 7.727 ms | 5.167 ms | 8.555 ms |
| 4 | 3.944 ms | 2.734 ms | 4.436 ms |

All returned maps and best-direction results matched the reference bit-for-bit.
This is a kernel measurement, not a whole-job speedup. At 1 or 64 valid
directions, the shared kernel was slower than the frozen original in this probe.

A separate synthetic stress test replaced the fixture's delay grid with one
repeated delay tuple. At four threads, grouped scanning took 9.605 ms versus
4.243 ms for singleton scanning: a single group cannot exploit direction-level
parallelism. Both modes returned identical results. This supports retaining the
explicit opt-out; it does not establish an automatic selection threshold.

Reproducible scripts, timing records and test logs are in the parent workspace's
`runs/sky_scan_unification/`. The new default has not received a fresh full-job
catalog or peak-memory benchmark.
