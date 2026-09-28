# Release comparison cleanup

The integration quality review used `release/v1.1.0` (`256291df`) as its baseline.
This follow-up fixes context ownership and offline/runtime validation, replaces
GPU function-global substitutions with explicit callbacks, and moves execution
policy, byte parsing and exact fingerprints below the workflow/scientific layers.
CPU and CUDA chirp scoring share native-order host sampling/finalization. The
original `ed13f26` CPU loop is frozen only in tests as an independent oracle.

## Removed and retained experiments

- Removed worker-side reconstruction and its wrapped-result transport/routing.
  The recorded LF 128-lag comparison passed exact output checks but was slower
  than parent output. Source evidence: enclosing workspace
  `runs/gpu_lf_exploration/FULL_GPU_BOUNDED.md`, “Remaining costs”. Parent output
  already overlaps worker analysis. `gpu.worker_output: true` now gives a
  migration error; false snapshots remain readable. Historical result artifacts
  are retained and do not become current performance claims.
- Excluded interpreter bytecode and Numba compilation caches from distributions.
  A package inspection found three Python 3.14 bytecode files under vendor data.
- Removed tests that required GPU workflow factories to clone private globals;
  contract/callback tests now check behavior. The cloning utility remains for
  diagnostic tracing and the GIL-releasing thread experiment.
- Retained the thread experiment: six threads were slower than six processes
  (projected 29.2 vs 20.0 minutes for 2,400 LF lags), but peak PSS was 3.36 vs
  6.34 GiB. These are projections from one 400-lag measurement, not complete-job
  timings. Evidence: `runs/lag_parallel_scaling_400_200/LF_REPORT.md`.
- Retained singleton sky-delay groups: they can win for tiny valid masks or
  extreme one-group workloads; see `unified_sky_scan.md`.
- Retained independent numerical references, benchmark drivers and compatibility
  imports. No evidence established that unrelated legacy prototypes were unused
  by their owners. Rejected GPU FFT arithmetic was already removed before this
  cleanup and has not been reinstated.

## Ownership and limits

Only workflows choose policy and own output. CPU frame/conditioning helpers take
explicit scheduling options. Native scientific entry points own selection/cut
policy and accept process-local backend callbacks. Context resets require new
stage objects because existing device buffers also become invalid.

The executor extracts input-memory and cache-reuse planning from process
supervision, and names the worker resources retained until clean exit. Existing
failure, pressure, resume and output acknowledgement tests remain mandatory.
Large numerical loops and unrelated legacy typing debt are intentionally not
rewritten to satisfy style limits.

Run `python tools/check_quality.py`, `python -m mypy`, the non-slow test suite,
and `make doc-check`. The type checker covers the 13 selected public boundary
files listed in `pyproject.toml`; it does not type-check every numerical kernel. The checked-in quality baseline allows inherited lint and
annotation debt, but rejects new diagnostics and missing public contracts. It is
not a claim that every legacy function has complete types or that every possible
configuration is scientifically validated.

## Local follow-up validation

Chirp fixed-stream comparisons cover 13, 32 and 128 micropixels, three seeds,
and both complete and exhausted streams, with bitwise equality including the
consumed stream cursor. A warmed, alternating-order 21-sample timing probe used
13/32/128/512 cells. Median shared/original runtime ratios were
0.917/0.943/0.958/0.964. These synthetic kernel results establish no observed
regression in this probe; they do not establish a whole-job speedup.

The exact environment and test/build logs are retained in the enclosing workspace
at `runs/quality_cleanup_20260929/`. The reference loop remains independent in
`likelihoodWP/tests/chirp_bootstrap_reference.py`; the runtime no longer imports
the CUDA package to prepare or finalize chirp trials.
