# Frozen sky-scan reference

Copied before consolidation from PycWB commit `6e9df56ce0e1e760565d20275c999304d010b98d`. These test-only modules preserve the independent allocating arithmetic for differential tests, including poisoned-buffer tests. Do not synchronize them with production refactors.

The companion `../data/sky_scan_preconsolidation.npz` stores the saved Phase-2 `BurstHF_job1` median-pixel likelihood scan input (8 pixels, 12,288 directions), replayed through this frozen reference. `arg_0`–`arg_13` are the original positional inputs; `expected_0`–`expected_12` are the returned tuple. Scalar entries are zero-dimensional arrays. The capture and comparison script is `runs/sky_scan_unification/benchmark.py` in the parent workspace.
