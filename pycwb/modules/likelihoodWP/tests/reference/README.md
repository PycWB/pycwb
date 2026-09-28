# Packet normalization reference

`packet_norm_sse.cpp` independently evaluates the four-lane SSE accumulation,
scalar reduction, float32 ratio cut, and normalization from
cWB 6.4.6.9 `wat/network.hh`, `network::_avx_norm_ps(I>0)`.
Reference commit: `e03cf7f02fa4d4666c5619ffcf9730e988328e3c`.
Compile with `g++ -O2 -ffp-contract=off packet_norm_sse.cpp -o packet_norm_sse`.
The two command-line arguments are binary input and output paths; the source
defines their layout.

`../packet_norm_release_golden.npz` contains inputs and the resulting float32
outputs for 20 cases. Each case has `p`, `q`, `xtalk`, `lookup`, `mask`, `energy`
and `expected` arrays, prefixed with its name and `__`. Expected output is the
concatenation of flattened detector SNR, norms, residual noise, and pixel norms.

Two cases are real Chunk 16a LF packets from lags 34 and 80. The other 18 cover
1–3 detectors, 1/4/21 pixels, isolated and dense cross-talk, and inactive masks
(NumPy random seed 61). Tests exercise both float32 and float64 callers and
verify that inputs remain unchanged. The separate unit-ratio test exposes the
branch discrepancy without a binary fixture.

These references validate this packet operation only. They do not establish
bitwise agreement of the complete cWB/PycWB pipelines.

## Sky localization reference

`sky_localization_release.npz` and `sky_localization_cases.json` record four calls to the separately installed cWB 6.4.6.9 sky-error implementation, covering the statistic scale, antenna prior and saved-map selection. The Python implementation reproduces the saved probabilities, indices and regions for these fixtures. Generation and full event comparisons are recorded under `runs/discrepancy_fixes` and `runs/faithfulness/full_scientific_outputs` in the parent workspace. Tied-statistic ordering outside these cases is not asserted to be identical to cWB.

## Micropixel chirp reference

`chirp_*` fixtures come from the installed cWB 6.4.6.9 `netcluster::getupixels` and `mchirp_upix` methods, plus TRandom3. They cover six synthetic cases, ten real accepted-pixel/seed cases and 60,000 uniforms. Fixture generators are `runs/chirp_fix/oracle.py` and `oracle_real.py` in the parent workspace. Tests require exact micropixel cells and all stored chirp fields on identical inputs. Full pipelines can differ when upstream pixel likelihoods differ.


Waveform summary fixtures (`waveform_*oracle*`) come from installed cWB 6.4.6.9 detector methods and one-off compiled release float expressions; generators live in `runs/waveform_stats`. Twelve explicit waveforms cover random/DC/Nyquist/mixed inputs; 32 scalar cases cover narrowing and network centroids. Runtime tests require neither ROOT nor a compiler.
