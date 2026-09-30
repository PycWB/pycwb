# Native conditioning and time-veto hooks

The native job processor supports two optional, ordered plugin stages. They run once per injection trial (once per job for background), after regression/whitening and before coherence or waveform caches are built. Empty lists preserve the previous processing path. The parallel and nogil wrappers that delegate to the native processor use the same hooks; independent processors must explicitly implement this API.

```yaml
conditioning:
  post_whitening:
    - module: pycwb.modules.conditioning_plugins.o3a_conditioning
      options:
        detectors: [L1, H1]
selection:
  time_vetoes:
    - module: pycwb.modules.conditioning_plugins.cwb_gating
      options:
        energy_threshold: 1000000
        integration_seconds: 0.5
        padding_seconds: 1.5
```

These containers are part of the built-in configuration schema. Each plugin exports `HOOK_STAGE`, an `OPTIONS_SCHEMA` JSON schema, and `apply(context, result, **options)`. Options are validated before invocation. Unknown options and wrong-stage modules raise errors.

`HookContext` supplies the configuration, detector order, and trial segment. `ConditioningResult` contains the conditioned strain list, noise-RMS maps, excluded GPS intervals, and diagnostics. Plugins must preserve detector count and strain sample count, start, and cadence. Time-veto plugins must leave strain and noise models unchanged. Return excluded intervals, not accepted intervals; the processor subtracts them from CAT2 accepted intervals and then applies existing injection-window and circular-lag semantics. An empty accepted list means no live data, not all live data.

## O3a narrow-band correction

`o3a_conditioning` ports the production cWB O3a 16–48 Hz correction: 32 Hz WDM layer, symmetric four-second LPR filter with right-tail rejection, reference order-statistic medians, magnitude and multiplicity corrections, and both inverse phases. Default detector selection is L1/H1, matching the reference plugin. Other detectors can be selected explicitly.

The correction returns an ordinary `pycwb.types.time_frequency_map.TimeFrequencyMap` on the layer's 64 Hz time lattice, matching cWB's use of `WSeries<float> nVAR` rather than a dedicated variability type. Its data has shape `(1, n_samples)` and float32 storage; `dt` and `start` describe sampling, while `f_low` and `f_high` give the affected 16–48 Hz band. The map contains correction factors, so `wavelet` is `None`. `NoiseRMSMap.variation` carries this map through coherence and likelihood pixel-noise lookups in `modules/data_conditioning/noise.py`, which delegates optional variation to `conditioning_plugins/noise_variation.py`, including partial overlap with the affected band and lagged detector sample indices. Physical waveform reconstruction consumes the resulting per-pixel noise RMS. A second variation-producing plugin is rejected until an explicit composition rule exists. Saved diagnostic arrays remain one-dimensional for compatibility.

Zero-energy/constant envelopes receive identity behavior; zero correction autocorrelation skips the multiplicity adjustment. These safeguards avoid reference divisions by zero. These safeguards do not change the tested ordinary-data reference result. This module is different from `data_conditioning/psd_correction.py`.

## Gate semantics

`cwb_gating` computes a rolling **sum of squared whitened samples**, without division by the sample rate, over 0.5 seconds. It applies the reference threshold and sample padding, scratch-edge clipping, interval rounding, and union across detectors. The strain is not zeroed. The combined exclusions affect pixel selection and reported livetime through the shared keep-window machinery, including circular lags. The CAT2/job-duration eligibility check uses the pre-gate windows; gating does not reapply that cut. Pixel selection and reported livetime use the post-gate windows, matching the reference stage ordering.

The gate is recomputed after injections and conditioning for every trial, as in the reference: a sufficiently loud injection can itself trigger a gate. Count gated injections explicitly in efficiency studies rather than silently removing them from the injected denominator.

## Outputs and validation

`conditioning/job_<index>/trial_<index>/diagnostics.json` records selected modules, explicitly supplied options, module source SHA256, detector diagnostics, excluded intervals, and variation-lattice metadata. `noise_variation.npz` stores the correction arrays by detector index. The saved configuration retains detector ordering and defaults are defined in the fingerprinted module source.

Validate three distinct levels:

1. Identical-input correction, variation-map, pixel-noise, and active-gate comparisons against the cWB release.
2. End-to-end LF runs with hooks enabled, plus regression coverage with hooks absent.
3. Independent-source population campaigns using final settings, many noise segments/banks, and predeclared equivalence tolerances.

Passing the first two does not certify injection/reconstruction population equivalence.
