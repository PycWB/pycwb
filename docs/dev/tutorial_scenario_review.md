# Tutorial scenario review

Date: 2026-09-29. Proposal based on the current documentation, examples, configuration schema, and the adjacent `pycwb-config` checkout. The physics research rigor skill was not applied, as requested. This is a source review; the candidate examples were not executed for this review. Configuration presence is evidence of intended use, not proof that a complete tutorial is ready or scientifically validated.

## Recommendation

Organize tutorials around research questions and observable results. A researcher should be able to move from a signal in detector data to a reconstructed candidate, a background-based significance estimate, and a population sensitivity measurement. The existing injection examples are valuable ingredients, but injection syntax alone does not demonstrate that full workflow.

Keep the first synthetic search in Getting started. Give Tutorials three small groups: Working with signals, Significance and sensitivity, and Specialized searches. Put scheduler and deployment details in Run analyses, and stage-by-stage Python internals in Concepts and methods. Cross-link these instead of requiring every reader to complete every branch.

## Core learning sequence

| Scenario | Research question and demonstrated functions | Material to reuse | Concrete result the learner should produce |
| --- | --- | --- | --- |
| 0. Recover a known synthetic signal | Can I run the pipeline and identify the injected signal? Configuration, generated noise, CLI, completion and trigger inspection. | `examples/demo/`, `start_here.rst`, `understanding_results.rst`, `tests/test_demo_e2e.py` | A completed run, a recovered trigger near the known signal, and a waveform plot. Keep this as the prerequisite, not another injection chapter. |
| 1. Search public detector data | How do I go from public strain and data-quality segments to a candidate? GWOSC download, frame lists, channels, GPS ranges, DQ, zero-lag search. | `examples/GW190521_search/`, `examples/gwosc/`, GW150914 notebooks | A catalog selection around the known event, time-frequency plots and reconstructed detector waveforms. Choose one canonical event and modernize its example. Recovery of an event in a short interval is not a reproduction of its published significance. |
| 2. Inspect a reconstructed event | What did the search recover? Catalog fields, detector timing, reconstructed versus injected strain, residuals, sky products. | `understanding_results.rst`, `examples/waveform_reconstruction/`, `examples/postproduction/waveform_reconstruction_workflow.yaml` | One small analysis script producing aligned injection/reconstruction plots and stated comparison conventions. Establish how saved native outputs feed the report: its current input is a consolidated `wave.h5`. |
| 3. Design an injection population | Which signals can I test, and how are trials scheduled? SG/SGE, WNB and CBC examples; sky/time distributions, repeated trials, fixed amplitude versus target SNR, truth and recovered-event matching. | `examples/sine_gaussian_injection/`, `white_noise_burst_injection/`, `multiple_injection/`, `new_injection_infra_with_gaussian_noise/`, `new_injection_infra_with_real_data/` | A truth table and matched table retaining recovered and missed injections. Start with a few explicit sources, then change one population dimension at a time. Show both generated noise and a public real-data segment. |
| 4. Estimate background and significance | How often would noise make a candidate this loud? Time slides, zero-lag separation, completed exposure, vetoes, ranking and finite-background limits. | `recipe_background.rst`, background guide, BKG templates in `pycwb-config`, selection/evaluation actions in the standard postproduction workflow | A background distribution, a cumulative FAR curve and a candidate's location on it, with exposure and units. Supply a small runnable workload plus larger prepared products; do not imply a short demo supports an extreme IFAR. |
| 5. Train and apply postproduction ranking | How do I combine event features and assess the ranking? XGBoost training, saved model, scoring, train/FAR/evaluation separation. | `examples/postproduction/standard_analysis_10pct_workflow.yaml`, `recipe_training.rst`, `pycwb-config` XGB configurations | A saved model and scored held-out catalogs, with before/after ranking comparisons on common evaluation data. Explain feature choice and which samples are allowed to train the model. |
| 6. Measure search sensitivity | At a chosen significance threshold, which injected sources are recovered? Matching, selection, all eligible injections as denominator, per-family efficiency and uncertainty. | Standard postproduction workflow, `recipe_efficiency.rst`, BurstLF/BurstHF simulation configurations | Efficiency versus amplitude, per-waveform curves and hrss50/hrss90 when supported by the sampled range. For CBC populations, use the relevant distance/population coordinate. Include missed signals and distinguish fixed-hrss and target-SNR populations. |

Lessons 4–6 should share one small BKG/SIM dataset and configuration. This makes the transition from triggers to FAR to sensitivity visible and avoids three disconnected placeholder workflows. Provide a fast route through recorded products and an optional route that regenerates them, with provenance and expected outputs for both.

## Specialized scenarios

These should be independently selectable after the relevant core lesson.

| Scenario | What it should teach | Existing basis and remaining work |
| --- | --- | --- |
| Detector networks and sky reconstruction | Run the same source with H1/L1 and H1/L1/V1; inspect antenna response, delays, detector PSDs, sky grids and reconstruction changes. Keep source parameters and shared detector-noise realizations controlled. | `examples/new_injection_infra_with_LHV/` and `injection_with_coordinate_system/`. Supply PSD assets and enable the output products used in the exercise. Do not promise that every added detector improves every metric. |
| Sky masks and externally triggered searches | A dedicated lesson: compare all-sky, fixed-direction, circular-patch and HEALPix-map masks on the same injected source; define the on-source time window and inspect the selected pixels and recovered candidates. | `targeted_search.rst`, `recipe_targeted.rst`, the `sky_mask` schema, and `likelihoodWP/sky_mask.py`. Explicitly distinguish injected-source sky distribution from the likelihood search mask. A HEALPix threshold mask is a selection, not automatically a probability-weighted likelihood. Background and significance must use the corresponding search procedure. |
| Different signal morphologies | Explain why short low-frequency bursts, high-frequency/ringdown signals and long-duration signals need different bands, resolutions, clustering gaps and segment margins. | `pycwb-config/config/{BurstLF,BurstHF,BurstLD,BBH,SN}` plus HF `VelaGlitch`, `NSGlitch`, `Vela_RD`, `STDINJ_RD` simulation folders. Build one small representative exercise per distinct method; use waveform-family variants within it. Folder names alone do not establish scientific readiness. |
| Bring your own waveform | Add a small waveform generator or adapt an external model. Explain polarizations, sample interval, epoch, support, units and detector projection. | Current generator loader and `examples/pyseobnr_injection/`. Start with a minimal local generator, then offer PySEOBNR as an optional dependency. Repair the current tutorial's interface before reuse. |
| Scale a finished analysis | Carry the same configuration to batch execution, monitor it, resume, merge and preserve catalog manifests and configuration snapshots. | `tutorial_batch_inj.rst`, `pycwb-config/README.md`, machine profiles, `examples/performance/bounded_cpu.yaml`. Use a few jobs and dry-run setup before full production. Keep this in Run analyses and link it from the population lesson. |
| Streaming search | Show ingestion, overlapping windows, duplicate handling and restart behavior with a local synthetic frame stream. | `examples/online_shm_run/`. An advanced operational lesson after offline analysis; check README paths against current output code before publication. External alert submission should be a separate, explicitly configured step. |

GPU/backend comparisons, cWB result conversion and parity studies belong in advanced technical guides. Their useful learning outcomes are compatibility, resource use and measured numerical differences; they should not interrupt the introductory scientific path.

## Problems in the current sequence

1. `tutorials.rst` mostly develops injection and batch skills. Background, ranking and efficiency are linked as follow-on references, not taught through a shared runnable dataset.
2. `tutorial_injection.rst` mixes running a search, calling pipeline internals and detailed cWB resampling conventions. The scientific injection lesson should focus on experiment design and recovery; preserve the internals and conventions in linked technical sections.
3. `tutorial_multi_injection.rst` points to directories but does not walk through building a population, checking the truth table or interpreting missed injections. Its stated duration and learning outcomes exceed the demonstrated exercise.
4. `tutorial_customized_wf_gen.rst` showed a generator mapping with `module`/`function`. The production path in `injection/strain.py` loads a function string and calls it with keyword arguments. It accepts a dictionary with `type: polarizations`, `hp` and `hc`, or `type: strain` with detector-keyed strain series; the former is projected into detectors. The older tuple result is deprecated. The tutorial now uses the production interface with a local working generator, rather than the obsolete helper in `wf_generator.py`.
5. The sky-patch example README uses old numeric-angle/unit syntax while its YAML uses unit-bearing angles and explicit `coordsys`. Use the current schema consistently.
6. The standard postproduction example contains `/path/to/` roots and expects several catalogs. It is a production template, so it needs a small supplied dataset or a complete generation procedure to serve as a tutorial.
7. The public-event examples contain historical and event-specific settings. For example, the GW190521 YAML contains a simulation-mode setting despite having no injection block. Review and run a cleaned canonical example before publishing it as a supported lesson.
8. The targeted recipe lists fewer triggers and improved localization at higher HEALPix resolution as expected checks. A tutorial should measure the actual outcome; a finer numerical grid does not by itself establish a better scientific localization.
9. `pycwb/modules/noise/glitch.py` and `non_gaussian.py` explicitly contain stubs. Do not advertise their planned noise generators as runnable features. Real open data can supply the non-Gaussian-noise teaching scenario.
10. `examples/colab/pycwb/` is a nested older checkout. Use the primary package and current schema as the implementation source; adapt notebook exposition rather than copying its embedded package.

## What makes each lesson complete

Every lesson should have one research question, a complete versioned input set, a short run path, and an output the reader can inspect. Annotate the few settings that change from the previous lesson. Include the commands or script that produce the promised plot/table and explain its axes, units and interpretation. State download requirements and measure runtime/memory on a recorded reference environment before publishing estimates. Show what a completed empty result means and how to distinguish it from a failed job.

Use `literalinclude` or equivalent inclusion of maintained example files rather than duplicating long YAML blocks. Preserve detailed parameters in Reference and pipeline internals in Concepts. Avoid expanding every waveform family and every scheduler into a separate sidebar entry.

## Implementation order

1. Modernize one public-event example and turn event inspection into a worked result.
2. Consolidate the injection lessons into population design plus an optional custom-generator lesson.
3. Package a small BKG/SIM dataset and teach background, XGBoost and efficiency as one connected analysis.
4. Add network/sky-constrained exercises, followed by morphology-specific variants and production scaling.
5. Add streaming and backend-extension lessons after their examples have been exercised with the current release.

The lifecycle animation is a useful orientation link at the start. Its simplified model should remain distinct from the production outputs used by these tutorials.

## Feature-coverage follow-up

The initial research-question sequence needs explicit feature exercises as well. Sky masks were mentioned under targeted searches but need a named lesson and concrete comparisons. The following matrix fills the other gaps without turning each configuration option into a separate page.

| Feature to demonstrate explicitly | Exercise and observable output | Placement / source evidence |
| --- | --- | --- |
| Sky masks | Run the same signal with no mask, `Fixed`, `Patch`, and `Custom`; plot accepted sky pixels and the reconstruction. Include a patch containing the source and a deliberately misplaced patch, and inspect the actual outcomes. | Dedicated tutorial. `likelihoodWP/sky_mask.py` implements all four types. Teach unit-bearing angles, `icrs`/`geo`/`cwb`, event GPS time, RING/NESTED ordering, map resolution and threshold. `Custom` currently selects map values strictly above the threshold; it does not interpret a threshold as a cumulative credible-region percentage. |
| Data quality, segment boundaries and vetoes | Start with a short interval containing known gaps/vetoes; show the accepted segments, padding, selected live intervals and catalog completion. Explain which intervals are excluded before search and which cuts reject candidates. | Core real-data lesson plus background lesson. `DQF`, `segLen`, `segMLS`, `segTHR`, `segEdge`, `segOverlap`, job-segment and progress logic. |
| Conditioning and gating | Inspect raw strain, conditioned strain and noise estimates; demonstrate an explicitly configured correction or time-veto hook and compare its diagnostics, accepted exposure and recovery. | A conditioning lesson. Current native workflow calls `conditioning_plugins.api.run_hooks`; schema exposes `conditioning.post_whitening` and `selection.time_vetoes`. The bundled `cwb_gating` adds excluded intervals and does not zero the strain. These hooks have implementation/tests, but a small runnable tutorial remains to be prepared. MESA is an advanced optional branch with its documented scope. |
| User-supplied data and stored waveform inputs | Replace the public-download input with local frames and correct channel names, sampling metadata and DQ. Separately illustrate reading stored detector strain or MDC inputs, checking epoch and resampling explicitly. | Extend data/injection lessons. `frFiles`, `channelNamesRaw`, `channelNamesMDC`, `gwdatafind`, `read_data`, `inj_generators.get_strain_from_file`, and `examples/benchmark/user_parameters_mdc.yaml`. The production loader accepts both typed polarization dictionaries and typed detector-strain dictionaries; the latter bypass detector projection. |
| Custom detector geometry | Add a detector through `detector_definitions_file`, select registry IDs with `detector_geometry`, supply its PSD/input stream and inspect antenna responses and delays. | Extend the network lesson. `detector_support.rst`, `config/detector_definitions.py`, `config/tests/test_custom_detectors.py`. This demonstrates the supported detector model; it does not establish support for arbitrary moving/space interferometers. |
| Amplitude normalization and scheduling | Compare fixed source hrss, a CBC distance change, and target network SNR; inspect realized detector signals and matched truth. Include an injection near a segment boundary and explain `t_start`, `t_end`, repeated trials and missed-event accounting. | Explicit exercises in the population lesson, not just a list of waveform names. Current injection infrastructure and target-SNR resampling documentation. Injection-only windows should be taught as a simulation shortcut, not background exposure. |
| Controlled configuration comparisons | Hold the signal/noise population fixed and vary one choice: network, sky mask, frequency band, time-frequency resolution or clustering gap. Combine catalogs and plot recovery, sky error and reconstruction changes, retaining missed injections. | Dedicated comparison exercise spanning the core and specialist lessons. Local `examples/postproduction/angle_error_comparison_workflow.yaml` and the multi-run workflow documentation provide an implementation example; the example is currently untracked and requires supplied catalogs. Only teach search modes actually supported by the selected processor. |
| Custom postproduction and reports | Build a small workflow that reads catalogs, selects events, runs a custom analysis/plot action and assembles a report. Show dependencies and reusable outputs. | Advanced researcher lesson. `postproduction_workflow.rst`, action interfaces, multi-run and generic report actions. This demonstrates modularity beyond using the standard XGBoost recipe. |
| Resource selection and restart | Use the same workload to demonstrate bounded CPU execution, an available GPU route, supported output combinations, interrupted-run resume and fragment merging. Record configuration and compare results before interpreting speed. | Run analyses, linked from tutorials. `execution`, `execution_profile`, `gpu`, `backends.rst`, `workflow_execution.rst`. Keep hardware-dependent routes optional. |

Sky-map outputs also need an explicit exercise: save and read the map, overlay injection truth and distinguish input mask, numerical sky grid, and reconstructed map. Do not interpret arbitrary likelihood-derived map values as calibrated posterior probabilities without establishing their definition.

Revised priority: make sky masks a first-class tutorial; make DQ/exposure and conditioning explicit parts of the real-data path; add custom geometry and controlled multi-run comparisons next. Other functionality can remain as well-defined exercises within the existing proposed lessons. Detailed scientific settings such as packet patterns, regulators, chirp and Q-veto features belong in linked advanced exercises rather than an introductory knob-by-knob parameter tour.

## Produced pages and checks

The first implementation adds 17 pages (three learning-path indexes and 14 lessons), updates the existing custom-waveform lesson, and preserves the earlier injection/batch pages. `tutorials.rst` now links Working with Signals, Significance and Sensitivity, and Extend and Scale an Analysis. Shared inputs live in `examples/tutorials/`; the preparation script creates 13 configurations plus mask/geometry assets without overwriting existing work.

Checks performed on 2026-09-29, using Python 3.13 in the local development environment:

- All 13 final generated YAML configurations pass offline `pycwb validate`.
- Synthetic searches exercised all-sky, fixed/patch/custom/displaced masks, HLV and custom geometry, custom waveforms, population, gating and bounded execution. Runs used the compatible local cross-talk catalog and one numerical thread for this check.
- The final four-source population recovered three sources. Simulation summary and a right match retained four rows, including the weakest source with no trigger ID.
- The multi-run comparison workflow generated its plots, manifest and HTML report from actual all-sky/fixed/patch catalogs.
- The event-inspection Python blocks executed against the native HDF output. Their waveform plot and the input-mask plot are included in the documentation. The generated binary mask selects 52 NSIDE-16 pixels.
- The deterministic gating exercise returned the documented four-second exclusion without changing samples. The DQ intersection exercise returned 50 seconds.
- The bounded run completed; a second run with unchanged YAML and `--force-overwrite` reused completed work. On this host its memory monitor required execution outside the filesystem/process sandbox. HTCondor files were generated with a test accounting group; no jobs were submitted.
- The custom-network run exposed a missing geometry handoff in injection arrival metadata. The fix reuses the configuration's initialized detector objects; 26 geometry/timing tests pass, including the new co-located custom-detector regression.
- Sphinx builds with warnings treated as errors, and the compiled tutorial content links resolve. Stale autogenerated test-module pages are excluded from documentation discovery.

Validation scope: public-data downloads/background runs, the large XGBoost/efficiency study, cluster execution, GPU routes and streaming replay were not executed in this pass. These pages describe their required external data or environment. The ranking/efficiency page is a guided adaptation of the maintained study template; a portable teaching dataset and a fully self-contained ranking lesson remain follow-up work. The checks above establish the exercised workflow behavior, not scientific sensitivity or significance validation.

## Tutorial and how-to boundary

The follow-up restructuring keeps prepared experiments in Tutorials and procedures for a reader's own data, configuration and environment in **How-to guides** (formerly Run analyses). The learning paths are now Working with Signals, Compare Recovery and Background, and Extend a Worked Example.

- Ranking and efficiency on supplied study catalogs lives in `postproduction_study.rst`; streaming setup lives in `online_search.rst`. Neither is presented as a completed, portable teaching experiment.
- `run_on_clusters.rst` owns batch submission and product collection. `workflow_execution.rst` owns resource selection and recovery. The resource tutorial retains the small baseline/bounded comparison and unchanged-run restart exercise.
- `config_repository.rst` owns adaptation to local frame and DQ files; the public-data tutorial links there. The injection Python internals are preserved under Concepts and methods.
- Recipes identify inputs, link the required procedures and state completion checks. They no longer maintain a second copy of commands and YAML settings.
- Former batch, multi-injection, ranking and streaming tutorial URLs remain as forwarding pages with their section anchors preserved. They are absent from the tutorial navigation.

This restructuring does not extend the scientific validation scope recorded above. Future full ranking/efficiency tutorials still need a supplied teaching dataset and an exercised path through its products.
