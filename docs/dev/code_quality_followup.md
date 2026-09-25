# Code quality follow-up — 19 September 2026

The `consistency-bugfix-optimization` branch records the existing consistency and optimization campaign in feature commits, followed by separate formatting and maintenance commits. The companion WDM, analysis-configuration and reference-build repositories use the same branch name. The pre-campaign revisions are pinned in the review site's `review-baselines.json`; committing the campaign must not erase its comparison baseline.

## Completed

- Formatted the 97 campaign Python files and tests to 120-column Ruff style. Parsed executable syntax was identical before/after formatting, excluding docstring whitespace. Public keyword spellings and numerical expressions were retained.
- Extracted chirp dispatch, reset and metadata writeback into `_update_cluster_chirp_statistics`. The environment switch is still read at call time; legacy fallback, disabled search families, reset behavior and seed propagation are retained. Eleven focused cases protect these contracts.
- Expanded NumPy-style docstrings for numerical helpers, sky scans, chirp estimation, delay-group caching and TD population. Corrected the original scan's noise-weight shape and omitted return-value documentation.
- Documented every scratch tuple's order, shape, dtype, overwrite requirements, alias restrictions and borrowed-result lifetime. Removed obsolete commented-out implementations while retaining reference arithmetic explanations.
- Replaced terse connectivity and inverse-gather locals with names describing runs, samples, bins, tiles and indices. Reverse-renaming reproduces the prior executable AST, excluding docstrings. Kept reference mathematical symbols and public argument names where renaming would obscure numerical provenance or break callers.
- Removed unused imports and unused temporary allocations, gave ambiguous subnet/cross-talk loop indices descriptive names, and replaced assigned local lambdas with named helpers. Preserved the existing TD-kernel re-export explicitly.

## Validation

The campaign's CPU-focused regression selection, including the new chirp-dispatch tests and companion WDM tests, passed **807 tests** after the maintenance and dead-code cleanup. Existing tests include bitwise sky-variant comparisons, poisoned/reused scratch arrays, chirp release oracles, staged TD lifecycle, detector geometry and bounded transforms. Third-party GWpy/Matplotlib deprecation warnings remain. A final focused rerun after removing an unused zero-initialized local passed **12 tests**.

Ruff formatting and the E4/E7/E9/F lint selection cover the 98 campaign/test Python files (97 original plus the new test module). NumPy docstring checks cover the ten newly documented helper modules. These are scoped checks, not a claim that all existing repository docstrings and typing pass a global audit.

No pipeline catalog, throughput, peak-memory benchmark or GPU validation was rerun. Historical performance and scientific acceptance limits remain those of the September 11 report. Formatting, documentation and helper extraction do not promote experimental options or resolve the outstanding numerical discrepancies.

## Remaining manual decisions

- **Q1 (completed 25 September):** The scans now share one compiled group-based kernel, and allocating numerical helpers delegate to their `*_into` implementations. Grouping defaults on with an explicit singleton opt-out. See [implementation, independent regression evidence and performance limits](unified_sky_scan.md).
- **Q2 (completed 25 September):** Native execution settings now use a validated, immutable YAML profile recorded in catalog metadata. See [defaults, migration and scope](execution_profile.md).
- **Q3:** Whether to enforce immutable delay grids or add explicit cache invalidation. The existing identity-based contract is now documented; no mutation policy was silently imposed.
- **Q5:** Whether to introduce explicit coarse/fine TD types or choose the simpler full-fine path. Stage lifecycle and frequency-offset contracts are documented, while the existing guarded staged behavior remains.
- Scientific, backend and experimental choices in the original ledger remain available for review. Completing code cleanup is not approval to enable a new default.

Q4's buffer documentation, Q6's chirp extraction and Q7's preservation of numerical contracts are addressed in this pass. The review page distinguishes these completed actions from remaining choices and provides full-campaign and cleanup-only comparisons with commit links.
