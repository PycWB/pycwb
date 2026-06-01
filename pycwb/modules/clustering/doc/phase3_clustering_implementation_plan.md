# Phase 3 Clustering Implementation Plan

## Scope

This plan covers the requested implementation sequence, but no code should be changed until the implementation is explicitly started.

The requested direction is:

1. Provide Phase 1 and Phase 2 support code under `pycwb/modules/clustering/validation`, but defer their testing because they require large data sets.
2. Implement and test Phase 3 as a replaceable pre-likelihood clustering pipeline.
3. Put each clustering method under `pycwb/modules/clustering/[method]` so existing coherence code is not broken.
4. Replace the current per-lag `coherence_single_lag` + `supercluster_single_lag` pair with a new clustering entry point, rather than inserting a method between them.
5. Keep lag-independent setup code available (`setup_coherence`, `setup_supercluster`, TD cache construction, XTalk loading), but move per-lag pixel selection and clustering ownership into the new Phase 3 path.
6. Add a test workflow function derived from `process_job_segment_native.py` to exercise the new replacement path.
7. Add test YAMLs similar to the existing `tests/sample` configs, using Gaussian noise and multiple injections.

## Current Replacement Point

The current native workflow does:

```python
frag_clusters_this_lag = coherence_single_lag(
    coherence_setup,
    lag,
    veto_windows=sub_job_seg.veto_windows,
)

fragment_cluster = supercluster_single_lag(
    supercluster_setup, config, frag_clusters_this_lag, lag,
    xtalk=xtalk, td_inputs_cache=td_inputs_cache,
)
```

`coherence_single_lag` currently performs two jobs:

1. Select significant pixels for each resolution and lag.
2. Cluster those pixels with connected components and apply coherence-level cuts.

`supercluster_single_lag` then performs the remaining pre-likelihood grouping and cuts:

1. Merge resolutions into one `FragmentCluster`.
2. Attach TD amplitudes from the TD cache.
3. Run supercluster linking, subnet cut, and defragmentation.
4. Return one likelihood-ready `FragmentCluster`.

The new Phase 3 direction is to replace both per-lag calls with a single clustering pipeline.  The workflow should no longer call `coherence_single_lag` and then `supercluster_single_lag`, and it should no longer insert a reclustering step between them.  Instead:

1. Keep the expensive lag-independent setup functions (`setup_coherence`, `setup_supercluster`, `build_td_inputs_cache`, `XTalk.load`) unchanged.
2. Extract the pixel-selection portion of `coherence_single_lag` into a reusable helper that performs no clustering.
3. Add a Phase 3 clustering entry point that accepts the selected per-resolution pixel candidates for one lag.
4. Each backend performs method-specific primary clustering, optional cross-resolution merging, TD amplitude attachment, subnet/supercluster-style cuts, and defragmentation as needed.
5. The entry point returns the final single `FragmentCluster` that the likelihood stage consumes.

This keeps production native behavior available while making the experimental path a true replacement for the existing per-lag clustering and superclustering pipeline.

## Proposed Module Layout

```text
pycwb/modules/clustering/
    __init__.py
    cluster.py
    common.py
    pipeline.py
    validation/
        __init__.py
        features.py
        datasets.py
        metrics.py
        reports.py
    connected_components/
        __init__.py
        method.py
    weighted_graph/
        __init__.py
        method.py
    dbscan/
        __init__.py
        method.py
    hdbscan/
        __init__.py
        method.py
    optics/
        __init__.py
        method.py
    doc/
        PycWB_ML.md
        clustering_method_investigation.md
        phase3_clustering_implementation_plan.md
```

Only Phase 3 methods need tests immediately. The `validation` package is for Phase 1 and Phase 2 scaffolding and should be importable, documented, and type-stable, but not tested against large catalogs yet.

## Entry Point Design

### `pycwb/modules/clustering/cluster.py`

The replacement entry point should expose one function:

```python
def cluster_lag_candidates(
    pixel_candidates_by_resolution: list[dict],
    method: str = "connected_components",
    config=None,
    lag_idx: int | None = None,
    setup: dict | None = None,
    xtalk=None,
    td_inputs_cache: dict | None = None,
    **kwargs,
) -> object | None:
    """Return a likelihood-ready FragmentCluster for one lag."""
```

Input:

- `pixel_candidates_by_resolution`: one selected-pixel candidate dictionary per resolution, produced by the pixel-selection helper. These dictionaries should be the same raw candidate payload currently passed to `cluster_pixels` inside `coherence_single_lag`.
- `method`: selected backend name.
- `config`: optional config object, used for thresholds and method parameters.
- `lag_idx`: optional for logging and reproducibility.
- `setup`: optional supercluster setup dictionary for sky arrays, TD range, subnet cut resources, and likelihood-compatible sky metadata.
- `xtalk`: optional XTalk catalog used by subnet-cut and related post-clustering cuts.
- `td_inputs_cache`: optional delay-vector cache used to attach TD amplitudes to clustered pixels before cuts.
- `kwargs`: method-specific overrides.

Output:

- A single `FragmentCluster` with the same contract as the return value of `supercluster_single_lag`.
- `None` if no candidates survive clustering/cuts for this lag.
- The output must be accepted directly by the likelihood loop.

Required behavior:

- `method="connected_components"` should reproduce native behavior as closely as possible: connected-component primary clustering followed by the same TD-amplitude, supercluster, subnet-cut, and defragmentation semantics currently provided by `supercluster_single_lag`.
- Unknown methods should raise a clear `ValueError` listing available methods.
- Empty inputs must return empty-compatible outputs without crashing.
- No backend should mutate selected-pixel candidate dictionaries unless that is explicitly documented. Prefer constructing new `PixelArrays`, `Cluster`, and `FragmentCluster` objects.
- Backends must preserve the final likelihood-facing `FragmentCluster` contract.

### Pixel-Selection Helper

Add a helper near the existing coherence code, or under the clustering package if ownership is clearer:

```python
def select_pixels_single_lag(
    coherence_setups: list[dict],
    lag_idx: int,
    veto_windows: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """Return selected pixel candidates for one lag, without clustering."""
```

This helper should contain the first half of `coherence_single_lag`:

1. Validate `lag_idx`.
2. Build optional veto masks.
3. Call `select_network_pixels` for each resolution.
4. Return raw candidate dictionaries, not `FragmentCluster` objects.

The existing `coherence_single_lag` can remain as a compatibility wrapper implemented in terms of this helper plus native `cluster_pixels`.

## Method Interface

Each method package should provide:

```python
def cluster(
    pixel_candidates_by_resolution: list[dict],
    config=None,
    lag_idx: int | None = None,
    setup: dict | None = None,
    xtalk=None,
    td_inputs_cache: dict | None = None,
    **kwargs,
):
    ...
```

The backend must preserve:

- Detector order.
- Pixel coordinate convention.
- Lag-adjusted `pixel_index` arrays.
- The `PixelArrays` fields needed by likelihood and TD extraction: `time`, `frequency`, `layers`, `rate`, `asnr`, `a_90`, `pixel_index`, `likelihood`, `noise_rms`, `td_amp`, and `core`.
- Final `Cluster` metadata needed by likelihood: cluster status, timing/frequency bounds, rates, and approximate `ClusterMeta` fields (`energy`, `like_net`, `sub_net`, `net_rho`, `c_time`, `c_freq`).

The backend may change:

- How pixels are grouped into clusters.
- Cluster-level approximate metadata.
- Which clusters are marked rejected with `cluster_status=1`.
- Whether cross-resolution merging happens through native supercluster logic, method-specific graph linking, density clustering, or a compatibility wrapper.

The backend must not change:

- Detector order.
- Pixel coordinate convention.
- Lag-adjusted `pixel_index` arrays.
- The expected likelihood-facing `FragmentCluster` output contract.

## Replacement Pipeline Responsibilities

The new clustering pipeline owns all per-lag work between pixel selection and likelihood:

1. **Primary clustering** — group selected pixels within each resolution, or across resolutions if the method supports it.
2. **Cluster construction** — build `Cluster` objects with valid `PixelArrays` and approximate `ClusterMeta`.
3. **TD amplitude attachment** — use `td_inputs_cache` and `setup["K_td"]` to populate TD amplitude fields exactly as the current `supercluster_single_lag` does.
4. **Cross-resolution aggregation** — merge clusters that should be treated as a single candidate.  The connected-components compatibility backend can call shared native-equivalent helpers; new methods may replace this with their own graph/density model.
5. **Subnet / sky cuts** — apply `apply_subnet_cut` or method-specific equivalents using `setup`, `xtalk`, and config thresholds.
6. **Defragmentation** — apply native `defragment` or method-specific final merging, controlled by existing `pattern`, `Tgap`, and `Fgap` semantics.
7. **Core marking** — mark surviving pixels as core before likelihood, matching the current `supercluster_single_lag` return contract.

To avoid duplicating complex supercluster logic across backends, add shared helpers in `pycwb/modules/clustering/pipeline.py`:

```python
def attach_td_amplitudes(fragment_cluster, config, setup, td_inputs_cache):
    ...

def finalize_clusters_for_likelihood(
    fragment_cluster,
    config,
    setup,
    xtalk,
    lag_idx: int,
):
    ...
```

Backends can then focus on primary grouping while the shared pipeline handles native-compatible finalization.

## Phase 1 and Phase 2 Validation Code

These modules should be produced but testing deferred.

### `validation/features.py`

Purpose: reusable feature extraction for catalog, cluster, and tensor-level experiments.

Planned functions:

```python
def trigger_to_feature_dict(trigger) -> dict[str, float]:
    ...

def cluster_to_feature_dict(cluster) -> dict[str, float]:
    ...

def pixel_arrays_to_table(pixel_arrays, n_ifo: int | None = None):
    ...

def cluster_to_tf_maps(cluster, keys=("likelihood", "null")) -> dict[str, object]:
    ...
```

Notes:

- Avoid heavy ML dependencies.
- Use NumPy and optional pandas only if already available.
- Keep functions deterministic and small enough to use later in tests.

### `validation/datasets.py`

Purpose: load catalogs and saved cluster artifacts for Phase 1/2 studies.

Planned functions:

```python
def load_trigger_table(path):
    ...

def build_feature_matrix(rows, feature_names=None):
    ...
```

Testing deferred because representative catalogs are large and not part of small unit tests.

### `validation/metrics.py`

Purpose: common metrics for later large-data evaluation.

Planned metrics:

- Cluster count and size distribution.
- Adjusted Rand index and normalized mutual information when labels exist.
- Injection-family purity.
- Background/injection separation summaries.
- Runtime and memory summaries.

### `validation/reports.py`

Purpose: generate summary tables/plots later, not required for Phase 3 unit tests.

## Phase 3 Method 1: Connected Components Compatibility

Package: `pycwb/modules/clustering/connected_components`

Initial behavior:

- Reproduce the current native per-lag path as the baseline replacement:
    1. Use the raw selected pixel candidates from `select_pixels_single_lag`.
    2. Call native `cluster_pixels(..., kt=2, kf=3)` or equivalent for network-pattern runs.
    3. Apply the same `subrho`/`subnet` coherence-level selections currently applied inside `coherence_single_lag`.
    4. Merge all resolutions and finalize through shared `pipeline.py` helpers that mirror `supercluster_single_lag`.
- This proves the new entry point can replace `coherence_single_lag` + `supercluster_single_lag` without changing science output.

Optional later behavior:

- Rebuild connected components from `PixelArrays` to make it a full standalone backend.

Tests:

- Compatibility backend produces the same final likelihood-ready clusters as the native `coherence_single_lag` + `supercluster_single_lag` path for small deterministic inputs.
- Compatibility backend preserves accepted trigger count, event GPS time, frequency band, and key statistics within numerical tolerance in the sample workflow.
- Empty selected-pixel inputs return `None` or an empty-compatible result without crashing.

## Phase 3 Method 2: Weighted Graph Clustering

Package: `pycwb/modules/clustering/weighted_graph`

This is the preferred first real Phase 3 method because it is close to the existing connected-component logic but allows physics-informed edge weights.

### Inputs

For each resolution-specific selected-pixel candidate dictionary, build a `PixelArrays` table directly from the raw candidate fields.  No pre-existing `FragmentCluster` is required.

Candidate per-pixel features:

- Time bin or seconds.
- Frequency bin or Hz.
- Resolution: `layers`, `rate`.
- Network pixel energy: `likelihood`.
- Per-detector energy proxy: `asnr**2 + a_90**2`.
- Detector balance ratio.
- Existing lag-adjusted `pixel_index` per detector.

### Edge Criteria

Start conservative:

- Pixels may connect only when their time/frequency distance is near the current `kt=2`, `kf=3` coherence neighborhood.
- Edge weight combines:
  - TF proximity.
  - Similar detector-energy ratio.
  - Compatible resolution.
  - Optional energy contrast.

Initial implementation should avoid new dependencies and use scipy sparse graph tools already available in the environment.

### Output Construction

For each weighted connected component after edge pruning:

1. Slice the selected-pixel `PixelArrays` into component pixels.
2. Recompute approximate metadata:
   - `energy = sum(likelihood)`.
   - `like_net = energy`.
   - `c_time = weighted mean time`.
   - `c_freq = weighted mean frequency`.
   - `net_rho` and `sub_net` copied or approximated from source clusters when available.
3. Create a new `Cluster`.
4. Merge resolution-level clusters into one candidate `FragmentCluster`.
5. Call the shared finalization helper to attach TD amplitudes, apply native-compatible subnet/defragmentation cuts, mark surviving pixels as core, and return a likelihood-ready `FragmentCluster`.

### Config Parameters

Add method parameters through config attributes if present, with safe defaults:

```yaml
clustering_method: "weighted_graph"
clustering:
  weighted_graph:
    time_radius_bins: 2
    freq_radius_bins: 3
    min_edge_weight: 0.25
    energy_balance_weight: 0.5
    resolution_weight: 0.25
    min_pixels: 1
```

If the config schema does not already allow arbitrary keys, pass options through `getattr` defensively or use an existing extension mechanism.

## Phase 3B MRA Methods: `mra_xxx`

The existing non-MRA methods remain behavior-preserving baselines.  New true
multi-resolution methods should use the `mra_` prefix so comparisons are
explicit and the previous code path stays intact.

Implemented methods:

- `mra_weighted_graph` — pools raw selected pixels from all WDM resolutions
    before primary clustering.  Same-resolution edges mirror the weighted-graph
    idea, while cross-resolution edges can connect overlapping or near-overlapping
    physical TF cells when detector-energy patterns and likelihoods are
    compatible.
- `mra_hdbscan` — pools raw selected pixels from all WDM resolutions, builds
        one scaled physical feature matrix, and runs HDBSCAN once over the pooled
        pixels.  This is the first adaptive-density MRA method; it is useful for
        morphology experiments but depends strongly on feature scaling and noise
        policy.

Example `mra_hdbscan` configuration:

```yaml
clustering_method: "mra_hdbscan"
clustering:
    mra_hdbscan:
        time_scale_seconds: 0.01
        freq_scale_hz: 64.0
        level_weight: 0.5
        log_energy_weight: 0.25
        detector_balance_weight: 0.5
        min_cluster_size: 2
        min_samples: null
        cluster_selection_epsilon: 0.0
        cluster_selection_method: "eom"
        noise_as_singletons: true
        min_pixels: 1
        final_defrag: false
```

Important distinction:

- Non-MRA methods cluster each resolution first, then call the native
    supercluster-style finalization path.
- `mra_xxx` methods perform primary clustering in pooled multi-resolution
    space and use MRA finalization without native `supercluster()` as the primary
    merger.

Planned naming for later variants:

- `mra_dbscan`
- `mra_optics`
- `mra_spectral_graph`

## Phase 3 Density Methods: DBSCAN, HDBSCAN, OPTICS

Packages:

- `pycwb/modules/clustering/dbscan`
- `pycwb/modules/clustering/hdbscan`
- `pycwb/modules/clustering/optics`

These methods use scikit-learn density clustering on selected pixels.  They should use the same replacement pipeline contract as connected-components and weighted-graph methods: build primary clusters from selected candidates, then return a likelihood-ready `FragmentCluster` after shared finalization.

Input features:

- Scaled time.
- Scaled frequency.
- Log energy.
- Detector balance.
- Resolution index.

Risks:

- DBSCAN parameters are sensitive to feature scaling.
- It may label too many pixels as noise unless defaults are tuned.
- HDBSCAN and OPTICS may split sparse CBC-like tracks unless `min_cluster_size`, `min_samples`, and feature scaling are tuned.
- Noise handling must be explicit: either discard noise pixels, keep them as singletons, or keep them as rejected clusters for diagnostics.

Recommendation:

- Keep defaults conservative and native-like for initial comparisons.
- Run all density methods against the same multi-injection YAML with isolated per-method catalogs.
- Use `connected_components` as the baseline for trigger count and key-statistic comparisons.

## New Test Workflow

Add a new workflow module, derived from native:

```text
pycwb/workflow/subflow/process_job_segment_clustering.py
```

Purpose:

- Keep production native workflow unchanged.
- Make clustering method selection explicit in a test YAML.

Minimal change from native:

```python
from pycwb.modules.cwb_coherence.coherence import select_pixels_single_lag
from pycwb.modules.clustering.cluster import cluster_lag_candidates

pixel_candidates_this_lag = select_pixels_single_lag(
    coherence_setup,
    lag,
    veto_windows=sub_job_seg.veto_windows,
)

fragment_cluster = cluster_lag_candidates(
    pixel_candidates_this_lag,
    method=getattr(config, "clustering_method", "connected_components"),
    config=config,
    lag_idx=lag,
    setup=supercluster_setup,
    xtalk=xtalk,
    td_inputs_cache=td_inputs_cache,
)
```

Everything after that remains identical: the likelihood loop receives `fragment_cluster` exactly as it currently receives the output of `supercluster_single_lag`.

This copy should be kept as close as possible to `process_job_segment_native.py`, with only the per-lag coherence/supercluster replacement and imports changed.

## Test YAML

Create a new sample config under:

```text
tests/sample/user_parameters_clustering_phase3.yaml
```

Base it on `tests/sample/user_parameters_injection.yaml`, with these changes:

- Use `segment_processer: pycwb.workflow.subflow.process_job_segment_clustering.process_job_segment`.
- Use Gaussian noise.
- Include multiple injections with different morphology or parameters.
- Keep runtime modest for local testing.
- Disable plotting and waveform saving by default.
- Set `clustering_method` to the method under test.

Example injection mix:

- One CBC-like injection using `IMRPhenomXPHM` or another existing supported approximant.
- One lower-amplitude CBC-like injection at a different GPS time.
- If supported by the existing injection generator, one burst-like waveform from `burst-waveform`; otherwise keep two or three CBC injections with different masses and distances.

Use deterministic seeds so test output is reproducible.

## Tests to Add for Phase 3

### Unit Tests

Suggested file:

```text
tests/clustering/test_cluster_entrypoint.py
```

Tests:

1. Pixel-selection helper returns one raw candidate dictionary per resolution.
2. Entry point with `method="connected_components"` returns a single `FragmentCluster` or `None` with the likelihood-facing contract.
3. Entry point rejects unknown method with clear error.
4. Weighted-graph backend returns a valid likelihood-facing `FragmentCluster` for small synthetic candidates.
5. DBSCAN, HDBSCAN, and OPTICS return valid likelihood-facing `FragmentCluster` objects for small synthetic candidates.
6. Backends preserve total selected pixel count unless explicitly configured to drop noise pixels.
7. Empty input is handled.
8. Shared finalization attaches TD amplitudes and marks surviving pixels as core.

These tests should use small synthetic selected-pixel candidate dictionaries and small `PixelArrays`, not full cWB data.

### Integration Test

Suggested file:

```text
tests/clustering/test_phase3_workflow.py
```

Test behavior:

- Load `tests/sample/user_parameters_clustering_phase3.yaml`.
- Use `Config.load_from_yaml` and `create_job_segment_from_config`.
- Run one job segment with `process_job_segment_clustering.process_job_segment`.
- Assert the workflow completes and writes catalog/progress output.
- Assert the workflow path does not call `coherence_single_lag` or `supercluster_single_lag` directly in the replacement branch.

This test may be marked slow if runtime is significant.

## Comparison Checks

For initial validation, run the same YAML with:

1. `clustering_method: "connected_components"`.
2. `clustering_method: "weighted_graph"`.
3. `clustering_method: "dbscan"`.
4. `clustering_method: "hdbscan"`.
5. `clustering_method: "optics"`.

Compare:

- Number of selected pixels before clustering.
- Number of primary clusters produced by each method.
- Number of final likelihood candidates after shared finalization.
- Number of accepted likelihood events.
- Trigger IDs and approximate event times/frequencies.
- Key catalog statistics: `rho`, `rho_alt`, `likelihood`, `coherent_energy`, `packet_norm`, `penalty`, `net_cc`, `sky_cc`, `subnet_cc`, `q_veto`, `q_factor`, `n_pixels_total`, `gps_time`, `freq_low_L1`, `freq_high_L1`, `duration_L1`, `bandwidth_L1`, `ra`, `dec`.
- Runtime per lag.

The connected-components compatibility backend should match native behavior exactly or very closely because it is the baseline replacement.  Other methods may differ, but deviations should be explainable by cluster grouping, noise handling, or finalization cuts.

## Implementation Order After Approval

1. Add `select_pixels_single_lag` by extracting the pixel-selection portion of `coherence_single_lag`.
2. Keep `coherence_single_lag` as a compatibility wrapper that calls `select_pixels_single_lag` plus native `cluster_pixels`.
3. Add `pycwb/modules/clustering/pipeline.py` with shared TD attachment and finalization helpers derived from `supercluster_single_lag`.
4. Add `cluster_lag_candidates` in `pycwb/modules/clustering/cluster.py` as the new replacement entry point.
5. Convert the connected-components backend from identity pass to native-compatible replacement backend.
6. Convert weighted-graph, DBSCAN, HDBSCAN, and OPTICS backends to consume selected-pixel candidates and call shared finalization.
7. Update `process_job_segment_clustering.py` so the per-lag workflow calls `select_pixels_single_lag` and `cluster_lag_candidates`, not `coherence_single_lag` or `supercluster_single_lag`.
8. Add unit tests for pixel selection, entry point dispatch, each backend, TD attachment, and finalization using synthetic inputs.
9. Add or update per-method YAML run directories under `tests/clustering/runs/<method>/`.
10. Run focused unit tests.
11. Run the phase-3 workflow for connected-components, weighted-graph, DBSCAN, HDBSCAN, and OPTICS if local runtime and dependencies allow.
12. Run the comparison script and record key-statistic differences.
13. Update this plan or the investigation document with results.

## Deferred Items

- Large-catalog Phase 1 evaluation.
- Phase 2 tensor extraction tests on real catalogs.
- Learned embedding models.
- Any change to production `process_job_segment_native.py`.
- Removing the native `coherence_single_lag` and `supercluster_single_lag` APIs; keep them for production compatibility while the replacement path is validated.

## Success Criteria

The implementation is successful when:

- Existing native workflow remains untouched.
- New clustering entry point replaces the per-lag `coherence_single_lag` + `supercluster_single_lag` pair in the test workflow.
- Pixel selection is reusable independently of clustering.
- New clustering entry point accepts per-resolution selected-pixel candidates and returns the same likelihood-facing `FragmentCluster` shape currently returned by `supercluster_single_lag`.
- The connected-components compatibility backend is behavior-preserving against the native path.
- Weighted-graph, DBSCAN, HDBSCAN, and OPTICS backends can run on synthetic candidates and the sample workflow.
- A YAML-driven phase-3 workflow can select the clustering method.
- Per-method run directories produce separate `catalog.parquet` files.
- The comparison script reports key-statistic differences across methods.
- Phase 1/2 validation scaffolding exists but does not require large data tests.