# Clustering Method Investigation for PycWB

## Purpose

This note investigates how to add new clustering methods to PycWB and how to test which method works better. It starts from the physical information already available in the current native cWB workflow, then compares two families of approaches:

1. Informed methods, where a waveform model or simulation family is available.
2. Uninformed methods, where only detector data, cWB reconstruction products, and generic gravitational-wave constraints are used.

The preliminary note in `PycWB_ML.md` correctly points toward embeddings and unsupervised learning. This document makes that idea more concrete for the current PycWB implementation.

## Where Clustering Can Enter the Current Pipeline

The native workflow in `pycwb/workflow/subflow/process_job_segment_native.py` has the following per-lag sequence:

1. `coherence_single_lag`: select significant WDM time-frequency pixels and build fragment clusters.
2. `supercluster_single_lag`: merge fragments across resolutions, extract time-delay amplitudes, apply subnet veto, and prepare clusters for likelihood.
3. `likelihood`: scan sky positions, reconstruct coherent signal and null energy, apply threshold cuts, and fill event-level statistics.
4. `Event.output_py` and `Trigger.from_event`: save the accepted event summary to catalog form.

This creates three useful insertion points:

| Stage | Object | Information available | Suitable clustering role |
| --- | --- | --- | --- |
| After coherence | `FragmentCluster`, `Cluster`, `PixelArrays` | WDM pixel geometry, per-detector pixel energy, subrho/subnet estimates | Replace or augment connected-component pixel grouping |
| After supercluster | `Cluster` with `td_amp` | Multi-resolution clusters, time-delay amplitude vectors, subnet sky consistency | Merge/split candidate clusters before expensive likelihood |
| After likelihood | `Cluster`, `Event`, `Trigger`, `SkyMapStatistics` | Coherent energy, null energy, sky map, reconstructed waveforms, catalog fields | Event morphology clustering, glitch families, ranking refinement |

The lowest-risk first prototype is post-likelihood event clustering, because it does not change detection behavior. The most scientifically powerful but riskiest prototype is replacing fragment clustering before likelihood, because it can change what becomes a trigger.

## Physical Information Available Before Likelihood

### WDM Time-Frequency Maps

`setup_coherence` builds one WDM transform per detector and resolution. For each resolution, the setup stores:

- `tf_maps`: per-detector WDM time-frequency maps after `max_energy` projection.
- `Eo`: pixel energy threshold from the configured black-pixel probability.
- `level`, `layers`, `rate`: resolution metadata.
- `pattern`, `select_subrho`, `select_subnet`, `segEdge`: configuration controlling selection.

The physical content at this point is mostly excess-power morphology:

- Time-frequency position of each selected pixel.
- Resolution level, rate, and number of WDM layers.
- Per-detector pixel energy after lag alignment.
- Total network pixel energy summed over detectors.
- Frequency band and segment-time constraints.

### Network Pixel Candidate Payload

`select_network_pixels` returns a dictionary with the fields most useful for alternative pixel clustering:

- `mask`: selected TF-pixel mask with shape `(n_freq, n_time)`.
- `time`, `frequency`: selected pixel coordinates.
- `energy`: summed network energy per pixel.
- `pix_det_energy`: per-detector pixel energy with shape `(n_pix, n_ifo)`.
- `pix_det_index`: detector-specific flat TF indices, already including lag shifts.
- `rate`, `layers`, `start`, `stop`, `f_low`, `f_high`: physical coordinate metadata.

This is the cleanest input for new pre-likelihood clustering methods. It has physical geometry and detector consistency, but not full sky reconstruction.

### Fragment Cluster and PixelArrays Fields

`cluster_pixels` currently uses connected components in the TF plane with rectangular adjacency (`kt`, `kf`). It converts candidates into `Cluster` objects backed by `PixelArrays`.

Useful `PixelArrays` fields before likelihood include:

- Per-pixel scalars: `time`, `frequency`, `layers`, `rate`, `core`, `likelihood`, `null`.
- Per-IFO arrays: `noise_rms`, `wave`, `w_90`, `asnr`, `a_90`, `pixel_index`.
- Time-delay amplitudes after supercluster: `td_amp_flat`, `td_amp_offsets`.

Before likelihood, `likelihood` is effectively the selected-pixel energy from coherence; `asnr` is initialized from per-detector pixel energy; `null`, `wave`, `w_90`, and `a_90` are not yet fully physical reconstruction outputs.

### Coherence-Level ClusterMeta

The first cluster metadata are approximate but useful:

- `energy`: summed selected-pixel energy.
- `like_net`: initially same energy proxy.
- `sub_net`: subnetwork statistic estimate.
- `net_rho`: subrho estimate.
- `c_time`, `c_freq`: simple cluster centroid.

These are fast features for a pre-likelihood gate or merge/split model.

## Physical Information Added by Supercluster

`supercluster_single_lag` adds important network-geometry information:

- Merges clusters across resolutions using TF proximity and detector time-index consistency.
- Extracts per-pixel, per-detector time-delay amplitude vectors from `td_inputs_cache`.
- Computes sky-delay and antenna-pattern arrays: `ml`, `FP`, `FX`.
- Applies subnet cuts using sky consistency, subnetwork energy, and MRA statistics.
- Marks surviving pixels as core for likelihood.

After this stage, a cluster carries richer information than plain TF geometry:

- Multi-resolution support.
- Per-detector time-delay amplitude vectors.
- Whether the event can be made coherent for at least one sky direction.
- Subnetwork robustness: whether one detector dominates or the event survives detector subsets.

This is the best stage for physically informed merge/split decisions that should still be cheaper than full likelihood.

## Physical Information Available After Likelihood

Likelihood is where the candidate becomes a physically reconstructed event. It scans the sky, reconstructs the coherent signal, computes residual/null energy, and fills `ClusterMeta`, `PixelArrays`, `SkyMapStatistics`, and finally `Event`/`Trigger` fields.

### ClusterMeta After Likelihood

Important network statistics include:

- `net_ecor`: packet coherent energy.
- `norm_cor`: normalized coherent energy.
- `like_net`: waveform likelihood, sum of reconstructed signal energy.
- `energy`: reconstructed data energy.
- `net_null`: null energy with Gaussian correction.
- `net_ed`: residual energy disbalance.
- `like_sky`: pixel-domain sky likelihood.
- `energy_sky`: TF-domain data energy.
- `net_cc`, `sky_cc`: network correlation coefficients.
- `sub_net`, `sub_net2`: subnetwork consistency statistics.
- `net_rho`, `net_rho2`: coherent SNR variants.
- `g_net`, `a_net`, `i_net`: network sensitivity, alignment, and index.
- `theta`, `phi`, `l_max`: best-fit sky location.
- `c_time`, `c_freq`: likelihood/reconstruction-weighted time and frequency.
- `ndof`, `sky_size`, `sky_chi2`, `g_noise`: effective pixel count and fit quality.
- `wave_snr`, `signal_snr`, `cross_snr`, `null_energy`: per-IFO waveform-domain energies.
- `signal_energy_physical`: physical strain energy for hrss.

These are excellent tabular features for fast clustering and ranking experiments.

### PixelArrays After Likelihood

Likelihood updates the pixel arrays with physical reconstruction outputs:

- `core`: final selected coherent pixels.
- `likelihood`: per-pixel coherent likelihood after xtalk correction.
- `null`: per-pixel null/residual contribution.
- `wave`, `w_90`: time-delayed data amplitudes.
- `asnr`, `a_90`: reconstructed signal amplitudes.
- `noise_rms`: per-pixel detector noise floor.

This supports image-like event tensors, for example channels such as:

- log network energy.
- coherent likelihood.
- null energy.
- coherence ratio `likelihood / (likelihood + null + eps)`.
- per-IFO reconstructed signal energy.
- detector imbalance measures.

The helper `Cluster.get_sparse_map(key=...)` can already turn `likelihood` or `null` into aligned 2D maps across resolutions.

### SkyMapStatistics

`likelihood` also returns `SkyMapStatistics`, including arrays over sky positions:

- `nLikelihood`, `nNullEnergy`, `nCorrEnergy`, `nCorrelation`.
- `nSkyStat`, `nDisbalance`, `nNetIndex`.
- `nAntennaPrior`, `nAlignment`.
- `nEllipticity`, `nPolarisation`.
- `nProbability`: normalized sky probability map.

These are valuable for clustering by localization structure, for example compact vs multi-modal sky solutions or events only accepted in poorly aligned network regions.

### Event and Trigger Catalog Features

`Event.output_py` and `Trigger.from_event` expose a catalog-ready summary. Particularly useful clustering features are:

- Network scalar features: `rho`, `rho_alt`, `net_cc`, `sky_cc`, `subnet_cc`, `subnet_cc2`, `likelihood`, `coherent_energy`, `net_null`, `net_energy`, `like_sky`, `energy_sky`, `packet_norm`, `penalty`, `strain`.
- Geometry and size: `n_pixels_total`, `n_pixels_core`, `n_pixels_positive`, `sky_size`, `duration`, `bandwidth`, `central_freq`, `freq_low`, `freq_high`.
- Detector balance: per-IFO `hrss`, `noise_rms`, `data_energy`, `signal_energy`, `cross_energy`, `null_energy`, `residual_energy`.
- Sky geometry: `ra`, `dec`, `theta`, `phi`, `psi`, `iota`, `network_sensitivity`, `network_alignment_factor`, antenna responses `fp`, `fx`.
- Post-processing: `q_veto`, `q_factor`, `ifar` when available.
- Simulation-only labels and parameters: `injection.name`, `injection.approximant`, `target_snr`, `hrss`, sky position, per-IFO injected SNR, overlap SNR, and complete waveform-specific JSON parameters.

This catalog layer is ideal for first comparisons because it is stable, cheap to load from Parquet, and does not require re-running cWB for every clustering experiment.

## Candidate Feature Representations

### A. Tabular Event Features

Use one row per event. Combine robust scalar features from `Trigger` and normalized per-IFO summaries:

- Signal strength: `rho`, `likelihood`, `coherent_energy`, `strain`.
- Coherence quality: `net_cc`, `sky_cc`, `subnet_cc`, `subnet_cc2`.
- Residual quality: `net_null`, `penalty`, `net_energy_disb`, `q_veto`.
- Morphology: log duration, log bandwidth, log central frequency, pixel counts, TF area, chirp features.
- Detector balance: ratios of per-IFO energy to total energy, max/min detector SNR, residual fractions.
- Sky/network geometry: network alignment, sensitivity, antenna response pattern, sky area statistics.

This is easy to prototype with scikit-learn and should be the baseline.

### B. Pixel-Cloud Features

Represent each event as a variable-size set of pixels:

`(time, frequency, resolution, coherent_likelihood, null, per-IFO energy, core flag)`

Possible encoders:

- Hand-crafted distribution summaries: weighted moments, skewness, TF covariance, occupancy, number of islands.
- Set encoders: DeepSets, PointNet-style MLPs, graph neural networks.
- Density-based clustering directly on pixels to split/merge event candidates.

This representation preserves more morphology than tabular features but needs careful normalization across WDM resolutions.

### C. Image/Tensor Features

Convert a cluster into fixed-size event-centric tensors:

`X in R^(channels x time x frequency)`

Recommended channels:

- Coherent likelihood map.
- Null energy map.
- Coherence ratio map.
- Network energy map.
- Per-detector energy maps or detector-imbalance maps.
- Optional sky-localization summary channels if saved.

This supports CNNs, ConvNeXt, Swin/ViT, masked autoencoders, and contrastive learning. It is the natural path for morphology discovery, but it requires storing intermediate cluster maps, not only catalog summaries.

### D. Waveform Features

Use reconstructed whitened or physical waveforms from post-processing:

- Time-domain reconstructed signal per detector.
- Whitened data and residual waveforms.
- Frequency-domain summaries and Q-veto outputs.
- Matched overlap with injections when available.

This is useful for model-informed methods and for distinguishing glitches with similar TF maps but different coherent reconstruction quality.

### E. Sky-Map Features

Use `SkyMapStatistics` arrays or compressed descriptors:

- Entropy of `nProbability`.
- Area above probability thresholds.
- Number of separated sky modes.
- Best-vs-second-best sky statistic gap.
- Correlation between sky probability and antenna prior/alignment.

This can identify events whose apparent coherence is driven by network degeneracy rather than robust morphology.

## Uninformed Methods

Uninformed methods do not assume a specific waveform family. They use detector data, cWB coherence, and generic GW constraints such as time-of-flight consistency, TF compactness, coherent energy, null energy, and detector balance.

### 1. Current Baseline: Connected Components plus Supercluster

Current coherence clustering groups selected pixels by local TF adjacency. Supercluster then merges across resolutions and applies subnet consistency.

Strengths:

- Fast and physically transparent.
- No training required.
- Easy to compare to legacy cWB behavior.

Weaknesses:

- Adjacency thresholds are local and hand-tuned.
- Does not learn common glitch morphologies.
- Can over-merge nearby structures or split extended morphologies.

Use as the required baseline for all comparisons.

### 2. Density-Based Clustering on Pixel Clouds

Methods: DBSCAN, HDBSCAN, OPTICS, Gaussian mixture variants.

Feature space:

- Time, frequency, log energy, resolution, detector imbalance, local coherence ratio.

Why it fits:

- Burst events have irregular TF shapes and variable pixel counts.
- Density clustering can separate islands without choosing the number of clusters.
- HDBSCAN can label noise pixels explicitly.

Risks:

- Distance metric must respect WDM resolution.
- HDBSCAN is not in the current dependencies; DBSCAN/OPTICS are available through scikit-learn.
- Pure TF density may ignore sky coherence unless supercluster features are included.

Recommendation:

- Prototype DBSCAN/OPTICS first using scikit-learn.
- If promising, add optional HDBSCAN for better variable-density behavior.

### 3. Graph-Based Pixel Clustering

Methods: weighted connected components, spectral clustering, Leiden/Louvain community detection, graph cuts.

Nodes are pixels. Edges encode:

- TF proximity across resolutions.
- Similar per-detector energy ratios.
- Compatible detector time-delay indices.
- Similar coherent/null likelihood after likelihood is available.

Why it fits:

- The existing connected-component method is already an unweighted graph. This is the most natural upgrade.
- Edge weights can include physics without a waveform model.

Risks:

- Spectral methods can be expensive for large clusters.
- Community-detection dependencies are not currently listed.

Recommendation:

- Start with a weighted graph using scipy sparse connected components and edge pruning.
- Treat it as a drop-in replacement for `cluster_pixels` before trying heavier graph packages.

### 4. Tabular Unsupervised Event Clustering

Methods: standardization + PCA/UMAP + k-means/GMM/agglomerative/DBSCAN/OPTICS.

Feature source:

- Catalog `Trigger` fields and optionally saved `ClusterMeta` extras.

Use cases:

- Glitch family discovery.
- Background trigger taxonomy.
- Quick comparison of candidate feature sets.
- Build interpretable labels for later supervised training.

Strengths:

- Fast and immediately feasible with existing dependencies.
- Works on large catalogs.
- Easy to evaluate with injections and background lags.

Weaknesses:

- Loses detailed TF morphology.
- Sensitive to scaling and correlated features.

Recommendation:

- First production-quality experiment should use this layer.
- Use robust scaling, log transforms for positive energy/count features, and explicit detector-balance ratios.

### 5. Contrastive Representation Learning

Methods: SimCLR/NT-Xent, triplet loss, BYOL/Barlow Twins variants.

Positive pairs can be generated by:

- Same injection in different noise realizations.
- Same event under mild detector dropout.
- Same event with channel masking.
- Same cluster represented at slightly different crop boundaries.

Negative pairs are different triggers, preferably matched by broad SNR/frequency bins to avoid trivial separation.

Strengths:

- Best fit for discovering morphology without labels.
- Can learn invariances that are hard to hand-code.

Risks:

- Bad augmentations can erase physical differences.
- Needs many events and careful validation.
- Requires a training stack beyond current core dependencies if using PyTorch; JAX/Flax could also be used but is not currently part of PycWB.

Recommendation:

- Use contrastive learning after tabular baselines define evaluation metrics.
- Start with image/tensor maps around accepted triggers, not raw full segments.

### 6. Autoencoder or Masked Reconstruction Embeddings

Methods: convolutional autoencoder, variational autoencoder, masked TF-patch prediction.

Strengths:

- Useful pretraining when labels are scarce.
- Reconstruction error can flag anomalies.

Weaknesses:

- May learn detector noise more than event identity.
- Embeddings are often less cluster-friendly than contrastive embeddings.

Recommendation:

- Treat as auxiliary or warm-up, not the main ranking method.

### 7. Self-Organizing Maps and Prototype Learning

Methods: SOM, neural gas, k-medoids on learned embeddings.

Strengths:

- Useful for human inspection of glitch/event families.
- Produces prototype events that can be plotted and reviewed.

Weaknesses:

- Less scalable and less statistically principled than modern density/contrastive methods.

Recommendation:

- Use for visualization and taxonomy, not as the primary clustering engine.

## Informed Methods

Informed methods use a waveform model, simulation family, or physically parameterized hypothesis. In PycWB this can come from injection metadata, generated waveforms, reconstructed injection waveforms, or external model families such as CBC approximants.

### 1. Template or Model-Feature Clustering

Given a waveform model, generate expected descriptors and compare each event to that model family.

Possible descriptors:

- Expected chirp track: time-frequency slope, curvature, merger time, chirp mass proxy.
- Expected duration-bandwidth relation.
- Expected detector arrival-time pattern for a sky position.
- Expected amplitude ratios from antenna responses.
- Expected polarization behavior.

Cluster on residuals:

- Observed minus model central time/frequency track.
- Observed detector energy ratios minus predicted antenna ratios.
- Reconstructed waveform overlap or mismatch.
- Null-energy residual after projecting onto model-consistent subspace.

This turns clustering into grouping by physical mismatch rather than raw morphology.

### 2. Simulation-Supervised or Semi-Supervised Embeddings

Use injected events as labels or weak labels:

- Waveform family: WNB, sine-Gaussian, CBC approximant, cosmic-string-like bursts, etc.
- Parameters: mass, spin, Q, central frequency, bandwidth, duration, hrss, distance.
- Recovery quality: overlap SNR, recovered SNR, injection residual, sky error.

Approaches:

- Train a classifier on injection families, then use penultimate-layer embeddings for clustering real/background events.
- Train metric learning with positives from the same injection family/parameter neighborhood.
- Train regression to physical parameters, then cluster in predicted-parameter plus residual space.

Strengths:

- High interpretability for known signal families.
- Directly tests whether clusters align with physical waveform classes.

Risks:

- Can bias discovery toward simulated families.
- May classify glitches as nearest known waveform if out-of-distribution handling is weak.

Recommendation:

- Keep an explicit unknown/anomaly branch.
- Evaluate on injections withheld by family, not only withheld by random split.

### 3. Matched-Filter or Overlap-Augmented Features

For known models, compute overlaps between reconstructed waveforms and model waveforms.

Available PycWB-related quantities:

- Reconstructed whitened waveforms from post-processing.
- Injected/recovered overlap fields when simulations are available: `iSNR`, `oSNR`, `ioSNR`, stored in `InjectionParams` as `snr_sq`, `rec_snr_sq`, `overlap_snr`.
- Effective distance and per-detector antenna responses.

Useful features:

- Match or mismatch to best template.
- Residual energy after subtracting best template.
- Best-fit model parameters and parameter uncertainty.
- Difference between cWB coherent reconstruction and model reconstruction.

This is especially useful for CBC-like or sine-Gaussian families.

### 4. Physics-Constrained Mixture Models

Instead of generic Gaussian mixtures over arbitrary features, define mixture components with physically meaningful axes:

- Short high-frequency bursts.
- Long low-frequency excess power.
- Chirping tracks.
- Detector-local glitches with poor network coherence.
- Coherent sky-localized candidates with low null energy.

Each component can have priors on duration, bandwidth, coherence, null fraction, detector balance, and sky localization quality.

Strengths:

- Interpretable and relatively data-efficient.
- Works with tabular event features.

Risks:

- Priors may be subjective.
- Can miss unexpected morphologies.

Recommendation:

- Use as a benchmark against uninformed density clustering, not as the only method.

### 5. Model-Aware Contrastive Learning

Positive pairs can be defined by physics instead of only augmentations:

- Same injection waveform with different noise.
- Same waveform family with nearby physical parameters.
- Same sky location but different amplitude.
- Same CBC intrinsic parameters but different extrinsic parameters.

Hard negatives:

- Similar duration/frequency but different waveform family.
- Same SNR but different coherence/null behavior.
- Glitches that overlap in TF morphology with simulated signals.

This is probably the strongest long-term informed approach, because it can learn both morphology and physically meaningful invariances.

## Methods That Mix Informed and Uninformed Information

### Anomaly Detection Around Known Signals

Train a model on known injected families and background glitches, then score new triggers by distance to known manifolds.

Methods:

- Isolation Forest or Local Outlier Factor on tabular features.
- One-class SVM on embeddings.
- Normalizing flows or density models on learned embeddings.
- Mahalanobis distance to family centroids.

Use case:

- Identify rare coherent events that do not look like common glitches or known injections.

### Hybrid Score for Ranking

Do not replace cWB likelihood. Add clustering-derived context:

`score = cWB statistic + anomaly/context term + family-specific consistency term`

Examples:

- Penalize clusters near known glitch families.
- Promote clusters far from background but close to plausible injected families.
- Flag events with high cWB likelihood but poor model consistency for review.

## Evaluation Strategy

### Ground Truth and Labels

Use multiple partial labels, because there is no single perfect ground truth:

- Injection metadata: waveform family and physical parameters.
- Recovery success: detected/not detected, reconstructed SNR, overlap SNR.
- Background lags: should mostly form noise/glitch clusters.
- Hardware/software injection campaigns if available.
- Human-vetted glitch families when available.

### Metrics for Event-Level Clustering

- Injection family separation: adjusted Rand index, normalized mutual information, purity.
- Parameter smoothness: nearby embedding distance should correlate with nearby waveform parameters.
- Background compactness: common background/glitch populations should form stable groups.
- Outlier quality: rare injections or held-out families should be high-anomaly but not simply high-SNR.
- Ranking impact: change in sensitive distance/volume at fixed false-alarm rate.
- Stability: cluster assignments should be stable across random seeds, nearby thresholds, and data splits.

### Metrics for Pre-Likelihood Pixel Clustering

- Trigger recovery efficiency vs FAR.
- Number of candidate clusters passed to likelihood.
- Wall time per lag.
- Memory use per lag.
- Agreement with baseline cWB for known regression segments.
- Over-merge/split rate for injections with known time-frequency support.

### Ablation Tests

Run the same clustering algorithm with feature groups removed:

- TF geometry only.
- Add detector energy balance.
- Add coherence/null statistics.
- Add sky-map descriptors.
- Add waveform reconstruction descriptors.
- Add injection/model features.

This will show whether the improvement comes from genuine physical information or from easy proxies such as SNR.

## Recommended Development Plan

### Phase 1: Catalog-Level Baseline

Goal: quickly learn whether clustering adds useful structure without touching detection.

Steps:

1. Load `Trigger` catalog tables.
2. Build a robust tabular feature matrix.
3. Apply log transforms to positive energy/count/duration/frequency fields.
4. Normalize per run or per detector network where needed.
5. Compare PCA, UMAP if available, k-means/GMM/agglomerative/DBSCAN/OPTICS.
6. Evaluate with injections, background lags, and recovery metrics.

Expected output:

- A baseline clustering report.
- Feature importance or cluster-profile tables.
- Plots of representative events per cluster.

### Phase 2: Save Event-Centric Pixel Tensors

Goal: make morphology learning possible.

Steps:

1. Add a feature extraction helper that converts accepted `Cluster.pixel_arrays` to aligned maps.
2. Save compressed tensors or sparse maps alongside trigger artifacts.
3. Include channels for likelihood, null, coherence ratio, and detector balance.
4. Train simple CNN/autoencoder/contrastive prototypes outside the main pipeline.

Expected output:

- Morphology embeddings.
- Comparison with tabular clustering.
- Representative TF maps for each cluster.

### Phase 3: Pre-Likelihood Alternative Pixel Clustering

Goal: test whether a new method improves candidate construction before likelihood.

Steps:

1. Implement a plugin-style clustering interface around `select_network_pixels` output.
2. Keep current connected components as `method="connected_components"`.
3. Add `method="dbscan"` or `method="weighted_graph"`.
4. Preserve `FragmentCluster`/`Cluster`/`PixelArrays` output contracts.
5. Compare recovery efficiency, FAR, runtime, and regression agreement.

Expected output:

- A safe switchable clustering backend.
- Quantitative comparison against the current baseline.

### Phase 4: Informed Waveform-Model Experiments

Goal: determine whether waveform-informed clustering improves interpretation or ranking.

Steps:

1. Use injection metadata and waveform reconstruction products.
2. Train or compute model-consistency features.
3. Cluster in combined space: cWB statistics + morphology embedding + model residuals.
4. Hold out entire waveform families to test discovery behavior.

Expected output:

- Family-aware embeddings.
- Unknown-family/anomaly tests.
- Ranking comparison at fixed background rate.

## Recommended First Methods to Test

### Fast Baseline

- Input: catalog `Trigger` features.
- Methods: robust scaling + PCA + GMM, agglomerative clustering, DBSCAN/OPTICS.
- Purpose: understand existing event populations and feature usefulness.
- Dependencies: already available through scikit-learn.

### Best Pre-Likelihood Upgrade Candidate

- Input: `select_network_pixels` output.
- Method: weighted graph clustering.
- Edge weights: TF proximity, resolution compatibility, detector energy-ratio similarity, lag/time-index consistency.
- Purpose: improve over binary rectangular adjacency while staying close to current cWB logic.
- Dependencies: scipy sparse graph tools already available.

### Best Long-Term Uninformed Method

- Input: event-centric TF tensors from post-likelihood clusters.
- Method: contrastive embedding learning with detector dropout, channel masking, and mild cropping.
- Purpose: morphology discovery and glitch taxonomy.
- Dependencies: likely needs a training framework decision.

### Best Long-Term Informed Method

- Input: TF tensors plus injection/model residual features.
- Method: model-aware contrastive learning or semi-supervised metric learning.
- Purpose: separate known waveform families while keeping sensitivity to unknown morphologies.
- Dependencies: waveform generation and training framework; PycWB already has `lalsuite`, `gwpy`, `burst-waveform`, and optional `pycbc` support.

## Important Pitfalls

- Do not cluster only on SNR. High-SNR events will separate trivially but not meaningfully.
- Do not use injection labels in validation splits that leak the same waveform parameters into train and test.
- Do not use strong time/frequency warping as augmentation unless the target physics is invariant under it.
- Do not replace cWB likelihood with ML clustering in the first iteration. Use clustering as an auxiliary diagnostic until it is validated.
- Preserve detector-network context. A two-detector event and a three-detector event can have different feature distributions.
- Watch selection bias: post-likelihood triggers are already filtered by cWB cuts, so they do not represent all possible excess-power clusters.

## Concrete Implementation Notes

### Feature Extractor API

A practical module layout could be:

```text
pycwb/modules/clustering/
    __init__.py
    features.py          # Trigger/Cluster/PixelArrays -> feature matrices or tensors
    methods.py           # sklearn wrappers and graph clustering methods
    evaluate.py          # clustering metrics and injection/background evaluation
    pixel_clustering.py  # optional replacement for cluster_pixels
```

Suggested feature functions:

```python
trigger_to_tabular_features(trigger) -> dict[str, float]
cluster_to_pixel_table(cluster) -> pandas.DataFrame
cluster_to_tf_tensor(cluster, channels, shape) -> np.ndarray
skymap_to_descriptors(skymap_statistics) -> dict[str, float]
```

### Pixel-Clustering Contract

Any replacement for `cluster_pixels` should return the same type:

```python
FragmentCluster(
    rate=...,
    start=...,
    stop=...,
    f_low=...,
    f_high=...,
    clusters=[Cluster(pixel_arrays=..., cluster_meta=...), ...],
)
```

This keeps `supercluster_single_lag` and `likelihood` unchanged.

### Minimum Saved Artifacts

For post-likelihood ML, catalog rows alone are enough for Phase 1. For tensor/embedding work, save at least:

- Trigger ID and bookkeeping: job, trial, lag, cluster ID.
- Aligned sparse TF maps for `likelihood` and `null`.
- Optional per-IFO energy maps.
- ClusterMeta scalar statistics.
- Injection metadata when available.

## Bottom Line

The current PycWB pipeline already exposes enough physical information for several clustering strategies. The most practical path is incremental:

1. Start with catalog-level tabular clustering to establish baselines.
2. Add saved post-likelihood TF tensors for morphology embeddings.
3. Test a weighted-graph or DBSCAN/OPTICS replacement for pre-likelihood pixel clustering.
4. Add waveform-informed metric learning only after the uninformed baselines are understood.

This keeps the physics-driven cWB core intact while making room for new clustering methods that can be compared quantitatively.