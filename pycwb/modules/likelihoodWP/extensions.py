"""Pure building blocks for optional likelihood features and cuts.

The likelihood kernels deliberately remain unaware of individual derived
statistics.  This module provides metadata registries, explicit numerical
products, and flat dispatchers called at fixed points in the likelihood flow.

The default configuration enables no optional features or cuts.  This matters
for production searches with millions of background triggers: the O(N log N)
HPD sort and all extra persisted scalars are paid for only when requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Mapping, Sequence

import numpy as np

from .sky_mask import compute_sky_valid_indices, sky_mask_requires_event_time

logger = logging.getLogger(__name__)

_FULL_SKY_DEG2 = 4.0 * np.pi * (180.0 / np.pi) ** 2
_LEGACY_SKY_LEVELS = tuple(i / 10.0 for i in range(1, 10))


@dataclass(frozen=True)
class FeatureSpec:
    """Cost, storage, and shared-product metadata for one output feature."""

    name: str
    cost: str
    purpose: str
    required_products: tuple[str, ...] = ()
    destination: str = "likelihood_feature_sidecar"
    version: int = 1
    heavy: bool = False
    columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class CutSpec:
    """Stage, cost, and shared-product metadata for one optional cut."""

    name: str
    stage: str
    cost: str
    required_products: tuple[str, ...] = ()
    version: int = 1


@dataclass(frozen=True)
class CutResult:
    passed: bool
    reason: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LikelihoodExtensionPlan:
    feature_names: tuple[str, ...]
    post_sky_cut_names: tuple[str, ...]
    post_reconstruction_cut_names: tuple[str, ...]
    feature_failure: str
    allow_heavy_features: bool
    required_products: tuple[str, ...] = ()


def _validate_level(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return value


def _configured_sky_levels(config) -> tuple[float, ...]:
    levels = getattr(config, "likelihood_sky_levels", (0.5, 0.9))
    normalized = tuple(
        sorted({_validate_level(v, "likelihood_sky_levels") for v in levels})
    )
    if not normalized:
        raise ValueError("likelihood_sky_levels must contain at least one level")
    return normalized


def validate_sky_temperature(value: float) -> float:
    """Return a positive finite softmax temperature."""
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("likelihood_sky_temperature must be positive and finite")
    return value


def build_sky_probability(
    sky_statistic: Sequence[float],
    sky_valid_indices: Sequence[int],
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize localization probability on positive evaluated directions."""
    statistic = np.asarray(sky_statistic, dtype=np.float64)
    valid = np.asarray(sky_valid_indices, dtype=np.int64)
    valid = valid[(valid >= 0) & (valid < len(statistic))]
    keep = np.isfinite(statistic[valid]) & (statistic[valid] > 0.0)
    evaluated = np.unique(valid[keep])
    if evaluated.size == 0:
        raise ValueError("No positive finite sky statistics are available")

    selected = statistic[evaluated]
    shifted = (selected - np.max(selected)) / validate_sky_temperature(temperature)
    weights = np.exp(shifted)
    norm = float(np.sum(weights))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Sky probability normalization is not finite")

    probability = np.zeros(statistic.shape, dtype=np.float32)
    probability[evaluated] = (weights / norm).astype(np.float32)
    return probability, evaluated


def rank_sky_probability(
    sky_probability: Sequence[float],
    evaluated_sky_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Sort evaluated directions once for all requested HPD products."""
    probability = np.asarray(sky_probability)
    evaluated = np.asarray(evaluated_sky_indices, dtype=np.int64)
    order = np.argsort(-probability[evaluated], kind="stable")
    ranked = evaluated[order]
    cumulative = np.cumsum(probability[ranked], dtype=np.float64)
    return ranked, cumulative


def hpd_region(
    level: float,
    ranked_sky_indices: Sequence[int],
    ranked_cumulative_probability: Sequence[float],
) -> np.ndarray:
    """Return the smallest discrete highest-posterior-density region."""
    level = _validate_level(level, "credible level")
    ranked = np.asarray(ranked_sky_indices, dtype=np.int64)
    cumulative = np.asarray(ranked_cumulative_probability, dtype=np.float64)
    stop = int(np.searchsorted(cumulative, level, side="left")) + 1
    return ranked[: min(stop, len(ranked))]


def sky_pixel_area_deg2(n_sky: int) -> float:
    """Equal-area sky pixel size for the active full HEALPix grid."""
    if n_sky <= 0:
        raise ValueError("Cannot calculate a sky area for an empty sky grid")
    return float(_FULL_SKY_DEG2 / n_sky)


def hpd_area_deg2(
    level: float,
    pixel_area_deg2: float,
    ranked_sky_indices: Sequence[int],
    ranked_cumulative_probability: Sequence[float],
) -> float:
    return float(
        len(hpd_region(level, ranked_sky_indices, ranked_cumulative_probability))
        * pixel_area_deg2
    )


def compute_sky_area_feature(
    config,
    *,
    pixel_area_deg2: float,
    ranked_sky_indices: Sequence[int],
    ranked_cumulative_probability: Sequence[float],
) -> Mapping[str, Any]:
    result: dict[str, Any] = {
        "sky_pixel_area_deg2": pixel_area_deg2,
    }
    for level in _configured_sky_levels(config):
        percent = int(round(level * 100.0))
        result[f"sky_area_{percent}_deg2"] = hpd_area_deg2(
            level,
            pixel_area_deg2,
            ranked_sky_indices,
            ranked_cumulative_probability,
        )
    return result


def compute_sky_shape_feature(
    sky_probability: Sequence[float],
    evaluated_sky_indices: Sequence[int],
    *,
    pixel_area_deg2: float,
    ranked_sky_indices: Sequence[int],
    ranked_cumulative_probability: Sequence[float],
) -> Mapping[str, Any]:
    probability = np.asarray(sky_probability)[
        np.asarray(evaluated_sky_indices, dtype=np.int64)
    ].astype(np.float64)
    entropy = -float(
        np.sum(probability * np.log(np.maximum(probability, 1e-300)))
    )
    effective_pixels = float(np.exp(entropy))
    area_50 = hpd_area_deg2(
        0.5,
        pixel_area_deg2,
        ranked_sky_indices,
        ranked_cumulative_probability,
    )
    area_90 = hpd_area_deg2(
        0.9,
        pixel_area_deg2,
        ranked_sky_indices,
        ranked_cumulative_probability,
    )
    return {
        "sky_log10_area_90_deg2": float(np.log10(max(area_90, 1e-300))),
        "sky_area_50_to_90_ratio": float(area_50 / area_90),
        "sky_probability_entropy": entropy,
        "sky_effective_pixel_count": effective_pixels,
        "sky_peak_probability": float(np.max(probability)),
    }


def _vmf_mean_direction_and_kappa(
    sky_probability: Sequence[float],
    evaluated_sky_indices: Sequence[int],
    phi_geo_arr: Sequence[float],
    latitude_arr: Sequence[float],
    l_max: int,
) -> tuple[np.ndarray, float, float]:
    """Moment/MLE fit of one von Mises--Fisher component on S2."""
    indices = np.asarray(evaluated_sky_indices, dtype=np.int64)
    probability = np.asarray(sky_probability)[indices].astype(np.float64)
    probability /= float(np.sum(probability))
    longitude_arr = np.asarray(phi_geo_arr, dtype=np.float64)
    latitude_arr = np.asarray(latitude_arr, dtype=np.float64)
    longitude = longitude_arr[indices]
    latitude = latitude_arr[indices]
    cos_latitude = np.cos(latitude)
    vectors = np.column_stack((
        cos_latitude * np.cos(longitude),
        cos_latitude * np.sin(longitude),
        np.sin(latitude),
    ))
    resultant = np.sum(probability[:, None] * vectors, axis=0)
    mean_resultant = float(np.linalg.norm(resultant))
    if mean_resultant < 1e-12:
        # An isotropic or exactly symmetric distribution has no unique mean
        # direction.  Use l_max only as a serialization convention; kappa=0
        # correctly describes the fit.
        mean_direction = np.array([
            np.cos(latitude_arr[l_max]) * np.cos(longitude_arr[l_max]),
            np.cos(latitude_arr[l_max]) * np.sin(longitude_arr[l_max]),
            np.sin(latitude_arr[l_max]),
        ])
        return mean_direction, 0.0, mean_resultant

    mean_direction = resultant / mean_resultant
    if mean_resultant >= 1.0 - 1e-10:
        return mean_direction, 1e10, mean_resultant

    # Banerjee et al. approximation for S2, followed by Newton refinement of
    # A3(kappa)=coth(kappa)-1/kappa=mean_resultant.
    kappa = max(
        mean_resultant * (3.0 - mean_resultant ** 2)
        / (1.0 - mean_resultant ** 2),
        1e-8,
    )
    for _ in range(8):
        if kappa < 1e-3:
            a3 = kappa / 3.0 - kappa ** 3 / 45.0
            derivative = 1.0 / 3.0 - kappa ** 2 / 15.0
        elif kappa > 50.0:
            a3 = 1.0 - 1.0 / kappa
            derivative = 1.0 / (kappa * kappa)
        else:
            a3 = 1.0 / np.tanh(kappa) - 1.0 / kappa
            derivative = 1.0 / (kappa * kappa) - 1.0 / np.sinh(kappa) ** 2
        step = (a3 - mean_resultant) / max(derivative, 1e-12)
        updated = max(kappa - step, 0.0)
        if abs(updated - kappa) <= 1e-10 * max(1.0, kappa):
            kappa = updated
            break
        kappa = updated
    return mean_direction, float(kappa), mean_resultant


def _vmf_cap_area_deg2(kappa: float, level: float) -> float:
    """Analytic spherical-cap area enclosing ``level`` of a vMF on S2."""
    level = _validate_level(level, "vMF credible level")
    if kappa < 1e-8:
        cos_radius = 1.0 - 2.0 * level
    else:
        dynamic_range = -np.expm1(-2.0 * kappa)
        cos_radius = 1.0 + np.log1p(-level * dynamic_range) / kappa
    cos_radius = float(np.clip(cos_radius, -1.0, 1.0))
    area_sr = 2.0 * np.pi * (1.0 - cos_radius)
    return float(area_sr * (180.0 / np.pi) ** 2)


def _vmf_resolution_kappa(pixel_area_deg2: float, level: float = 0.9) -> float:
    """Largest concentration whose continuous cap is at least one pixel."""
    target = min(float(pixel_area_deg2), level * _FULL_SKY_DEG2)
    low = 0.0
    high = 1.0
    while _vmf_cap_area_deg2(high, level) > target and high < 1e12:
        high *= 2.0
    for _ in range(64):
        middle = 0.5 * (low + high)
        if _vmf_cap_area_deg2(middle, level) >= target:
            low = middle
        else:
            high = middle
    return float(low)


def compute_sky_vmf_feature(
    sky_probability: Sequence[float],
    evaluated_sky_indices: Sequence[int],
    phi_geo_arr: Sequence[float],
    latitude_arr: Sequence[float],
    l_max: int,
    *,
    pixel_area_deg2: float,
) -> Mapping[str, Any]:
    """Lossy one-component spherical fit with an explicit KL quality metric."""
    mean_direction, kappa, mean_resultant = _vmf_mean_direction_and_kappa(
        sky_probability,
        evaluated_sky_indices,
        phi_geo_arr,
        latitude_arr,
        l_max,
    )
    # Do not let a fit to a discrete map claim sub-pixel localization.  Cap the
    # concentration where the continuous 90% vMF area reaches one grid pixel.
    resolution_kappa = _vmf_resolution_kappa(pixel_area_deg2)
    resolution_limited = bool(kappa > resolution_kappa)
    kappa = min(kappa, resolution_kappa)
    longitude = float(
        np.arctan2(mean_direction[1], mean_direction[0]) % (2.0 * np.pi)
    )
    latitude = float(np.arcsin(np.clip(mean_direction[2], -1.0, 1.0)))

    indices = np.asarray(evaluated_sky_indices, dtype=np.int64)
    probability = np.asarray(sky_probability)[indices].astype(np.float64)
    probability /= float(np.sum(probability))
    longitude_arr = np.asarray(phi_geo_arr, dtype=np.float64)
    latitude_arr = np.asarray(latitude_arr, dtype=np.float64)
    cos_latitude = np.cos(latitude_arr[indices])
    vectors = np.column_stack((
        cos_latitude * np.cos(longitude_arr[indices]),
        cos_latitude * np.sin(longitude_arr[indices]),
        np.sin(latitude_arr[indices]),
    ))
    log_q = kappa * (vectors @ mean_direction)
    log_q -= float(np.max(log_q))
    log_q -= float(np.log(np.sum(np.exp(log_q))))
    kl_nats = float(np.sum(
        probability * (np.log(np.maximum(probability, 1e-300)) - log_q)
    ))

    return {
        "sky_vmf_longitude_deg": float(np.degrees(longitude)),
        "sky_vmf_latitude_deg": float(np.degrees(latitude)),
        "sky_vmf_kappa": kappa,
        "sky_vmf_resolution_limited": resolution_limited,
        "sky_vmf_mean_resultant": mean_resultant,
        "sky_vmf_kl_nats": max(kl_nats, 0.0),
        "sky_vmf_area_50_deg2": _vmf_cap_area_deg2(kappa, 0.5),
        "sky_vmf_area_90_deg2": _vmf_cap_area_deg2(kappa, 0.9),
    }


def compute_sky_sparse_map_feature(
    config,
    sky_probability: Sequence[float],
    phi_geo_arr: Sequence[float],
    latitude_arr: Sequence[float],
    *,
    healpix_order: int | None,
    ranked_sky_indices: Sequence[int],
    ranked_cumulative_probability: Sequence[float],
) -> Mapping[str, Any]:
    """Capped ranked-pixel representation that preserves multimodality."""
    level = _validate_level(
        getattr(config, "likelihood_sky_sparse_level", 0.99),
        "likelihood_sky_sparse_level",
    )
    max_pixels = int(getattr(config, "likelihood_sky_sparse_max_pixels", 2048))
    if max_pixels < 1:
        raise ValueError("likelihood_sky_sparse_max_pixels must be at least 1")
    full_region = hpd_region(
        level, ranked_sky_indices, ranked_cumulative_probability
    )
    stored_indices = full_region[:max_pixels]
    probability = np.asarray(sky_probability)
    longitude = np.asarray(phi_geo_arr)
    latitude = np.asarray(latitude_arr)
    stored_probability = probability[stored_indices]
    return {
        "sky_sparse_indices": stored_indices.astype(np.int32),
        "sky_sparse_longitude_geo_deg": np.degrees(longitude[stored_indices]).astype(
            np.float32
        ),
        "sky_sparse_latitude_geo_deg": np.degrees(latitude[stored_indices]).astype(
            np.float32
        ),
        "sky_sparse_probability": stored_probability.astype(np.float32),
        "sky_sparse_grid_size": int(len(longitude)),
        "sky_sparse_healpix_order": healpix_order,
        "sky_sparse_ordering": "ring",
        "sky_sparse_coordinate_frame": "geo",
        "sky_sparse_requested_level": level,
        "sky_sparse_stored_mass": float(
            np.sum(stored_probability, dtype=np.float64)
        ),
        "sky_sparse_hpd_pixel_count": int(len(full_region)),
        "sky_sparse_stored_pixel_count": int(len(stored_indices)),
        "sky_sparse_truncated": bool(len(stored_indices) < len(full_region)),
    }


def trigger_gps(cluster, setup: Mapping[str, Any]) -> float | None:
    """Convert a cluster's segment-relative time to GPS when possible."""
    segment_start = setup.get("segment_start_gps")
    if segment_start is None:
        return None
    offset = float(getattr(cluster, "cluster_time", 0.0) or 0.0)
    if offset == 0.0:
        meta = getattr(cluster, "cluster_meta", None)
        offset = float(getattr(meta, "c_time", 0.0) or 0.0)
    return float(segment_start) + offset


def build_target_sky_indices(
    target_region: Mapping[str, Any] | None,
    phi_geo_arr: Sequence[float],
    latitude_arr: Sequence[float],
    *,
    t_ref: float | None,
) -> np.ndarray:
    """Build target-region pixels independently of the likelihood scan mask."""
    if not target_region:
        raise ValueError(
            "likelihood_target_region is required by target_sky_metrics "
            "and target_sky_consistency"
        )
    if sky_mask_requires_event_time(target_region) and t_ref is None:
        raise ValueError(
            "An ICRS likelihood_target_region requires segment_start_gps"
        )
    return np.asarray(
        compute_sky_valid_indices(
            phi_geo_arr,
            latitude_arr,
            target_region,
            t_ref=t_ref,
        ),
        dtype=np.int64,
    )


def compute_target_sky_metrics(
    sky_statistic: Sequence[float],
    sky_probability: Sequence[float],
    evaluated_sky_indices: Sequence[int],
    target_sky_indices: Sequence[int],
    *,
    target_level: float,
    ranked_sky_indices: Sequence[int],
    ranked_cumulative_probability: Sequence[float],
) -> dict[str, Any]:
    """Return target probability, fit loss, and HPD overlap diagnostics."""
    probability = np.asarray(sky_probability)
    statistic = np.asarray(sky_statistic, dtype=np.float64)
    evaluated = np.asarray(evaluated_sky_indices, dtype=np.int64)
    target = np.intersect1d(
        np.asarray(target_sky_indices, dtype=np.int64),
        evaluated,
        assume_unique=False,
    )
    if target.size == 0:
        metrics: dict[str, Any] = {
            "target_pixel_count": 0,
            "target_probability_mass": 0.0,
            "target_credible_level": 1.0,
            "target_delta_sky_stat": None,
        }
    else:
        target_max_probability = float(np.max(probability[target]))
        credible_level = float(
            np.sum(
                probability[evaluated][
                    probability[evaluated] >= target_max_probability
                ],
                dtype=np.float64,
            )
        )
        metrics = {
            "target_pixel_count": int(target.size),
            "target_probability_mass": float(
                np.sum(probability[target], dtype=np.float64)
            ),
            "target_credible_level": min(credible_level, 1.0),
            "target_delta_sky_stat": float(
                np.max(statistic[evaluated]) - np.max(statistic[target])
            ),
        }

    level = _validate_level(target_level, "likelihood_target_level")
    hpd = hpd_region(level, ranked_sky_indices, ranked_cumulative_probability)
    overlap_count = int(np.intersect1d(hpd, target).size)
    metrics.update({
        "target_level": level,
        "target_hpd_overlap": bool(overlap_count > 0),
        "target_hpd_overlap_pixels": overlap_count,
        "target_hpd_overlap_fraction": (
            float(overlap_count / len(hpd)) if len(hpd) else 0.0
        ),
    })
    return metrics


def evaluate_target_sky_consistency(
    target_metrics: Mapping[str, Any],
    *,
    rule: str,
    min_probability: float | None,
    max_delta_sky_stat: float | None,
    min_overlap_fraction: float,
) -> CutResult:
    """Apply one configured targeted-search rule to precomputed metrics."""
    metrics = dict(target_metrics)
    if rule == "credible_touch":
        # Use the actual discrete HPD set, not a floating-point comparison of
        # cumulative probability at its boundary.
        passed = metrics["target_hpd_overlap"]
        threshold = metrics["target_level"]
        observed = metrics["target_credible_level"]
    elif rule == "probability_mass":
        if min_probability is None:
            raise ValueError(
                "likelihood_target_min_probability is required for "
                "likelihood_target_rule='probability_mass'"
            )
        threshold = float(min_probability)
        passed = metrics["target_probability_mass"] >= threshold
        observed = metrics["target_probability_mass"]
    elif rule == "delta_sky_stat":
        if max_delta_sky_stat is None:
            raise ValueError(
                "likelihood_target_max_delta_sky_stat is required for "
                "likelihood_target_rule='delta_sky_stat'"
            )
        threshold = float(max_delta_sky_stat)
        observed = metrics["target_delta_sky_stat"]
        passed = observed is not None and observed <= threshold
    elif rule == "overlap_fraction":
        threshold = float(min_overlap_fraction)
        passed = (
            metrics["target_hpd_overlap"]
            and metrics["target_hpd_overlap_fraction"] >= threshold
        )
        observed = metrics["target_hpd_overlap_fraction"]
    else:
        raise ValueError(f"Unknown likelihood_target_rule: {rule!r}")

    metrics["target_rule"] = rule
    metrics["target_rule_threshold"] = threshold
    reason = "" if passed else (
        f"target_sky_consistency failed rule={rule!r} "
        f"with observed={observed} at threshold={threshold}"
    )
    return CutResult(bool(passed), reason=reason, metrics=metrics)


FEATURE_REGISTRY: dict[str, FeatureSpec] = {
    "sky_area": FeatureSpec(
        name="sky_area",
        cost="O(N log N) time, O(N) transient memory; a few persisted scalars",
        purpose="Localization validation and event follow-up; optional for classification",
        required_products=("sky_ranking", "target_indices"),
    ),
    "sky_shape": FeatureSpec(
        name="sky_shape",
        cost="Reuses sky_area sort; five persisted scalars",
        purpose="Candidate morphology feature for offline XGBoost ablation studies",
        required_products=("sky_ranking",),
    ),
    "sky_vmf_fit": FeatureSpec(
        name="sky_vmf_fit",
        cost="O(N) time; eight persisted scalars",
        purpose=(
            "Lossy single-component 2D compression for advanced studies; "
            "KL divergence flags arcs or multimodal maps that it cannot represent"
        ),
    ),
    "sky_sparse_map": FeatureSpec(
        name="sky_sparse_map",
        cost="O(N log N) sort; capped index/coordinate/probability arrays",
        purpose="Multimodal-safe compressed sky map for selected-event follow-up",
        required_products=("sky_ranking",),
        heavy=True,
    ),
    "target_sky_metrics": FeatureSpec(
        name="target_sky_metrics",
        cost="O(N) target mask plus shared HPD sort; small scalar output",
        purpose="Targeted-search diagnostics and selection-efficiency studies",
        required_products=("sky_ranking", "target_indices", "target_metrics"),
    ),
}

CUT_REGISTRY: dict[str, CutSpec] = {
    "target_sky_consistency": CutSpec(
        name="target_sky_consistency",
        stage="post_sky",
        cost="O(N) target mask plus O(N log N) HPD sort",
        required_products=("sky_ranking", "target_indices", "target_metrics"),
    ),
}


def resolve_extension_plan(config) -> LikelihoodExtensionPlan:
    """Validate flat YAML selections and resolve their fixed execution stages."""
    feature_names = tuple(getattr(config, "likelihood_features", ()) or ())
    cut_names = tuple(getattr(config, "likelihood_cuts", ()) or ())
    unknown_features = sorted(set(feature_names) - set(FEATURE_REGISTRY))
    unknown_cuts = sorted(set(cut_names) - set(CUT_REGISTRY))
    if unknown_features:
        raise ValueError(f"Unknown likelihood feature(s): {unknown_features}")
    if unknown_cuts:
        raise ValueError(f"Unknown likelihood cut(s): {unknown_cuts}")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("likelihood_features must not contain duplicates")
    if len(set(cut_names)) != len(cut_names):
        raise ValueError("likelihood_cuts must not contain duplicates")

    failure = str(getattr(config, "likelihood_feature_failure", "warn"))
    if failure not in {"warn", "error"}:
        raise ValueError("likelihood_feature_failure must be 'warn' or 'error'")
    allow_heavy = bool(
        getattr(config, "likelihood_allow_heavy_features", False)
    )
    blocked = [
        name for name in feature_names
        if FEATURE_REGISTRY[name].heavy and not allow_heavy
    ]
    if blocked:
        raise ValueError(
            "Heavy likelihood feature(s) require "
            f"likelihood_allow_heavy_features: true: {blocked}"
        )

    post_sky = tuple(
        name for name in cut_names if CUT_REGISTRY[name].stage == "post_sky"
    )
    post_reconstruction = tuple(
        name for name in cut_names
        if CUT_REGISTRY[name].stage == "post_reconstruction"
    )
    required_products = {
        product
        for name in feature_names
        for product in FEATURE_REGISTRY[name].required_products
    }
    required_products.update(
        product
        for name in cut_names
        for product in CUT_REGISTRY[name].required_products
    )
    return LikelihoodExtensionPlan(
        feature_names=feature_names,
        post_sky_cut_names=post_sky,
        post_reconstruction_cut_names=post_reconstruction,
        feature_failure=failure,
        allow_heavy_features=allow_heavy,
        required_products=tuple(sorted(required_products)),
    )


def run_likelihood_cuts(
    plan: LikelihoodExtensionPlan,
    *,
    stage: str,
    config,
    target_metrics: Mapping[str, Any] | None,
) -> CutResult:
    """Run pure cut functions selected for one fixed extension stage."""
    if stage == "post_sky":
        names = plan.post_sky_cut_names
    elif stage == "post_reconstruction":
        names = plan.post_reconstruction_cut_names
    else:
        raise ValueError(f"Unknown likelihood extension stage: {stage!r}")

    all_metrics: dict[str, Any] = {}
    for name in names:
        if name == "target_sky_consistency":
            if target_metrics is None:
                raise ValueError(
                    "target_sky_consistency requires target sky metrics"
                )
            result = evaluate_target_sky_consistency(
                target_metrics,
                rule=str(
                    getattr(config, "likelihood_target_rule", "credible_touch")
                ),
                min_probability=getattr(
                    config, "likelihood_target_min_probability", None
                ),
                max_delta_sky_stat=getattr(
                    config, "likelihood_target_max_delta_sky_stat", None
                ),
                min_overlap_fraction=float(
                    getattr(
                        config,
                        "likelihood_target_min_overlap_fraction",
                        0.0,
                    )
                ),
            )
        else:  # pragma: no cover - plan validation prevents this path
            raise ValueError(f"Unknown likelihood cut: {name!r}")
        all_metrics[name] = dict(result.metrics)
        if not result.passed:
            return CutResult(False, reason=result.reason, metrics=all_metrics)
    return CutResult(True, metrics=all_metrics)


def compute_likelihood_feature(
    name: str,
    *,
    config,
    sky_probability: np.ndarray,
    evaluated_sky_indices: np.ndarray,
    phi_geo_arr: np.ndarray,
    latitude_arr: np.ndarray,
    healpix_order: int | None,
    l_max: int,
    ranked_sky_indices: np.ndarray | None,
    ranked_cumulative_probability: np.ndarray | None,
    target_metrics: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Dispatch one named feature to a pure function with explicit inputs."""
    pixel_area = sky_pixel_area_deg2(len(sky_probability))
    if name in {"sky_area", "sky_shape", "sky_sparse_map"}:
        if ranked_sky_indices is None or ranked_cumulative_probability is None:
            raise ValueError(f"{name} requires ranked sky probability")

    if name == "sky_area":
        return compute_sky_area_feature(
            config,
            pixel_area_deg2=pixel_area,
            ranked_sky_indices=ranked_sky_indices,
            ranked_cumulative_probability=ranked_cumulative_probability,
        )
    if name == "sky_shape":
        return compute_sky_shape_feature(
            sky_probability,
            evaluated_sky_indices,
            pixel_area_deg2=pixel_area,
            ranked_sky_indices=ranked_sky_indices,
            ranked_cumulative_probability=ranked_cumulative_probability,
        )
    if name == "sky_vmf_fit":
        return compute_sky_vmf_feature(
            sky_probability,
            evaluated_sky_indices,
            phi_geo_arr,
            latitude_arr,
            l_max,
            pixel_area_deg2=pixel_area,
        )
    if name == "sky_sparse_map":
        return compute_sky_sparse_map_feature(
            config,
            sky_probability,
            phi_geo_arr,
            latitude_arr,
            healpix_order=healpix_order,
            ranked_sky_indices=ranked_sky_indices,
            ranked_cumulative_probability=ranked_cumulative_probability,
        )
    if name == "target_sky_metrics":
        if target_metrics is None:
            raise ValueError("target_sky_metrics could not be computed")
        return dict(target_metrics)
    raise ValueError(f"Unknown likelihood feature: {name!r}")


def run_likelihood_features(
    plan: LikelihoodExtensionPlan,
    *,
    config,
    sky_probability: np.ndarray,
    evaluated_sky_indices: np.ndarray,
    phi_geo_arr: np.ndarray,
    latitude_arr: np.ndarray,
    healpix_order: int | None,
    l_max: int,
    ranked_sky_indices: np.ndarray | None,
    ranked_cumulative_probability: np.ndarray | None,
    target_metrics: Mapping[str, Any] | None,
    preparation_errors: Mapping[str, Exception] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Evaluate selected output features with explicit per-feature status."""
    preparation_errors = preparation_errors or {}
    values: dict[str, Any] = {}
    status: dict[str, dict[str, Any]] = {}
    for name in plan.feature_names:
        spec = FEATURE_REGISTRY[name]
        try:
            if name in preparation_errors:
                raise preparation_errors[name]
            feature_values = dict(
                compute_likelihood_feature(
                    name,
                    config=config,
                    sky_probability=sky_probability,
                    evaluated_sky_indices=evaluated_sky_indices,
                    phi_geo_arr=phi_geo_arr,
                    latitude_arr=latitude_arr,
                    healpix_order=healpix_order,
                    l_max=l_max,
                    ranked_sky_indices=ranked_sky_indices,
                    ranked_cumulative_probability=(
                        ranked_cumulative_probability
                    ),
                    target_metrics=target_metrics,
                )
            )
        except Exception as exc:
            if plan.feature_failure == "error":
                raise
            logger.warning("Likelihood feature %s failed: %s", name, exc)
            status[name] = {
                "ok": False,
                "version": spec.version,
                "error": f"{type(exc).__name__}: {exc}",
            }
            for column in spec.columns:
                values[column] = None
            continue
        overlap = set(values).intersection(feature_values)
        if overlap:
            raise ValueError(
                f"Likelihood feature {name!r} overwrote output(s): {sorted(overlap)}"
            )
        values.update(feature_values)
        status[name] = {"ok": True, "version": spec.version}
    return values, status


def build_legacy_sky_area(
    *,
    pixel_area_deg2: float,
    sky_probability: Sequence[float],
    evaluated_sky_indices: Sequence[int],
    ranked_sky_indices: Sequence[int],
    ranked_cumulative_probability: Sequence[float],
    target_sky_indices: Sequence[int] | None,
) -> list[float]:
    """Build cWB-compatible ``erA`` values for an enabled sky-area feature.

    cWB stores the square root of area in square degrees for the 10--90 %
    regions.  Entries 0 and 10 use the best target-region pixel when a target
    is configured; injection-only values remain unavailable at this stage.
    """
    legacy = [0.0]
    legacy.extend(
        float(
            np.sqrt(
                hpd_area_deg2(
                    level,
                    pixel_area_deg2,
                    ranked_sky_indices,
                    ranked_cumulative_probability,
                )
            )
        )
        for level in _LEGACY_SKY_LEVELS
    )
    legacy.append(0.0)

    if target_sky_indices is not None:
        target = np.intersect1d(
            target_sky_indices,
            evaluated_sky_indices,
            assume_unique=False,
        )
        if target.size:
            probability = np.asarray(sky_probability)
            searched = int(target[np.argmax(probability[target])])
            positions = np.where(
                np.asarray(ranked_sky_indices) == searched
            )[0]
            if positions.size:
                position = int(positions[0])
                credible_level = float(
                    ranked_cumulative_probability[position]
                )
                legacy[0] = float(
                    np.sqrt((position + 1) * pixel_area_deg2)
                )
                legacy[10] = min(credible_level, 1.0)

    return legacy


__all__ = [
    "FEATURE_REGISTRY",
    "CUT_REGISTRY",
    "FeatureSpec",
    "CutSpec",
    "CutResult",
    "LikelihoodExtensionPlan",
    "resolve_extension_plan",
    "validate_sky_temperature",
    "build_sky_probability",
    "rank_sky_probability",
    "hpd_region",
    "sky_pixel_area_deg2",
    "hpd_area_deg2",
    "trigger_gps",
    "build_target_sky_indices",
    "compute_target_sky_metrics",
    "evaluate_target_sky_consistency",
    "compute_sky_area_feature",
    "compute_sky_shape_feature",
    "compute_sky_vmf_feature",
    "compute_sky_sparse_map_feature",
    "compute_likelihood_feature",
    "run_likelihood_cuts",
    "run_likelihood_features",
    "build_legacy_sky_area",
]
