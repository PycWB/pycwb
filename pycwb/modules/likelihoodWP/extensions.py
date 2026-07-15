"""Optional likelihood features and cuts built from shared cached products.

The likelihood kernels deliberately remain unaware of individual derived
statistics.  This module provides a small, explicit registry and flat runners
called by the shared likelihood flow at fixed extension points.

The default configuration enables no optional features or cuts.  This matters
for production searches with millions of background triggers: the O(N log N)
HPD sort and all extra persisted scalars are paid for only when requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .sky_mask import compute_sky_valid_indices, sky_mask_requires_event_time

logger = logging.getLogger(__name__)

_FULL_SKY_DEG2 = 4.0 * np.pi * (180.0 / np.pi) ** 2
_LEGACY_SKY_LEVELS = tuple(i / 10.0 for i in range(1, 10))


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata and implementation for one optional output feature."""

    name: str
    compute: Callable[["LikelihoodExtensionContext"], Mapping[str, Any]]
    cost: str
    purpose: str
    destination: str = "likelihood_feature_sidecar"
    version: int = 1
    heavy: bool = False
    columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class CutSpec:
    """Metadata and implementation for one optional cluster cut."""

    name: str
    evaluate: Callable[["LikelihoodExtensionContext"], "CutResult"]
    stage: str
    cost: str
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


class LikelihoodExtensionContext:
    """Per-cluster inputs and lazily cached shared likelihood products."""

    def __init__(
        self,
        *,
        config,
        cluster,
        setup: Mapping[str, Any],
        skymap_statistics,
        sky_statistics,
        sky_valid_indices: Sequence[int],
        phi_geo_arr: Sequence[float],
        latitude_arr: Sequence[float],
        healpix_order: int | None = None,
    ) -> None:
        self.config = config
        self.cluster = cluster
        self.setup = setup
        self.skymap_statistics = skymap_statistics
        self.sky_statistics = sky_statistics
        self.sky_valid_indices = np.asarray(sky_valid_indices, dtype=np.int64)
        self.phi_geo_arr = np.asarray(phi_geo_arr, dtype=np.float64)
        self.latitude_arr = np.asarray(latitude_arr, dtype=np.float64)
        self.healpix_order = (
            int(healpix_order) if healpix_order is not None else None
        )
        self._cache: dict[str, Any] = {}

    @property
    def pixel_area_deg2(self) -> float:
        """Equal-area sky pixel size for the active full HEALPix grid."""
        n_sky = len(np.asarray(self.skymap_statistics.nSkyStat))
        if n_sky <= 0:
            raise ValueError("Cannot calculate a sky area for an empty sky grid")
        return float(_FULL_SKY_DEG2 / n_sky)

    @property
    def evaluated_sky_indices(self) -> np.ndarray:
        """Scanned directions with a finite, positive localization statistic.

        The native sky scan leaves masked and netCC-rejected directions at zero.
        Including those zeros in a softmax assigns probability outside the
        evaluated likelihood support, especially for targeted sky scans.
        cWB's sky-area calculation likewise ignores non-positive map values.
        """
        if "evaluated_sky_indices" not in self._cache:
            stat = np.asarray(self.skymap_statistics.nSkyStat, dtype=np.float64)
            valid = self.sky_valid_indices
            valid = valid[(valid >= 0) & (valid < len(stat))]
            keep = np.isfinite(stat[valid]) & (stat[valid] > 0.0)
            indices = np.unique(valid[keep])
            if indices.size == 0:
                raise ValueError("No positive finite sky statistics are available")
            self._cache["evaluated_sky_indices"] = indices
        return self._cache["evaluated_sky_indices"]

    @property
    def sky_temperature(self) -> float:
        value = float(
            getattr(self.config, "likelihood_sky_temperature", 1.0)
        )
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                "likelihood_sky_temperature must be positive and finite"
            )
        return value

    @property
    def sky_probability(self) -> np.ndarray:
        """Normalized probability on evaluated directions, zero elsewhere."""
        if "sky_probability" not in self._cache:
            stat = np.asarray(self.skymap_statistics.nSkyStat, dtype=np.float64)
            indices = self.evaluated_sky_indices
            selected = stat[indices]
            shifted = (selected - np.max(selected)) / self.sky_temperature
            weights = np.exp(shifted)
            norm = float(np.sum(weights))
            if not np.isfinite(norm) or norm <= 0.0:
                raise ValueError("Sky probability normalization is not finite")
            probability = np.zeros(stat.shape, dtype=np.float32)
            probability[indices] = (weights / norm).astype(np.float32)
            self._cache["sky_probability"] = probability
        return self._cache["sky_probability"]

    @property
    def ranked_sky_indices(self) -> np.ndarray:
        """Evaluated sky directions sorted by descending probability."""
        if "ranked_sky_indices" not in self._cache:
            indices = self.evaluated_sky_indices
            probability = self.sky_probability
            order = np.argsort(-probability[indices], kind="stable")
            self._cache["ranked_sky_indices"] = indices[order]
        return self._cache["ranked_sky_indices"]

    @property
    def ranked_cumulative_probability(self) -> np.ndarray:
        if "ranked_cumulative_probability" not in self._cache:
            ranked = self.ranked_sky_indices
            self._cache["ranked_cumulative_probability"] = np.cumsum(
                self.sky_probability[ranked], dtype=np.float64
            )
        return self._cache["ranked_cumulative_probability"]

    def hpd_region(self, level: float) -> np.ndarray:
        """Return pixel indices in the smallest discrete HPD region."""
        level = _validate_level(level, "credible level")
        key = f"hpd_region:{level:.12g}"
        if key not in self._cache:
            ranked = self.ranked_sky_indices
            cumulative = self.ranked_cumulative_probability
            stop = int(np.searchsorted(cumulative, level, side="left")) + 1
            self._cache[key] = ranked[: min(stop, len(ranked))]
        return self._cache[key]

    def hpd_area_deg2(self, level: float) -> float:
        return float(len(self.hpd_region(level)) * self.pixel_area_deg2)

    @property
    def trigger_gps(self) -> float | None:
        if "trigger_gps" not in self._cache:
            segment_start = self.setup.get("segment_start_gps")
            if segment_start is None:
                value = None
            else:
                offset = float(getattr(self.cluster, "cluster_time", 0.0) or 0.0)
                if offset == 0.0:
                    meta = getattr(self.cluster, "cluster_meta", None)
                    offset = float(getattr(meta, "c_time", 0.0) or 0.0)
                value = float(segment_start) + offset
            self._cache["trigger_gps"] = value
        return self._cache["trigger_gps"]

    @property
    def target_indices(self) -> np.ndarray:
        """Target-region pixels, independent of the likelihood scan mask."""
        if "target_indices" not in self._cache:
            target = getattr(self.config, "likelihood_target_region", None)
            if not target:
                raise ValueError(
                    "likelihood_target_region is required by target_sky_metrics "
                    "and target_sky_consistency"
                )
            t_ref = self.trigger_gps
            if sky_mask_requires_event_time(target) and t_ref is None:
                raise ValueError(
                    "An ICRS likelihood_target_region requires segment_start_gps"
                )
            indices = compute_sky_valid_indices(
                self.phi_geo_arr,
                self.latitude_arr,
                target,
                t_ref=t_ref,
            )
            self._cache["target_indices"] = np.asarray(indices, dtype=np.int64)
        return self._cache["target_indices"]

    @property
    def target_metrics(self) -> dict[str, Any]:
        if "target_metrics" not in self._cache:
            probability = self.sky_probability
            statistic = np.asarray(
                self.skymap_statistics.nSkyStat, dtype=np.float64
            )
            evaluated = self.evaluated_sky_indices
            target = np.intersect1d(
                self.target_indices, evaluated, assume_unique=False
            )
            if target.size == 0:
                metrics = {
                    "target_pixel_count": 0,
                    "target_probability_mass": 0.0,
                    "target_credible_level": 1.0,
                    "target_delta_sky_stat": None,
                }
            else:
                target_max_probability = float(np.max(probability[target]))
                # Include ties at the target threshold.  This is the credible
                # mass at which the HPD set first touches the target region.
                credible_level = float(
                    np.sum(probability[evaluated][
                        probability[evaluated] >= target_max_probability
                    ], dtype=np.float64)
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

            level = _validate_level(
                getattr(self.config, "likelihood_target_level", 0.9),
                "likelihood_target_level",
            )
            hpd = self.hpd_region(level)
            overlap_count = int(np.intersect1d(hpd, target).size)
            metrics.update({
                "target_level": level,
                "target_hpd_overlap": bool(overlap_count > 0),
                "target_hpd_overlap_pixels": overlap_count,
                "target_hpd_overlap_fraction": (
                    float(overlap_count / len(hpd)) if len(hpd) else 0.0
                ),
            })
            self._cache["target_metrics"] = metrics
        return self._cache["target_metrics"]


def _validate_level(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return value


def _configured_sky_levels(config) -> tuple[float, ...]:
    levels = getattr(config, "likelihood_sky_levels", (0.5, 0.9))
    normalized = tuple(sorted({_validate_level(v, "likelihood_sky_levels") for v in levels}))
    if not normalized:
        raise ValueError("likelihood_sky_levels must contain at least one level")
    return normalized


def build_sky_probability(context: LikelihoodExtensionContext) -> np.ndarray:
    """Public helper used by both likelihood backends and unit tests."""
    probability = context.sky_probability
    context.skymap_statistics.nProbability = probability
    return probability


def _feature_sky_area(context: LikelihoodExtensionContext) -> Mapping[str, Any]:
    result: dict[str, Any] = {
        "sky_pixel_area_deg2": context.pixel_area_deg2,
    }
    for level in _configured_sky_levels(context.config):
        percent = int(round(level * 100.0))
        result[f"sky_area_{percent}_deg2"] = context.hpd_area_deg2(level)
    return result


def _feature_sky_shape(context: LikelihoodExtensionContext) -> Mapping[str, Any]:
    probability = context.sky_probability[context.evaluated_sky_indices].astype(
        np.float64
    )
    entropy = -float(np.sum(probability * np.log(np.maximum(probability, 1e-300))))
    effective_pixels = float(np.exp(entropy))
    area_50 = context.hpd_area_deg2(0.5)
    area_90 = context.hpd_area_deg2(0.9)
    return {
        "sky_log10_area_90_deg2": float(np.log10(max(area_90, 1e-300))),
        "sky_area_50_to_90_ratio": float(area_50 / area_90),
        "sky_probability_entropy": entropy,
        "sky_effective_pixel_count": effective_pixels,
        "sky_peak_probability": float(np.max(probability)),
    }


def _vmf_mean_direction_and_kappa(
    context: LikelihoodExtensionContext,
) -> tuple[np.ndarray, float, float]:
    """Moment/MLE fit of one von Mises--Fisher component on S2."""
    indices = context.evaluated_sky_indices
    probability = context.sky_probability[indices].astype(np.float64)
    probability /= float(np.sum(probability))
    longitude = context.phi_geo_arr[indices]
    latitude = context.latitude_arr[indices]
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
        l_max = int(context.skymap_statistics.l_max)
        mean_direction = np.array([
            np.cos(context.latitude_arr[l_max]) * np.cos(context.phi_geo_arr[l_max]),
            np.cos(context.latitude_arr[l_max]) * np.sin(context.phi_geo_arr[l_max]),
            np.sin(context.latitude_arr[l_max]),
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


def _feature_sky_vmf_fit(context: LikelihoodExtensionContext) -> Mapping[str, Any]:
    """Lossy one-component spherical fit with an explicit KL quality metric."""
    mean_direction, kappa, mean_resultant = _vmf_mean_direction_and_kappa(context)
    # Do not let a fit to a discrete map claim sub-pixel localization.  Cap the
    # concentration where the continuous 90% vMF area reaches one grid pixel.
    resolution_kappa = _vmf_resolution_kappa(context.pixel_area_deg2)
    resolution_limited = bool(kappa > resolution_kappa)
    kappa = min(kappa, resolution_kappa)
    longitude = float(np.arctan2(mean_direction[1], mean_direction[0]) % (2.0 * np.pi))
    latitude = float(np.arcsin(np.clip(mean_direction[2], -1.0, 1.0)))

    indices = context.evaluated_sky_indices
    probability = context.sky_probability[indices].astype(np.float64)
    probability /= float(np.sum(probability))
    cos_latitude = np.cos(context.latitude_arr[indices])
    vectors = np.column_stack((
        cos_latitude * np.cos(context.phi_geo_arr[indices]),
        cos_latitude * np.sin(context.phi_geo_arr[indices]),
        np.sin(context.latitude_arr[indices]),
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


def _feature_sky_sparse_map(context: LikelihoodExtensionContext) -> Mapping[str, Any]:
    """Capped ranked-pixel representation that preserves multimodality."""
    level = _validate_level(
        getattr(context.config, "likelihood_sky_sparse_level", 0.99),
        "likelihood_sky_sparse_level",
    )
    max_pixels = int(
        getattr(context.config, "likelihood_sky_sparse_max_pixels", 2048)
    )
    if max_pixels < 1:
        raise ValueError("likelihood_sky_sparse_max_pixels must be at least 1")
    full_region = context.hpd_region(level)
    stored_indices = full_region[:max_pixels]
    stored_probability = context.sky_probability[stored_indices]
    return {
        "sky_sparse_indices": stored_indices.astype(np.int32),
        "sky_sparse_longitude_geo_deg": np.degrees(
            context.phi_geo_arr[stored_indices]
        ).astype(np.float32),
        "sky_sparse_latitude_geo_deg": np.degrees(
            context.latitude_arr[stored_indices]
        ).astype(np.float32),
        "sky_sparse_probability": stored_probability.astype(np.float32),
        "sky_sparse_grid_size": int(len(context.phi_geo_arr)),
        "sky_sparse_healpix_order": context.healpix_order,
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


def _feature_target_sky_metrics(
    context: LikelihoodExtensionContext,
) -> Mapping[str, Any]:
    return dict(context.target_metrics)


def _cut_target_sky_consistency(
    context: LikelihoodExtensionContext,
) -> CutResult:
    metrics = dict(context.target_metrics)
    rule = str(
        getattr(context.config, "likelihood_target_rule", "credible_touch")
    )
    if rule == "credible_touch":
        # Use the actual discrete HPD set, not a floating-point comparison of
        # cumulative probability at its boundary.
        passed = metrics["target_hpd_overlap"]
        threshold = metrics["target_level"]
        observed = metrics["target_credible_level"]
    elif rule == "probability_mass":
        configured = getattr(
            context.config, "likelihood_target_min_probability", None
        )
        if configured is None:
            raise ValueError(
                "likelihood_target_min_probability is required for "
                "likelihood_target_rule='probability_mass'"
            )
        threshold = float(configured)
        passed = metrics["target_probability_mass"] >= threshold
        observed = metrics["target_probability_mass"]
    elif rule == "delta_sky_stat":
        configured = getattr(
            context.config, "likelihood_target_max_delta_sky_stat", None
        )
        if configured is None:
            raise ValueError(
                "likelihood_target_max_delta_sky_stat is required for "
                "likelihood_target_rule='delta_sky_stat'"
            )
        threshold = float(configured)
        observed = metrics["target_delta_sky_stat"]
        passed = observed is not None and observed <= threshold
    elif rule == "overlap_fraction":
        threshold = float(
            getattr(context.config, "likelihood_target_min_overlap_fraction", 0.0)
        )
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
        compute=_feature_sky_area,
        cost="O(N log N) time, O(N) transient memory; a few persisted scalars",
        purpose="Localization validation and event follow-up; optional for classification",
    ),
    "sky_shape": FeatureSpec(
        name="sky_shape",
        compute=_feature_sky_shape,
        cost="Reuses sky_area sort; five persisted scalars",
        purpose="Candidate morphology feature for offline XGBoost ablation studies",
    ),
    "sky_vmf_fit": FeatureSpec(
        name="sky_vmf_fit",
        compute=_feature_sky_vmf_fit,
        cost="O(N) time; eight persisted scalars",
        purpose=(
            "Lossy single-component 2D compression for advanced studies; "
            "KL divergence flags arcs or multimodal maps that it cannot represent"
        ),
    ),
    "sky_sparse_map": FeatureSpec(
        name="sky_sparse_map",
        compute=_feature_sky_sparse_map,
        cost="O(N log N) sort; capped index/coordinate/probability arrays",
        purpose="Multimodal-safe compressed sky map for selected-event follow-up",
        heavy=True,
    ),
    "target_sky_metrics": FeatureSpec(
        name="target_sky_metrics",
        compute=_feature_target_sky_metrics,
        cost="O(N) target mask plus shared HPD sort; small scalar output",
        purpose="Targeted-search diagnostics and selection-efficiency studies",
    ),
}

CUT_REGISTRY: dict[str, CutSpec] = {
    "target_sky_consistency": CutSpec(
        name="target_sky_consistency",
        evaluate=_cut_target_sky_consistency,
        stage="post_sky",
        cost="O(N) target mask plus O(N log N) HPD sort",
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
    return LikelihoodExtensionPlan(
        feature_names=feature_names,
        post_sky_cut_names=post_sky,
        post_reconstruction_cut_names=post_reconstruction,
        feature_failure=failure,
        allow_heavy_features=allow_heavy,
    )


def run_likelihood_cuts(
    context: LikelihoodExtensionContext,
    plan: LikelihoodExtensionPlan,
    *,
    stage: str,
) -> CutResult:
    """Run cuts for one fixed stage; cut errors always fail loudly."""
    if stage == "post_sky":
        names = plan.post_sky_cut_names
    elif stage == "post_reconstruction":
        names = plan.post_reconstruction_cut_names
    else:
        raise ValueError(f"Unknown likelihood extension stage: {stage!r}")

    all_metrics: dict[str, Any] = {}
    for name in names:
        result = CUT_REGISTRY[name].evaluate(context)
        all_metrics[name] = dict(result.metrics)
        if not result.passed:
            return CutResult(False, reason=result.reason, metrics=all_metrics)
    return CutResult(True, metrics=all_metrics)


def run_likelihood_features(
    context: LikelihoodExtensionContext,
    plan: LikelihoodExtensionPlan,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Evaluate selected output features with explicit per-feature status."""
    values: dict[str, Any] = {}
    status: dict[str, dict[str, Any]] = {}
    for name in plan.feature_names:
        spec = FEATURE_REGISTRY[name]
        try:
            feature_values = dict(spec.compute(context))
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


def apply_legacy_sky_area(
    context: LikelihoodExtensionContext,
    plan: LikelihoodExtensionPlan,
) -> None:
    """Populate cWB-compatible ``erA`` values only when sky_area is enabled.

    cWB stores the square root of area in square degrees for the 10--90 %
    regions.  Entries 0 and 10 use the best target-region pixel when a target
    is configured; injection-only values remain unavailable at this stage.
    """
    if "sky_area" not in plan.feature_names:
        return
    legacy = [0.0]
    legacy.extend(
        float(np.sqrt(context.hpd_area_deg2(level)))
        for level in _LEGACY_SKY_LEVELS
    )
    legacy.append(0.0)

    if getattr(context.config, "likelihood_target_region", None):
        target = np.intersect1d(
            context.target_indices,
            context.evaluated_sky_indices,
            assume_unique=False,
        )
        if target.size:
            probability = context.sky_probability
            searched = int(target[np.argmax(probability[target])])
            positions = np.where(context.ranked_sky_indices == searched)[0]
            if positions.size:
                position = int(positions[0])
                credible_level = float(
                    context.ranked_cumulative_probability[position]
                )
                legacy[0] = float(
                    np.sqrt((position + 1) * context.pixel_area_deg2)
                )
                legacy[10] = min(credible_level, 1.0)

    context.cluster.sky_area = legacy


def attach_extension_outputs(
    context: LikelihoodExtensionContext,
    *,
    features: Mapping[str, Any],
    feature_status: Mapping[str, Any],
    cut_metrics: Mapping[str, Any],
) -> None:
    """Attach optional sidecar products without changing likelihood returns."""
    skymap = context.skymap_statistics
    skymap.likelihood_features = dict(features) or None
    skymap.likelihood_feature_status = dict(feature_status) or None
    skymap.likelihood_cut_metrics = dict(cut_metrics) or None
    skymap.likelihood_metadata = (
        {
            "sky_probability_temperature": context.sky_temperature,
            "evaluated_sky_pixel_count": int(
                len(context.evaluated_sky_indices)
            ),
        }
        if features or feature_status or cut_metrics else None
    )


__all__ = [
    "FEATURE_REGISTRY",
    "CUT_REGISTRY",
    "FeatureSpec",
    "CutSpec",
    "CutResult",
    "LikelihoodExtensionContext",
    "LikelihoodExtensionPlan",
    "resolve_extension_plan",
    "build_sky_probability",
    "run_likelihood_cuts",
    "run_likelihood_features",
    "apply_legacy_sky_area",
    "attach_extension_outputs",
]
