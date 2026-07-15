from types import SimpleNamespace

import numpy as np
import orjson
import pytest
from jsonschema import ValidationError, validate

from pycwb.config.config import Config
from pycwb.constants.user_parameters_schema import schema
from pycwb.modules.likelihoodWP.detection_statistics import (
    compute_sky_error_region,
)
from pycwb.modules.likelihoodWP.extensions import (
    LikelihoodExtensionContext,
    apply_legacy_sky_area,
    attach_extension_outputs,
    build_sky_probability,
    resolve_extension_plan,
    run_likelihood_cuts,
    run_likelihood_features,
)
from pycwb.modules.workflow_utils.trigger_utils import save_trigger


def _fixed_target(longitude_deg):
    return {
        "type": "Fixed",
        "coordsys": "geo",
        "coordinates": {
            "longitude": f"{longitude_deg} deg",
            "latitude": "0 deg",
        },
    }


def _context(
    probability,
    *,
    valid_indices=None,
    target_region=None,
    features=(),
    cuts=(),
    target_level=0.9,
):
    probability = np.asarray(probability, dtype=np.float64)
    # A positive common shift makes every intended evaluated statistic positive
    # while retaining exactly the requested softmax probabilities.
    statistic = np.log(probability) + 10.0
    skymap = SimpleNamespace(
        l_max=int(np.argmax(probability)),
        nSkyStat=statistic.astype(np.float32),
        nProbability=None,
        likelihood_features=None,
        likelihood_feature_status=None,
        likelihood_cut_metrics=None,
        likelihood_metadata=None,
    )
    config = SimpleNamespace(
        likelihood_features=list(features),
        likelihood_cuts=list(cuts),
        likelihood_feature_failure="warn",
        likelihood_allow_heavy_features=False,
        likelihood_sky_levels=[0.5, 0.9],
        likelihood_sky_temperature=1.0,
        likelihood_sky_sparse_level=0.99,
        likelihood_sky_sparse_max_pixels=2048,
        likelihood_target_region=target_region,
        likelihood_target_rule="credible_touch",
        likelihood_target_level=target_level,
        likelihood_target_min_probability=None,
        likelihood_target_max_delta_sky_stat=None,
        likelihood_target_min_overlap_fraction=0.0,
    )
    cluster = SimpleNamespace(
        cluster_time=2.0,
        cluster_meta=SimpleNamespace(c_time=0.0),
        sky_area=[],
    )
    phi = np.linspace(0.0, 2.0 * np.pi, len(probability), endpoint=False)
    latitude = np.zeros(len(probability))
    if valid_indices is None:
        valid_indices = np.arange(len(probability), dtype=np.int64)
    return LikelihoodExtensionContext(
        config=config,
        cluster=cluster,
        setup={"segment_start_gps": 1_000_000_000.0},
        skymap_statistics=skymap,
        sky_statistics=None,
        sky_valid_indices=valid_indices,
        phi_geo_arr=phi,
        latitude_arr=latitude,
        healpix_order=None,
    )


def test_probability_excludes_masked_and_nonpositive_sky_pixels():
    context = _context([0.6, 0.2, 0.1, 0.1], valid_indices=[0, 2])
    probability = build_sky_probability(context)

    assert probability[1] == 0.0
    assert probability[3] == 0.0
    assert float(np.sum(probability)) == pytest.approx(1.0)
    assert probability[0] > probability[2]


def test_shared_probability_and_hpd_products_are_cached():
    context = _context([0.6, 0.2, 0.11, 0.09])

    assert context.sky_probability is context.sky_probability
    assert context.ranked_sky_indices is context.ranked_sky_indices
    assert context.hpd_region(0.9) is context.hpd_region(0.9)
    assert len(context.hpd_region(0.5)) == 1
    assert len(context.hpd_region(0.9)) == 3


def test_probability_temperature_is_explicit_and_persisted_with_features():
    context = _context([0.8, 0.1, 0.05, 0.05], features=["sky_area"])
    native_peak = float(context.sky_probability[0])

    warmer = _context([0.8, 0.1, 0.05, 0.05], features=["sky_area"])
    warmer.config.likelihood_sky_temperature = 4.0
    assert float(warmer.sky_probability[0]) < native_peak

    values, status = run_likelihood_features(
        warmer, resolve_extension_plan(warmer.config)
    )
    attach_extension_outputs(
        warmer,
        features=values,
        feature_status=status,
        cut_metrics={},
    )
    assert warmer.skymap_statistics.likelihood_metadata == {
        "sky_probability_temperature": 4.0,
        "evaluated_sky_pixel_count": 4,
    }


def test_sky_area_feature_is_optional_and_uses_compact_scalars():
    context = _context(
        [0.6, 0.2, 0.11, 0.09],
        features=["sky_area"],
    )
    plan = resolve_extension_plan(context.config)
    values, status = run_likelihood_features(context, plan)

    pixel_area = 4.0 * np.pi * (180.0 / np.pi) ** 2 / 4.0
    assert values["sky_area_50_deg2"] == pytest.approx(pixel_area)
    assert values["sky_area_90_deg2"] == pytest.approx(3.0 * pixel_area)
    assert status["sky_area"]["ok"] is True

    apply_legacy_sky_area(context, plan)
    assert len(context.cluster.sky_area) == 11
    assert context.cluster.sky_area[5] == pytest.approx(np.sqrt(pixel_area))
    assert context.cluster.sky_area[9] == pytest.approx(np.sqrt(3.0 * pixel_area))


def test_default_plan_adds_no_feature_output_or_hpd_sort():
    context = _context([0.6, 0.2, 0.11, 0.09])
    plan = resolve_extension_plan(context.config)
    build_sky_probability(context)
    values, status = run_likelihood_features(context, plan)

    assert values == {}
    assert status == {}
    assert "ranked_sky_indices" not in context._cache
    assert context.cluster.sky_area == []


def test_target_90_percent_consistency_accepts_overlap_and_rejects_no_overlap():
    probability = [0.7, 0.2, 0.08, 0.02]

    overlapping = _context(
        probability,
        target_region=_fixed_target(90),
        cuts=["target_sky_consistency"],
    )
    overlap_result = run_likelihood_cuts(
        overlapping,
        resolve_extension_plan(overlapping.config),
        stage="post_sky",
    )
    assert overlap_result.passed is True
    metrics = overlap_result.metrics["target_sky_consistency"]
    assert metrics["target_hpd_overlap"] is True
    assert metrics["target_credible_level"] == pytest.approx(0.9, abs=1e-6)

    outside = _context(
        probability,
        target_region=_fixed_target(-90),
        cuts=["target_sky_consistency"],
    )
    outside_result = run_likelihood_cuts(
        outside,
        resolve_extension_plan(outside.config),
        stage="post_sky",
    )
    assert outside_result.passed is False
    metrics = outside_result.metrics["target_sky_consistency"]
    assert metrics["target_hpd_overlap"] is False
    assert metrics["target_credible_level"] > 0.9


def test_target_metrics_are_an_optional_output_not_a_catalog_requirement():
    context = _context(
        [0.7, 0.2, 0.08, 0.02],
        target_region=_fixed_target(90),
        features=["target_sky_metrics"],
    )
    values, status = run_likelihood_features(
        context, resolve_extension_plan(context.config)
    )

    assert status["target_sky_metrics"]["ok"] is True
    assert values["target_probability_mass"] == pytest.approx(0.2)
    assert values["target_hpd_overlap"] is True


def test_vmf_fit_is_fixed_size_and_reports_fit_quality():
    context = _context(
        [0.85, 0.05, 0.05, 0.05],
        features=["sky_vmf_fit"],
    )
    values, status = run_likelihood_features(
        context, resolve_extension_plan(context.config)
    )

    assert status["sky_vmf_fit"]["ok"] is True
    assert values["sky_vmf_longitude_deg"] == pytest.approx(0.0, abs=1e-6)
    assert values["sky_vmf_latitude_deg"] == pytest.approx(0.0, abs=1e-6)
    assert values["sky_vmf_kappa"] > 0.0
    assert values["sky_vmf_kl_nats"] >= 0.0
    assert values["sky_vmf_area_90_deg2"] > values["sky_vmf_area_50_deg2"]
    assert values["sky_vmf_area_90_deg2"] >= context.pixel_area_deg2


def test_sparse_map_requires_heavy_opt_in_and_reports_truncation():
    context = _context(
        [0.6, 0.2, 0.11, 0.09],
        features=["sky_sparse_map"],
    )
    context.config.likelihood_sky_sparse_level = 0.9
    context.config.likelihood_sky_sparse_max_pixels = 2
    with pytest.raises(ValueError, match="Heavy likelihood feature"):
        resolve_extension_plan(context.config)

    context.config.likelihood_allow_heavy_features = True
    values, status = run_likelihood_features(
        context, resolve_extension_plan(context.config)
    )
    assert status["sky_sparse_map"]["ok"] is True
    np.testing.assert_array_equal(values["sky_sparse_indices"], [0, 1])
    assert values["sky_sparse_hpd_pixel_count"] == 3
    assert values["sky_sparse_stored_pixel_count"] == 2
    assert values["sky_sparse_stored_mass"] == pytest.approx(0.8)
    assert values["sky_sparse_truncated"] is True


def test_lightweight_feature_sidecar_does_not_require_full_sky_map(tmp_path):
    skymap = SimpleNamespace(
        likelihood_features={"sky_area_90_deg2": 12.5},
        likelihood_feature_status={"sky_area": {"ok": True, "version": 1}},
        likelihood_cut_metrics=None,
    )
    trigger_folder = tmp_path / "trigger"
    save_trigger(
        str(trigger_folder),
        (SimpleNamespace(hash_id="event"), SimpleNamespace(), skymap),
        save_cluster=False,
        save_sky_map=False,
        save_likelihood_features=True,
    )

    assert not (trigger_folder / "skymap_statistics.json").exists()
    payload = orjson.loads((trigger_folder / "likelihood_features.json").read_bytes())
    assert payload["features"]["sky_area_90_deg2"] == 12.5


def test_legacy_error_region_helper_populates_searched_location():
    cluster = SimpleNamespace(sky_area=[])
    result = compute_sky_error_region(
        cluster,
        [0.6, 0.2, 0.11, 0.09],
        pixel_area_deg2=4.0,
        searched_sky_index=2,
    )

    assert result is cluster.sky_area
    assert len(result) == 11
    assert result[0] == pytest.approx(np.sqrt(12.0))
    assert result[10] == pytest.approx(0.91)


def test_unknown_extension_names_fail_during_plan_resolution():
    context = _context([0.6, 0.2, 0.11, 0.09])
    context.config.likelihood_features = ["unknown_feature"]
    with pytest.raises(ValueError, match="Unknown likelihood feature"):
        resolve_extension_plan(context.config)


def test_flat_yaml_schema_accepts_known_extensions_and_rejects_unknown_names():
    parameters = {
        "analysis": "2G",
        "ifo": ["L1", "H1"],
        "refIFO": "L1",
        "likelihood_features": ["sky_area", "target_sky_metrics"],
        "likelihood_cuts": ["target_sky_consistency"],
        "likelihood_target_region": _fixed_target(90),
    }
    validate(instance=parameters, schema=schema)

    parameters["likelihood_features"] = ["not_registered"]
    with pytest.raises(ValidationError):
        validate(instance=parameters, schema=schema)


def test_config_requires_a_target_region_for_target_consumers(tmp_path, monkeypatch):
    monkeypatch.setattr(
        Config,
        "add_derived_key",
        lambda self: setattr(self, "MRAcatalog", "unused"),
    )
    for method_name in (
        "check_xtalk_file",
        "check_MRA_catalog",
        "check_lagStep",
        "check_analyze_injection_only",
    ):
        monkeypatch.setattr(Config, method_name, lambda *args, **kwargs: None)

    config_path = tmp_path / "missing_target.yaml"
    config_path.write_text(
        "analysis: 2G\n"
        "ifo: [L1, H1]\n"
        "refIFO: L1\n"
        "likelihood_features: [target_sky_metrics]\n"
    )
    with pytest.raises(ValueError, match="likelihood_target_region is required"):
        Config().load_from_yaml(config_path)

    config_path.write_text(
        "analysis: 2G\n"
        "ifo: [L1, H1]\n"
        "refIFO: L1\n"
        "likelihood_features: [target_sky_metrics]\n"
        "likelihood_target_region:\n"
        "  type: Fixed\n"
        "  coordsys: geo\n"
        "  coordinates:\n"
        "    longitude: 0 deg\n"
        "    latitude: 0 deg\n"
    )
    config = Config()
    config.load_from_yaml(config_path)
    assert config.likelihood_target_region["type"] == "Fixed"
