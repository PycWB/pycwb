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
    build_legacy_sky_area,
    build_sky_probability,
    build_target_sky_indices,
    compute_target_sky_metrics,
    hpd_region,
    rank_sky_probability,
    resolve_extension_plan,
    run_likelihood_cuts,
    run_likelihood_features,
    sky_pixel_area_deg2,
    trigger_gps,
    validate_sky_temperature,
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


def _case(
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
    return SimpleNamespace(
        config=config,
        cluster=cluster,
        setup={"segment_start_gps": 1_000_000_000.0},
        skymap=skymap,
        sky_valid_indices=valid_indices,
        phi=phi,
        latitude=latitude,
        healpix_order=None,
    )


def _products(case, plan=None):
    if plan is None:
        plan = resolve_extension_plan(case.config)
    temperature = validate_sky_temperature(
        case.config.likelihood_sky_temperature
    )
    probability, evaluated = build_sky_probability(
        case.skymap.nSkyStat,
        case.sky_valid_indices,
        temperature,
    )
    ranked = cumulative = None
    if "sky_ranking" in plan.required_products:
        ranked, cumulative = rank_sky_probability(probability, evaluated)

    target_indices = target_metrics = None
    needs_target_metrics = "target_metrics" in plan.required_products
    if "target_indices" in plan.required_products and (
        case.config.likelihood_target_region or needs_target_metrics
    ):
        target_indices = build_target_sky_indices(
            case.config.likelihood_target_region,
            case.phi,
            case.latitude,
            t_ref=trigger_gps(case.cluster, case.setup),
        )
        if needs_target_metrics:
            target_metrics = compute_target_sky_metrics(
                case.skymap.nSkyStat,
                probability,
                evaluated,
                target_indices,
                target_level=case.config.likelihood_target_level,
                ranked_sky_indices=ranked,
                ranked_cumulative_probability=cumulative,
            )
    return SimpleNamespace(
        probability=probability,
        evaluated=evaluated,
        ranked=ranked,
        cumulative=cumulative,
        target_indices=target_indices,
        target_metrics=target_metrics,
        temperature=temperature,
    )


def _run_features(case):
    plan = resolve_extension_plan(case.config)
    products = _products(case, plan)
    return run_likelihood_features(
        plan,
        config=case.config,
        sky_probability=products.probability,
        evaluated_sky_indices=products.evaluated,
        phi_geo_arr=case.phi,
        latitude_arr=case.latitude,
        healpix_order=case.healpix_order,
        l_max=case.skymap.l_max,
        ranked_sky_indices=products.ranked,
        ranked_cumulative_probability=products.cumulative,
        target_metrics=products.target_metrics,
    ), products


def _run_cuts(case, *, stage):
    plan = resolve_extension_plan(case.config)
    products = _products(case, plan)
    return run_likelihood_cuts(
        plan,
        stage=stage,
        config=case.config,
        target_metrics=products.target_metrics,
    )


def test_probability_excludes_masked_and_nonpositive_sky_pixels():
    case = _case([0.6, 0.2, 0.1, 0.1], valid_indices=[0, 2])
    probability, evaluated = build_sky_probability(
        case.skymap.nSkyStat,
        case.sky_valid_indices,
        temperature=1.0,
    )

    np.testing.assert_array_equal(evaluated, [0, 2])
    assert probability[1] == 0.0
    assert probability[3] == 0.0
    assert float(np.sum(probability)) == pytest.approx(1.0)
    assert probability[0] > probability[2]


def test_shared_probability_and_hpd_products_are_explicit_arrays():
    case = _case(
        [0.6, 0.2, 0.11, 0.09],
        features=["sky_area"],
    )
    products = _products(case)

    assert len(hpd_region(0.5, products.ranked, products.cumulative)) == 1
    assert len(hpd_region(0.9, products.ranked, products.cumulative)) == 3


def test_probability_temperature_is_explicit_and_persisted_with_features():
    case = _case([0.8, 0.1, 0.05, 0.05], features=["sky_area"])
    native_peak = float(_products(case).probability[0])

    warmer = _case([0.8, 0.1, 0.05, 0.05], features=["sky_area"])
    warmer.config.likelihood_sky_temperature = 4.0
    (values, status), products = _run_features(warmer)

    assert float(products.probability[0]) < native_peak
    assert values
    assert status["sky_area"]["ok"] is True
    metadata = {
        "sky_probability_temperature": 4.0,
        "evaluated_sky_pixel_count": 4,
    }
    assert metadata == {
        "sky_probability_temperature": products.temperature,
        "evaluated_sky_pixel_count": len(products.evaluated),
    }


def test_sky_area_feature_is_optional_and_uses_compact_scalars():
    case = _case(
        [0.6, 0.2, 0.11, 0.09],
        features=["sky_area"],
    )
    (values, status), products = _run_features(case)

    pixel_area = 4.0 * np.pi * (180.0 / np.pi) ** 2 / 4.0
    assert values["sky_area_50_deg2"] == pytest.approx(pixel_area)
    assert values["sky_area_90_deg2"] == pytest.approx(3.0 * pixel_area)
    assert status["sky_area"]["ok"] is True

    case.cluster.sky_area = build_legacy_sky_area(
        pixel_area_deg2=pixel_area,
        sky_probability=products.probability,
        evaluated_sky_indices=products.evaluated,
        ranked_sky_indices=products.ranked,
        ranked_cumulative_probability=products.cumulative,
        target_sky_indices=products.target_indices,
    )
    assert len(case.cluster.sky_area) == 11
    assert case.cluster.sky_area[5] == pytest.approx(np.sqrt(pixel_area))
    assert case.cluster.sky_area[9] == pytest.approx(np.sqrt(3.0 * pixel_area))


def test_legacy_sky_area_keeps_target_searched_location():
    case = _case(
        [0.6, 0.2, 0.11, 0.09],
        features=["sky_area"],
        target_region=_fixed_target(90),
    )
    (_, status), products = _run_features(case)
    assert status["sky_area"]["ok"] is True

    legacy = build_legacy_sky_area(
        pixel_area_deg2=sky_pixel_area_deg2(4),
        sky_probability=products.probability,
        evaluated_sky_indices=products.evaluated,
        ranked_sky_indices=products.ranked,
        ranked_cumulative_probability=products.cumulative,
        target_sky_indices=products.target_indices,
    )
    assert legacy[0] > 0.0
    assert legacy[10] > 0.0


def test_default_plan_adds_no_feature_output_or_hpd_sort():
    case = _case([0.6, 0.2, 0.11, 0.09])
    plan = resolve_extension_plan(case.config)
    (values, status), products = _run_features(case)

    assert values == {}
    assert status == {}
    assert plan.required_products == ()
    assert products.ranked is None
    assert case.cluster.sky_area == []


def test_target_90_percent_consistency_accepts_overlap_and_rejects_no_overlap():
    probability = [0.7, 0.2, 0.08, 0.02]

    overlapping = _case(
        probability,
        target_region=_fixed_target(90),
        cuts=["target_sky_consistency"],
    )
    overlap_result = _run_cuts(overlapping, stage="post_sky")
    assert overlap_result.passed is True
    metrics = overlap_result.metrics["target_sky_consistency"]
    assert metrics["target_hpd_overlap"] is True
    assert metrics["target_credible_level"] == pytest.approx(0.9, abs=1e-6)

    outside = _case(
        probability,
        target_region=_fixed_target(-90),
        cuts=["target_sky_consistency"],
    )
    outside_result = _run_cuts(outside, stage="post_sky")
    assert outside_result.passed is False
    metrics = outside_result.metrics["target_sky_consistency"]
    assert metrics["target_hpd_overlap"] is False
    assert metrics["target_credible_level"] > 0.9


def test_target_metrics_are_an_optional_output_not_a_catalog_requirement():
    case = _case(
        [0.7, 0.2, 0.08, 0.02],
        target_region=_fixed_target(90),
        features=["target_sky_metrics"],
    )
    (values, status), _ = _run_features(case)

    assert status["target_sky_metrics"]["ok"] is True
    assert values["target_probability_mass"] == pytest.approx(0.2)
    assert values["target_hpd_overlap"] is True


def test_vmf_fit_is_fixed_size_and_reports_fit_quality():
    case = _case(
        [0.85, 0.05, 0.05, 0.05],
        features=["sky_vmf_fit"],
    )
    (values, status), _ = _run_features(case)

    assert status["sky_vmf_fit"]["ok"] is True
    assert values["sky_vmf_longitude_deg"] == pytest.approx(0.0, abs=1e-6)
    assert values["sky_vmf_latitude_deg"] == pytest.approx(0.0, abs=1e-6)
    assert values["sky_vmf_kappa"] > 0.0
    assert values["sky_vmf_kl_nats"] >= 0.0
    assert values["sky_vmf_area_90_deg2"] > values["sky_vmf_area_50_deg2"]
    assert values["sky_vmf_area_90_deg2"] >= sky_pixel_area_deg2(4)


def test_sparse_map_requires_heavy_opt_in_and_reports_truncation():
    case = _case(
        [0.6, 0.2, 0.11, 0.09],
        features=["sky_sparse_map"],
    )
    case.config.likelihood_sky_sparse_level = 0.9
    case.config.likelihood_sky_sparse_max_pixels = 2
    with pytest.raises(ValueError, match="Heavy likelihood feature"):
        resolve_extension_plan(case.config)

    case.config.likelihood_allow_heavy_features = True
    (values, status), _ = _run_features(case)
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
    case = _case([0.6, 0.2, 0.11, 0.09])
    case.config.likelihood_features = ["unknown_feature"]
    with pytest.raises(ValueError, match="Unknown likelihood feature"):
        resolve_extension_plan(case.config)


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
