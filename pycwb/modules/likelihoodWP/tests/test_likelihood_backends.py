"""Contracts for the shared likelihood flow and selectable backends."""

from types import SimpleNamespace

import numpy as np
import pytest
from jsonschema import ValidationError, validate

from pycwb.config.config import Config
from pycwb.constants.user_parameters_schema import schema
from pycwb.modules.likelihoodWP.backends import (
    LikelihoodKernels,
    get_likelihood_backend,
    normalize_likelihood_backend,
)
from pycwb.modules.likelihoodWP.extensions import (
    CutResult,
    LikelihoodExtensionPlan,
)
from pycwb.modules.likelihoodWP.likelihood import evaluate_cluster_likelihood
from pycwb.modules.likelihoodWP.typing import SkyMapStatistics


class _Pixels(list):
    def populate_noise_rms(self, maps):
        self.noise_maps = maps


def _cluster():
    return SimpleNamespace(
        pixel_arrays=_Pixels([object(), object()]),
        cluster_meta=SimpleNamespace(
            l_max=0,
            theta=0.0,
            phi=0.0,
            c_time=0.0,
            c_freq=0.0,
        ),
        cluster_time=12.5,
        cluster_freq=64.0,
        cluster_status=0,
        sky_time_delay=[],
        sky_area=[],
    )


def _skymap():
    values = np.zeros(3, dtype=np.float32)
    return SkyMapStatistics(
        l_max=1,
        nAntennaPrior=values.copy(),
        nAlignment=values.copy(),
        nLikelihood=values.copy(),
        nNullEnergy=values.copy(),
        nCorrEnergy=values.copy(),
        nCorrelation=values.copy(),
        nSkyStat=np.array([1.0, 3.0, 2.0], dtype=np.float32),
        nDisbalance=values.copy(),
        nNetIndex=values.copy(),
        nEllipticity=values.copy(),
        nPolarisation=values.copy(),
        sky_stat_max=3.0,
    )


class _XTalk:
    def get_xtalk_pixels(self, pixel_arrays, check):
        return (
            np.zeros((2, 2), dtype=np.int32),
            np.zeros((1, 8), dtype=np.float32),
        )


def _recording_kernels():
    calls = []

    def calculate_dpf_regulator(*args):
        calls.append("dpf")
        return 0.25

    def scan_sky(*args):
        calls.append("scan")
        skymap = _skymap()
        return (
            skymap.l_max,
            skymap.nAntennaPrior,
            skymap.nAlignment,
            skymap.nLikelihood,
            skymap.nNullEnergy,
            skymap.nCorrEnergy,
            skymap.nCorrelation,
            skymap.nSkyStat,
            skymap.nDisbalance,
            skymap.nNetIndex,
            skymap.nEllipticity,
            skymap.nPolarisation,
            skymap.sky_stat_max,
        )

    def statistics_at_best_fit(*args, **kwargs):
        calls.append("best_fit")
        return SimpleNamespace(pixel_mask=np.ones(2, dtype=np.int32))

    return LikelihoodKernels(
        name="recording",
        calculate_dpf_regulator=calculate_dpf_regulator,
        scan_sky=scan_sky,
        statistics_at_best_fit=statistics_at_best_fit,
    ), calls


def _patch_data_prep(monkeypatch, flow, calls):
    def extract(*args, **kwargs):
        calls.append("prepare")
        return (
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 2, 3), dtype=np.float32),
            np.ones((2, 2, 3), dtype=np.float32),
            np.ones((2, 2, 3), dtype=np.float32),
        )

    monkeypatch.setattr(flow, "extract_pixel_time_delay_data", extract)


def _patch_reconstruction(monkeypatch, flow, calls):
    monkeypatch.setattr(flow, "_create_wdm_set_python", lambda config: [])
    monkeypatch.setattr(
        flow,
        "populate_detection_statistics",
        lambda *args, **kwargs: calls.append("reconstruct"),
    )
    monkeypatch.setattr(
        flow,
        "update_chirp_mass_statistics",
        lambda *args, **kwargs: None,
    )


def _setup():
    extension_plan = LikelihoodExtensionPlan(
        feature_names=(),
        post_sky_cut_names=(),
        post_reconstruction_cut_names=(),
        feature_failure="warn",
        allow_heavy_features=False,
    )
    return {
        "network_energy_threshold": 1.0,
        "xgb_rho_mode": False,
        "gamma_regulator": 0.2,
        "delta_regulator": 0.1,
        "net_rho_threshold": 0.0,
        "netEC_threshold": 0.0,
        "netCC": 0.5,
        "ml": np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int32),
        "FP_t": np.ones((3, 2), dtype=np.float32),
        "FX_t": np.ones((3, 2), dtype=np.float32),
        "n_sky": 3,
        "sky_valid_indices": np.arange(3, dtype=np.int64),
        "phi_geo_arr": np.array([0.0, 1.0, 2.0]),
        "latitude_arr": np.array([0.0, 0.1, -0.1]),
        "healpix_order": None,
        "ml_big_cluster": None,
        "likelihood_extension_plan": extension_plan,
    }


def _config():
    return SimpleNamespace(
        precision=0,
        nRES=1,
        pattern=10,
        likelihood_features=[],
        likelihood_cuts=[],
        likelihood_feature_failure="warn",
        likelihood_allow_heavy_features=False,
        likelihood_sky_levels=[0.5, 0.9],
        likelihood_sky_temperature=1.0,
        likelihood_target_region=None,
    )


def test_shared_flow_calls_only_the_backend_numerical_boundary(monkeypatch):
    import importlib

    flow = importlib.import_module("pycwb.modules.likelihoodWP.likelihood")

    backend, calls = _recording_kernels()
    cluster = _cluster()
    _patch_data_prep(monkeypatch, flow, calls)
    _patch_reconstruction(monkeypatch, flow, calls)
    monkeypatch.setattr(flow, "sky_valid_indices_for_cluster", lambda *a, **k: None)
    monkeypatch.setattr(
        flow,
        "get_likelihood_rejection_reason",
        lambda *args, **kwargs: None,
    )

    result, skymap = evaluate_cluster_likelihood(
        nIFO=2,
        cluster=cluster,
        config=_config(),
        setup=_setup(),
        xtalk=_XTalk(),
        backend=backend,
        cluster_id=7,
    )

    assert result is cluster
    assert calls == ["prepare", "dpf", "scan", "best_fit", "reconstruct"]
    assert cluster.cluster_status == -1
    assert cluster.cluster_meta.l_max == 1
    assert cluster.sky_time_delay == [1.0, 1.0]
    assert skymap.likelihood_backend == "recording"
    assert set(skymap.stage_timings) >= {
        "data_prep", "sky_scan", "likelihood_features", "total"
    }


def test_shared_flow_computes_selected_feature_without_context(monkeypatch):
    import importlib

    flow = importlib.import_module("pycwb.modules.likelihoodWP.likelihood")
    backend, calls = _recording_kernels()
    cluster = _cluster()
    config = _config()
    config.likelihood_features = ["sky_area"]
    setup = _setup()
    setup.pop("likelihood_extension_plan")

    _patch_data_prep(monkeypatch, flow, calls)
    _patch_reconstruction(monkeypatch, flow, calls)
    monkeypatch.setattr(flow, "sky_valid_indices_for_cluster", lambda *a, **k: None)
    monkeypatch.setattr(
        flow,
        "get_likelihood_rejection_reason",
        lambda *args, **kwargs: None,
    )

    result, skymap = evaluate_cluster_likelihood(
        nIFO=2,
        cluster=cluster,
        config=config,
        setup=setup,
        xtalk=_XTalk(),
        backend=backend,
    )

    assert result is cluster
    assert len(cluster.sky_area) == 11
    assert skymap.likelihood_features["sky_area_90_deg2"] > 0.0
    assert skymap.likelihood_feature_status["sky_area"]["ok"] is True


def test_post_sky_rejection_skips_shared_reconstruction(monkeypatch):
    import importlib

    flow = importlib.import_module("pycwb.modules.likelihoodWP.likelihood")

    backend, calls = _recording_kernels()
    _patch_data_prep(monkeypatch, flow, calls)
    monkeypatch.setattr(flow, "sky_valid_indices_for_cluster", lambda *a, **k: None)
    monkeypatch.setattr(
        flow,
        "get_likelihood_rejection_reason",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        flow,
        "run_likelihood_cuts",
        lambda *a, stage, **k: CutResult(
            passed=stage != "post_sky", reason="target mismatch"
        ),
    )
    monkeypatch.setattr(
        flow,
        "populate_detection_statistics",
        lambda *args, **kwargs: pytest.fail(
            "reconstruction should have been skipped"
        ),
    )

    result = evaluate_cluster_likelihood(
        nIFO=2,
        cluster=_cluster(),
        config=_config(),
        setup=_setup(),
        xtalk=_XTalk(),
        backend=backend,
    )

    assert result == (None, None)
    assert calls == ["prepare", "dpf", "scan", "best_fit"]


def test_backend_names_are_normalized_and_invalid_names_fail():
    assert normalize_likelihood_backend(None) == "numba"
    assert normalize_likelihood_backend("CPU") == "numba"
    assert normalize_likelihood_backend("GPU") == "jax"
    assert get_likelihood_backend("numba").name == "numba"
    with pytest.raises(ValueError, match="likelihood_backend"):
        normalize_likelihood_backend("cuda")


def test_numba_selection_exposes_real_kernels_without_method_wrappers():
    from pycwb.modules.likelihoodWP.dpf import calculate_dpf
    from pycwb.modules.likelihoodWP.sky_scan import scan_sky_for_best_fit
    from pycwb.modules.likelihoodWP.sky_statistics import (
        compute_statistics_at_sky_position,
    )

    kernels = get_likelihood_backend("numba")
    assert kernels.calculate_dpf_regulator is calculate_dpf
    assert kernels.scan_sky is scan_sky_for_best_fit
    assert kernels.statistics_at_best_fit is compute_statistics_at_sky_position


def test_config_and_schema_expose_flat_backend_switch():
    assert Config().likelihood_backend == "numba"
    backend_schema = schema["properties"]["likelihood_backend"]
    validate("jax", backend_schema)
    with pytest.raises(ValidationError):
        validate("cuda", backend_schema)


def test_public_likelihood_dispatches_from_segment_setup(monkeypatch):
    import importlib

    facade = importlib.import_module("pycwb.modules.likelihoodWP.likelihood")
    selected, calls = _recording_kernels()
    observed = {}

    def fake_resolve(name):
        observed["name"] = name
        return selected

    monkeypatch.setattr(facade, "get_likelihood_backend", fake_resolve)
    monkeypatch.setattr(
        facade, "sky_valid_indices_for_cluster", lambda *a, **k: None
    )
    monkeypatch.setattr(
        facade,
        "get_likelihood_rejection_reason",
        lambda *args, **kwargs: None,
    )
    _patch_data_prep(monkeypatch, facade, calls)
    _patch_reconstruction(monkeypatch, facade, calls)

    setup = _setup()
    setup["likelihood_backend"] = "jax"

    result = facade.likelihood(
        2,
        _cluster(),
        SimpleNamespace(likelihood_backend="numba"),
        setup=setup,
        xtalk=_XTalk(),
    )

    cluster, skymap = result
    assert cluster is not None
    assert skymap.likelihood_backend == "recording"
    assert observed == {"name": "jax"}
