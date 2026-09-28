"""Protect estimator dispatch, reset semantics and seed propagation."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from pycwb.modules.likelihoodWP.chirp_micropixel import ChirpResult

likelihood = import_module("pycwb.modules.likelihoodWP.likelihood")
chirp = import_module("pycwb.modules.likelihoodWP.chirp_micropixel")
FIELDS = (
    "mchirp",
    "mchirp_error",
    "chirp_merger_time",
    "chirp_merger_time_error",
    "chirp_ellipticity",
    "chirp_energy_fraction",
    "chirp_symmetry",
)


def cluster_with_stale_metadata():
    return SimpleNamespace(cluster_meta=SimpleNamespace(**dict.fromkeys(FIELDS, 99.0)), pixel_arrays=object())


@pytest.mark.parametrize("overrides", [{"optim": True}, {"cfg_search": "x"}, {"Search": "Burst"}])
def test_ineligible_native_search_clears_every_chirp_field(monkeypatch, overrides):
    cluster = cluster_with_stale_metadata()
    config = SimpleNamespace(**(dict(optim=False, cfg_search="i", Search="CBC", rateANA=4096) | overrides))

    def unexpected(*args, **kwargs):
        pytest.fail("A disabled native search must not call either estimator")

    monkeypatch.setattr(chirp, "estimate_chirp", unexpected)
    monkeypatch.setattr(likelihood, "_update_chirp_mass_statistics", unexpected)
    likelihood._update_cluster_chirp_statistics(
        cluster,
        config,
        xgb_rho_mode=True,
        chirp_seed=42,
        use_native_chirp=True,
    )
    assert [getattr(cluster.cluster_meta, name) for name in FIELDS] == [0.0] * len(FIELDS)


def test_native_seed_and_result_mapping(monkeypatch):
    cluster = cluster_with_stale_metadata()
    config = SimpleNamespace(optim=False, cfg_search="i", Search="CBC", rateANA=4096)

    def estimate(pixels, rate, seed):
        assert pixels is cluster.pixel_arrays
        assert (rate, seed) == (4096, 42)
        assert all(getattr(cluster.cluster_meta, name) == 0.0 for name in FIELDS)
        return ChirpResult(1, 2, 3, 4, 5, 6, 7)

    monkeypatch.setattr(chirp, "estimate_chirp", estimate)
    likelihood._update_cluster_chirp_statistics(
        cluster,
        config,
        xgb_rho_mode=True,
        chirp_seed=42,
        use_native_chirp=True,
    )
    assert [getattr(cluster.cluster_meta, name) for name in FIELDS] == list(range(1, 8))


@pytest.mark.parametrize("native,xgb", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("pattern", [0, 10])
def test_legacy_dispatch_keeps_pattern_and_metadata_contract(monkeypatch, native, xgb, pattern):
    cluster = cluster_with_stale_metadata()
    calls = []

    def update(received, **kwargs):
        assert received is cluster
        calls.append(kwargs)

    monkeypatch.setattr(likelihood, "_update_chirp_mass_statistics", update)
    likelihood._update_cluster_chirp_statistics(
        cluster,
        SimpleNamespace(pattern=pattern),
        xgb_rho_mode=xgb,
        chirp_seed=42,
        use_native_chirp=native,
    )
    assert calls == [{"xgb_rho_mode": xgb, "pat0": pattern == 0}]
    assert all(getattr(cluster.cluster_meta, name) == 99.0 for name in FIELDS)


def test_missing_configuration_skips_native_estimation():
    cluster = cluster_with_stale_metadata()
    likelihood._update_cluster_chirp_statistics(
        cluster,
        None,
        xgb_rho_mode=True,
        chirp_seed=1,
        use_native_chirp=True,
    )
    assert all(getattr(cluster.cluster_meta, name) == 0.0 for name in FIELDS)
