import numpy as np
from types import SimpleNamespace

from pycwb.modules.cwb_coherence.lag_plan import (
    build_lag_plan_from_config,
    build_lag_plan_from_network,
)


class DummyIfo:
    def __init__(self, shifts):
        self.lagShift = SimpleNamespace(data=np.asarray(shifts, dtype=float))


class DummyNet:
    def __init__(self, n_lag, shifts_by_ifo):
        self.nLag = n_lag
        self._ifos = [DummyIfo(s) for s in shifts_by_ifo]

    def get_ifo(self, idx):
        return self._ifos[idx]


class DummyTFMap:
    def __init__(self, n_freq=4, n_time=64, dt=0.5):
        self.data = np.zeros((n_freq, n_time), dtype=float)
        self.dt = float(dt)


def test_build_lag_plan_from_network_pads_and_truncates():
    net = DummyNet(n_lag=3, shifts_by_ifo=[[0.0, 1.0, 2.0, 3.0], [10.0]])

    plan = build_lag_plan_from_network(net, n_ifo=2)

    assert plan.n_lag == 3
    assert plan.lag_shifts.shape == (3, 2)
    np.testing.assert_allclose(plan.lag_shifts[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(plan.lag_shifts[:, 1], [10.0, 0.0, 0.0])


def test_build_lag_plan_from_config_standard_mode_basic():
    config = SimpleNamespace(
        nIFO=2,
        lagSize=4,
        lagStep=0.5,
        lagOff=1,
        lagMax=0,
        segEdge=0.0,
        lagSite=None,
    )
    tf_maps = [DummyTFMap(n_time=64, dt=0.5), DummyTFMap(n_time=64, dt=0.5)]

    plan = build_lag_plan_from_config(config, tf_maps)

    assert plan.n_lag == 4
    assert plan.lag_shifts.shape == (4, 2)
    np.testing.assert_allclose(plan.lag_shifts[:, 0], [0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(plan.lag_shifts[:, 1], [0.0, 0.0, 0.0, 0.0])


def test_build_lag_plan_from_config_applies_segment_boundary_filter():
    config = SimpleNamespace(
        nIFO=2,
        lagSize=4,
        lagStep=1.0,
        lagOff=4,
        lagMax=0,
        segEdge=2.0,
        lagSite=None,
    )
    tf_maps = [DummyTFMap(n_time=10, dt=1.0), DummyTFMap(n_time=10, dt=1.0)]

    plan = build_lag_plan_from_config(config, tf_maps)

    assert plan.n_lag == 2
    assert plan.lag_shifts.shape == (2, 2)
    np.testing.assert_allclose(plan.lag_shifts[:, 0], [4.0, 5.0])


def test_build_lag_plan_from_config_extended_mode_properties():
    config = SimpleNamespace(
        nIFO=3,
        lagSize=6,
        lagStep=1.0,
        lagOff=0,
        lagMax=5,
        segEdge=0.0,
        lagSite=None,
    )
    tf_maps = [DummyTFMap(n_time=200, dt=1.0) for _ in range(3)]

    plan = build_lag_plan_from_config(config, tf_maps)

    assert 1 <= plan.n_lag <= config.lagSize
    assert plan.lag_shifts.shape == (plan.n_lag, config.nIFO)
    assert np.all(plan.lag_shifts >= 0.0)
    assert np.allclose(plan.lag_shifts, np.round(plan.lag_shifts / config.lagStep) * config.lagStep)
    assert np.unique(plan.lag_shifts, axis=0).shape[0] == plan.n_lag
