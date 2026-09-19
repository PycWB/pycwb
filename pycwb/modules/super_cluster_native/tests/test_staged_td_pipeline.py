"""Verify the boundary between subnet sampling and final likelihood vectors."""

import importlib
from types import SimpleNamespace
import numpy as np
import pytest
from pycwb.types.network_cluster import Cluster
from .test_supercluster_optimizations import _make_pixel_arrays


@pytest.mark.parametrize("pattern", [0, 1])
@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("aligned", [False, True])
def test_survivors_receive_fine_vectors(monkeypatch, pattern, keep, aligned):
    sc = importlib.import_module("pycwb.modules.super_cluster_native.super_cluster")
    monkeypatch.setenv("PYCWB_STAGED_TD", "1")
    calls = []

    class Inputs:
        def extract_td_vecs(self, indices, K, delay_stride=1):
            calls.append((len(indices), K, delay_stride))
            phase = np.arange(-K, K + 1, dtype=np.float32) * delay_stride
            return np.tile(np.concatenate([phase, phase + 100]), (len(indices), 1))

    clusters = [
        Cluster(pixel_arrays=_make_pixel_arrays(time=[16], frequency=[2]), cluster_status=-1),
        Cluster(pixel_arrays=_make_pixel_arrays(time=[32], frequency=[2]), cluster_status=-1),
    ]
    fragment = SimpleNamespace(clusters=clusters)
    config = SimpleNamespace(
        nIFO=2,
        upTDF=4,
        Acore=1.0,
        subacor=1.0,
        pattern=pattern,
        TFgap=1.0,
        Tgap=1.0,
        Fgap=1.0,
        subrho=4.5,
        netRHO=4.5,
        LOUD=100,
        subnet=0.5,
        subcut=0.5,
        subnorm=2.5,
    )
    delay = 4 if aligned else 3
    ml = np.array([[0, 0], [delay, -delay]], dtype=np.int32)
    setup = dict(K_td=9, ml=ml, FP=np.ones((2, 2)), FX=np.ones((2, 2)), n_sky=2)
    monkeypatch.setattr(sc, "supercluster", lambda cs, *args: cs)
    monkeypatch.setattr(sc, "defragment", lambda cs, *args: cs)

    def subnet(cs, loud, actual_ml, *args, **kw):
        np.testing.assert_array_equal(actual_ml, ml // 4 if aligned else ml)
        for c in cs:
            td = c.pixel_arrays.td_amp_dense()
            assert td.shape[-1] == (10 if aligned else 38)
            half = td.shape[-1] // 2
            for ifo in range(2):
                np.testing.assert_array_equal(td[0, ifo, actual_ml[ifo] + half // 2], ml[ifo])
        return cs[:1] if keep else []

    monkeypatch.setattr(sc, "apply_subnet_cut", subnet)
    result = sc.supercluster_single_lag(setup, config, [fragment], 0, None, {16: [Inputs(), Inputs()]})
    assert len(result.clusters) == int(keep)
    if keep:
        assert result.clusters[0].pixel_arrays.td_amp_dense().shape == (1, 2, 38)
        np.testing.assert_array_equal(result.clusters[0].pixel_arrays.get_td_amp(0, 0)[:19], np.arange(-9, 10))
    expected = [(2, 2, 4)] * 2 if aligned else [(2, 9, 1)] * 2
    if aligned and keep:
        expected += [(1, 9, 1)] * 2
    assert calls == expected
