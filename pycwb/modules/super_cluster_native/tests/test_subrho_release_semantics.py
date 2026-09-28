import importlib
import pytest
import numpy as np


@pytest.mark.parametrize("threshold", [-6.5, 6.5])
@pytest.mark.parametrize("rho,expected", [(6.0, False), (6.5, False), (7.0, True)])
def test_subnet_uses_absolute_subrho_threshold(monkeypatch, threshold, rho, expected):
    module = importlib.import_module("pycwb.modules.super_cluster_native.sub_net_cut")
    monkeypatch.setattr(module, "optimze_sky_loc", lambda *args: (0, 0, 20, 0, 0, 0, 1, 0))
    monkeypatch.setattr(module, "mra_statistics", lambda *args: (1, rho, 2, 0, 0, 0))
    result = module._sub_net_cut_prepared_arrays(
        np.zeros((1, 2)),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        1.0,
        1.0,
        2,
        1,
        0.5,
        0.0,
        0.0,
        threshold,
    )
    assert result["subrho_passed"] is expected
