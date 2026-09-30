"""Regression against cWB's analysis-rate subnet delay table."""

from pathlib import Path
from pycwb.config import Config

import numpy as np
import pytest

from pycwb.modules.super_cluster_native.super_cluster import setup_supercluster

REFERENCE = Path(__file__).with_name("subnet_grid_reference.npz")


def config(upsampling, healpix):
    return _config(
        ifo=["L1", "H1"],
        refIFO="L1",
        rateANA=4096,
        TDRate=4096 * upsampling,
        upTDF=upsampling,
        TDSize=12,
        max_delay=0.010012846152266964,
        healpix=healpix,
        MIN_SKYRES_HEALPIX=4,
    )


@pytest.mark.parametrize("upsampling", [1, 2, 4])
@pytest.mark.parametrize("healpix", [4, 5])
def test_subnet_grid_matches_release(upsampling, healpix):
    setup = setup_supercluster(config(upsampling, healpix), 1387221730)
    with np.load(REFERENCE) as data:
        expected = data["cwb_subnet_delays"] * upsampling
    # Indices address the existing high-rate TD buffers, but sample only the
    # release's analysis-rate delay grid, even when both sky maps have order 4.
    np.testing.assert_array_equal(setup["ml"], expected)
    np.testing.assert_array_equal(setup["ml_subnet_i32"], expected)
    assert setup["ml_subnet_i32"].dtype == np.int32
    assert setup["n_sky"] == 3072


def test_likelihood_keeps_fine_delay_grid():
    setup = setup_supercluster(config(4, 5), 1387221730)
    with np.load(REFERENCE) as data:
        np.testing.assert_array_equal(setup["ml_likelihood"], data["native_likelihood_delays"])
    assert np.any(setup["ml_likelihood"] % 4 != 0)
    assert setup["K_td"] == 165


@pytest.mark.parametrize("rate", [1024, 2048, 4096, 8192])
@pytest.mark.parametrize("upsampling", [1, 2, 4, 8])
def test_both_sky_grids_fit_time_delay_buffers(rate, upsampling):
    cfg = config(upsampling, 4)
    cfg.rateANA = rate
    cfg.TDRate = rate * upsampling
    setup = setup_supercluster(cfg, 1387221730)
    half_width = setup["K_td"]
    for key in ("ml", "ml_likelihood"):
        indices = setup[key] + half_width
        assert indices.min() >= 0
        assert indices.max() < 2 * half_width + 1
    # Staged TD extraction samples the same buffer at analysis-rate steps.
    coarse_half_width = half_width // upsampling
    coarse_indices = setup["ml"] // upsampling + coarse_half_width
    assert coarse_indices.min() >= 0
    assert coarse_indices.max() < 2 * coarse_half_width + 1


def _config(**kwargs):
    result = Config()
    result.load_from_dict(kwargs)
    return result
