"""Regression tests for cWB-compatible cluster statistic plots."""

import numpy as np
import pytest

from pycwb.modules.plot.cluster_statistics import (
    _prepare_statistics_map,
    plot_statistics,
)
from pycwb.types.network_cluster import Cluster
from pycwb.types.pixel_arrays import PixelArrays


def _cluster():
    # All pixels describe a 4096 Hz WDM analysis:
    # 64 * (65 - 1) == 256 * (17 - 1) == 4096.
    return Cluster(
        pixel_arrays=PixelArrays.from_arrays(
            time=np.array([1_672_830, 1_750_129, 1_750_129]),
            frequency=np.array([55, 13, 1]),
            layers=np.array([65, 17, 17]),
            rate=np.array([64, 256, 256]),
            core=np.array([True, True, False]),
            likelihood=np.array([15.0, 5.0, 1000.0]),
            null=np.array([7.0, -2.0, 1000.0]),
            noise_rms=np.ones((2, 3)),
            pixel_index=np.zeros((2, 3)),
            n_ifo=2,
        )
    )


def _half_bin_overlap_cluster():
    # The 256 Hz pixel begins 1.5 finest-grid bins after the 128 Hz pixel.
    # cWB truncates that start to bin 3, where it overlaps the second bin of
    # the coarser pixel.  Rounding would move it to bin 4 and lose the sum.
    return Cluster(
        pixel_arrays=PixelArrays.from_arrays(
            time=np.array([33_000, 34_017]),
            frequency=np.array([28, 14]),
            layers=np.array([33, 17]),
            rate=np.array([128, 256]),
            core=np.array([True, True]),
            likelihood=np.array([12.0, 14.0]),
            null=np.zeros(2),
            noise_rms=np.ones((2, 2)),
            pixel_index=np.zeros((2, 2)),
            n_ifo=2,
        )
    )


def test_prepare_statistics_map_uses_cwb_zoom_and_ignores_halo_pixels():
    result = _prepare_statistics_map(_cluster(), "likelihood")

    assert result.total == pytest.approx(20.0)
    assert result.npix == 2
    assert result.min_dt == pytest.approx(1 / 256)
    assert result.max_dt == pytest.approx(1 / 64)
    assert result.min_df == pytest.approx(32)
    assert result.max_df == pytest.approx(128)
    assert result.display_df == pytest.approx(16)

    # The cWB view is centered around the cluster, not 0 Hz.
    assert result.ylim == pytest.approx((1408.0, 2016.0))
    assert result.xlim[0] > 400
    assert result.xlim[1] < 403

    # cWB's TH2 grid uses half-frequency cells.  The 256 Hz pixel is
    # expanded over eight 16 Hz display bins (a 128 Hz WDM pixel).
    nonzero_frequency_bins = np.flatnonzero(np.any(result.values > 0, axis=0))
    assert set(range(100, 108)).issubset(nonzero_frequency_bins)
    assert np.max(result.values) == pytest.approx(15.0)


def test_prepare_statistics_map_preserves_cwb_half_bin_truncation():
    result = _prepare_statistics_map(
        _half_bin_overlap_cluster(),
        "likelihood",
    )

    assert np.max(result.values) == pytest.approx(26.0)


def test_null_statistics_are_clipped_at_zero_like_cwb():
    result = _prepare_statistics_map(_cluster(), "null")

    assert result.total == pytest.approx(7.0)
    assert result.npix == 1
    assert np.max(result.values) == pytest.approx(7.0)


def test_plot_statistics_writes_cwb_shaped_image(tmp_path):
    output = tmp_path / "likelihood.png"
    figure = plot_statistics(
        _cluster(),
        "likelihood",
        gps_shift=1_421_073_590,
        filename=output,
    )

    assert output.exists()
    assert figure.get_size_inches() == pytest.approx((8.0, 5.75))
    axis = figure.axes[0]
    assert axis.get_ylim()[0] > 0
    assert axis.get_xlabel() == (
        "Time (sec) : GPS OFFSET = 1421073590.000"
    )
    assert not axis.xaxis.get_major_formatter().get_useOffset()
    assert axis.get_title().startswith("Likelihood  20")


def test_prepare_statistics_map_rejects_unknown_statistic():
    with pytest.raises(ValueError, match="likelihood.*null"):
        _prepare_statistics_map(_cluster(), "energy")
