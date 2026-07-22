import matplotlib

matplotlib.use("Agg")

import healpy as hp
import numpy as np
import pytest

from pycwb.modules.plot.skymap import plot_skymap_contour


def test_cwb_degree_marker_is_converted_to_geo_degrees():
    skymap = np.ones(hp.nside2npix(128), dtype=float)
    fig, ax = plot_skymap_contour(
        {"nProbability": skymap},
        reconstructed_loc=(300.0, 60.0),
        detection_loc=(120.0, 100.0),
        injected_loc=(210.0, 45.0),
        resolution=1,
    )
    marker = next(c for c in ax.collections if c.get_label() == "reconstructed position")
    np.testing.assert_allclose(marker.get_offsets()[0], [-60.0, 30.0], atol=1e-12)

    detection = next(c for c in ax.collections if c.get_label() == "detection position")
    np.testing.assert_allclose(detection.get_offsets()[0], [120.0, -10.0], atol=1e-12)

    injected = next(c for c in ax.collections if c.get_label() == "injected position")
    np.testing.assert_allclose(injected.get_offsets()[0], [-150.0, 45.0], atol=1e-12)
    assert ax.get_xlabel() == "Earth-fixed longitude (deg)"
    assert ax.get_ylabel() == "Earth-fixed latitude (deg)"
    fig.clear()


def test_skymap_resolution_is_inferred_from_array_length():
    skymap = np.ones(hp.nside2npix(16), dtype=float)
    fig, _ = plot_skymap_contour(
        {"nProbability": skymap},
        resolution=1,
    )
    fig.clear()


def test_detector_loc_remains_a_deprecated_alias():
    skymap = np.ones(hp.nside2npix(16), dtype=float)
    with pytest.warns(DeprecationWarning, match="detection_loc"):
        fig, ax = plot_skymap_contour(
            {"nProbability": skymap},
            detector_loc=(120.0, 100.0),
            resolution=1,
        )
    detection = next(c for c in ax.collections if c.get_label() == "detection position")
    np.testing.assert_allclose(detection.get_offsets()[0], [120.0, -10.0], atol=1e-12)
    fig.clear()
