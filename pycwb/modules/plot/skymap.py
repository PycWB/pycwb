"""Skymap and sky-location plotting helpers."""

from __future__ import annotations

import logging

import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np

from pycwb.utils.skymap_coord import convert_cwb_to_geo

logger = logging.getLogger(__name__)


def plot_world_map(phi, theta, filename=None):
    """Plot one sky location on a cylindrical world map."""
    from mpl_toolkits.basemap import Basemap

    fig, ax = plt.subplots(figsize=(12, 8))
    phi -= 360 if phi > 180 else 0
    theta = -(theta - 90)

    basemap = Basemap(projection="cyl", ax=ax)
    basemap.drawcoastlines(linewidth=0.25)
    basemap.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0])
    basemap.drawmeridians(np.arange(basemap.lonmin, basemap.lonmax + 30, 30), labels=[0, 0, 0, 1])

    x, y = basemap([phi], [theta])
    basemap.scatter(x, y, marker="*", color="blue", s=200)

    if filename:
        fig.savefig(filename)
        plt.close(fig)
        return None
    return fig, ax



def plot_skymap_contour(
    skymap_statistic,
    key="nProbability",
    reconstructed_loc=None,
    detector_loc=None,
    filename=None,
    resolution=2,
):
    """Plot a HEALPix skymap statistic as an RA/Dec histogram."""
    mplstyle.use("fast")

    skymap = np.array(skymap_statistic[key])
    theta = np.linspace(0, np.pi, 180 * resolution + 1)[:-1]
    phi = np.linspace(0, 2 * np.pi, 360 * resolution + 1)[:-1]
    theta, phi = np.meshgrid(theta, phi)
    flatten_theta = theta.flatten()
    flatten_phi = phi.flatten()

    healpix_indices = hp.ang2pix(2 ** 7, flatten_theta, flatten_phi)
    if len(skymap) < len(healpix_indices):
        raise ValueError("The length of the skymap is smaller than the length of the healpix indices")

    values = skymap[healpix_indices]
    values = np.reshape(values, theta.shape)

    phi, theta = convert_cwb_to_geo(phi, theta)
    phi_deg = np.rad2deg(phi)
    theta_deg = np.rad2deg(theta)
    values[values == 0] = np.nan

    fig, ax = plt.subplots(figsize=(12, 6))
    hist = ax.hist2d(
        phi_deg.ravel(),
        theta_deg.ravel(),
        weights=values.ravel(),
        bins=(360 * resolution, 180 * resolution),
    )

    if reconstructed_loc:
        rec_x, rec_y = convert_cwb_to_geo(reconstructed_loc[0], reconstructed_loc[1])
        ax.scatter(rec_x, rec_y, marker="*", color="red", label="reconstructed position")

    if detector_loc:
        det_x, det_y = convert_cwb_to_geo(detector_loc[0], detector_loc[1])
        ax.scatter(det_x, det_y, marker=".", color="blue", label="detector position")

    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-90, -45, 0, 45, 90])
    fig.colorbar(hist[3], ax=ax)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(f"{key} skymap")
    ax.legend(loc=1)

    if filename:
        fig.savefig(filename)
        plt.close(fig)
        logger.info("Plot %s skymap saved to %s", key, filename)
        return None
    return fig, ax
