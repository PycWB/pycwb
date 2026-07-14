"""Skymap and sky-location plotting helpers."""

from __future__ import annotations

import logging
import warnings

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
    *,
    detection_loc=None,
    injected_loc=None,
):
    """Plot a cWB HEALPix statistic in Earth-fixed lon/lat coordinates.

    ``reconstructed_loc``, ``detection_loc``, and ``injected_loc`` are
    ``(phi_geo, theta_cwb)`` pairs in degrees. The first two match the
    corresponding ``Event.phi/theta`` slots. ``detector_loc`` is a deprecated
    compatibility name for ``detection_loc``.
    """
    if detector_loc is not None:
        if detection_loc is not None:
            raise ValueError(
                "Use only detection_loc; detector_loc is its deprecated alias"
            )
        warnings.warn(
            "detector_loc is deprecated; use detection_loc for the cWB "
            "detection position",
            DeprecationWarning,
            stacklevel=2,
        )
        detection_loc = detector_loc

    mplstyle.use("fast")

    skymap = np.array(skymap_statistic[key])
    theta = np.linspace(0, np.pi, 180 * resolution + 1)[:-1]
    phi = np.linspace(0, 2 * np.pi, 360 * resolution + 1)[:-1]
    theta, phi = np.meshgrid(theta, phi)
    flatten_theta = theta.flatten()
    flatten_phi = phi.flatten()

    try:
        nside = hp.npix2nside(len(skymap))
    except ValueError as exc:
        raise ValueError(
            f"Skymap length {len(skymap)} is not a valid HEALPix map size"
        ) from exc
    healpix_indices = hp.ang2pix(nside, flatten_theta, flatten_phi)

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

    if reconstructed_loc is not None:
        rec_phi, rec_theta = np.deg2rad(np.asarray(reconstructed_loc, dtype=float))
        rec_x, rec_y = convert_cwb_to_geo(rec_phi, rec_theta)
        rec_x, rec_y = np.rad2deg([rec_x, rec_y])
        ax.scatter(
            rec_x, rec_y, marker="*", color="red", s=120, zorder=5,
            label="reconstructed position",
        )

    if detection_loc is not None:
        det_phi, det_theta = np.deg2rad(np.asarray(detection_loc, dtype=float))
        det_x, det_y = convert_cwb_to_geo(det_phi, det_theta)
        det_x, det_y = np.rad2deg([det_x, det_y])
        ax.scatter(
            det_x, det_y, marker="o", facecolors="none", edgecolors="blue",
            s=100, linewidths=1.5, zorder=4, label="detection position",
        )

    if injected_loc is not None:
        inj_phi, inj_theta = np.deg2rad(np.asarray(injected_loc, dtype=float))
        inj_x, inj_y = convert_cwb_to_geo(inj_phi, inj_theta)
        inj_x, inj_y = np.rad2deg([inj_x, inj_y])
        ax.scatter(
            inj_x, inj_y, marker="x", color="limegreen", s=100,
            linewidths=2.0, zorder=6, label="injected position",
        )

    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-90, -45, 0, 45, 90])
    fig.colorbar(hist[3], ax=ax)
    ax.set_xlabel("Earth-fixed longitude (deg)")
    ax.set_ylabel("Earth-fixed latitude (deg)")
    ax.set_title(f"{key} skymap")
    if any(loc is not None for loc in (reconstructed_loc, detection_loc, injected_loc)):
        ax.legend(loc=1)

    if filename:
        fig.savefig(filename)
        plt.close(fig)
        logger.info("Plot %s skymap saved to %s", key, filename)
        return None
    return fig, ax
