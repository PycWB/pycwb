"""Detector and network antenna pattern sky maps (matplotlib + cartopy).

Provides standalone functions for plotting gravitational-wave antenna
patterns on world maps.  These replace the deprecated
:meth:`Detector.draw_antenna_pattern` and
:meth:`DetectorNetwork.draw_antenna_pattern` methods.

All heavy imports (matplotlib, cartopy) are deferred to function-call
time so that importing geometry types remains cheap.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers (migrated from DetectorNetwork private methods)
# ---------------------------------------------------------------------------

def _create_plot_figure(projection):
    """Create a matplotlib figure + axes for the requested projection."""
    import matplotlib.pyplot as plt

    proj_lower = projection.lower()
    if proj_lower == "hammer":
        return plt.subplots(figsize=(12, 8), subplot_kw={"projection": "aitoff"})
    elif proj_lower == "mollweide":
        return plt.subplots(figsize=(12, 8), subplot_kw={"projection": "mollweide"})
    elif proj_lower == "sinusoidal":
        from cartopy.crs import Sinusoidal

        return plt.subplots(
            figsize=(12, 8),
            subplot_kw={"projection": Sinusoidal(central_longitude=0)},
        )
    else:  # rectilinear / PlateCarree
        from cartopy.crs import PlateCarree

        return plt.subplots(
            figsize=(12, 6), subplot_kw={"projection": PlateCarree()}
        )


def _plot_pattern(ax, pattern, lon_deg, lat_deg, palette, projection, vmin=0.0, vmax=None):
    """Render the antenna pattern as a pcolormesh."""
    pattern_flipped = np.flipud(pattern)
    proj_lower = projection.lower()

    if proj_lower in ("hammer", "mollweide"):
        lon_plot = lon_deg * np.pi / 180.0
        lat_plot = lat_deg * np.pi / 180.0
        im = ax.pcolormesh(
            lon_plot, lat_plot, pattern_flipped,
            cmap=palette, shading="auto", vmin=vmin, vmax=vmax,
        )
        ax.grid(True, linestyle="--", alpha=0.5)
    else:
        from cartopy.crs import PlateCarree

        im = ax.pcolormesh(
            lon_deg, lat_deg, pattern_flipped,
            transform=PlateCarree(),
            cmap=palette, shading="auto", vmin=vmin, vmax=vmax,
        )
    return im


def _add_world_map(ax, display_world_map, projection):
    """Optionally overlay cartopy coastlines / features."""
    if not display_world_map:
        return
    proj_lower = projection.lower()
    if proj_lower in ("hammer", "mollweide"):
        ax.grid(True, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
        return

    try:
        from cartopy.crs import PlateCarree
        import cartopy.feature as cfeature

        ax.coastlines(linewidth=0.5, alpha=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5, linestyle=":")
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.2)
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.1)

        if proj_lower != "sinusoidal":
            gl = ax.gridlines(
                crs=PlateCarree(),
                draw_labels=True,
                linewidth=0.5,
                color="gray",
                alpha=0.5,
                linestyle="--",
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 8}
            gl.ylabel_style = {"size": 8}
    except Exception:
        ax.grid(True, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)


def _plot_detector_sites(ax, detectors, projection):
    """Draw detector markers and arm lines on the map.

    ``detectors`` is a list of dicts with keys ``lat``, ``lon``, ``code``,
    ``x_az``, ``x_alt``, ``y_az``, ``y_alt`` (as returned by
    :meth:`DetectorNetwork._get_detector_info`).
    """
    # Import here to avoid circular issues; DetectorNetwork is imported
    # lazily because it lives in the same types package.
    from pycwb.types.detector import DetectorNetwork

    from cartopy.crs import PlateCarree

    proj_lower = projection.lower()
    for det in detectors:
        lat_deg = np.degrees(det["lat"])
        lon_deg = np.degrees(det["lon"])
        (x_lon, x_lat), (y_lon, y_lat) = DetectorNetwork._compute_arm_endpoints(
            det, arm_length_factor=6.0
        )

        if proj_lower in ("hammer", "mollweide"):
            lon_plot = np.radians(lon_deg)
            if lon_plot > np.pi:
                lon_plot -= 2 * np.pi
            lat_plot = np.radians(lat_deg)

            x_lon_plot = np.radians(x_lon)
            if x_lon_plot > np.pi:
                x_lon_plot -= 2 * np.pi
            x_lat_plot = np.radians(x_lat)

            y_lon_plot = np.radians(y_lon)
            if y_lon_plot > np.pi:
                y_lon_plot -= 2 * np.pi
            y_lat_plot = np.radians(y_lat)

            ax.plot(
                lon_plot, lat_plot, "k.", markersize=10,
                markeredgewidth=2, transform=ax.transData,
            )
            ax.plot(
                [lon_plot, x_lon_plot], [lat_plot, x_lat_plot],
                "k-", linewidth=2.0, transform=ax.transData,
            )
            ax.plot(
                [lon_plot, y_lon_plot], [lat_plot, y_lat_plot],
                "k-", linewidth=2.0, transform=ax.transData,
            )
            ax.text(
                lon_plot + 0.05, lat_plot + 0.05, det["code"],
                fontsize=12, fontweight="bold", ha="left", va="bottom",
                transform=ax.transData,
            )
        else:
            ax.plot(
                lon_deg, lat_deg, "k.", markersize=10,
                transform=PlateCarree(), markeredgewidth=2,
            )
            ax.plot(
                [lon_deg, x_lon], [lat_deg, x_lat],
                "k-", linewidth=2.0, transform=PlateCarree(),
            )
            ax.plot(
                [lon_deg, y_lon], [lat_deg, y_lat],
                "k-", linewidth=2.0, transform=PlateCarree(),
            )
            ax.text(
                lon_deg + 1, lat_deg + 1, det["code"],
                transform=PlateCarree(), fontsize=12,
                fontweight="bold", ha="left", va="bottom",
            )


# ---------------------------------------------------------------------------
# Public plotting API
# ---------------------------------------------------------------------------

def plot_detector_antenna_pattern(
    detector,
    polarization=3,
    palette="turbo",
    resolution=2,
    projection="rectilinear",
    display_world_map=True,
    add_title=True,
    ax=None,
    vmin=0.0,
    vmax=None,
):
    """Draw a single-detector antenna pattern sky map.

    Parameters
    ----------
    detector : Detector
        A :class:`~pycwb.types.detector.Detector` instance.
    polarization : int
        Polarization quantity to plot:

        - 0: |Fx| (DPF)
        - 1: |F+| (DPF)
        - 2: |Fx| / |F+| (DPF)
        - 3: sqrt(|F+|² + |Fx|²) (DPF, default)
        - 4: |Fx|² (DPF)
        - 5: |F+|² (DPF)
    palette : str
        Matplotlib colormap name.
    resolution : int
        Sky-map resolution (1=low, 2=medium, 4=high).
    projection : str
        ``'hammer'``, ``'mollweide'``, ``'rectilinear'``, or ``'sinusoidal'``.
    display_world_map : bool
        Overlay cartopy coastlines and features.
    add_title : bool
        Add a descriptive title.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.
    vmin : float
        Minimum colour-bar value.
    vmax : float, optional
        Maximum colour-bar value (auto-detected if *None*).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    # --- sky grid ----------------------------------------------------------
    n_lon = 360 * resolution
    n_lat = 180 * resolution
    lon_rad = np.linspace(0, 2 * np.pi, n_lon)
    lat_rad = np.linspace(0, np.pi, n_lat)
    lon_grid, lat_grid = np.meshgrid(lon_rad, lat_rad)

    # --- antenna patterns --------------------------------------------------
    F_plus, F_cross = detector.compute_antenna_pattern_for_grid(lat_grid, lon_grid)

    # --- polarisation quantity ---------------------------------------------
    if polarization == 0:
        pattern = np.abs(F_cross)
    elif polarization == 1:
        pattern = np.abs(F_plus)
    elif polarization == 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            pattern = np.abs(F_cross) / np.abs(F_plus)
        pattern[~np.isfinite(pattern)] = 0.0
    elif polarization == 3:
        pattern = np.sqrt(F_plus ** 2 + F_cross ** 2)
    elif polarization == 4:
        pattern = F_cross ** 2
    elif polarization == 5:
        pattern = F_plus ** 2
    else:
        raise ValueError(f"Unsupported polarization: {polarization}")

    # --- figure / axes ----------------------------------------------------
    if ax is None:
        fig, ax = _create_plot_figure(projection)
    else:
        fig = ax.get_figure()

    # --- coordinate transform for plotting ---------------------------------
    lon_deg = np.degrees(lon_grid) - 180.0
    lat_deg = 90.0 - np.degrees(lat_grid)

    if vmax is None:
        vmax = float(np.max(pattern))

    im = _plot_pattern(ax, pattern, lon_deg, lat_deg, palette, projection,
                       vmin=vmin, vmax=vmax)

    _add_world_map(ax, display_world_map, projection)

    # --- detector marker ---------------------------------------------------
    lat_deg_det = np.degrees(detector.latitude)
    lon_deg_det = np.degrees(detector.longitude)

    proj_lower = projection.lower()
    if proj_lower in ("hammer", "mollweide"):
        lon_plot = np.radians(lon_deg_det)
        if lon_plot > np.pi:
            lon_plot -= 2 * np.pi
        lat_plot = np.radians(lat_deg_det)
        ax.plot(lon_plot, lat_plot, "k.", markersize=10,
                markeredgewidth=2, transform=ax.transData)
        ax.text(lon_plot + 0.05, lat_plot + 0.05, detector.name,
                fontsize=12, fontweight="bold", ha="left", va="bottom",
                transform=ax.transData)
    else:
        from cartopy.crs import PlateCarree

        ax.plot(lon_deg_det, lat_deg_det, "k.", markersize=10,
                transform=PlateCarree(), markeredgewidth=2)
        ax.text(lon_deg_det + 1, lat_deg_det + 1, detector.name,
                transform=PlateCarree(), fontsize=12,
                fontweight="bold", ha="left", va="bottom")

    # --- colour bar --------------------------------------------------------
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.05, shrink=0.45)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Pattern Value", fontsize=11)

    # --- title -------------------------------------------------------------
    if add_title:
        _POLARIZATION_NAMES = {
            0: r"$|F_x|$ (DPF)",
            1: r"$|F_+|$ (DPF)",
            2: r"$|F_x|/|F_+|$ (DPF)",
            3: r"$\sqrt{|F_+|^2 + |F_x|^2}$ (DPF)",
            4: "$|F_x|^2$ (DPF)",
            5: "$|F_+|^2$ (DPF)",
        }
        title = (
            f"Detector: {detector.name} - "
            f"{_POLARIZATION_NAMES.get(polarization, f'Polarization {polarization}')}"
        )
        ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()
    return fig, ax


def plot_network_antenna_pattern(
    network,
    polarization=3,
    palette="turbo",
    resolution=2,
    projection="rectilinear",
    display_world_map=True,
    add_title=True,
    uniform_colorbar=True,
    ax=None,
    detector_scales=None,
):
    """Draw a network antenna pattern sky map.

    Parameters
    ----------
    network : DetectorNetwork
        A :class:`~pycwb.types.detector.DetectorNetwork` instance.
    polarization : int
        See :func:`plot_detector_antenna_pattern`.
    palette : str
        Matplotlib colormap name.
    resolution : int
        Sky-map resolution.
    projection : str
        ``'hammer'``, ``'mollweide'``, ``'rectilinear'``, or ``'sinusoidal'``.
    display_world_map : bool
        Overlay cartopy coastlines and features.
    add_title : bool
        Add a descriptive title.
    uniform_colorbar : bool
        Use a single colour-bar range across all detectors.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.
    detector_scales : dict or array-like, optional
        Per-detector scale factors.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    detectors = network._get_detector_info()
    if not detectors:
        raise ValueError("No valid detectors found in network")

    # --- per-detector scales -----------------------------------------------
    scales = np.ones(len(detectors))
    if detector_scales is not None:
        if isinstance(detector_scales, dict):
            scales = np.array(
                [detector_scales.get(d["code"], 1.0) for d in detectors]
            )
        elif isinstance(detector_scales, (list, np.ndarray)):
            if len(detector_scales) == len(detectors):
                scales = np.array(detector_scales)

    # --- sky grid & antenna patterns ---------------------------------------
    lon_grid, lat_grid, _, _ = network._create_sky_grid(resolution)
    F_plus, F_cross = network._compute_antenna_patterns(
        lat_grid, lon_grid, detectors
    )

    reference_max = None
    if uniform_colorbar:
        reference_max = network._compute_reference_max(F_plus, F_cross, scales)

    pattern, pattern_max = network._compute_antenna_pattern(
        F_plus, F_cross, polarization, scales
    )

    # --- figure / axes ----------------------------------------------------
    if ax is None:
        fig, ax = _create_plot_figure(projection)
    else:
        fig = ax.get_figure()

    lon_deg = np.degrees(lon_grid) - 180.0
    lat_deg = 90.0 - np.degrees(lat_grid)

    vmax = reference_max if (uniform_colorbar and reference_max is not None) else pattern_max
    im = _plot_pattern(ax, pattern, lon_deg, lat_deg, palette, projection,
                       vmin=0.0, vmax=vmax)

    _add_world_map(ax, display_world_map, projection)

    # --- colour bar --------------------------------------------------------
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.05, shrink=0.45)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Pattern Value", fontsize=11)

    # --- detector sites ----------------------------------------------------
    _plot_detector_sites(ax, detectors, projection)

    # --- title -------------------------------------------------------------
    if add_title:
        _POLARIZATION_NAMES = {
            0: r"$|F_x|$ (DPF)",
            1: r"$|F_+|$ (DPF)",
            2: r"$|F_x|/|F_+|$ (DPF)",
            3: r"$\sqrt{|F_+|^2 + |F_x|^2}$ (DPF)",
            4: "$|F_x|^2$ (DPF)",
            5: "$|F_+|^2$ (DPF)",
        }
        codes = " ".join(d["code"] for d in detectors)
        title = (
            f"Network: {codes} - "
            f"{_POLARIZATION_NAMES.get(polarization, f'Polarization {polarization}')}"
        )
        ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()
    return fig, ax
