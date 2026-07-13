"""Interactive Plotly globe visualization of detector arm geometry.

Provides standalone functions for plotting gravitational-wave detector
arms on a 3D globe using Plotly.  These replace the deprecated
:meth:`Detector.plot_on_globe` method.
"""

from __future__ import annotations

import numpy as np


def plot_detector_on_globe(
    detector,
    fig=False,
    distance_km=500,
    projection_type="orthographic",
    width=800,
    height=800,
    color="red",
):
    """Plot a detector's X and Y arms on a globe using Plotly.

    Parameters
    ----------
    detector : Detector
        A :class:`~pycwb.types.detector.Detector` instance.
    fig : go.Figure or bool, optional
        A Plotly figure to add traces to.  If ``False`` (default), a new
        figure is created.
    distance_km : float
        Arm endpoint distance in kilometres (default 500).
    projection_type : str
        Plotly globe projection type (``'orthographic'``, ``'natural earth'``,
        etc.).
    width, height : int
        Figure dimensions in pixels.
    color : str
        Line colour for the arm traces.

    Returns
    -------
    go.Figure
        A Plotly figure with the detector arms plotted on a globe.

    Examples
    --------
    >>> from pycwb.types.detector import Detector
    >>> from pycwb.modules.plot.detector_globe import plot_detector_on_globe
    >>> h1 = Detector("H1")
    >>> l1 = Detector("L1")
    >>> fig = plot_detector_on_globe(h1)
    >>> plot_detector_on_globe(l1, fig=fig, color="blue")
    >>> fig.show()
    """
    import plotly.graph_objects as go

    if fig is False:
        fig = go.Figure()
        fig.update_layout(
            geo=dict(
                showland=True,
                showcountries=False,
                lataxis_showgrid=True,
                lonaxis_showgrid=True,
                projection_type=projection_type,
            ),
            width=width,
            height=height,
        )

    x_arm_endpoint = np.rad2deg(detector.get_x_arm_endpoint_in_geo(distance_km))
    y_arm_endpoint = np.rad2deg(detector.get_y_arm_endpoint_in_geo(distance_km))
    longitude = np.rad2deg(detector.longitude)
    latitude = np.rad2deg(detector.latitude)

    fig.add_trace(
        go.Scattergeo(
            lon=[longitude, x_arm_endpoint[0]],
            lat=[latitude, x_arm_endpoint[1]],
            mode="lines",
            line=dict(width=2, color=color),
            marker=dict(size=5),
            name=f"{detector.name} X Arm",
        )
    )

    fig.add_trace(
        go.Scattergeo(
            lon=[longitude, y_arm_endpoint[0]],
            lat=[latitude, y_arm_endpoint[1]],
            mode="lines",
            line=dict(width=2, color=color),
            marker=dict(size=5),
            name=f"{detector.name} Y Arm",
        )
    )

    return fig
