"""Smoke tests for detector plotting functions.

Tests verify that plotting functions run without errors and return
correct types.  Full pixel-level output validation is deferred.
"""

import ast
from pathlib import Path
import warnings

import numpy as np
import pytest

# Optional dependencies — skip all tests if not available
plt = pytest.importorskip("matplotlib.pyplot")
cartopy = pytest.importorskip("cartopy")
ccrs = pytest.importorskip("cartopy.crs")
if not all(hasattr(ccrs, name) for name in ("PlateCarree", "Sinusoidal")):
    pytest.skip("cartopy projection classes unavailable", allow_module_level=True)
plotly = pytest.importorskip("plotly")


from pycwb.types.detector import Detector, DetectorNetwork  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h1():
    return Detector("H1")


@pytest.fixture(scope="module")
def l1():
    return Detector("L1")


@pytest.fixture(scope="module")
def h1l1_network(h1, l1):
    return DetectorNetwork(detectors=[h1, l1])


@pytest.fixture(scope="module")
def h1l1v1_network(h1, l1):
    v1 = Detector("V1")
    return DetectorNetwork(detectors=[h1, l1, v1])


# ---------------------------------------------------------------------------
# New public API
# ---------------------------------------------------------------------------

class TestPlotDetectorAntennaPattern:
    """Smoke tests for :func:`plot_detector_antenna_pattern`."""

    def test_returns_figure_and_axes(self, h1):
        from pycwb.modules.plot.detector_antenna import (
            plot_detector_antenna_pattern,
        )
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        fig, ax = plot_detector_antenna_pattern(h1)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_with_existing_axes(self, h1):
        """Plotting on an existing cartopy GeoAxes should work.
        Plain matplotlib Axes with cartopy transforms may fail due to
        gwpy monkeypatching (known issue); skip if GeoAxes unavailable."""
        from pycwb.modules.plot.detector_antenna import (
            plot_detector_antenna_pattern,
        )
        from cartopy.mpl.geoaxes import GeoAxes

        # Create a proper GeoAxes for cartopy plotting
        from cartopy.crs import PlateCarree
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=PlateCarree())
        assert isinstance(ax, GeoAxes)

        fig2, ax2 = plot_detector_antenna_pattern(h1, ax=ax)
        assert fig2 is fig
        assert ax2 is ax
        plt.close(fig)

    @pytest.mark.parametrize("polarization", [0, 1, 2, 3, 4, 5])
    def test_all_polarizations(self, h1, polarization):
        from pycwb.modules.plot.detector_antenna import (
            plot_detector_antenna_pattern,
        )

        fig, ax = plot_detector_antenna_pattern(
            h1, polarization=polarization,
        )
        assert fig is not None
        plt.close(fig)

    def test_invalid_polarization_raises(self, h1):
        from pycwb.modules.plot.detector_antenna import (
            plot_detector_antenna_pattern,
        )

        with pytest.raises(ValueError):
            plot_detector_antenna_pattern(h1, polarization=99)

    @pytest.mark.parametrize("projection", [
        "rectilinear", "hammer", "mollweide", "sinusoidal",
    ])
    def test_all_projections(self, h1, projection):
        from pycwb.modules.plot.detector_antenna import (
            plot_detector_antenna_pattern,
        )

        fig, ax = plot_detector_antenna_pattern(
            h1, projection=projection,
        )
        assert fig is not None
        plt.close(fig)

    def test_no_world_map(self, h1):
        from pycwb.modules.plot.detector_antenna import (
            plot_detector_antenna_pattern,
        )

        fig, ax = plot_detector_antenna_pattern(
            h1, display_world_map=False,
        )
        assert fig is not None
        plt.close(fig)

    def test_no_title(self, h1):
        from pycwb.modules.plot.detector_antenna import (
            plot_detector_antenna_pattern,
        )

        fig, ax = plot_detector_antenna_pattern(h1, add_title=False)
        assert fig is not None
        plt.close(fig)

    def test_custom_colorbar_range(self, h1):
        from pycwb.modules.plot.detector_antenna import (
            plot_detector_antenna_pattern,
        )

        fig, ax = plot_detector_antenna_pattern(
            h1, vmin=0.2, vmax=0.8,
        )
        assert fig is not None
        plt.close(fig)

    def test_with_resolution(self, h1):
        from pycwb.modules.plot.detector_antenna import (
            plot_detector_antenna_pattern,
        )

        fig, ax = plot_detector_antenna_pattern(h1, resolution=1)
        assert fig is not None
        plt.close(fig)


class TestPlotNetworkAntennaPattern:
    """Smoke tests for :func:`plot_network_antenna_pattern`."""

    def test_returns_figure_and_axes(self, h1l1_network):
        from pycwb.modules.plot.detector_antenna import (
            plot_network_antenna_pattern,
        )
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        fig, ax = plot_network_antenna_pattern(h1l1_network)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_three_detector_network(self, h1l1v1_network):
        from pycwb.modules.plot.detector_antenna import (
            plot_network_antenna_pattern,
        )

        fig, ax = plot_network_antenna_pattern(h1l1v1_network)
        assert fig is not None
        plt.close(fig)

    def test_non_uniform_colorbar(self, h1l1_network):
        from pycwb.modules.plot.detector_antenna import (
            plot_network_antenna_pattern,
        )

        fig, ax = plot_network_antenna_pattern(
            h1l1_network, uniform_colorbar=False,
        )
        assert fig is not None
        plt.close(fig)

    def test_with_detector_scales(self, h1l1_network):
        from pycwb.modules.plot.detector_antenna import (
            plot_network_antenna_pattern,
        )

        fig, ax = plot_network_antenna_pattern(
            h1l1_network, detector_scales={"H1": 2.0, "L1": 1.0},
        )
        assert fig is not None
        plt.close(fig)

    def test_empty_network_raises(self):
        from pycwb.modules.plot.detector_antenna import (
            plot_network_antenna_pattern,
        )

        empty = DetectorNetwork()
        with pytest.raises(ValueError):
            plot_network_antenna_pattern(empty)


class TestPlotDetectorOnGlobe:
    """Smoke tests for :func:`plot_detector_on_globe`."""

    def test_returns_plotly_figure(self, h1):
        from pycwb.modules.plot.detector_globe import (
            plot_detector_on_globe,
        )
        import plotly.graph_objects as go

        fig = plot_detector_on_globe(h1)
        assert isinstance(fig, go.Figure)

    def test_multiple_detectors_same_figure(self, h1, l1):
        from pycwb.modules.plot.detector_globe import (
            plot_detector_on_globe,
        )
        import plotly.graph_objects as go

        fig = plot_detector_on_globe(h1)
        fig = plot_detector_on_globe(l1, fig=fig, color="blue")
        assert isinstance(fig, go.Figure)
        # Should now have more traces
        assert len(fig.data) >= 4  # 2 per detector = 4 traces

    def test_custom_distance(self, h1):
        from pycwb.modules.plot.detector_globe import (
            plot_detector_on_globe,
        )

        fig = plot_detector_on_globe(h1, distance_km=200)
        assert fig is not None

    def test_custom_projection(self, h1):
        from pycwb.modules.plot.detector_globe import (
            plot_detector_on_globe,
        )

        fig = plot_detector_on_globe(
            h1, projection_type="natural earth",
        )
        assert fig is not None


# ---------------------------------------------------------------------------
# Deprecation shims
# ---------------------------------------------------------------------------

class TestDeprecationShims:
    """Verify that deprecated methods emit warnings and delegate correctly."""

    def test_plot_on_globe_deprecation(self, h1):
        import plotly.graph_objects as go

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            fig = h1.plot_on_globe()
            # At least one DeprecationWarning should be emitted
            dep_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
                and "plot_on_globe" in str(x.message)
            ]
            assert len(dep_warnings) >= 1, (
                f"Expected DeprecationWarning, got {[str(x.message) for x in w]}"
            )
        assert isinstance(fig, go.Figure)

    def test_draw_antenna_pattern_deprecation(self, h1):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            fig, ax = h1.draw_antenna_pattern()
            dep_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
                and "draw_antenna_pattern" in str(x.message)
            ]
            assert len(dep_warnings) >= 1
        assert fig is not None
        plt.close(fig)

    def test_network_draw_antenna_pattern_deprecation(self, h1l1_network):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            fig, ax = h1l1_network.draw_antenna_pattern()
            dep_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
                and "draw_antenna_pattern" in str(x.message)
            ]
            assert len(dep_warnings) >= 1
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# Import-time check: geometry-only path doesn't load plotting libs
# ---------------------------------------------------------------------------

class TestNoPlottingLeak:
    """Geometry imports should not pull in the plotting stack."""

    def test_detector_module_no_matplotlib(self):
        """Verify detector module itself doesn't import matplotlib."""
        # The overall pycwb import may load matplotlib (pre-existing issue
        # in pycwb/__init__.py), but detector.py itself should not.
        import pycwb.types.detector as det_mod
        # The module's own source shouldn't reference plt/plotly/cartopy
        source = open(det_mod.__file__).read()
        assert "import matplotlib.pyplot" not in source
        assert "import plotly" not in source
        assert "import cartopy" not in source

    @pytest.mark.parametrize(
        "module_path",
        [
            "pycwb/modules/plot/detector_antenna.py",
            "pycwb/modules/plot/detector_globe.py",
        ],
    )
    def test_plotting_modules_have_no_top_level_heavy_imports(self, module_path):
        tree = ast.parse(Path(module_path).read_text())
        top_level_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                top_level_imports.append(node.module)

        heavy_imports = {
            name
            for name in top_level_imports
            if name.split(".")[0] in {"matplotlib", "cartopy", "plotly"}
        }
        assert heavy_imports == set()
