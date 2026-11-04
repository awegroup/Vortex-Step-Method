"""Tests for the fancy Plotly plot: inflatable tubes and curved canopy.

Unit tests cover the pure geometry helpers (ring math, contour handling,
interpolation). The integration tests exercise the full SurfplanAdapter
conversion and figure assembly against the bundled TUDELFT V3 Surfplan export;
they are skipped if SurfplanAdapter or the export data are unavailable.
"""

from pathlib import Path

import numpy as np
import pytest

from VSM.plotly.canopy_geometry import (
    interpolate_contour_at,
    read_upper_contour,
    resample_contour,
)
from VSM.plotly.tube_geometry import circle_points

REPO_ROOT = Path(__file__).resolve().parents[2]
SURFPLAN_DIR = REPO_ROOT / "data" / "TUDELFT_V3_KITE" / "Surfplan_export"
CP_DIR = REPO_ROOT / "data" / "TUDELFT_V3_KITE" / "cpx_distributions"
DRAWN_YAML = (
    REPO_ROOT
    / "data"
    / "TUDELFT_V3_KITE"
    / "CAD_derived_geometry"
    / "aero_geometry_CAD_breukels_regression.yaml"
)


def _has_surfplan():
    try:
        import SurfplanAdapter  # noqa: F401
    except ImportError:
        return False
    return SURFPLAN_DIR.is_dir()


requires_surfplan = pytest.mark.skipif(
    not _has_surfplan(), reason="SurfplanAdapter or Surfplan export data unavailable"
)


# --------------------------------------------------------------------------- #
# Unit tests -- pure geometry helpers
# --------------------------------------------------------------------------- #


def test_circle_points_radius_and_orthogonality():
    center = np.array([1.0, 2.0, 3.0])
    axis = np.array([0.0, 1.0, 0.0])
    diameter = 0.4
    ring = circle_points(center, axis, diameter, n=32)

    # Every point lies at the radius from the centre.
    radii = np.linalg.norm(ring - center, axis=1)
    assert np.allclose(radii, 0.5 * diameter, atol=1e-9)

    # The ring lies in the plane normal to the axis (constant projection).
    projections = (ring - center) @ axis
    assert np.allclose(projections, 0.0, atol=1e-9)


def test_circle_points_axis_normalisation_invariant():
    center = np.zeros(3)
    ring_unit = circle_points(center, np.array([0.0, 0.0, 1.0]), 1.0)
    ring_scaled = circle_points(center, np.array([0.0, 0.0, 5.0]), 1.0)
    assert np.allclose(ring_unit, ring_scaled, atol=1e-9)


def test_resample_contour_count_and_endpoints():
    contour = np.array([[0.0, 0.0], [0.3, 0.08], [0.7, 0.05], [1.0, 0.0]])
    resampled = resample_contour(contour, n_points=25)
    assert resampled.shape == (25, 2)
    assert np.allclose(resampled[0], contour[0], atol=1e-9)
    assert np.allclose(resampled[-1], contour[-1], atol=1e-9)


def test_interpolate_contour_at_blends_and_clamps():
    c_lo = np.array([[0.0, 0.0], [1.0, 0.0]])
    c_hi = np.array([[0.0, 0.0], [1.0, 1.0]])
    table = [(0.0, c_lo), (1.0, c_hi)]

    # Midpoint is the average of the two contours.
    mid = interpolate_contour_at(0.5, table)
    assert np.allclose(mid, 0.5 * (c_lo + c_hi))

    # Outside the range clamps to the end contours.
    assert np.allclose(interpolate_contour_at(-1.0, table), c_lo)
    assert np.allclose(interpolate_contour_at(2.0, table), c_hi)


def test_read_upper_contour_ordering(tmp_path):
    # Minimal Selig-style .dat: TE -> upper -> LE -> lower -> TE.
    dat = tmp_path / "prof.dat"
    dat.write_text(
        "test_profile\n" "1.0 0.0\n" "0.5 0.08\n" "0.0 0.0\n" "0.5 -0.04\n" "1.0 0.0\n"
    )
    upper = read_upper_contour(dat)
    # Ordered LE (x=0) to TE (x=1), all upper-surface points.
    assert upper[0, 0] == pytest.approx(0.0)
    assert upper[-1, 0] == pytest.approx(1.0)
    assert np.all(np.diff(upper[:, 0]) > 0)
    assert np.all(upper[:, 1] >= 0.0)


# --------------------------------------------------------------------------- #
# Integration tests -- full conversion + figure assembly
# --------------------------------------------------------------------------- #


@requires_surfplan
def test_ensure_surfplan_processed_and_cache():
    from VSM.plotly.surfplan_runner import ensure_surfplan_processed

    processed = ensure_surfplan_processed(SURFPLAN_DIR)
    aero = processed / "aero_geometry.yaml"
    struc = processed / "struc_geometry_all_in_surfplan.yaml"
    assert aero.exists() and struc.exists()

    # Second call is a cache hit: the generated file is not rewritten.
    first_mtime = aero.stat().st_mtime
    ensure_surfplan_processed(SURFPLAN_DIR)
    assert aero.stat().st_mtime == first_mtime


@requires_surfplan
def test_build_tube_and_canopy_shapes():
    from VSM.core.BodyAerodynamics import BodyAerodynamics
    from VSM.plotly.surfplan_runner import ensure_surfplan_processed
    from VSM.plotly.tube_geometry import build_tube_data, load_strut_span_fractions
    from VSM.plotly.canopy_geometry import build_canopy_grid, load_contour_table

    n_panels = 36
    body = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=DRAWN_YAML,
        spanwise_panel_distribution="uniform",
    )
    panels = body.panels
    processed = ensure_surfplan_processed(SURFPLAN_DIR)

    tube = build_tube_data(panels, processed)
    n_struts = len(load_strut_span_fractions(processed))
    # LE tube has one ring per panel edge, plus the wrap-around tip extensions.
    assert len(tube["le"]) >= n_panels + 1
    assert len(tube["struts"]) == n_struts

    x, y, z = build_canopy_grid(panels, load_contour_table(processed))
    assert x.shape[0] == n_panels + 1
    assert x.shape == y.shape == z.shape

    # Tubes track the drawn wing span (within a tube radius at the tips).
    wing_le = np.array([p.LE_point_1 for p in panels] + [p.LE_point_2 for p in panels])
    tube_pts = np.vstack(list(tube["le"]))
    assert abs(tube_pts[:, 1].max() - wing_le[:, 1].max()) < 0.2


@requires_surfplan
def test_interactive_plot_fancy_traces():
    import plotly.graph_objects as go
    from VSM.core.BodyAerodynamics import BodyAerodynamics
    from VSM.plot_geometry_plotly import interactive_plot
    from VSM.plotly.tube_geometry import load_strut_span_fractions
    from VSM.plotly.surfplan_runner import ensure_surfplan_processed

    body = BodyAerodynamics.instantiate(
        n_panels=36,
        file_path=DRAWN_YAML,
        spanwise_panel_distribution="uniform",
    )
    n_struts = len(load_strut_span_fractions(ensure_surfplan_processed(SURFPLAN_DIR)))

    # Fancy plot with canopy (panels default off): surfaces = canopy + LE + struts.
    fig = interactive_plot(body, surfplan_dir=SURFPLAN_DIR, is_show=False)
    surfaces = [t for t in fig.data if isinstance(t, go.Surface)]
    flat_panels = [
        t for t in fig.data if isinstance(t, go.Mesh3d) and t.name == "Panel Surface"
    ]
    le_outline = [t for t in fig.data if getattr(t, "name", None) == "Leading Edge"]
    assert len(surfaces) == 1 + 1 + n_struts
    assert len(flat_panels) == 0
    assert len(le_outline) == 0  # panel outlines hidden by default in fancy mode

    # Tubes only with panels explicitly on: flat panels kept, no canopy surface.
    fig2 = interactive_plot(
        body,
        surfplan_dir=SURFPLAN_DIR,
        is_with_canopy=False,
        is_with_panels=True,
        is_show=False,
    )
    surfaces2 = [t for t in fig2.data if isinstance(t, go.Surface)]
    flat_panels2 = [
        t for t in fig2.data if isinstance(t, go.Mesh3d) and t.name == "Panel Surface"
    ]
    assert len(surfaces2) == 1 + n_struts
    assert len(flat_panels2) == len(body.panels)

    # Plain plot is unchanged: panels shown, no surfaces at all.
    fig3 = interactive_plot(body, is_show=False)
    assert not [t for t in fig3.data if isinstance(t, go.Surface)]
    assert [t for t in fig3.data if getattr(t, "name", None) == "Leading Edge"]


# --------------------------------------------------------------------------- #
# Cp distribution database
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not CP_DIR.is_dir(), reason="Cp distributions unavailable")
def test_load_upper_surface_cp_ordering():
    from VSM.plotly.cp_distributions import load_upper_surface_cp

    x, cp = load_upper_surface_cp(CP_DIR / "cp_AOA_8.dat")
    assert np.all(np.diff(x) >= 0)  # ascending x/c
    assert x.min() >= 0.0 and x.max() <= 1.01  # normalised chord (small CFD overshoot)
    assert cp.min() < 0  # suction side has negative Cp


@pytest.mark.skipif(not CP_DIR.is_dir(), reason="Cp distributions unavailable")
def test_build_cp_magnitude_fn_closest_aoa():
    from VSM.plotly.cp_distributions import build_cp_magnitude_fn

    # Exact and nearest matches over the available {2, 6, 8} degrees.
    assert build_cp_magnitude_fn(CP_DIR, 8)[1] == 8
    assert build_cp_magnitude_fn(CP_DIR, 5)[1] == 6
    assert build_cp_magnitude_fn(CP_DIR, 3)[1] == 2
    assert build_cp_magnitude_fn(CP_DIR, 100)[1] == 8

    magnitude, _ = build_cp_magnitude_fn(CP_DIR, 8)
    values = magnitude(np.array([0.05, 0.5, 0.95]))
    assert np.all(values >= 0)
    # Loading is front-heavy: near-LE |Cp| exceeds near-TE |Cp|.
    assert values[0] > values[-1]


@pytest.mark.skipif(
    not (_has_surfplan() and CP_DIR.is_dir()),
    reason="SurfplanAdapter or data unavailable",
)
def test_interactive_plot_cp_scaled_vectors_run():
    from VSM.core.BodyAerodynamics import BodyAerodynamics
    from VSM.plot_geometry_plotly import interactive_plot

    body = BodyAerodynamics.instantiate(
        n_panels=36,
        file_path=DRAWN_YAML,
        spanwise_panel_distribution="uniform",
    )
    fig = interactive_plot(
        body,
        angle_of_attack=8,
        surfplan_dir=SURFPLAN_DIR,
        cp_distributions_dir=CP_DIR,
        is_show=False,
    )
    # The vector grid is still drawn (stems + arrowheads present).
    assert [t for t in fig.data if getattr(t, "name", None) == "Aerodynamic Vectors"]
