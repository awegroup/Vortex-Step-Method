"""Helpers for the interactive Plotly wing plot.

The public ``interactive_plot`` entry point lives in ``VSM.plot_geometry_plotly``
(unchanged location for backward compatibility). This subpackage holds the
supporting machinery for the opt-in "fancy" plot that renders the inflatable
tubes and the curved single-skin canopy of a LEI kite from a raw Surfplan export.
"""

from VSM.plotly.surfplan_runner import ensure_surfplan_processed
from VSM.plotly.tube_geometry import (
    add_tube_rings,
    add_tube_surfaces,
    build_tube_data,
)
from VSM.plotly.canopy_geometry import build_canopy_grid, load_contour_table

__all__ = [
    "ensure_surfplan_processed",
    "build_tube_data",
    "add_tube_surfaces",
    "add_tube_rings",
    "build_canopy_grid",
    "load_contour_table",
]
