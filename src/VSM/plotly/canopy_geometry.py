"""Curved single-skin canopy geometry for the interactive Plotly wing plot.

The canopy is rendered as a loft of the real airfoil top-surfaces (the ``.dat``
profiles produced by SurfplanAdapter) placed on the *drawn* wing panels. Only the
upper contour is used (LEI kites are single-skin), so the inflatable tubes stay
visible from below. See ``VSM.plotly.tube_geometry`` for the tubes and
``VSM.plotly.surfplan_runner`` for how the source files are generated.

All geometry is anchored on the drawn ``Panel`` objects (their LE/TE points and
local frame), so the result is independent of the coordinate frame of the source
Surfplan export.
"""

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import yaml


def read_upper_contour(dat_file_path: Path) -> np.ndarray:
    """Read an airfoil ``.dat`` file and return its upper contour, LE-to-TE.

    The ``.dat`` files use the standard ordering starting at the trailing edge
    ``(1, 0)``, running forward over the upper surface to the leading edge
    ``(0, 0)``, then back over the lower surface to the trailing edge. The upper
    surface is the run up to the minimum-x (leading edge) point.

    Args:
        dat_file_path (Path): Path to the airfoil coordinate file. The first line
            is a name label and is skipped; remaining lines are ``x y`` pairs
            normalised by the chord.

    Returns:
        np.ndarray: Upper contour as an ``(n, 2)`` array ordered from the leading
            edge ``(0, 0)`` to the trailing edge ``(1, 0)``.
    """
    points = []
    for line in Path(dat_file_path).read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            points.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    points = np.asarray(points, dtype=float)
    le_index = int(np.argmin(points[:, 0]))
    upper = points[: le_index + 1]  # TE -> LE
    return upper[::-1]  # LE -> TE


def resample_contour(contour: np.ndarray, n_points: int = 40) -> np.ndarray:
    """Resample a contour to a fixed point count by arc length.

    Uses cumulative chord-length parameterisation so that the leading- and
    trailing-edge endpoints are preserved. This lets contours from different
    airfoils be blended point-by-point.

    Args:
        contour (np.ndarray): ``(n, 2)`` contour ordered LE-to-TE.
        n_points (int): Number of output points. Defaults to 40.

    Returns:
        np.ndarray: ``(n_points, 2)`` resampled contour.
    """
    segment_lengths = np.linalg.norm(np.diff(contour, axis=0), axis=1)
    arc_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = arc_length[-1]
    if total <= 0:
        return np.repeat(contour[:1], n_points, axis=0)
    sample = np.linspace(0.0, total, n_points)
    x = np.interp(sample, arc_length, contour[:, 0])
    y = np.interp(sample, arc_length, contour[:, 1])
    return np.column_stack([x, y])


def load_contour_table(
    processed_dir: Path, n_points: int = 40
) -> List[Tuple[float, np.ndarray]]:
    """Build a span-fraction -> upper-contour table from the generated aero yaml.

    Reads ``aero_geometry.yaml`` written by SurfplanAdapter: each wing section
    provides its span position (``LE_y``) and an ``airfoil_id`` whose
    ``info_dict`` carries the ``dat_file_path``. Span positions are normalised to
    fractions of the half-span so the table can be sampled against any drawn wing
    regardless of scale or frame.

    Args:
        processed_dir (Path): Directory holding ``aero_geometry.yaml`` and the
            ``profiles/`` referenced by relative ``dat_file_path`` entries.
        n_points (int): Point count each contour is resampled to. Defaults to 40.

    Returns:
        List[Tuple[float, np.ndarray]]: ``(span_fraction, contour)`` pairs sorted
            by ascending span fraction, each contour an ``(n_points, 2)`` array.
    """
    processed_dir = Path(processed_dir)
    aero = yaml.safe_load((processed_dir / "aero_geometry.yaml").read_text())
    sections = aero["wing_sections"]
    airfoils = aero["wing_airfoils"]

    section_headers = sections["headers"]
    idx_id = section_headers.index("airfoil_id")
    idx_le_y = section_headers.index("LE_y")

    airfoil_headers = airfoils["headers"]
    a_idx_id = airfoil_headers.index("airfoil_id")
    a_idx_info = airfoil_headers.index("info_dict")
    dat_by_id = {
        row[a_idx_id]: row[a_idx_info]["dat_file_path"] for row in airfoils["data"]
    }

    contour_cache = {}
    span_y = np.array([abs(row[idx_le_y]) for row in sections["data"]])
    half_span = span_y.max() if span_y.max() > 0 else 1.0

    table = []
    for row in sections["data"]:
        airfoil_id = row[idx_id]
        if airfoil_id not in contour_cache:
            dat_path = processed_dir / dat_by_id[airfoil_id]
            contour_cache[airfoil_id] = resample_contour(
                read_upper_contour(dat_path), n_points
            )
        fraction = abs(row[idx_le_y]) / half_span
        table.append((fraction, contour_cache[airfoil_id]))

    table.sort(key=lambda item: item[0])
    return table


def interpolate_contour_at(
    span_fraction: float, contour_table: List[Tuple[float, np.ndarray]]
) -> np.ndarray:
    """Linearly blend the two contours bracketing a span fraction.

    Args:
        span_fraction (float): Target fraction of the half-span, in ``[0, 1]``.
        contour_table (List[Tuple[float, np.ndarray]]): Output of
            :func:`load_contour_table`; contours share a common point count.

    Returns:
        np.ndarray: Interpolated contour, clamped to the end contours outside the
            table's fraction range.
    """
    fractions = [f for f, _ in contour_table]
    if span_fraction <= fractions[0]:
        return contour_table[0][1]
    if span_fraction >= fractions[-1]:
        return contour_table[-1][1]
    upper = int(np.searchsorted(fractions, span_fraction))
    lower = upper - 1
    f_lo, c_lo = contour_table[lower]
    f_hi, c_hi = contour_table[upper]
    weight = (span_fraction - f_lo) / (f_hi - f_lo) if f_hi > f_lo else 0.0
    return (1.0 - weight) * c_lo + weight * c_hi


def _edge_stations(panels: List[Any]) -> List[dict]:
    """Collect the ``n_panels + 1`` spanwise edge stations of the drawn wing.

    Each station carries its leading- and trailing-edge points, the chord vector
    and length, and the local up-vector (averaged across adjacent panels), all
    taken from the ``Panel`` objects so the canopy sits exactly on the drawn wing.
    """
    stations = []

    def make_station(le_point, te_point, up_vectors):
        chord_vector = np.asarray(te_point) - np.asarray(le_point)
        chord_length = float(np.linalg.norm(chord_vector))
        chord_hat = chord_vector / max(chord_length, 1e-12)
        up = np.mean(up_vectors, axis=0)
        up = up / max(np.linalg.norm(up), 1e-12)
        return {
            "le_point": np.asarray(le_point, dtype=float),
            "te_point": np.asarray(te_point, dtype=float),
            "chord_length": chord_length,
            "chord_hat": chord_hat,
            "up": up,
        }

    stations.append(
        make_station(panels[0].LE_point_1, panels[0].TE_point_1, [panels[0].x_airf])
    )
    for index, panel in enumerate(panels):
        neighbours = [panel.x_airf]
        if index + 1 < len(panels):
            neighbours.append(panels[index + 1].x_airf)
        stations.append(make_station(panel.LE_point_2, panel.TE_point_2, neighbours))
    return stations


def build_canopy_grid(
    panels: List[Any], contour_table: List[Tuple[float, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the lofted canopy surface grid over the drawn wing panels.

    For every panel-edge station the local upper contour (blended from the rib
    profiles at that span fraction) is placed on the drawn chord: the contour's
    chordwise coordinate runs along the station chord and its height along the
    local up-vector, both scaled by the station chord length.

    Args:
        panels (List[Any]): Drawn ``Panel`` objects, ordered tip to tip.
        contour_table (List[Tuple[float, np.ndarray]]): Output of
            :func:`load_contour_table`.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ``X``, ``Y``, ``Z`` arrays of
            shape ``(n_panels + 1, n_points)`` suitable for a Plotly surface.
    """
    stations = _edge_stations(panels)
    all_y = np.array([abs(station["le_point"][1]) for station in stations])
    half_span = all_y.max() if all_y.max() > 0 else 1.0

    rows = []
    for station in stations:
        fraction = abs(station["le_point"][1]) / half_span
        contour = interpolate_contour_at(fraction, contour_table)
        chord_length = station["chord_length"]
        points = (
            station["le_point"][None, :]
            + contour[:, 0:1] * chord_length * station["chord_hat"][None, :]
            + contour[:, 1:2] * chord_length * station["up"][None, :]
        )
        rows.append(points)

    grid = np.stack(rows, axis=0)  # (n_stations, n_points, 3)
    return grid[:, :, 0], grid[:, :, 1], grid[:, :, 2]


def distributed_surface_vector_field(
    canopy_grid: Tuple[np.ndarray, np.ndarray, np.ndarray],
    forces_of_panels: List[np.ndarray],
    scale: float,
    n_chord: int = 10,
    n_span: int = None,
    cp_magnitude_fn=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chord-normal force-vector field over the canopy (origins, directions, lengths).

    One row of ``n_chord`` nodes is placed at each panel centre, uniformly over
    the chord (at the true 5%-95% chord fractions). Every node's arrow points
    normal to the local chord line (the flat-panel "up", outward/suction side).
    Each panel's VSM force is distributed over its chord nodes, weighted by the
    ``|Cp|`` chordwise loading (an even split without a Cp distribution) and
    normalised to length so the largest panel force plots at the maximum chord
    length (``scale``).

    Args:
        canopy_grid: ``(X, Y, Z)`` canopy arrays, each ``(S, P)``.
        forces_of_panels: Per-panel VSM force vectors.
        scale: Length scale (typically the maximum chord); the largest panel
            force plots at this length.
        n_chord: Nodes per row over the chord. Defaults to 10.
        n_span: Number of spanwise rows. Defaults to one per panel (``S - 1``).
        cp_magnitude_fn: Optional ``x/c -> |Cp|`` weighting callable.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ``origins`` ``(N, 3)``,
            unit ``directions`` ``(N, 3)`` and ``lengths`` ``(N,)``. Empty arrays
            if there is no force.
    """
    x, y, z = canopy_grid
    grid = np.stack([x, y, z], axis=-1)  # (S, P, 3)

    force_magnitudes = np.linalg.norm(np.asarray(forces_of_panels), axis=1)
    if force_magnitudes.size == 0 or np.max(force_magnitudes) <= 0:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0,))

    panel_grid = 0.5 * (grid[:-1] + grid[1:])  # (S - 1, P, 3), one row per panel
    n_rows, n_points, _ = panel_grid.shape

    if n_span is None:
        n_span = n_rows
    span_rows = np.unique(
        np.clip(np.round(np.linspace(0, n_rows - 1, n_span)), 0, n_rows - 1).astype(int)
    )
    d_span = np.gradient(panel_grid, axis=0)  # (R, P, 3)

    targets = np.linspace(0.05, 0.95, n_chord)
    if cp_magnitude_fn is not None:
        chord_weights = np.asarray(cp_magnitude_fn(targets), dtype=float)
    else:
        chord_weights = np.ones(n_chord)
    chord_weights = chord_weights / max(chord_weights.sum(), 1e-12)

    origins = []
    directions = []
    node_forces = []
    for row in span_rows:
        pts = panel_grid[row]
        chord_vector = pts[-1] - pts[0]
        chord_length = np.linalg.norm(chord_vector)
        chord_hat = chord_vector / max(chord_length, 1e-12)

        span_vector = d_span[row].mean(axis=0)
        chord_normal = np.cross(chord_vector, span_vector)
        cn = np.linalg.norm(chord_normal)
        if cn < 1e-9:
            continue
        chord_normal = chord_normal / cn
        if chord_normal[2] < 0:  # outward (suction) side
            chord_normal = -chord_normal

        panel_force = force_magnitudes[row] if row < len(force_magnitudes) else 0.0

        chord_fraction = ((pts - pts[0]) @ chord_hat) / max(chord_length, 1e-12)
        order = np.argsort(chord_fraction)
        frac_sorted = chord_fraction[order]
        pts_sorted = pts[order]
        for k, target in enumerate(targets):
            point = np.array(
                [np.interp(target, frac_sorted, pts_sorted[:, j]) for j in range(3)]
            )
            origins.append(point)
            directions.append(chord_normal)
            node_forces.append(panel_force * chord_weights[k])

    if not origins:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0,))

    origins = np.array(origins)
    directions = np.array(directions)
    node_forces = np.array(node_forces)
    newton_to_metre = scale / max(float(force_magnitudes.max()), 1e-12)
    # Anchor the per-vector size to a base of 10 chordwise nodes, so adding more
    # nodes makes the field denser without shrinking each arrow (at n_chord=10 the
    # factor is 1 and the largest panel's arrows still sum to the max chord).
    lengths = node_forces * newton_to_metre * (n_chord / 10.0)
    return origins, directions, lengths
