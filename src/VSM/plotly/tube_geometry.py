"""Inflatable-tube geometry for the interactive Plotly wing plot.

Builds lofted 3D surfaces for the leading-edge tube and the chordwise strut tubes
of a LEI kite, using the physical tube-diameter tables produced by SurfplanAdapter
(``leading_edge_tubes`` and ``strut_tubes`` in
``struc_geometry_all_in_surfplan.yaml``) together with the airfoil top-surface
contours (see ``VSM.plotly.canopy_geometry``).

Diameters are physical (metres) and sourced from the generated tables; positions
are anchored on the *drawn* ``Panel`` objects, so tubes always sit on the wing
that is actually plotted, independent of the Surfplan export's coordinate frame:

* Leading-edge tube: centred just aft of the drawn LE polyline, diameter
  interpolated from the ``leading_edge_tubes`` table by span fraction (this
  captures the tapering tip, which ``t * chord`` does not).
* Struts: placed at the ``is_strut`` rib span fractions, shaped to follow the
  local airfoil top-surface and offset downward so the tube top touches the
  canopy underside. Per-strut LE/TE diameters come from the ``strut_tubes`` table
  paired tip-to-tip with the strut ribs (both are symmetric and span-ordered, so
  the pairing does not rely on the table's node indices).
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import yaml

from VSM.plotly.canopy_geometry import interpolate_contour_at, load_contour_table

TUBE_COLOR = "black"
TUBE_OPACITY = 1.0
RING_COLOR = "#08306b"


def circle_points(
    center: np.ndarray,
    axis: np.ndarray,
    diameter: float,
    n: int = 24,
    ref: np.ndarray = None,
) -> np.ndarray:
    """Ring of ``n`` points of the given diameter, perpendicular to ``axis``.

    Args:
        center (np.ndarray): Ring centre, shape ``(3,)``.
        axis (np.ndarray): Tube axis at this station; the ring lies in the plane
            normal to it.
        diameter (float): Ring diameter.
        n (int): Number of points around the ring. Defaults to 24.
        ref (np.ndarray): Optional reference vector fixing the azimuthal origin of
            the ring. Passing the *same* ``ref`` for every ring of a tube keeps
            consecutive rings consistently oriented, which avoids the surface
            twisting where the tube tangent swings (e.g. at a wingtip). Falls back
            to an automatic helper when ``ref`` is parallel to the axis.

    Returns:
        np.ndarray: ``(n, 3)`` ring points, closed (first point repeated at end).
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    if ref is not None:
        helper = np.asarray(ref, dtype=float)
        if abs(axis @ (helper / max(np.linalg.norm(helper), 1e-12))) > 0.98:
            helper = np.array([0.0, 0.0, 1.0])
            if abs(axis @ helper) > 0.9:
                helper = np.array([0.0, 1.0, 0.0])
    else:
        helper = np.array([0.0, 0.0, 1.0])
        if abs(axis @ helper) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, helper)
    e1 = e1 / max(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(axis, e1)
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    radius = 0.5 * float(diameter)
    return (
        np.asarray(center, dtype=float)[None, :]
        + radius * np.cos(theta)[:, None] * e1[None, :]
        + radius * np.sin(theta)[:, None] * e2[None, :]
    )


def rings_along_centerline(
    centers: np.ndarray, diameters: np.ndarray, n: int = 24
) -> List[np.ndarray]:
    """Sweep circular rings along a centreline with twist-free framing.

    The ring frame is parallel-transported from one station to the next (the
    previous in-plane axis is projected onto the new normal plane) instead of
    being chosen independently per ring. This keeps the swept surface from
    twisting or pinching where the centreline curves sharply -- e.g. where the
    leading-edge tube turns chordwise at a wingtip.

    Args:
        centers (np.ndarray): ``(m, 3)`` centreline points.
        diameters (np.ndarray): ``(m,)`` local tube diameters.
        n (int): Number of points around each ring. Defaults to 24.

    Returns:
        List[np.ndarray]: One ``(n, 3)`` ring per centreline station.
    """
    centers = np.asarray(centers, dtype=float)
    diameters = np.asarray(diameters, dtype=float)
    m = len(centers)

    tangents = np.zeros_like(centers)
    tangents[1:-1] = centers[2:] - centers[:-2]
    tangents[0] = centers[1] - centers[0]
    tangents[-1] = centers[-1] - centers[-2]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True).clip(1e-12)

    def _seed_axis(tangent):
        ref = np.array([0.0, 0.0, 1.0])
        if abs(tangent @ ref) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        axis = ref - (ref @ tangent) * tangent
        return axis / max(np.linalg.norm(axis), 1e-12)

    theta = np.linspace(0.0, 2.0 * np.pi, n)
    e1 = _seed_axis(tangents[0])
    rings = []
    for index in range(m):
        # Parallel-transport the previous in-plane axis onto this normal plane.
        e1 = e1 - (e1 @ tangents[index]) * tangents[index]
        if np.linalg.norm(e1) < 1e-8:
            e1 = _seed_axis(tangents[index])
        e1 /= max(np.linalg.norm(e1), 1e-12)
        e2 = np.cross(tangents[index], e1)
        radius = 0.5 * diameters[index]
        rings.append(
            centers[index][None, :]
            + radius * np.cos(theta)[:, None] * e1[None, :]
            + radius * np.sin(theta)[:, None] * e2[None, :]
        )
    return rings


def load_le_diameter_interp(processed_dir: Path) -> Callable[[float], float]:
    """Return a leading-edge diameter(span-fraction) interpolator.

    Builds the interpolator from the ``leading_edge_tubes`` segment table: each
    segment midpoint gives a ``(|y| fraction, diameter)`` sample. Fractions are
    of the half-span so the curve can be sampled against any drawn wing.

    Args:
        processed_dir (Path): Directory holding
            ``struc_geometry_all_in_surfplan.yaml``.

    Returns:
        Callable[[float], float]: Function mapping a span fraction in ``[0, 1]``
            to a leading-edge tube diameter in metres.
    """
    struc = yaml.safe_load(
        (Path(processed_dir) / "struc_geometry_all_in_surfplan.yaml").read_text()
    )
    particles = {
        int(row[0]): np.asarray(row[1:4], dtype=float)
        for row in struc["wing_particles"]["data"]
    }
    le = struc["leading_edge_tubes"]
    idx_ci = le["headers"].index("ci")
    idx_cj = le["headers"].index("cj")
    idx_d = le["headers"].index("diameter")

    y_mid = []
    diam = []
    for row in le["data"]:
        ci = particles[int(row[idx_ci])]
        cj = particles[int(row[idx_cj])]
        y_mid.append(0.5 * (ci[1] + cj[1]))
        diam.append(float(row[idx_d]))
    y_mid = np.abs(np.asarray(y_mid))
    diam = np.asarray(diam)
    half_span = y_mid.max() if y_mid.max() > 0 else 1.0
    fractions = y_mid / half_span

    order = np.argsort(fractions)
    frac_sorted = fractions[order]
    diam_sorted = diam[order]

    def interp(span_fraction: float) -> float:
        return float(np.interp(span_fraction, frac_sorted, diam_sorted))

    return interp


def load_strut_diameter_table(processed_dir: Path) -> List[Tuple[float, float]]:
    """Per-strut ``(diam_le, diam_te)`` pairs in span order (tip to tip).

    Args:
        processed_dir (Path): Directory holding
            ``struc_geometry_all_in_surfplan.yaml``.

    Returns:
        List[Tuple[float, float]]: Leading- and trailing-edge strut-tube diameters
            in metres, in the table's listed (span) order.
    """
    struc = yaml.safe_load(
        (Path(processed_dir) / "struc_geometry_all_in_surfplan.yaml").read_text()
    )
    st = struc["strut_tubes"]
    idx_le = st["headers"].index("strut_diam_le")
    idx_te = st["headers"].index("strut_diam_te")
    return [(float(row[idx_le]), float(row[idx_te])) for row in st["data"]]


def load_strut_span_fractions(processed_dir: Path) -> np.ndarray:
    """Signed span fractions of the strut ribs, from the generated aero yaml.

    Uses the ``is_strut`` flag in each section's airfoil ``info_dict``. Fractions
    are ``LE_y`` normalised by the half-span, returned sorted by descending span
    (``+tip`` to ``-tip``) to match the ``strut_tubes`` table order.

    Args:
        processed_dir (Path): Directory holding ``aero_geometry.yaml``.

    Returns:
        np.ndarray: Signed span fractions of the strut ribs, high to low.
    """
    aero = yaml.safe_load((Path(processed_dir) / "aero_geometry.yaml").read_text())
    sections = aero["wing_sections"]
    airfoils = aero["wing_airfoils"]
    idx_id = sections["headers"].index("airfoil_id")
    idx_le_y = sections["headers"].index("LE_y")
    a_idx_id = airfoils["headers"].index("airfoil_id")
    a_idx_info = airfoils["headers"].index("info_dict")
    is_strut_by_id = {
        row[a_idx_id]: bool(row[a_idx_info].get("is_strut", False))
        for row in airfoils["data"]
    }

    span_y = np.array([abs(row[idx_le_y]) for row in sections["data"]])
    half_span = span_y.max() if span_y.max() > 0 else 1.0
    strut_y = [
        row[idx_le_y]
        for row in sections["data"]
        if is_strut_by_id.get(row[idx_id], False)
    ]
    fractions = np.array(sorted(strut_y, reverse=True)) / half_span
    return fractions


def _drawn_stations(panels: List[Any]) -> Dict[str, np.ndarray]:
    """Spanwise edge stations of the drawn wing as stacked arrays.

    Returns leading/trailing points, chord lengths, chord and up unit vectors, and
    signed span fractions for the ``n_panels + 1`` panel edges, sorted by span.
    """
    le = [panels[0].LE_point_1] + [p.LE_point_2 for p in panels]
    te = [panels[0].TE_point_1] + [p.TE_point_2 for p in panels]
    up = [panels[0].x_airf]
    for index, panel in enumerate(panels):
        neighbours = [panel.x_airf]
        if index + 1 < len(panels):
            neighbours.append(panels[index + 1].x_airf)
        up.append(np.mean(neighbours, axis=0))

    le = np.asarray(le, dtype=float)
    te = np.asarray(te, dtype=float)
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up, axis=1, keepdims=True).clip(1e-12)
    chord_vec = te - le
    chord_len = np.linalg.norm(chord_vec, axis=1)
    chord_hat = chord_vec / chord_len[:, None].clip(1e-12)

    half_span = np.abs(le[:, 1]).max()
    half_span = half_span if half_span > 0 else 1.0
    fraction = le[:, 1] / half_span

    order = np.argsort(fraction)
    return {
        "le": le[order],
        "te": te[order],
        "up": up[order],
        "chord_len": chord_len[order],
        "chord_hat": chord_hat[order],
        "fraction": fraction[order],
    }


def _station_at(stations: Dict[str, np.ndarray], signed_fraction: float) -> dict:
    """Interpolate a drawn-wing station at a signed span fraction."""
    frac = stations["fraction"]
    le = np.array(
        [np.interp(signed_fraction, frac, stations["le"][:, i]) for i in range(3)]
    )
    te = np.array(
        [np.interp(signed_fraction, frac, stations["te"][:, i]) for i in range(3)]
    )
    up = np.array(
        [np.interp(signed_fraction, frac, stations["up"][:, i]) for i in range(3)]
    )
    up = up / max(np.linalg.norm(up), 1e-12)
    chord_vec = te - le
    chord_len = float(np.linalg.norm(chord_vec))
    chord_hat = chord_vec / max(chord_len, 1e-12)
    return {
        "le": le,
        "te": te,
        "up": up,
        "chord_len": chord_len,
        "chord_hat": chord_hat,
    }


def _tip_extension(le_center, te_point, le_diameter, n=8, taper=1.0):
    """Centreline points + diameters continuing from a tip LE to the TE.

    Extends the leading-edge tube at the wingtip so it runs on to the trailing
    edge of the outer panel, at (by default) constant diameter for a clean tube.
    """
    ts = np.linspace(0.0, 1.0, n + 1)[1:]  # exclude 0 (the LE centre already exists)
    le_center = np.asarray(le_center, dtype=float)
    te_point = np.asarray(te_point, dtype=float)
    points = [le_center + t * (te_point - le_center) for t in ts]
    diams = [le_diameter * (1.0 - (1.0 - taper) * t) for t in ts]
    return points, diams


def build_le_tube(
    panels: List[Any],
    le_diameter_interp: Callable[[float], float],
    extend_tips: bool = True,
) -> List[np.ndarray]:
    """Build the leading-edge tube as a list of cross-section rings.

    The tube runs spanwise along the drawn LE polyline; each ring is centred just
    aft of the LE point (so the tube front touches the drawn leading edge) and its
    diameter is looked up from the LE table at the local span fraction. With
    ``extend_tips`` the tube wraps around each wingtip and continues toward the
    trailing edge, tapering closed.

    Args:
        panels (List[Any]): Drawn ``Panel`` objects, ordered tip to tip.
        le_diameter_interp (Callable[[float], float]): Output of
            :func:`load_le_diameter_interp`.
        extend_tips (bool): Wrap the tube from each tip toward the trailing edge.

    Returns:
        List[np.ndarray]: One ``(n, 3)`` ring per centreline station.
    """
    stations = _drawn_stations(panels)
    le = stations["le"]
    te = stations["te"]
    fraction = np.abs(stations["fraction"])
    diameters = np.array([le_diameter_interp(f) for f in fraction])
    centers = le + 0.5 * diameters[:, None] * stations["chord_hat"]

    center_list = list(centers)
    diam_list = list(diameters)
    if extend_tips:
        neg_pts, neg_d = _tip_extension(centers[0], te[0], diameters[0])
        pos_pts, pos_d = _tip_extension(centers[-1], te[-1], diameters[-1])
        # Prepend the -tip wrap (reversed: TE -> LE) and append the +tip wrap.
        center_list = neg_pts[::-1] + center_list + pos_pts
        diam_list = neg_d[::-1] + diam_list + pos_d

    return rings_along_centerline(np.array(center_list), np.array(diam_list))


def build_strut(
    stations: Dict[str, np.ndarray],
    contour_table,
    signed_fraction: float,
    diam_le: float,
    diam_te: float,
    le_tube_diameter: float = 0.0,
) -> List[np.ndarray]:
    """Build one chordwise strut tube hugging the canopy underside.

    The local airfoil top-surface is placed on the drawn wing at the strut span,
    then each chord station is offset downward by the local strut radius so the
    tube top touches the canopy. Diameter tapers linearly from ``diam_le`` at the
    leading edge to ``diam_te`` at the trailing edge.

    The strut starts at the aft edge of the leading-edge tube (chord fraction
    ``le_tube_diameter / chord`` -- the LE tube spans that far back from the drawn
    LE) rather than at the leading edge, so it butts against the LE tube instead
    of passing straight through it.

    Args:
        stations (Dict[str, np.ndarray]): Output of :func:`_drawn_stations`.
        contour_table: Output of ``canopy_geometry.load_contour_table``.
        signed_fraction (float): Signed span fraction of the strut rib.
        diam_le (float): Strut-tube diameter at the leading edge, metres.
        diam_te (float): Strut-tube diameter at the trailing edge, metres.
        le_tube_diameter (float): Local leading-edge tube diameter, metres; sets
            where the strut starts. Zero means start at the leading edge.

    Returns:
        List[np.ndarray]: One ``(n, 3)`` ring per chordwise station.
    """
    station = _station_at(stations, signed_fraction)
    contour = interpolate_contour_at(abs(signed_fraction), contour_table)
    chord_len = station["chord_len"]

    # Start the strut partway into the leading-edge tube (half its depth) so the
    # forward cap is buried inside the LE tube -- no gap, no through-poke.
    le_tube_depth_fraction = le_tube_diameter / max(chord_len, 1e-12)
    start_fraction = min(0.5 * le_tube_depth_fraction, 0.9)
    cx = contour[:, 0]
    cy = contour[:, 1]
    keep = cx > start_fraction
    if start_fraction > 0 and keep.any():
        cx = np.concatenate([[start_fraction], cx[keep]])
        cy = np.concatenate(
            [[float(np.interp(start_fraction, contour[:, 0], cy))], cy[keep]]
        )

    canopy_points = (
        station["le"][None, :]
        + cx[:, None] * chord_len * station["chord_hat"][None, :]
        + cy[:, None] * chord_len * station["up"][None, :]
    )
    diameters = diam_le + (diam_te - diam_le) * cx
    centers = canopy_points - 0.5 * diameters[:, None] * station["up"][None, :]
    return rings_along_centerline(centers, diameters)


def build_tube_data(panels: List[Any], processed_dir: Path) -> Dict[str, Any]:
    """Assemble leading-edge and strut tube rings for the drawn wing.

    Args:
        panels (List[Any]): Drawn ``Panel`` objects, ordered tip to tip.
        processed_dir (Path): Directory of SurfplanAdapter-generated files.

    Returns:
        Dict[str, Any]: ``{"le": [rings], "struts": [[rings], ...]}``.
    """
    processed_dir = Path(processed_dir)
    le_interp = load_le_diameter_interp(processed_dir)
    contour_table = load_contour_table(processed_dir)
    strut_fractions = load_strut_span_fractions(processed_dir)
    strut_table = load_strut_diameter_table(processed_dir)

    le_rings = build_le_tube(panels, le_interp)

    stations = _drawn_stations(panels)
    struts = []
    for index, signed_fraction in enumerate(strut_fractions):
        if index < len(strut_table):
            diam_le, diam_te = strut_table[index]
        else:
            # Fallback: size the strut from the local LE tube diameter.
            diam_le = le_interp(abs(signed_fraction))
            diam_te = 0.6 * diam_le
        le_tube_diameter = le_interp(abs(signed_fraction))
        struts.append(
            build_strut(
                stations,
                contour_table,
                signed_fraction,
                diam_le,
                diam_te,
                le_tube_diameter=le_tube_diameter,
            )
        )
    return {"le": le_rings, "struts": struts}


def _rings_to_surface(
    rings: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack a list of rings into ``X``, ``Y``, ``Z`` surface arrays."""
    grid = np.stack(rings, axis=0)  # (n_rings, n_circle, 3)
    return grid[:, :, 0], grid[:, :, 1], grid[:, :, 2]


def _add_surface(fig: go.Figure, rings: List[np.ndarray], name: str, show_legend: bool):
    """Add one tube as a uniformly coloured, semi-transparent surface."""
    x, y, z = _rings_to_surface(rings)
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=np.zeros_like(z),
            colorscale=[[0, TUBE_COLOR], [1, TUBE_COLOR]],
            showscale=False,
            opacity=TUBE_OPACITY,
            name=name,
            showlegend=show_legend,
            lighting=dict(ambient=0.75, diffuse=0.8, specular=0.05, roughness=0.9),
        )
    )


def _add_ring_cap(fig: go.Figure, ring: np.ndarray) -> None:
    """Close an open tube end with a filled (fan-triangulated) cap disc."""
    ring = np.asarray(ring)[:-1]  # drop the duplicated closing point
    n = len(ring)
    center = ring.mean(axis=0)
    verts = np.vstack([center[None, :], ring])  # index 0 is the centre
    i = [0] * n
    j = [1 + t for t in range(n)]
    k = [1 + (t + 1) % n for t in range(n)]
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=i,
            j=j,
            k=k,
            color=TUBE_COLOR,
            opacity=TUBE_OPACITY,
            name="Tube cap",
            showlegend=False,
        )
    )


def _add_tube_with_caps(
    fig: go.Figure, rings: List[np.ndarray], name: str, show_legend: bool
):
    """Add a tube surface and close both open ends with caps."""
    _add_surface(fig, rings, name, show_legend)
    _add_ring_cap(fig, rings[0])
    _add_ring_cap(fig, rings[-1])


def add_strut_surfaces(fig: go.Figure, tube_data: Dict[str, Any]) -> None:
    """Add the strut tube surfaces (with end caps) to the figure."""
    for index, strut in enumerate(tube_data["struts"]):
        _add_tube_with_caps(fig, strut, "Strut tube", show_legend=(index == 0))


def add_le_tube_surface(fig: go.Figure, tube_data: Dict[str, Any]) -> None:
    """Add the leading-edge tube surface (with end caps) to the figure."""
    _add_tube_with_caps(fig, tube_data["le"], "Leading-edge tube", show_legend=True)


def add_tube_surfaces(fig: go.Figure, tube_data: Dict[str, Any]) -> None:
    """Add the leading-edge and strut tube surfaces to the figure."""
    add_le_tube_surface(fig, tube_data)
    add_strut_surfaces(fig, tube_data)


def add_tube_rings(fig: go.Figure, tube_data: Dict[str, Any]) -> None:
    """Add the construction rings (wireframe circles) for inspection."""
    all_rings = list(tube_data["le"]) + [
        r for strut in tube_data["struts"] for r in strut
    ]
    for index, ring in enumerate(all_rings):
        fig.add_trace(
            go.Scatter3d(
                x=ring[:, 0],
                y=ring[:, 1],
                z=ring[:, 2],
                mode="lines",
                line=dict(color=RING_COLOR, width=1),
                name="Tube rings",
                showlegend=(index == 0),
            )
        )
