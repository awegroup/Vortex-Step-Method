from numba import jit
import numpy as np


@jit(nopython=True)
def jit_cross(a, b):
    return np.cross(a, b)


@jit(nopython=True)
def jit_norm(value):
    return np.linalg.norm(value.astype(np.float64))


@jit(nopython=True)
def jit_dot(a, b):
    return np.dot(a.astype(np.float64), b.astype(np.float64))


# =====================================================================
# JIT-compiled AIC assembly
#
# Numba ports of the per-filament induced-velocity kernels in
# core/Filament.py (velocity_3D_bound_vortex, velocity_3D_trailing_vortex,
# velocity_3D_trailing_vortex_semiinfinite) and of the 2D bound correction
# in core/Panel.py, assembled over all control-point/panel pairs in one
# compiled double loop. The formulas, branch structure (vortex-core
# projections included), and summation order match the Python originals
# term for term, so the assembled matrices agree to floating-point
# round-off; Filament.py remains the readable reference implementation.
# The only intentional difference: the bound-vortex in-core branch cannot
# emit the logging.info diagnostic from compiled code.
# =====================================================================

_ALPHA0_OSEEN = 1.25643  # Oseen parameter (Filament._alpha0)
_NU_AIR = 1.48e-5  # kinematic viscosity of air [m^2/s] (Filament._nu)


@jit(nopython=True, cache=True)
def _vel_bound_vortex(XV1, XV2, XVP, gamma, core_radius_fraction):
    """Bound-filament induced velocity (Vatistas core), as in
    Filament.velocity_3D_bound_vortex."""
    r0 = XV2 - XV1
    r1 = XVP - XV1
    r2 = XVP - XV2

    r1Xr0 = np.cross(r1, r0)
    epsilon = core_radius_fraction * np.linalg.norm(r0)
    dist = np.linalg.norm(r1Xr0) / np.linalg.norm(r0)
    if dist > epsilon:
        r1Xr2 = np.cross(r1, r2)
        return (
            gamma
            / (4 * np.pi)
            * r1Xr2
            / (np.linalg.norm(r1Xr2) ** 2)
            * np.dot(r0, r1 / np.linalg.norm(r1) - r2 / np.linalg.norm(r2))
        )
    elif dist < 1e-12 * epsilon:
        return np.zeros(3)
    else:
        r1_radial = r1 - np.dot(r1, r0) * r0 / (np.linalg.norm(r0) ** 2)
        r2_radial = r2 - np.dot(r2, r0) * r0 / (np.linalg.norm(r0) ** 2)
        r1_proj = np.dot(r1, r0) * r0 / (
            np.linalg.norm(r0) ** 2
        ) + epsilon * r1_radial / np.linalg.norm(r1_radial)
        r2_proj = np.dot(r2, r0) * r0 / (
            np.linalg.norm(r0) ** 2
        ) + epsilon * r2_radial / np.linalg.norm(r2_radial)
        r1Xr2_proj = np.cross(r1_proj, r2_proj)
        vel_ind_proj = (
            gamma
            / (4 * np.pi)
            * r1Xr2_proj
            / (np.linalg.norm(r1Xr2_proj) ** 2)
            * np.dot(
                r0,
                r1_proj / np.linalg.norm(r1_proj) - r2_proj / np.linalg.norm(r2_proj),
            )
        )
        return dist / epsilon * vel_ind_proj


@jit(nopython=True, cache=True)
def _vel_trailing_vortex(XV1, XV2, XVP, gamma, Uinf):
    """Finite trailing-leg induced velocity (viscous core), as in
    Filament.velocity_3D_trailing_vortex."""
    r0 = XV2 - XV1
    r1 = XVP - XV1
    r2 = XVP - XV2

    r_perp = r1 - np.dot(r1, r0) * r0 / (np.linalg.norm(r0) ** 2)
    epsilon = np.sqrt(4 * _ALPHA0_OSEEN * _NU_AIR * np.linalg.norm(r_perp) / Uinf)

    r1Xr0 = np.cross(r1, r0)
    dist = np.linalg.norm(r1Xr0) / np.linalg.norm(r0)
    if dist > epsilon:
        r1Xr2 = np.cross(r1, r2)
        return (
            gamma
            / (4 * np.pi)
            * r1Xr2
            / (np.linalg.norm(r1Xr2) ** 2)
            * np.dot(r0, r1 / np.linalg.norm(r1) - r2 / np.linalg.norm(r2))
        )
    elif dist < 1e-12 * epsilon:
        return np.zeros(3)
    else:
        r1_radial = r1 - np.dot(r1, r0) * r0 / (np.linalg.norm(r0) ** 2)
        r2_radial = r2 - np.dot(r2, r0) * r0 / (np.linalg.norm(r0) ** 2)
        r1_proj = np.dot(r1, r0) * r0 / (
            np.linalg.norm(r0) ** 2
        ) + epsilon * r1_radial / np.linalg.norm(r1_radial)
        r2_proj = np.dot(r2, r0) * r0 / (
            np.linalg.norm(r0) ** 2
        ) + epsilon * r2_radial / np.linalg.norm(r2_radial)
        r1Xr2_proj = np.cross(r1_proj, r2_proj)
        vel_ind_proj = (
            gamma
            / (4 * np.pi)
            * r1Xr2_proj
            / (np.linalg.norm(r1Xr2_proj) ** 2)
            * np.dot(
                r0,
                r1_proj / np.linalg.norm(r1_proj) - r2_proj / np.linalg.norm(r2_proj),
            )
        )
        return dist / epsilon * vel_ind_proj


@jit(nopython=True, cache=True)
def _vel_semiinfinite(XV1, Vf, XVP, gamma, Uinf, filament_direction):
    """Semi-infinite trailing-vortex induced velocity, as in
    Filament.velocity_3D_trailing_vortex_semiinfinite."""
    GAMMA = -gamma * filament_direction

    r1 = XVP - XV1
    r1XVf = np.cross(r1, Vf)
    r_perp = r1 - np.dot(r1, Vf) * Vf

    if np.linalg.norm(r_perp) < 1e-16:
        return np.zeros(3)

    epsilon = np.sqrt(4 * _ALPHA0_OSEEN * _NU_AIR * np.linalg.norm(r_perp) / Uinf)

    dist = np.linalg.norm(r1XVf) / np.linalg.norm(Vf)
    if dist > epsilon:
        K = (
            GAMMA
            / 4
            / np.pi
            / np.linalg.norm(r1XVf) ** 2
            * (1 + np.dot(r1, Vf) / np.linalg.norm(r1))
        )
        return K * r1XVf
    elif dist < 1e-12 * epsilon:
        return np.zeros(3)
    else:
        r1_radial = r1 - np.dot(r1, Vf) * Vf
        r1_proj = np.dot(r1, Vf) * Vf + epsilon * r1_radial / np.linalg.norm(r1_radial)
        r1XVf_proj = np.cross(r1_proj, Vf)
        K = (
            GAMMA
            / 4
            / np.pi
            / np.linalg.norm(r1XVf_proj) ** 2
            * (1 + np.dot(r1_proj, Vf) / np.linalg.norm(r1_proj))
        )
        return K * r1XVf_proj


@jit(nopython=True, cache=True)
def _vel_bound_2D(XVP, bound_point_1, bound_point_2):
    """2D bound-vortex correction at the evaluation point, as in
    Panel.compute_velocity_induced_bound_2D (gamma = 1)."""
    r3 = XVP - (bound_point_1 + bound_point_2) / 2
    r0 = bound_point_1 - bound_point_2
    r0_unit_X_r3 = np.cross(r0 / np.linalg.norm(r0), r3)
    return (1 / (2 * np.pi)) * (r0_unit_X_r3 / (np.linalg.norm(r0_unit_X_r3) ** 2))


@jit(nopython=True, cache=True)
def assemble_AIC_matrices(
    eval_points,
    bound_point_1,
    bound_point_2,
    TE_point_1,
    TE_point_2,
    wake_units,
    wake_speeds,
    core_radius_fraction,
    evaluation_point_on_bound,
    subtract_bound_2D,
):
    """Assemble the (3, N, N) unit-circulation induced-velocity matrices for
    the horseshoe system of every panel (bound + two chordwise legs + two
    semi-infinite wake filaments), evaluated at every evaluation point.

    Filament layout and summation order match
    Panel.compute_velocity_induced_single_ring_semiinfinite: bound filament
    runs bound_point_2 -> bound_point_1; legs run bound_point_1 -> TE_point_1
    and TE_point_2 -> bound_point_2; semi-infinite filaments start at
    TE_point_1 (direction +1) and TE_point_2 (direction -1) along the ring's
    own wake direction wake_units[j] (the panel's apparent velocity, so a
    rotating body or a distributed inflow gets a locally aligned frozen wake;
    a uniform inflow gives one direction for all). wake_speeds[j] sets the
    viscous-core radius of that ring's trailing filaments.
    ``evaluation_point_on_bound`` (LLT) zeroes the bound contribution;
    ``subtract_bound_2D`` (VSM) subtracts the 2D bound correction on the
    diagonal.
    """
    n = eval_points.shape[0]
    AIC = np.empty((3, n, n))
    for icp in range(n):
        ep = eval_points[icp]
        for jring in range(n):
            if evaluation_point_on_bound:
                velocity_induced = np.zeros(3)
            else:
                velocity_induced = _vel_bound_vortex(
                    bound_point_2[jring],
                    bound_point_1[jring],
                    ep,
                    1.0,
                    core_radius_fraction,
                )
            wake_unit = wake_units[jring]
            wake_speed = wake_speeds[jring]
            velocity_induced = velocity_induced + _vel_trailing_vortex(
                bound_point_1[jring], TE_point_1[jring], ep, 1.0, wake_speed
            )
            velocity_induced = velocity_induced + _vel_trailing_vortex(
                TE_point_2[jring], bound_point_2[jring], ep, 1.0, wake_speed
            )
            velocity_induced = velocity_induced + _vel_semiinfinite(
                TE_point_1[jring], wake_unit, ep, 1.0, wake_speed, 1.0
            )
            velocity_induced = velocity_induced + _vel_semiinfinite(
                TE_point_2[jring], wake_unit, ep, 1.0, wake_speed, -1.0
            )
            if subtract_bound_2D and icp == jring:
                velocity_induced = velocity_induced - _vel_bound_2D(
                    ep, bound_point_1[jring], bound_point_2[jring]
                )
            AIC[0, icp, jring] = velocity_induced[0]
            AIC[1, icp, jring] = velocity_induced[1]
            AIC[2, icp, jring] = velocity_induced[2]
    return AIC


@jit(nopython=True, cache=True)
def assemble_curved_bound_correction_AIC(
    eval_points,
    bound_point_1,
    bound_point_2,
    TE_point_1,
    TE_point_2,
    chord_dir,
    chords,
    va_norm,
    ac_fraction,
    core_radius_fraction,
    strip_x,
    strip_w,
):
    """Curvature, sweep and rib-skew correction to the quarter-chord (direction)
    velocity, (3, N, N), to be ADDED to the trailed-vortex-only (LLT) AIC
    evaluated on the quarter chord.

    A lifting line evaluated on itself has a logarithmically singular
    self-induction wherever it is curved (Li, Gaunaa, Pirrung, Ramos-Garcia &
    Horcas 2020), so neither adding the other panels' bound segments at the
    quarter chord (Phillips & Snyder) nor leaving them out converges with the
    panel count on a curved or swept wing. Here, for every evaluation panel i,
    the wing is UNROLLED onto the tangent line of panel i (nodes keep their
    arc-length coordinate, legs perpendicular to the line in panel i's chord
    direction), and the quarter-chord velocity is built as

        V(1/4c) = LLT downwash of the unrolled wing
                + [ actual vortex system - unrolled vortex system ],

    where in the bracket the bound vorticity of every ring and the starting
    points of its legs are spread along the chord with the thin-airfoil
    distribution gamma(x) ~ sqrt((c - x)/x) (stations strip_x, weights strip_w;
    strip nodes lie on each node's own leg line so the strip legs are collinear
    with the actual legs). The first term is Prandtl's classical downwash, exact
    for straight wings; the bracket is identically zero on a straight wing with
    perpendicular legs and finite on curved ones, with the cut-off fixed at the
    chord scale by the distribution itself (effective radius 0.15 c). This is the
    discrete counterpart of the inner solution of curved lifting-line theory
    (Guermond 1990) and shares its mechanism with the effective-lifting-line /
    jointed-vortex method of Reid & Hunsaker (2020), without their two tuning
    lengths. On-blade induced drag then matches the Trefftz plane within 1 % on
    arc, swept and kite geometries independently of the panel count.

    Since the LLT AIC already holds the actual legs and wake, the returned
    matrix is (unrolled legs - actual legs) + (smeared actual - smeared
    unrolled); the semi-infinite wake filaments cancel identically.
    """
    n = eval_points.shape[0]
    n_strips = strip_x.shape[0]
    A = np.zeros((3, n, n))
    seg = bound_point_1 - bound_point_2
    L = np.empty(n)
    for j in range(n):
        L[j] = np.linalg.norm(seg[j])
    # Panel sequence direction relative to the filament direction bp2 -> bp1.
    fwd = True
    if n > 1:
        fwd = np.linalg.norm(bound_point_1[0] - bound_point_2[1]) < np.linalg.norm(
            bound_point_2[0] - bound_point_1[1]
        )
    sgn = 1.0 if fwd else -1.0
    s_node = np.zeros(n + 1)
    for j in range(n):
        s_node[j + 1] = s_node[j] + L[j]
    for i in range(n):
        ep = eval_points[i]
        t = seg[i] / L[i]
        c_i = chord_dir[i]
        s_P = 0.5 * (s_node[i] + s_node[i + 1])
        U = va_norm[i]
        for j in range(n):
            ra = ep + sgn * (s_node[j] - s_P) * t
            rb = ep + sgn * (s_node[j + 1] - s_P) * t
            if fwd:
                rb2 = ra
                rb1 = rb
            else:
                rb2 = rb
                rb1 = ra
            leg = (1.0 - ac_fraction) * chords[j] * c_i
            rte1 = rb1 + leg
            rte2 = rb2 + leg
            bp1 = bound_point_1[j]
            bp2 = bound_point_2[j]
            te1 = TE_point_1[j]
            te2 = TE_point_2[j]
            # unrolled legs (single filaments) minus actual legs (single filaments)
            v = _vel_trailing_vortex(rb1, rte1, ep, 1.0, U) + _vel_trailing_vortex(
                rte2, rb2, ep, 1.0, U
            )
            v = v - _vel_trailing_vortex(bp1, te1, ep, 1.0, U) - _vel_trailing_vortex(
                te2, bp2, ep, 1.0, U
            )
            # chordwise-spread actual system minus chordwise-spread unrolled system
            for q in range(n_strips):
                f = (strip_x[q] - ac_fraction) / (1.0 - ac_fraction)
                w = strip_w[q]
                n1 = bp1 + f * (te1 - bp1)
                n2 = bp2 + f * (te2 - bp2)
                u1 = rb1 + f * (rte1 - rb1)
                u2 = rb2 + f * (rte2 - rb2)
                v = v + w * (
                    _vel_bound_vortex(n2, n1, ep, 1.0, core_radius_fraction)
                    - _vel_bound_vortex(u2, u1, ep, 1.0, core_radius_fraction)
                    + _vel_trailing_vortex(n1, te1, ep, 1.0, U)
                    + _vel_trailing_vortex(te2, n2, ep, 1.0, U)
                    - _vel_trailing_vortex(u1, rte1, ep, 1.0, U)
                    - _vel_trailing_vortex(rte2, u2, ep, 1.0, U)
                )
            A[0, i, j] = v[0]
            A[1, i, j] = v[1]
            A[2, i, j] = v[2]
    return A


def thin_airfoil_strips(n_strips=24):
    """Chordwise stations x/c and weights of the thin-airfoil bound vorticity
    distribution gamma(x) ~ sqrt((c - x)/x), via x = c sin^2(phi) with phi at
    the midpoints of n_strips equal intervals on (0, pi/2), so that
    sqrt((c - x)/x) dx = 2 c cos^2(phi) dphi. The centroid of the weights is at
    x/c = 1/4, so the far field of the spread vorticity equals that of the
    single quarter-chord filament."""
    phi = (np.arange(n_strips) + 0.5) * (np.pi / 2) / n_strips
    x = np.sin(phi) ** 2
    w = np.cos(phi) ** 2
    return np.ascontiguousarray(x, dtype=float), np.ascontiguousarray(w / w.sum(), dtype=float)


@jit(nopython=True, cache=True)
def _on_segment_line(XV1, XV2, XVP):
    """True if XVP lies on the line through XV1 and XV2 (to round-off)."""
    r0 = XV2 - XV1
    r1 = XVP - XV1
    len0 = np.linalg.norm(r0)
    if len0 == 0.0:
        return True
    return np.linalg.norm(np.cross(r1, r0)) <= 1e-10 * len0 * max(np.linalg.norm(r1), len0)


@jit(nopython=True, cache=True)
def induced_velocity_at_points(
    points,
    bound_point_1,
    bound_point_2,
    TE_point_1,
    TE_point_2,
    gamma,
    wake_units,
    wake_speeds,
    core_radius_fraction,
):
    """Velocity induced by the complete horseshoe system of every panel
    (bound + two chordwise legs + two semi-infinite wake filaments, same layout
    and per-ring wake directions as assemble_AIC_matrices) at M arbitrary
    points, for the circulation distribution ``gamma``. Returns an (M, 3) array. A point lying on a
    filament gets no contribution from that filament (the kernels return zero
    on the line), so this can be evaluated on the attached trailed vortex
    legs themselves.
    """
    m = points.shape[0]
    n = bound_point_1.shape[0]
    out = np.zeros((m, 3))
    for ip in range(m):
        ep = points[ip]
        vel = np.zeros(3)
        for j in range(n):
            g = gamma[j]
            wake_unit = wake_units[j]
            wake_speed = wake_speeds[j]
            vel = vel + _vel_bound_vortex(
                bound_point_2[j], bound_point_1[j], ep, g, core_radius_fraction
            )
            # a straight filament induces nothing on its own line; skip the
            # legs the point lies on (the kernel would divide by zero there)
            if not _on_segment_line(bound_point_1[j], TE_point_1[j], ep):
                vel = vel + _vel_trailing_vortex(
                    bound_point_1[j], TE_point_1[j], ep, g, wake_speed
                )
            if not _on_segment_line(TE_point_2[j], bound_point_2[j], ep):
                vel = vel + _vel_trailing_vortex(
                    TE_point_2[j], bound_point_2[j], ep, g, wake_speed
                )
            vel = vel + _vel_semiinfinite(
                TE_point_1[j], wake_unit, ep, g, wake_speed, 1.0
            )
            vel = vel + _vel_semiinfinite(
                TE_point_2[j], wake_unit, ep, g, wake_speed, -1.0
            )
        out[ip, 0] = vel[0]
        out[ip, 1] = vel[1]
        out[ip, 2] = vel[2]
    return out


def intersect_line_with_plane(x_cp, F_unit, plane_point, plane_normal):
    numerator = np.dot(plane_normal, (plane_point - x_cp))
    denominator = np.dot(plane_normal, F_unit)
    if np.abs(denominator) < 1e-6:
        return None  # No intersection
    lam = numerator / denominator
    intersection = x_cp + lam * F_unit
    return intersection


def point_in_triangle(pt, v0, v1, v2, tol=1e-8):
    # Barycentric coordinate check with tolerance
    u = v1 - v0
    v = v2 - v0
    w = pt - v0

    uu = np.dot(u, u)
    uv = np.dot(u, v)
    vv = np.dot(v, v)
    wu = np.dot(w, u)
    wv = np.dot(w, v)

    denom = uv * uv - uu * vv
    if np.abs(denom) < 1e-12:
        return False  # Degenerate triangle

    s = (uv * wv - vv * wu) / denom
    t = (uv * wu - uu * wv) / denom

    return (s >= -tol) and (t >= -tol) and (s + t <= 1 + tol)


def point_in_quad(pt, corners):
    """
    Checks if a point is inside a quadrilateral.
    corners: array of shape (4, 3)
    """
    return point_in_triangle(
        pt, corners[0], corners[1], corners[2]
    ) or point_in_triangle(pt, corners[0], corners[2], corners[3])


def perpendicular_leg_end_points(
    bound_nodes, te_nodes, chord_dir, chords, ac_fraction, direction=None, normals=None
):
    """End points of attached-trailed legs perpendicular to the quarter-chord line.

    Gaunaa, Li & Pirrung (TORQUE 2026, Sec. 3) run the attached part of the
    trailed vortex from the bound vortex "along the local chordwise direction
    up to the trailing edge", the chordwise direction being that of the
    Crossflow Principle (CP1): perpendicular to the local quarter-chord line.
    The legs then depend only on the quarter-chord line and the trailing-edge
    curve, not on where the input ribs were drawn.

    Where the line turns by more than about one radian within one leg length
    (a planform kink, a tip whose curvature radius is below a chord) plain
    perpendiculars cross. The line tangent is therefore averaged over plus and
    minus one leg length, ``(1 - ac) c``, of arc length around each node before
    the perpendicular is taken. On a straight or uniformly swept line this is
    the plain perpendicular. At a kink it turns the leg at the kink into the
    bisector (streamwise at the root of a symmetric swept wing) and blends to
    the perpendicular within one leg length. The averaging length is the leg's
    own scale, not a free parameter. Legs that still cross raise an error.

    Args:
        bound_nodes (np.ndarray): (n+1, 3) bound-vortex nodes in panel order
            (node ``k`` is ``bound_point_1`` of panel ``k``; node ``n`` is
            ``bound_point_2`` of the last panel).
        te_nodes (np.ndarray): (n+1, 3) section trailing-edge points in the
            same order; their polyline is the trailing-edge curve.
        chord_dir (np.ndarray): (n, 3) span-perpendicular chord directions of
            the panels, leading edge to trailing edge (``Panel.y_airf``).
        chords (np.ndarray): (n,) perpendicular chords of the panels.
        ac_fraction (float): chord fraction of the bound vortex.
        direction (np.ndarray, optional): a fixed leg direction (3,), e.g. the
            apparent-velocity direction for streamwise legs. It is projected
            onto the local surface (perpendicular to ``normals``) at each
            node and used instead of the perpendicular to the quarter-chord
            line; the tangent averaging is then not needed.
        normals (np.ndarray, optional): (n, 3) panel normals (``Panel.x_airf``),
            required with ``direction``.

    Returns:
        np.ndarray: (n+1, 3) leg end points, on the trailing-edge polyline
        (or on the straight extension of its first or last segment, for a
        leg that leaves a skewed tip through the tip edge).

    Raises:
        ValueError: if a leg does not reach the trailing-edge curve or two
            neighbouring legs cross (end points not ordered along the edge).
    """
    B = np.asarray(bound_nodes, dtype=float)
    T = np.asarray(te_nodes, dtype=float)
    Y = np.asarray(chord_dir, dtype=float)
    c = np.asarray(chords, dtype=float).ravel()
    n = len(c)
    if B.shape != (n + 1, 3) or T.shape != (n + 1, 3) or Y.shape != (n, 3):
        raise ValueError("perpendicular_leg_end_points: inconsistent array shapes")

    seg = B[1:] - B[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    if np.any(seg_len < 1e-12):
        raise ValueError("repeated bound-vortex nodes")
    tangent = seg / seg_len[:, None]
    s_node = np.concatenate(([0.0], np.cumsum(seg_len)))

    te_seg = T[1:] - T[:-1]
    te_len = np.linalg.norm(te_seg, axis=1)
    if np.any(te_len < 1e-12):
        raise ValueError("repeated trailing-edge points")
    s_te = np.concatenate(([0.0], np.cumsum(te_len)))

    ends = np.zeros_like(B)
    s_end = np.zeros(n + 1)
    for k in range(n + 1):
        adjacent = [j for j in (k - 1, k) if 0 <= j < n]
        leg_length = (1.0 - ac_fraction) * float(np.mean(c[adjacent]))

        # tangent of the quarter-chord line averaged over +- one leg length
        lo, hi = s_node[k] - leg_length, s_node[k] + leg_length
        overlap = np.minimum(s_node[1:], hi) - np.maximum(s_node[:-1], lo)
        overlap = np.clip(overlap, 0.0, None)
        t_avg = overlap @ tangent
        if np.linalg.norm(t_avg) < 1e-12:
            t_avg = tangent[adjacent].sum(axis=0)
        t_avg = t_avg / np.linalg.norm(t_avg)

        if direction is None:
            # leg direction: the chord direction with its component along the
            # averaged tangent removed (points from the bound vortex aft)
            y = Y[adjacent].sum(axis=0)
            d = y - np.dot(y, t_avg) * t_avg
        else:
            # fixed direction projected onto the local surface
            nrm = np.asarray(normals, dtype=float)[adjacent].sum(axis=0)
            nrm = nrm / np.linalg.norm(nrm)
            d = np.asarray(direction, dtype=float)
            d = d - np.dot(d, nrm) * nrm
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-9:
            raise ValueError(f"leg direction at node {k} is degenerate (along the span or the normal)")
        d = d / d_norm

        # The leg ends at the point of the trailing-edge curve closest to the
        # ray B_k + t d, t > 0. A ray in 3D does not meet a polyline on a
        # curved surface exactly, so the closest point is used rather than an
        # intersection; it lies on a segment, or on the straight extension of
        # the first or last segment, so that a leg leaving a skewed tip
        # through the tip edge still ends on the trailing-edge line.
        w0 = B[k][None, :] - T[:-1]
        b = te_seg @ d
        cc = te_len**2
        dd = w0 @ d
        e0 = np.einsum("ij,ij->i", te_seg, w0)
        denom = cc - b * b
        regular = denom > 1e-10 * cc
        u = np.where(regular, (e0 - b * dd) / np.where(regular, denom, 1.0), 0.0)
        lower = np.zeros(len(te_len))
        upper = np.ones(len(te_len))
        lower[0], upper[-1] = -np.inf, np.inf
        u = np.clip(u, lower, upper)
        P = T[:-1] + u[:, None] * te_seg
        t = (P - B[k][None, :]) @ d
        ahead = t > 0.0
        if not np.any(ahead):
            raise ValueError(
                f"the perpendicular attached-trailed leg of node {k} does not reach "
                "the trailing-edge curve; use leg_direction='rib'"
            )
        dist = np.linalg.norm(B[k][None, :] + t[:, None] * d[None, :] - P, axis=1)
        dist[~ahead] = np.inf
        m = int(np.argmin(dist))
        u_clip = u
        ends[k] = P[m]
        s_end[k] = s_te[m] + u_clip[m] * te_len[m]

    crossing = np.where(np.diff(s_end) <= 0.0)[0]
    if crossing.size:
        k = int(crossing[0])
        raise ValueError(
            f"attached-trailed legs cross between nodes {k} and {k + 1}: the "
            "quarter-chord line turns by more than one leg length there; use "
            "leg_direction='rib'"
        )
    return ends
