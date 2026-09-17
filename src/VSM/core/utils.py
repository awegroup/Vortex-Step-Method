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
def assemble_bound_vortex_AIC(eval_points, bound_point_1, bound_point_2, core_radius_fraction):
    """Unit-circulation induced velocity of every panel's BOUND vortex alone at
    every evaluation point, (3, N, N), with the panel's own bound vortex left
    out (a straight segment induces nothing on its own line, and the quarter
    chord point sits on it). Used to add the curved/swept bound-vortex
    influence to a trailed-vortex-only AIC evaluated on the quarter chord.
    """
    n = eval_points.shape[0]
    AIC = np.zeros((3, n, n))
    for icp in range(n):
        ep = eval_points[icp]
        for jring in range(n):
            if icp == jring:
                continue
            vel = _vel_bound_vortex(
                bound_point_2[jring], bound_point_1[jring], ep, 1.0, core_radius_fraction
            )
            AIC[0, icp, jring] = vel[0]
            AIC[1, icp, jring] = vel[1]
            AIC[2, icp, jring] = vel[2]
    return AIC


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
