"""Section force law of the VSM as ``xp`` kernels.

Pure functions over a math namespace ``xp``, either ``numpy`` (the default)
or ``casadi``, so the same formula serves ``BodyAerodynamics.compute_results``
and a CasADi graph (``Solver.casadi_panel_force_function``, or a caller's own
graph such as AWETrim's CasADi trim). The formula lives here only; callers do
not restate it.

Conventions:

* Per-panel vectors are rows of an ``(n, 3)`` array; per-panel scalars are
  ``(n,)`` arrays in NumPy and ``(n, 1)`` columns in CasADi.
* ``x_airf`` is the chord normal, ``y_airf`` the chord (leading to trailing
  edge), ``z_airf`` the span direction of each panel (``Panel`` axes).
* Angles are in radians. ``alpha = atan2(v . x_airf, v . y_airf)``.
* ``umag = |v_rel x z_airf|`` is the speed the 2-D section sees; the dynamic
  pressure of the section force is ``0.5 rho umag^2``, not the freestream one.

The force law (Gaunaa, Li & Pirrung, TORQUE 2026): the MAGNITUDES of lift,
drag and moment come from the polar at the 3/4-chord control-point angle
``alpha``; the DIRECTIONS come from the direction angle ``alpha_dir``. With
``is_aoa_corrected=True`` (TAT3, VSM model) ``alpha_dir`` is the angle of the
flow at the quarter chord, ``va + AIC_qc gamma`` with the matrices of
``BodyAerodynamics.compute_quarter_chord_AIC_matrices``; with ``False`` it is
``alpha`` itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# --------------------------------------------------------------------- xp rows
def _is_casadi(xp: Any) -> bool:
    return hasattr(xp, "horzcat")


def hstack_columns(xp: Any, *cols):
    """``(n, k)`` from ``k`` per-row columns."""
    if _is_casadi(xp):
        return xp.horzcat(*cols)
    return np.column_stack(cols)


def cross_rows(a, b, xp=np):
    """Row-wise cross product of two ``(n, 3)`` arrays."""
    return hstack_columns(
        xp,
        a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
        a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
        a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
    )


def dot_rows(a, b, xp=np):
    """Row-wise dot product of two ``(n, 3)`` arrays -> per-row scalar."""
    if _is_casadi(xp):
        return xp.sum2(a * b)
    return np.sum(a * b, axis=1)


def norm_rows(a, xp=np):
    return xp.sqrt(dot_rows(a, a, xp))


def scale_rows(s, a, xp=np):
    """Per-row scalar ``s`` times the ``(n, 3)`` rows of ``a``."""
    if _is_casadi(xp):
        return xp.repmat(s, 1, a.shape[1]) * a
    return np.asarray(s)[:, None] * a


def unit_rows(a, xp=np):
    """Rows of ``a`` divided by their norms."""
    norms = norm_rows(a, xp)
    if _is_casadi(xp):
        return a / xp.repmat(norms, 1, a.shape[1])
    return a / norms[:, None]


def _atan2(y, x, xp):
    return xp.atan2(y, x) if _is_casadi(xp) else np.arctan2(y, x)


def _matvec(matrix, vector, xp):
    if _is_casadi(xp):
        return xp.mtimes(matrix, vector)
    return matrix @ vector


# ------------------------------------------------------------------ section flow
def relative_flow(v_rel, x_airf, y_airf, z_airf, xp=np):
    """Angle of attack and section speed of the relative velocity ``v_rel``
    (``Solver.compute_aerodynamic_quantities``):
    ``alpha = atan2(v_rel . x_airf, v_rel . y_airf)``, ``umag = |v_rel x z_airf|``."""
    alpha = direction_angle(v_rel, x_airf, y_airf, xp)
    umag = norm_rows(cross_rows(v_rel, z_airf, xp), xp)
    return alpha, umag


def quarter_chord_velocity(va, gamma, AIC_x, AIC_y, AIC_z, xp=np):
    """Relative flow at the panel quarter chords, ``va + AIC_qc gamma``.

    ``AIC_*`` are the ``(n, n)`` quarter-chord induction matrices of
    ``BodyAerodynamics.compute_quarter_chord_AIC_matrices`` (trailed system on
    the quarter chord plus the curved bound-vortex correction), ``va`` the
    ``(n, 3)`` panel inflow and ``gamma`` the ``(n,)`` circulation.
    """
    induced = hstack_columns(
        xp,
        _matvec(AIC_x, gamma, xp),
        _matvec(AIC_y, gamma, xp),
        _matvec(AIC_z, gamma, xp),
    )
    return va + induced


def direction_angle(v, x_airf, y_airf, xp=np):
    """Angle of the flow ``v`` (``(n, 3)``) in each panel's section plane,
    ``atan2(v . x_airf, v . y_airf)``. Applied to the quarter-chord flow it is
    the TAT3 direction angle of the force law."""
    return _atan2(dot_rows(x_airf, v, xp), dot_rows(y_airf, v, xp), xp)


# ------------------------------------------------------------------- force law
def section_force_directions(alpha_dir, x_airf, y_airf, z_airf, xp=np):
    """Unit flow and lift directions of each section at direction angle
    ``alpha_dir``: ``dir_flow = unit(cos(a) y_airf + sin(a) x_airf)`` (in the
    plane normal to ``z_airf``; drag acts along it, TAT3's local inner flow,
    not the global span axis) and ``dir_lift = unit(dir_flow x z_airf)``."""
    dir_flow = scale_rows(xp.cos(alpha_dir), y_airf, xp) + scale_rows(
        xp.sin(alpha_dir), x_airf, xp
    )
    dir_flow = unit_rows(dir_flow, xp)
    dir_lift = unit_rows(cross_rows(dir_flow, z_airf, xp), xp)
    return dir_flow, dir_lift


def section_force_magnitudes(umag, cl, cd, cm, chord, rho):
    """2-D lift, drag [N/m] and pitching moment [Nm/m] about the quarter
    chord, ``c_x 0.5 rho umag^2 chord`` (``chord^2`` for the moment). The
    coefficients are the polar values at the 3/4-chord angle."""
    lift = cl * 0.5 * rho * umag**2 * chord
    drag = cd * 0.5 * rho * umag**2 * chord
    moment = cm * 0.5 * rho * umag**2 * chord**2
    return lift, drag, moment


def section_force_vectors(
    alpha, umag, cl, cd, cm, chord, x_airf, y_airf, z_airf, rho, alpha_dir=None, xp=np
):
    """2-D (per unit span) lift and drag vectors and the flow direction.

    ``alpha_dir`` None uses ``alpha`` (the ``is_aoa_corrected=False`` law).
    Returns ``(lift_2d (n, 3), drag_2d (n, 3), moment (n,), dir_flow (n, 3))``.
    """
    if alpha_dir is None:
        alpha_dir = alpha
    lift, drag, moment = section_force_magnitudes(umag, cl, cd, cm, chord, rho)
    dir_flow, dir_lift = section_force_directions(alpha_dir, x_airf, y_airf, z_airf, xp)
    return scale_rows(lift, dir_lift, xp), scale_rows(drag, dir_flow, xp), moment, dir_flow


def panel_forces(
    alpha,
    umag,
    cl,
    cd,
    cm,
    chord,
    width,
    x_airf,
    y_airf,
    z_airf,
    rho,
    alpha_dir=None,
    xp=np,
):
    """Per-panel 3-D force [N] and local pitching moment [Nm].

    ``force = width (lift_2d + drag_2d)`` (see :func:`section_force_vectors`);
    ``moment_local = width (cm q chord^2) z_airf``, the quarter-chord moment as
    a vector along the span axis. ``alpha_dir`` None gives the 3/4-chord
    directions (``is_aoa_corrected=False``); pass :func:`direction_angle` of
    :func:`quarter_chord_velocity` for the quarter-chord law (``True``).

    Not included: the attached-trailed vortex force
    (:func:`attached_trailed_forces`) and the viscous drag correction
    (``BodyAerodynamics.viscous_drag_correction``, NumPy only).

    Returns ``(force (n, 3), moment_local (n, 3))``.
    """
    lift_2d, drag_2d, moment, _ = section_force_vectors(
        alpha, umag, cl, cd, cm, chord, x_airf, y_airf, z_airf, rho, alpha_dir, xp
    )
    force = scale_rows(width, lift_2d + drag_2d, xp)
    return force, panel_moment_local(moment, width, z_airf, xp)


def panel_moment_local(moment, width, z_airf, xp=np):
    """3-D local pitching moment [Nm] of each panel: the 2-D quarter-chord
    moment [Nm/m] (:func:`section_force_magnitudes`) as a vector along the
    span axis ``z_airf``, times the panel width. ``(n, 3)``."""
    return scale_rows(width, scale_rows(moment, z_airf, xp), xp)


# ---------------------------------------------------- attached trailed vortices
def attached_trailed_forces(gamma, va_panels, v_induced, legs, rho, xp=np):
    """Kutta-Joukowski force on the attached trailed (AT) legs (Gaunaa, Li &
    Pirrung 2026, Sec. 3), ``rho (gamma_j - gamma_{j-1}) (v_rel_j x l_j)`` on
    boundary ``j = 0..n`` with ``gamma_{-1} = gamma_n = 0``.

    ``v_rel_j`` is the mean of the neighbouring panels' inflow ``va_panels``
    (the tip boundaries take their one panel's) plus ``v_induced``, the full
    vortex system's induction at the AT force points, ``(n+1, 3)``. ``legs``
    are the ``(n+1, 3)`` leg vectors, bound vortex to trailing edge. Returns
    the ``(n+1, 3)`` boundary forces.
    """
    n = va_panels.shape[0]
    to_boundary = np.zeros((n + 1, n))  # gamma_net = to_boundary @ gamma
    to_boundary[np.arange(n), np.arange(n)] += 1.0
    to_boundary[np.arange(1, n + 1), np.arange(n)] -= 1.0
    # boundary inflow = mean of the neighbouring panels (halving is exact)
    mean = np.abs(to_boundary)
    mean[1:n] *= 0.5
    if _is_casadi(xp):
        gamma_net = xp.mtimes(xp.DM(to_boundary), gamma)
        va_boundary = xp.mtimes(xp.DM(mean), va_panels)
        if isinstance(legs, np.ndarray):
            legs = xp.DM(legs)
    else:
        gamma_net = to_boundary @ np.asarray(gamma, dtype=float).ravel()
        va_boundary = mean @ va_panels
    return scale_rows(rho * gamma_net, cross_rows(va_boundary + v_induced, legs, xp), xp)
