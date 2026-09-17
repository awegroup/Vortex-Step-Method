"""Tests for ``gamma_loop_type="casadi_newton"``: the exact-Jacobian Newton /
pseudo-transient circulation solve.

Contract: same fixed point as ``base`` (to base's own converged residual),
far fewer iterations, an assembled Jacobian that matches finite differences
of the numpy residual, and convergence where the plain loops fail.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

casadi = pytest.importorskip("casadi")

from VSM.core.Solver import Solver  # noqa: E402
from tests.Solver.test_solver import (  # noqa: E402,F401
    _rectangular_body,
    _stalled_polar_data,
    body_aero,  # fixture (and its dependencies), re-exported for this module
    inviscid_polar_data,
    simple_wing,
)


def _numpy_residual(
    solver: Solver, gamma: np.ndarray, averaged_cl: bool = False
) -> np.ndarray:
    """R(gamma) = (I - diag(mu) L) gamma - G_raw(gamma), from the numpy loop's
    own building blocks (compute_aerodynamic_quantities + the viscosity
    context), independent of the CasADi graph. ``averaged_cl`` swaps the
    polar for the two-point average 0.5 (Cl(a + d) + Cl(a - d)), d = 0.5 deg,
    which is what the loop's JACOBIAN surrogate differentiates (its slope is
    the mean of the one-sided slopes at a table corner)."""
    ctx = solver._build_viscosity_ctx()
    alpha, umag, cl = solver.compute_aerodynamic_quantities(gamma)
    if averaged_cl:
        d = np.deg2rad(0.5)
        cl = np.array(
            [
                0.5 * (panel.compute_cl(a + d) + panel.compute_cl(a - d))
                for panel, a in zip(solver.panels, alpha)
            ]
        )
    gamma_raw = 0.5 * umag * cl * solver.chord_array
    if ctx is None or not (
        np.any(alpha > ctx["stall_angles"])
        or np.any(alpha < ctx["stall_angles_neg"])
    ):
        return gamma - gamma_raw
    slope = solver._lift_slope_from_ctx(alpha, ctx)
    mu = np.maximum(
        0.0,
        -solver.artificial_viscosity_factor
        * ctx["planform_area"]
        * slope
        / solver.width_array**2,
    )
    laplacian = solver._build_spanwise_laplacian()
    return (np.eye(gamma.size) - mu[:, None] * laplacian) @ gamma - gamma_raw


def _assembled_residual_and_jacobian(solver: Solver, gamma: np.ndarray):
    """Replicates the loop's evaluate(): CasADi per-panel physics + numpy
    chain rule through v_rel = va + AIC gamma."""
    fn = solver._casadi_newton_function()
    n = solver.n_panels
    av = solver.is_with_artificial_viscosity
    laplacian = solver._build_spanwise_laplacian() if av else np.zeros((n, n))
    stall = solver._panel_stall_angles() if av else np.full(n, np.inf)
    stall_neg = solver._panel_negative_stall_angles() if av else np.full(n, -np.inf)
    aic = (solver.AIC_x, solver.AIC_y, solver.AIC_z)
    v_rel = solver.va_array + np.column_stack([a @ gamma for a in aic])
    h, dh, mu = fn(
        v_rel,
        solver.x_airf_array,
        solver.y_airf_array,
        solver.z_airf_array,
        solver.chord_array,
        solver.width_array,
        stall,
        stall_neg,
        laplacian @ gamma,
    )[:3]
    h = np.asarray(h, dtype=float).ravel()
    dh = np.asarray(dh, dtype=float)
    mu = np.asarray(mu, dtype=float).ravel()
    jac = np.eye(n) - mu[:, None] * laplacian
    for k in range(3):
        jac -= dh[:, k][:, None] * aic[k]
    return gamma - h, jac


@pytest.mark.parametrize("with_viscosity", [False, True])
def test_casadi_newton_matches_base_attached(body_aero, with_viscosity):
    """Attached flow: the Newton root is the base loop's fixed point, in a
    handful of iterations instead of ~1000. Agreement is bounded by BASE's
    converged residual: at allowed_error 1e-8 base stops at a relaxed error
    of 1e-8, i.e. a true residual of 1e-6 relative to peak circulation."""
    base = Solver(
        relaxation_factor=0.01,  # the ~1000-iteration reference of the docstring
        gamma_loop_type="base", allowed_error=1e-8, is_with_artificial_viscosity=with_viscosity
    )
    body_aero.va_initialize(Umag=10.0, angle_of_attack=5.0)
    res_base = base.solve(body_aero)

    newton = Solver(
        gamma_loop_type="casadi_newton",
        allowed_error=1e-8,
        is_with_artificial_viscosity=with_viscosity,
        newton_fallback_to_base=False,
    )
    body_aero.va_initialize(Umag=10.0, angle_of_attack=5.0)
    res_newton = newton.solve(body_aero)

    assert res_newton["gamma_converged"]
    assert not newton.last_fallback
    peak = np.max(np.abs(res_base["gamma_distribution"]))
    np.testing.assert_allclose(
        res_newton["gamma_distribution"],
        res_base["gamma_distribution"],
        atol=2e-6 * peak,
    )
    assert np.isclose(res_newton["cl"], res_base["cl"], atol=1e-5)
    assert newton.last_iterations <= 10
    assert newton.last_iterations * 20 < base.last_iterations


def test_casadi_newton_residual_is_tighter_than_base():
    """The stopping rule is the UN-relaxed residual, so at the same
    allowed_error the Newton solution satisfies the fixed point ~1e-2 x
    tighter than base's relaxed stopping rule."""
    body = _rectangular_body(16, _stalled_polar_data())
    tol = 1e-8
    out = {}
    for kind in ("base", "casadi_newton"):
        body.va_initialize(Umag=10.0, angle_of_attack=6.0)
        # explicit 0.01 so base's relaxed stopping rule is the 100x looser one
        # this test asserts against (the default is the adaptive factor)
        solver = Solver(gamma_loop_type=kind, allowed_error=tol, relaxation_factor=0.01)
        res = solver.solve(body)
        gamma = np.asarray(res["gamma_distribution"], dtype=float)
        residual = _numpy_residual(solver, gamma)
        out[kind] = np.max(np.abs(residual)) / np.max(np.abs(gamma))
    assert out["casadi_newton"] < tol
    assert out["casadi_newton"] < 1e-2 * out["base"]


@pytest.mark.parametrize(
    "with_viscosity, angle_of_attack", [(False, 5.0), (True, 5.0), (True, 18.0)]
)
def test_casadi_newton_residual_and_jacobian_match_numpy(with_viscosity, angle_of_attack):
    """The CasADi residual equals the numpy fixed-point residual to rounding,
    and the assembled Jacobian matches central differences of it -- with
    sideslip and a yaw rate so no term is trivially zero, and post-stall so the
    viscosity block (mu, its slope, the gate) is exercised."""
    body = _rectangular_body(20, _stalled_polar_data())
    body.va_initialize(
        Umag=10.0,
        angle_of_attack=angle_of_attack,
        side_slip=4.0,
        body_rates=np.array([0.0, 0.0, 0.2]),
    )
    solver = Solver(
        gamma_loop_type="casadi_newton",
        allowed_error=1e-10,
        is_with_artificial_viscosity=with_viscosity,
    )
    res = solver.solve(body)
    assert res["gamma_converged"]
    gamma = np.asarray(res["gamma_distribution"], dtype=float)
    if with_viscosity and angle_of_attack > 15.0:
        ctx = solver._build_viscosity_ctx()
        alpha, *_ = solver.compute_aerodynamic_quantities(gamma)
        assert np.any(alpha > ctx["stall_angles"])  # the gate really is open

    point = 0.9 * gamma + 0.01  # off the root, so the residual is not ~0
    residual, jac = _assembled_residual_and_jacobian(solver, point)
    np.testing.assert_allclose(residual, _numpy_residual(solver, point), atol=1e-12)

    # The Jacobian differentiates the corner-averaged polar (exact slope
    # wherever the +-0.5 deg window straddles no change of slope).
    eps = 1e-6
    jac_fd = np.zeros_like(jac)
    for j in range(point.size):
        e = np.zeros(point.size)
        e[j] = eps
        jac_fd[:, j] = (
            _numpy_residual(solver, point + e, averaged_cl=True)
            - _numpy_residual(solver, point - e, averaged_cl=True)
        ) / (2.0 * eps)
    np.testing.assert_allclose(jac, jac_fd, atol=1e-7, rtol=1e-7)


def test_casadi_newton_converges_post_stall_with_viscosity():
    """Deep post-stall with the Li/Gaunaa viscosity: converges to a genuine
    root (true residual below tolerance) and to a smooth distribution."""
    body = _rectangular_body(20, _stalled_polar_data())
    body.va_initialize(Umag=10.0, angle_of_attack=18.0)
    solver = Solver(
        gamma_loop_type="casadi_newton",
        allowed_error=1e-8,
        is_with_artificial_viscosity=True,
        newton_fallback_to_base=False,
    )
    res = solver.solve(body)
    assert res["gamma_converged"]
    gamma = np.asarray(res["gamma_distribution"], dtype=float)
    peak = np.max(np.abs(gamma))
    assert np.max(np.abs(_numpy_residual(solver, gamma))) / peak < 1e-8
    sawtooth = np.mean(np.abs(gamma[:-2] - 2 * gamma[1:-1] + gamma[2:])) / peak
    assert sawtooth < 0.05

    # Same fixed point as the base loop from the same (cold) seed.
    body.va_initialize(Umag=10.0, angle_of_attack=18.0)
    base = Solver(gamma_loop_type="base", allowed_error=1e-8, is_with_artificial_viscosity=True)
    res_base = base.solve(body)
    assert res_base["gamma_converged"]
    np.testing.assert_allclose(gamma, res_base["gamma_distribution"], atol=2e-6 * peak)


def test_casadi_newton_converges_where_relaxed_picard_cannot():
    """Without viscosity the post-stall fixed-point map is expansive: base and
    Anderson exhaust their budgets. The root still exists and Newton finds it.
    (10 panels: with the inner-velocity circulation map, Gamma = 0.5 |V_inner|
    c Cl, the 20-panel unregularized post-stall root is not reached by Newton
    either; the regularized loop converges there in a handful of steps.)"""
    body = _rectangular_body(10, _stalled_polar_data())
    body.va_initialize(Umag=10.0, angle_of_attack=18.0)
    # explicit 0.01: with the adaptive default the relaxed loop happens to reach
    # this post-stall root too, which is not the situation the test is about
    base = Solver(
        gamma_loop_type="base", allowed_error=1e-8, max_iterations=2000, relaxation_factor=0.01
    )
    assert not base.solve(body)["gamma_converged"]

    body.va_initialize(Umag=10.0, angle_of_attack=18.0)
    newton = Solver(
        gamma_loop_type="casadi_newton", allowed_error=1e-8, newton_fallback_to_base=False
    )
    res = newton.solve(body)
    assert res["gamma_converged"]
    gamma = np.asarray(res["gamma_distribution"], dtype=float)
    assert np.max(np.abs(_numpy_residual(newton, gamma))) / np.max(np.abs(gamma)) < 1e-8


def test_casadi_newton_seed_is_respected_and_zero_iterations_at_root(body_aero):
    """A converged circulation passed as the seed is recognised as the root
    without a single Newton step (the trim solvers warm-start this way)."""
    solver = Solver(gamma_loop_type="casadi_newton", allowed_error=1e-8)
    body_aero.va_initialize(Umag=10.0, angle_of_attack=5.0)
    res = solver.solve(body_aero)
    gamma = np.asarray(res["gamma_distribution"], dtype=float)
    body_aero.va_initialize(Umag=10.0, angle_of_attack=5.0)
    res2 = solver.solve(body_aero, gamma_distribution=gamma)
    assert res2["gamma_converged"]
    assert solver.last_iterations == 0
    assert solver.last_newton_evaluations == 1
    np.testing.assert_allclose(res2["gamma_distribution"], gamma, atol=1e-12)


def test_casadi_newton_function_is_cached_per_polar_set(body_aero):
    """The CasADi function depends only on the panel count and polar tables:
    a second solve on the same body reuses it; a body with different polars
    builds a new one; the panel geometry and inflow are numeric parameters."""
    solver = Solver(gamma_loop_type="casadi_newton", allowed_error=1e-8)
    body_aero.va_initialize(Umag=10.0, angle_of_attack=5.0)
    solver.solve(body_aero)
    assert len(solver._casadi_newton_cache) == 1
    fn = next(iter(solver._casadi_newton_cache.values()))

    body_aero.va_initialize(Umag=12.0, angle_of_attack=-2.0, side_slip=3.0)
    solver.solve(body_aero)
    assert len(solver._casadi_newton_cache) == 1
    assert next(iter(solver._casadi_newton_cache.values())) is fn

    other = _rectangular_body(4, _stalled_polar_data())
    other.va_initialize(Umag=10.0, angle_of_attack=5.0)
    solver.solve(other)
    assert len(solver._casadi_newton_cache) == 2


def test_casadi_newton_per_panel_polars_match_shared_grid_path():
    """Panels carrying different alpha grids take the per-panel interpolant
    branch; it must give the same root as the shared-grid 2-D interpolant."""
    from VSM.core.BodyAerodynamics import BodyAerodynamics
    from VSM.core.WingGeometry import Wing

    polar = _stalled_polar_data()
    # Same physics, resampled onto a different grid for the odd sections.
    alpha_fine = np.deg2rad(np.arange(-10.0, 41.0, 0.5))
    polar_fine = np.column_stack(
        [alpha_fine] + [np.interp(alpha_fine, polar[:, 0], polar[:, k]) for k in (1, 2, 3)]
    )
    n = 8
    ys = np.linspace(-4.0, 4.0, n + 1)

    def make(mixed):
        wing = Wing(n_panels=n, spanwise_panel_distribution="uniform")
        for i, y in enumerate(ys):
            table = polar_fine if (mixed and i % 2) else polar
            wing.add_section(np.array([0.0, y, 0.0]), np.array([1.0, y, 0.0]), table)
        return BodyAerodynamics([wing])

    out = []
    for mixed in (False, True):
        body = make(mixed)
        body.va_initialize(Umag=10.0, angle_of_attack=6.0)
        solver = Solver(gamma_loop_type="casadi_newton", allowed_error=1e-10)
        res = solver.solve(body)
        assert res["gamma_converged"]
        out.append(np.asarray(res["gamma_distribution"], dtype=float))
    # The resampled table is the same piecewise-linear function, so the
    # interpolated Cl agrees to rounding wherever the panel polars are
    # interpolated (VSM averages section polars onto panels).
    np.testing.assert_allclose(out[0], out[1], rtol=1e-6, atol=1e-9)


def test_casadi_panel_function_polar_outputs_match_tables():
    """The extra outputs (cl, cd, cm at alpha) are the panels' own np.interp
    lookups, and with bspline polars the averaged outputs equal the exact ones
    while linear polars average across the +-0.5 deg window."""
    body = _rectangular_body(6, _stalled_polar_data())
    body.va_initialize(Umag=10.0, angle_of_attack=7.0)
    for kind in ("linear", "bspline"):
        solver = Solver(gamma_loop_type="casadi_newton", polar_interpolation=kind)
        res = solver.solve(body)
        gamma = np.asarray(res["gamma_distribution"], dtype=float)
        fn = solver._casadi_newton_function()
        n = solver.n_panels
        aic = (solver.AIC_x, solver.AIC_y, solver.AIC_z)
        v_rel = solver.va_array + np.column_stack([a @ gamma for a in aic])
        out = fn(
            v_rel, solver.x_airf_array, solver.y_airf_array,
            solver.z_airf_array, solver.chord_array, solver.width_array,
            np.full(n, np.inf), np.full(n, -np.inf), np.zeros(n),
        )
        alpha = np.asarray(out[3]).ravel()
        cl, cd, cm = (np.asarray(o).ravel() for o in out[7:10])
        cl_avg, cd_avg, cm_avg = (np.asarray(o).ravel() for o in out[10:13])
        expected_cl = np.array([p.compute_cl(a) for p, a in zip(solver.panels, alpha)])
        expected_cd_cm = np.array([p.compute_cd_cm(a) for p, a in zip(solver.panels, alpha)])
        if kind == "linear":
            np.testing.assert_allclose(cl, expected_cl, atol=1e-12)
            np.testing.assert_allclose(cd, expected_cd_cm[:, 0], atol=1e-12)
            np.testing.assert_allclose(cm, expected_cd_cm[:, 1], atol=1e-12)
            d = np.deg2rad(0.5)
            np.testing.assert_allclose(
                cl_avg,
                [0.5 * (p.compute_cl(a + d) + p.compute_cl(a - d)) for p, a in zip(solver.panels, alpha)],
                atol=1e-12,
            )
        else:
            # Spline through the same nodes: close to, not equal to, the table.
            np.testing.assert_allclose(cl, expected_cl, atol=2e-2)
            np.testing.assert_allclose(cl_avg, cl, atol=1e-14)
            np.testing.assert_allclose(cd_avg, cd, atol=1e-14)
            np.testing.assert_allclose(cm_avg, cm, atol=1e-14)


def test_casadi_newton_rejects_non_increasing_polar_grid():
    body = _rectangular_body(4, _stalled_polar_data())
    body.va_initialize(Umag=10.0, angle_of_attack=5.0)
    solver = Solver(gamma_loop_type="casadi_newton")
    solver.solve(body)  # populates panels
    bad = [np.array([0.0, 0.1, 0.1, 0.2])] * 4
    with pytest.raises(ValueError, match="strictly increasing"):
        solver._build_casadi_newton_function(bad, [np.zeros(4)] * 4, [np.zeros(4)] * 4, [np.zeros(4)] * 4)


def test_casadi_newton_fallback_to_base_flag(monkeypatch, body_aero, caplog):
    """When the Newton attempt reports failure the caller runs the base loop
    (flag on, default) or returns the unconverged result (flag off), and the
    diagnostic ``last_fallback`` says which happened."""
    body_aero.va_initialize(Umag=10.0, angle_of_attack=5.0)
    solver = Solver(gamma_loop_type="casadi_newton", allowed_error=1e-8)

    def fake_newton(gamma_initial):
        n = gamma_initial.size
        return False, np.zeros(n), np.zeros(n), np.ones(n)

    monkeypatch.setattr(solver, "gamma_loop_casadi_newton", fake_newton)
    with caplog.at_level(logging.INFO):
        res = solver.solve(body_aero)
    assert res["gamma_converged"]
    assert solver.last_fallback

    body_aero.va_initialize(Umag=10.0, angle_of_attack=5.0)
    solver_nofb = Solver(
        gamma_loop_type="casadi_newton", allowed_error=1e-8, newton_fallback_to_base=False
    )
    monkeypatch.setattr(solver_nofb, "gamma_loop_casadi_newton", fake_newton)
    res = solver_nofb.solve(body_aero)
    assert not res["gamma_converged"]
    assert not solver_nofb.last_fallback


if __name__ == "__main__":
    pytest.main([__file__])
