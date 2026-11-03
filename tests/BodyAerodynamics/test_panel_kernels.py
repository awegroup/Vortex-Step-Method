"""The section force law as ``xp`` kernels (``VSM.core.panel_kernels``).

Contract:

* moving the force law into the kernels changed no output: every numeric
  result of ``Solver.solve`` equals the one of VSM ``1201f6f`` (the commit
  before the move) to round-off, for ``is_aoa_corrected`` True and False, on
  a straight, a swept, a curved (arc) wing and the LEI V3 kite;
* the public quarter-chord induction matrices are the ones the force
  directions are computed from;
* every kernel gives the same numbers with ``xp=casadi`` as with
  ``xp=numpy``;
* a CasADi graph with the induction matrices as parameters reproduces the
  per-panel forces of ``compute_results``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from panel_kernel_cases import CASES, REFERENCE_FILE, run_case, v3_body, arc_body  # noqa: E402
from VSM.core import panel_kernels  # noqa: E402
from VSM.core.Solver import Solver  # noqa: E402
from VSM.core.utils import thin_airfoil_strips, assemble_curved_bound_correction_AIC  # noqa: E402

RTOL = 1e-12

_FORCE_KEYS = ("Fx", "Fy", "Fz")
_MOMENT_KEYS = ("Mx", "My", "Mz")


# ----------------------------------------------------------- no-change regression
@pytest.fixture(scope="module")
def reference():
    return np.load(REFERENCE_FILE)


@pytest.mark.parametrize("is_aoa_corrected", [True, False])
@pytest.mark.parametrize("case", list(CASES))
def test_solve_outputs_match_reference_commit(reference, case, is_aoa_corrected):
    """Every numeric output equals VSM 1201f6f to round-off. Entries that
    vanish by symmetry (side force of a symmetric wing, ...) are compared
    against the case's force or moment scale instead of their own."""
    outputs = run_case(case, is_aoa_corrected)
    prefix = f"{case}|{is_aoa_corrected}|"
    expected_keys = {k[len(prefix):] for k in reference.files if k.startswith(prefix)}
    assert set(outputs) == expected_keys
    family_scale = {
        "force": max(abs(float(reference[prefix + k])) for k in _FORCE_KEYS),
        "moment": max(abs(float(reference[prefix + k])) for k in _MOMENT_KEYS),
        "other": max(abs(float(reference[prefix + k])) for k in ("cl", "cd")),
    }
    for key, value in outputs.items():
        ref = reference[prefix + key]
        assert value.shape == ref.shape, key
        if not ref.size:
            continue
        if key[0] == "F" or key in ("lift", "drag", "side", "drag_induced_trefftz", "bridle_line_forces"):
            family = "force"
        elif key[0] == "M":
            family = "moment"
        else:
            family = "other"
        scale = max(np.max(np.abs(ref)), 1e-3 * family_scale[family])
        np.testing.assert_allclose(value, ref, rtol=0.0, atol=RTOL * scale, err_msg=key)


# ------------------------------------------------------ quarter-chord induction
@pytest.mark.parametrize("factory", [arc_body, v3_body])
def test_quarter_chord_matrices_are_llt_plus_curved_correction(factory):
    """The public matrices equal the trailed system on the quarter chord plus
    the curved bound-vortex correction, assembled independently here."""
    body = factory()
    va = np.array([p.va for p in body.panels])
    va_norm = np.linalg.norm(va, axis=1)
    va_unit = va / va_norm[:, None]
    core = 0.05
    public = body.compute_quarter_chord_AIC_matrices(core, va_norm, va_unit)

    llt = body.compute_AIC_matrices("LLT", core, va_norm, va_unit)
    rows = lambda name: np.ascontiguousarray([getattr(p, name) for p in body.panels], dtype=float)  # noqa: E731
    strip_x, strip_w = thin_airfoil_strips()
    correction = assemble_curved_bound_correction_AIC(
        rows("aerodynamic_center"), rows("bound_point_1"), rows("bound_point_2"),
        rows("TE_point_1"), rows("TE_point_2"), rows("y_airf"),
        np.array([p.chord for p in body.panels]), va_norm,
        float(body._aerodynamic_center_location), core, strip_x, strip_w,
    )
    for k in range(3):
        assert public[k].shape == (body.n_panels, body.n_panels)
        np.testing.assert_allclose(public[k], llt[k] + correction[k], rtol=1e-13, atol=1e-15)
    if factory is arc_body:  # curved: the correction is not negligible
        assert np.max(np.abs(correction)) > 1e-3 * np.max(np.abs(llt))


def test_quarter_chord_direction_angle_uses_the_public_matrices():
    body = v3_body()
    solver = Solver(allowed_error=1e-10)
    results = solver.solve(body)
    va_norm = np.linalg.norm(solver.va_array, axis=1)
    AIC_qc = body.compute_quarter_chord_AIC_matrices(
        solver.core_radius_fraction, va_norm, solver.va_array / va_norm[:, None]
    )
    gamma = results["gamma_distribution"]
    v_qc = solver.va_array + np.column_stack([A @ gamma for A in AIC_qc])
    alpha_qc = np.arctan2(
        np.sum(solver.x_airf_array * v_qc, axis=1), np.sum(solver.y_airf_array * v_qc, axis=1)
    )
    np.testing.assert_allclose(np.ravel(results["alpha_at_ac"]), alpha_qc, rtol=0, atol=1e-14)


# --------------------------------------------------------- casadi == numpy
@pytest.fixture(scope="module")
def solved_v3():
    """A converged V3 solve and the arrays a force-law caller needs."""
    body = v3_body()
    solver = Solver(
        allowed_error=1e-10,
        is_aoa_corrected=True,
        is_with_attached_trailed_vortex_force=True,
    )
    results = solver.solve(body)
    va_norm = np.linalg.norm(solver.va_array, axis=1)
    AIC_qc = body.compute_quarter_chord_AIC_matrices(
        solver.core_radius_fraction, va_norm, solver.va_array / va_norm[:, None]
    )
    return body, solver, results, AIC_qc


def _numpy_inputs(solver, results, AIC_qc):
    gamma = np.asarray(results["gamma_distribution"], dtype=float)
    aic = (solver.AIC_x, solver.AIC_y, solver.AIC_z)
    v_rel = panel_kernels.quarter_chord_velocity(solver.va_array, gamma, *aic)
    alpha, umag = panel_kernels.relative_flow(
        v_rel, solver.x_airf_array, solver.y_airf_array, solver.z_airf_array
    )
    panels = solver.panels
    return {
        "gamma": gamma,
        "v_rel": v_rel,
        "alpha": alpha,
        "umag": umag,
        "cl": np.array([p.compute_cl(a) for p, a in zip(panels, alpha)]),
        "cd": np.array([p.compute_cd_cm(a)[0] for p, a in zip(panels, alpha)]),
        "cm": np.array([p.compute_cd_cm(a)[1] for p, a in zip(panels, alpha)]),
        "AIC_qc": AIC_qc,
    }


def test_kernels_casadi_equals_numpy(solved_v3):
    ca = pytest.importorskip("casadi")
    body, solver, results, AIC_qc = solved_v3
    d = _numpy_inputs(solver, results, AIC_qc)
    x, y, z = solver.x_airf_array, solver.y_airf_array, solver.z_airf_array
    chord, width, rho = solver.chord_array, solver.width_array, solver.rho

    col = lambda v: ca.DM(np.asarray(v, dtype=float).reshape(-1, 1))  # noqa: E731
    mat = lambda a: ca.DM(np.asarray(a, dtype=float))  # noqa: E731

    def same(ca_value, np_value):
        ca_array = np.array(ca_value, dtype=float).reshape(np.shape(np_value))
        scale = max(np.max(np.abs(np_value)), 1e-300)
        np.testing.assert_allclose(ca_array, np_value, rtol=0, atol=RTOL * scale)

    # flow
    same(panel_kernels.relative_flow(mat(d["v_rel"]), mat(x), mat(y), mat(z), xp=ca)[0], d["alpha"])
    same(panel_kernels.relative_flow(mat(d["v_rel"]), mat(x), mat(y), mat(z), xp=ca)[1], d["umag"])
    v_qc_np = panel_kernels.quarter_chord_velocity(solver.va_array, d["gamma"], *AIC_qc)
    v_qc_ca = panel_kernels.quarter_chord_velocity(
        mat(solver.va_array), col(d["gamma"]), *[mat(A) for A in AIC_qc], xp=ca
    )
    same(v_qc_ca, v_qc_np)
    alpha_dir_np = panel_kernels.direction_angle(v_qc_np, x, y)
    same(panel_kernels.direction_angle(v_qc_ca, mat(x), mat(y), xp=ca), alpha_dir_np)

    # force law, both direction laws
    for alpha_dir in (None, alpha_dir_np):
        np_out = panel_kernels.panel_forces(
            d["alpha"], d["umag"], d["cl"], d["cd"], d["cm"], chord, width, x, y, z, rho,
            alpha_dir=alpha_dir,
        )
        ca_out = panel_kernels.panel_forces(
            col(d["alpha"]), col(d["umag"]), col(d["cl"]), col(d["cd"]), col(d["cm"]),
            col(chord), col(width), mat(x), mat(y), mat(z), rho,
            alpha_dir=None if alpha_dir is None else col(alpha_dir), xp=ca,
        )
        for ca_value, np_value in zip(ca_out, np_out):
            same(ca_value, np_value)
        np_vec = panel_kernels.section_force_vectors(
            d["alpha"], d["umag"], d["cl"], d["cd"], d["cm"], chord, x, y, z, rho,
            alpha_dir=alpha_dir,
        )
        ca_vec = panel_kernels.section_force_vectors(
            col(d["alpha"]), col(d["umag"]), col(d["cl"]), col(d["cd"]), col(d["cm"]),
            col(chord), mat(x), mat(y), mat(z), rho,
            alpha_dir=None if alpha_dir is None else col(alpha_dir), xp=ca,
        )
        for ca_value, np_value in zip(ca_vec, np_vec):
            same(ca_value, np_value)

    # attached trailed vortex force
    legs = np.random.default_rng(0).normal(size=(body.n_panels + 1, 3))
    v_ind = np.random.default_rng(1).normal(size=(body.n_panels + 1, 3))
    same(
        panel_kernels.attached_trailed_forces(
            col(d["gamma"]), mat(solver.va_array), mat(v_ind), legs, rho, xp=ca
        ),
        panel_kernels.attached_trailed_forces(d["gamma"], solver.va_array, v_ind, legs, rho),
    )


# ------------------------------------------- casadi graph == compute_results
def _boundary_shares(n):
    """(n, n+1) split of boundary loads onto panels, compute_results' rule."""
    shares = np.zeros((n, n + 1))
    shares[0, 0] = 1.0
    shares[n - 1, n] = 1.0
    for j in range(1, n):
        shares[j - 1, j] = shares[j, j] = 0.5
    return shares


def _compute_results_at(body, solver, gamma):
    """compute_results for this ``gamma`` with ``alpha``/``umag`` evaluated at
    the same ``gamma``. (``solve`` passes the loop's last alpha, which lags the
    returned gamma by one iteration, i.e. agrees with it only to the
    convergence tolerance.)"""
    alpha, umag, _ = solver.compute_aerodynamic_quantities(gamma)
    va_norm = np.linalg.norm(solver.va_array, axis=1)
    return body.compute_results(
        gamma, solver.rho, solver.aerodynamic_model_type, solver.core_radius_fraction,
        solver.mu, alpha, umag, solver.chord_array, solver.x_airf_array,
        solver.y_airf_array, solver.z_airf_array, solver.va_array, va_norm,
        solver.va_array / va_norm[:, None], solver.panels, False,
        solver.is_with_viscous_drag_correction, solver.reference_point,
        solver.is_aoa_corrected,
        relative_velocity_array=solver.compute_relative_velocity(gamma),
        is_with_attached_trailed_vortex_force=solver.is_with_attached_trailed_vortex_force,
    )


@pytest.mark.parametrize("is_aoa_corrected", [True, False])
def test_casadi_graph_reproduces_compute_results_forces(is_aoa_corrected):
    """Graph: gamma, va and the 3/4-chord and quarter-chord matrices are MX
    parameters; ``v_rel`` and ``v_qc`` come from ``quarter_chord_velocity``,
    the section forces from ``Solver.casadi_panel_force_function``. Evaluated
    at a converged gamma it reproduces compute_results' panel forces and
    moments (the attached-trailed part removed, as it is not in the
    function)."""
    ca = pytest.importorskip("casadi")
    body = v3_body()
    solver = Solver(allowed_error=1e-10, is_aoa_corrected=is_aoa_corrected)
    gamma_converged = solver.solve(body)["gamma_distribution"]
    results = _compute_results_at(body, solver, gamma_converged)
    n = solver.n_panels
    va_norm = np.linalg.norm(solver.va_array, axis=1)
    AIC_qc = body.compute_quarter_chord_AIC_matrices(
        solver.core_radius_fraction, va_norm, solver.va_array / va_norm[:, None]
    )

    gamma = ca.MX.sym("gamma", n)
    va = ca.MX.sym("va", n, 3)
    A_cp = [ca.MX.sym(f"A_cp_{k}", n, n) for k in range(3)]
    A_qc = [ca.MX.sym(f"A_qc_{k}", n, n) for k in range(3)]
    v_rel = panel_kernels.quarter_chord_velocity(va, gamma, *A_cp, xp=ca)
    v_qc = panel_kernels.quarter_chord_velocity(va, gamma, *A_qc, xp=ca) if is_aoa_corrected else v_rel
    forces_fn = solver.casadi_panel_force_function()
    force, moment_local = forces_fn(
        v_rel, v_qc, solver.x_airf_array, solver.y_airf_array, solver.z_airf_array,
        solver.chord_array, solver.width_array, solver.rho,
    )[:2]
    graph = ca.Function("forces", [gamma, va, *A_cp, *A_qc], [force, moment_local])

    force_num, moment_num = graph(
        gamma_converged, solver.va_array,
        solver.AIC_x, solver.AIC_y, solver.AIC_z, *AIC_qc,
    )
    force_num = np.array(force_num)
    moment_num = np.array(moment_num)

    shares = _boundary_shares(n)
    f_at = np.array(results["F_attached_trailed_distribution"])
    assert f_at.shape == (n + 1, 3)  # AT force on (the default)
    expected_force = np.array(results["F_distribution"]) - shares @ f_at
    reference_point = np.zeros(3)
    ac = np.array([p.aerodynamic_center for p in solver.panels])
    at_points = body.compute_attached_trailed_vortex_forces(
        gamma_converged, solver.rho, solver.core_radius_fraction, solver.va_array
    )[1]
    expected_moment = (
        np.array(results["M_distribution"])
        - shares @ np.cross(at_points - reference_point, f_at)
        - np.cross(ac - reference_point, expected_force)
    )
    f_scale = np.max(np.abs(expected_force))
    np.testing.assert_allclose(force_num, expected_force, rtol=0, atol=RTOL * f_scale)
    np.testing.assert_allclose(
        moment_num, expected_moment, rtol=0, atol=RTOL * np.max(np.abs(expected_moment))
    )
