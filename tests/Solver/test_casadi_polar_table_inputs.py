"""Polar tables as inputs of the CasADi panel functions (``table_inputs=True``).

Contract: with the tables passed as inputs, the residual and panel-force
functions give the same outputs as the functions with the tables embedded;
one function, built once, serves a second shape whose blended polars differ
(no rebuild); omitted tables give NaN, not a silently zero polar; and the
function still expands to SX.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ca = pytest.importorskip("casadi")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "BodyAerodynamics"))

from panel_kernel_cases import V3_FILE  # noqa: E402
from VSM.core.BodyAerodynamics import BodyAerodynamics  # noqa: E402
from VSM.core.Solver import Solver  # noqa: E402

RTOL = 1e-12


def _v3(distribution):
    body = BodyAerodynamics.instantiate(
        n_panels=20, file_path=V3_FILE, spanwise_panel_distribution=distribution
    )
    body.va_initialize(10.0, 8.0, 3.0)
    return body


def _solved(body, **solver_kwargs):
    solver = Solver(gamma_loop_type="casadi_newton", allowed_error=1e-10, **solver_kwargs)
    results = solver.solve(body)
    return solver, np.asarray(results["gamma_distribution"], dtype=float)


def _newton_inputs(solver, gamma):
    v_rel = solver.compute_relative_velocity(gamma)
    n = solver.n_panels
    laplacian = (
        solver._build_spanwise_laplacian()
        if solver.is_with_artificial_viscosity
        else np.zeros((n, n))
    )
    return {
        "v_rel": v_rel,
        "x_airf": solver.x_airf_array,
        "y_airf": solver.y_airf_array,
        "z_airf": solver.z_airf_array,
        "chord": solver.chord_array,
        "width": solver.width_array,
        "l_gamma": laplacian @ gamma,
        **solver.casadi_stall_angle_inputs(),
    }


def _force_inputs(solver, gamma):
    v_rel = solver.compute_relative_velocity(gamma)
    return {
        "v_rel": v_rel,
        "v_qc": v_rel,
        "x_airf": solver.x_airf_array,
        "y_airf": solver.y_airf_array,
        "z_airf": solver.z_airf_array,
        "chord": solver.chord_array,
        "width": solver.width_array,
        "rho": solver.rho,
    }


def _assert_outputs_equal(out_tables, out_const):
    assert set(out_tables) == set(out_const)
    for key, value in out_const.items():
        expected = np.array(value, dtype=float)
        actual = np.array(out_tables[key], dtype=float)
        scale = max(np.max(np.abs(expected)), 1e-300)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=RTOL * scale, err_msg=key)


@pytest.mark.parametrize("viscosity", [False, True])
def test_table_inputs_reproduce_embedded_tables(viscosity):
    """Residual function (all 13 outputs, incl. the corner-averaged ones and
    the viscosity lift slope) and panel-force function, against the
    embedded-table functions of the same shape."""
    solver, gamma = _solved(_v3("uniform"), is_with_artificial_viscosity=viscosity)
    tables = solver.casadi_polar_table_inputs()
    assert tables["cl_table"].shape == (solver.n_panels, solver.panels[0].panel_polar_data.shape[0])

    args = _newton_inputs(solver, gamma)
    if viscosity:  # put a panel past stall so the gate and mu are exercised
        args["stall_angles"] = args["stall_angles"] - 0.3
    _assert_outputs_equal(
        solver._casadi_newton_function(table_inputs=True)(**args, **tables),
        solver._casadi_newton_function()(**args),
    )
    f_args = _force_inputs(solver, gamma)
    _assert_outputs_equal(
        solver.casadi_panel_force_function(table_inputs=True)(**f_args, **tables),
        solver.casadi_panel_force_function()(**f_args),
    )


def test_one_function_serves_a_second_shape_without_rebuild():
    """Shapes A and B have the same panel count and alpha grid but differently
    blended polars. The table-input function built for A, given B's tables and
    geometry, equals a fresh embedded-table build for B, and no new
    table-input function is built."""
    solver_a, gamma_a = _solved(_v3("uniform"))
    fn = solver_a._casadi_newton_function(table_inputs=True)
    force_fn = solver_a.casadi_panel_force_function(table_inputs=True)
    cache_size = len(solver_a._casadi_table_input_cache)

    body_b = _v3("cosine")
    tables_a = solver_a.casadi_polar_table_inputs()
    solver_a.solve(body_b)  # same solver object, now on shape B
    gamma_b = solver_a.solve(body_b)["gamma_distribution"]
    tables_b = solver_a.casadi_polar_table_inputs()
    assert np.max(np.abs(tables_b["cl_table"] - tables_a["cl_table"])) > 1e-3  # the polars differ

    assert solver_a._casadi_newton_function(table_inputs=True) is fn
    assert solver_a.casadi_panel_force_function(table_inputs=True) is force_fn
    assert len(solver_a._casadi_table_input_cache) == cache_size

    fresh_b, _ = _solved(_v3("cosine"))
    _assert_outputs_equal(
        fn(**_newton_inputs(solver_a, gamma_b), **tables_b),
        fresh_b._casadi_newton_function()(**_newton_inputs(solver_a, gamma_b)),
    )
    _assert_outputs_equal(
        force_fn(**_force_inputs(solver_a, gamma_b), **tables_b),
        fresh_b.casadi_panel_force_function()(**_force_inputs(solver_a, gamma_b)),
    )


def test_omitted_tables_give_nan_not_zero():
    solver, gamma = _solved(_v3("uniform"))
    out = solver._casadi_newton_function(table_inputs=True)(**_newton_inputs(solver, gamma))
    assert np.all(np.isnan(np.array(out["h"], dtype=float)))
    out = solver.casadi_panel_force_function(table_inputs=True)(**_force_inputs(solver, gamma))
    assert np.all(np.isnan(np.array(out["force"], dtype=float)))


def test_table_input_functions_expand_to_sx():
    solver, gamma = _solved(_v3("uniform"), is_with_artificial_viscosity=True)
    tables = solver.casadi_polar_table_inputs()
    for fn, args in (
        (solver._casadi_newton_function(table_inputs=True), _newton_inputs(solver, gamma)),
        (solver.casadi_panel_force_function(table_inputs=True), _force_inputs(solver, gamma)),
    ):
        _assert_outputs_equal(fn.expand()(**args, **tables), fn(**args, **tables))


def test_newton_loop_is_unchanged_and_does_not_build_table_functions():
    solver, _ = _solved(_v3("uniform"))
    assert solver._casadi_table_input_cache == {}


def test_unsupported_cases_raise_clearly():
    solver, _ = _solved(_v3("uniform"), polar_interpolation="bspline")
    with pytest.raises(ValueError, match="polar_interpolation='linear'"):
        solver._casadi_newton_function(table_inputs=True)
    with pytest.raises(ValueError, match="polar_interpolation='linear'"):
        solver.casadi_polar_table_inputs()
