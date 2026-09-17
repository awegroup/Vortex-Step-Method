"""The frozen wake of every ring follows that panel's own apparent velocity:
identical to the classical straight wake in a uniform inflow, locally aligned
under body rates or a distributed inflow."""
import numpy as np
import pytest
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.WingGeometry import Wing
from VSM.core.Solver import Solver


def _wing(n_panels=12, span=8.0):
    alpha = np.radians(np.linspace(-25, 25, 101))
    polar = np.column_stack((alpha, 2 * np.pi * alpha, 0 * alpha, 0 * alpha))
    wing = Wing(n_panels=n_panels, spanwise_panel_distribution="uniform")
    for y in np.linspace(-span / 2, span / 2, n_panels + 1):
        wing.add_section(np.array([0.0, y, 0.0]), np.array([1.0, y, 0.0]), polar)
    return BodyAerodynamics([wing])


def _va_arrays(body):
    va = np.array([p.va for p in body.panels], dtype=float)
    norm = np.linalg.norm(va, axis=1)
    return norm, va / norm[:, None]


def test_wake_directions_follow_each_panel_apparent_velocity():
    body = _wing()
    body.va_initialize(10.0, 5.0, 0.0)
    units, speeds = body._wake_directions(*_va_arrays(body))
    # uniform inflow: one direction and one speed for every ring
    np.testing.assert_allclose(units, np.tile(units[0], (body.n_panels, 1)))
    np.testing.assert_allclose(speeds, 10.0)

    body.va_initialize(10.0, 5.0, 0.0, body_rates=np.array([0.0, 0.0, 0.3]))
    units, speeds = body._wake_directions(*_va_arrays(body))
    va = np.array([p.va for p in body.panels])
    np.testing.assert_allclose(units, va / np.linalg.norm(va, axis=1)[:, None])
    # a yaw rate gives the two tips different wake directions and speeds
    angle = np.degrees(np.arccos(np.clip(units[0] @ units[-1], -1.0, 1.0)))
    assert angle > 0.5
    assert abs(speeds[0] - speeds[-1]) > 1.0


def test_wake_direction_changes_the_solution_only_under_body_rates():
    body = _wing()
    solver = Solver(allowed_error=1e-9)

    def solve_with_mean_wake(body):
        """The previous behaviour: one wake direction, the mean freestream."""
        original = body.compute_AIC_matrices

        def mean_wake_AIC(model, core, va_norm, va_unit):
            mean = np.mean(va_unit * va_norm[:, None], axis=0)
            n = body.n_panels
            return original(
                model, core, np.full(n, np.linalg.norm(mean)), np.tile(mean / np.linalg.norm(mean), (n, 1))
            )

        body.compute_AIC_matrices = mean_wake_AIC
        try:
            return solver.solve(body)
        finally:
            del body.compute_AIC_matrices

    body.va_initialize(10.0, 5.0, 0.0)
    ref = solve_with_mean_wake(body)
    body.va_initialize(10.0, 5.0, 0.0)
    new = solver.solve(body)
    np.testing.assert_allclose(new["gamma_distribution"], ref["gamma_distribution"], rtol=1e-12)

    body.va_initialize(10.0, 5.0, 0.0, body_rates=np.array([0.0, 0.0, 0.3]))
    ref = solve_with_mean_wake(body)
    body.va_initialize(10.0, 5.0, 0.0, body_rates=np.array([0.0, 0.0, 0.3]))
    new = solver.solve(body)
    assert new["gamma_converged"] and ref["gamma_converged"]
    assert not np.allclose(new["gamma_distribution"], ref["gamma_distribution"], rtol=1e-4)
    # a small correction, as the wake direction is a second-order effect
    np.testing.assert_allclose(new["Mz"], ref["Mz"], rtol=0.1)


def test_wake_directions_reject_non_positive_speed():
    body = _wing(4)
    body.va_initialize(10.0, 5.0, 0.0)
    norm, unit = _va_arrays(body)
    norm = norm.copy()
    norm[1] = 0.0
    with pytest.raises(ValueError):
        body._wake_directions(norm, unit)
