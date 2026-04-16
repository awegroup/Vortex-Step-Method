import inspect

import numpy as np

from VSM.quasi_steady_state import (
    compute_quasi_steady_trim_jacobian,
    linearize_fast_dynamics_from_trim_jacobian,
    run_quasi_steady_sweep,
    solve_quasi_steady_state,
)


def test_solve_quasi_steady_state_required_arguments():
    signature = inspect.signature(solve_quasi_steady_state)

    required_arguments = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is inspect.Parameter.empty
    ]

    assert required_arguments == [
        "body_aero",
        "center_of_gravity",
        "reference_point",
        "system_model",
        "x_guess",
    ]


def test_run_quasi_steady_sweep_required_keyword_arguments():
    signature = inspect.signature(run_quasi_steady_sweep)

    required_keyword_arguments = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
    ]

    assert required_keyword_arguments == [
        "build_body",
        "system_model",
        "center_of_gravity",
        "reference_point",
        "x_guess",
        "principal_axis",
        "secondary_axis",
        "sweep_values",
    ]


def test_compute_quasi_steady_trim_jacobian_required_arguments():
    signature = inspect.signature(compute_quasi_steady_trim_jacobian)

    required_arguments = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is inspect.Parameter.empty
    ]

    assert required_arguments == [
        "body_aero",
        "center_of_gravity",
        "reference_point",
        "system_model",
        "x_state",
    ]


def test_linearize_fast_dynamics_from_trim_jacobian_shapes_and_stability_flags():
    jac = np.zeros((5, 5), dtype=float)
    # Longitudinal: d(cfx)/d(vtau) and d(cmy)/d(pitch_rad)
    jac[3, 0] = -0.2
    jac[1, 2] = -0.15
    # Lateral: simple diagonal-dominant stable block in yaw/course/roll channels
    jac[4, 3] = -0.08
    jac[2, 4] = -0.12
    jac[0, 1] = -0.10

    result = linearize_fast_dynamics_from_trim_jacobian(
        jacobian=jac,
        x_state=np.array([20.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        mass=10.0,
        Ixx=50.0,
        Iyy=50.0,
        Izz=50.0,
        rho=1.0,
        reference_area=2.0,
        reference_chord=1.0,
        radial_distance=200.0,
    )

    assert result["A_long"].shape == (2, 2)
    assert result["A_lateral"].shape == (3, 3)
    assert result["eig_long"].shape == (2,)
    assert result["eig_lateral"].shape == (3,)
    assert result["stable_long"] is True
    assert result["stable_lateral"] is True
    assert np.all(np.isfinite(result["Tfast_long"]))
    assert np.all(np.isfinite(result["Tfast_lateral"]))


def test_linearize_fast_dynamics_from_trim_jacobian_input_validation():
    with np.testing.assert_raises(ValueError):
        linearize_fast_dynamics_from_trim_jacobian(
            jacobian=np.zeros((4, 4), dtype=float),
            x_state=np.zeros(5, dtype=float),
        )
