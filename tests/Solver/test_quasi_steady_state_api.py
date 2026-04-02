import inspect

from VSM.quasi_steady_state import (
    compute_quasi_steady_trim_jacobian,
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
