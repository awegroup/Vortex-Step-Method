import inspect

from VSM.quasi_steady_state import solve_quasi_steady_state


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
