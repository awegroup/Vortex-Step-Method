"""Geometry and solver cases for the force-law regression in
``test_panel_kernels.py``.

``python tests/BodyAerodynamics/panel_kernel_cases.py OUT.npz`` writes every
numeric output of ``Solver.solve`` for every case. The committed reference
``data/panel_kernel_reference.npz`` was written with VSM at ``1201f6f``, the
last commit before the force law moved into ``VSM.core.panel_kernels``, so
the regression test pins the refactor to the original per-panel loop.
"""

import sys
from pathlib import Path

import numpy as np

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.core.WingGeometry import Wing

REPO = Path(__file__).resolve().parents[2]
V3_FILE = REPO / "data/TUDELFT_V3_KITE/CAD_derived_geometry/aero_geometry_CAD_CFD_polars.yaml"
REFERENCE_FILE = Path(__file__).resolve().parent / "data" / "panel_kernel_reference.npz"

_alpha = np.radians(np.linspace(-25, 25, 101))
# Thin-airfoil lift with a small drag and moment, so all three magnitudes are exercised.
POLAR = np.column_stack(
    (_alpha, 2 * np.pi * _alpha, 0.01 + 0.05 * _alpha**2, -0.1 - 0.02 * _alpha)
)


def _wing_body(leading_edges, chord_vector, n_panels):
    wing = Wing(n_panels=n_panels, spanwise_panel_distribution="uniform")
    for le in leading_edges:
        wing.add_section(le, le + chord_vector, POLAR)
    return BodyAerodynamics([wing])


def straight_body(n_panels=16):
    ys = np.linspace(-4.0, 4.0, n_panels + 1)
    body = _wing_body([np.array([0.0, y, 0.0]) for y in ys], np.array([1.0, 0.0, 0.0]), n_panels)
    body.va_initialize(20.0, 5.0, 0.0)
    return body


def swept_body(n_panels=16):
    """Symmetric 30 deg swept-back wing."""
    ys = np.linspace(-4.0, 4.0, n_panels + 1)
    sweep = np.tan(np.radians(30.0))
    body = _wing_body(
        [np.array([sweep * abs(y), y, 0.0]) for y in ys], np.array([1.0, 0.0, 0.0]), n_panels
    )
    body.va_initialize(15.0, 6.0, 2.0)
    return body


def arc_body(n_panels=16):
    """Circular-arc wing (kite-like anhedral), constant chord along x."""
    th = np.radians(np.linspace(-55.0, 55.0, n_panels + 1))
    les = [np.array([0.0, 4.0 * np.sin(t), 4.0 * np.cos(t)]) for t in th]
    body = _wing_body(les, np.array([1.0, 0.0, 0.0]), n_panels)
    body.va_initialize(10.0, 8.0, 0.0)
    return body


def v3_body(n_panels=20, yaw_rate=0.0):
    body = BodyAerodynamics.instantiate(n_panels=n_panels, file_path=V3_FILE)
    body.va_initialize(10.0, 8.0, 3.0, np.array([0.0, 0.0, yaw_rate]))
    return body


# name -> (body factory, solver kwargs)
CASES = {
    "straight": (straight_body, {}),
    "swept": (swept_body, {}),
    "arc": (arc_body, {}),
    "v3": (v3_body, {}),
    "v3_viscous_noAT": (
        v3_body,
        {"is_with_viscous_drag_correction": True, "is_with_attached_trailed_vortex_force": False},
    ),
    "v3_yaw_rate": (lambda: v3_body(yaw_rate=0.4), {"reference_point": [0.1, 0.0, 0.5]}),
}


def numeric_outputs(results: dict) -> dict:
    """Every numeric entry of a ``solve`` result, as float arrays."""
    out = {}
    for key, value in results.items():
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            continue
        out[key] = array
    return out


def run_case(name: str, is_aoa_corrected: bool) -> dict:
    factory, kwargs = CASES[name]
    solver = Solver(allowed_error=1e-10, is_aoa_corrected=is_aoa_corrected, **kwargs)
    results = solver.solve(factory())
    assert results["gamma_converged"]
    return numeric_outputs(results)


def all_outputs() -> dict:
    flat = {}
    for name in CASES:
        for corrected in (True, False):
            for key, value in run_case(name, corrected).items():
                flat[f"{name}|{corrected}|{key}"] = value
    return flat


if __name__ == "__main__":
    np.savez(sys.argv[1], **all_outputs())
