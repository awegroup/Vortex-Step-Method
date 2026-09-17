"""The quarter-chord direction velocity on curved lifting lines.

A lifting line evaluated on itself has a logarithmically singular
self-induction wherever it is curved. The regularised correction in
``assemble_curved_bound_correction_AIC`` must (a) vanish on a straight wing,
so the classical exact result is kept, and (b) make the on-blade induced drag
equal to the Trefftz-plane value on curved wings independently of the panel
count. Inviscid polars, so the on-blade drag is pure induced drag.
"""

import numpy as np
import pytest

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.core.WingGeometry import Wing
from VSM.core.utils import assemble_curved_bound_correction_AIC, thin_airfoil_strips

_alpha = np.radians(np.linspace(-25, 25, 101))
INVISCID = np.column_stack((_alpha, 2 * np.pi * _alpha, 0 * _alpha, 0 * _alpha))


def _straight(n_panels, half_span=4.0):
    wing = Wing(n_panels=n_panels, spanwise_panel_distribution="uniform")
    for y in np.linspace(-half_span, half_span, n_panels + 1):
        wing.add_section(np.array([0.0, y, 0.0]), np.array([1.0, y, 0.0]), INVISCID)
    body = BodyAerodynamics([wing])
    body.va_initialize(20.0, 5.0, 0.0)
    return body


def _arc(n_panels, radius=4.0, half_angle_deg=55.0):
    """Circular-arc wing (kite-like anhedral), constant chord 1 along x."""
    wing = Wing(n_panels=n_panels, spanwise_panel_distribution="uniform")
    for th in np.radians(np.linspace(-half_angle_deg, half_angle_deg, n_panels + 1)):
        le = np.array([0.0, radius * np.sin(th), radius * np.cos(th)])
        wing.add_section(le, le + np.array([1.0, 0.0, 0.0]), INVISCID)
    body = BodyAerodynamics([wing])
    body.va_initialize(10.0, 8.0, 0.0)
    return body


def _ratio(body):
    results = Solver(allowed_error=1e-7, max_iterations=4000).solve(body)
    assert results["gamma_converged"]
    return results["drag"] / results["drag_induced_trefftz"]


def test_strip_weights_are_thin_airfoil_with_quarter_chord_centroid():
    x, w = thin_airfoil_strips(24)
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(np.sum(w * x), 0.25, atol=2e-3)
    # effective cut-off exp(<ln|x - 1/4|>) of the distribution is 0.15 c
    assert np.isclose(np.exp(np.sum(w * np.log(np.abs(x - 0.25)))), 0.15, atol=0.02)


def test_correction_vanishes_on_a_straight_wing():
    body = _straight(20)
    panels = body.panels
    ep = np.array([p.aerodynamic_center for p in panels])
    bp1 = np.array([p.bound_point_1 for p in panels])
    bp2 = np.array([p.bound_point_2 for p in panels])
    te1 = np.array([p.TE_point_1 for p in panels])
    te2 = np.array([p.TE_point_2 for p in panels])
    y_airf = np.array([p.y_airf for p in panels])
    chords = np.array([p.chord for p in panels])
    x, w = thin_airfoil_strips()
    A = assemble_curved_bound_correction_AIC(
        ep, bp1, bp2, te1, te2, y_airf, chords, np.full(20, 20.0), 0.25, 0.05, x, w
    )
    assert np.max(np.abs(A)) < 1e-9


def test_straight_wing_on_blade_drag_equals_trefftz():
    assert np.isclose(_ratio(_straight(40)), 1.0, atol=2e-3)


@pytest.mark.parametrize("n_panels", [20, 40, 80])
def test_arc_wing_on_blade_drag_equals_trefftz_at_every_resolution(n_panels):
    # Before the regularisation this ratio was 0.977 / 0.957 / 0.937 at
    # 20 / 40 / 80 panels and kept falling (log-singular self-induction).
    assert np.isclose(_ratio(_arc(n_panels)), 1.0, atol=1e-2)


def test_arc_wing_ratio_is_mesh_independent():
    r40 = _ratio(_arc(40))
    r80 = _ratio(_arc(80))
    assert abs(r80 - r40) < 3e-3
