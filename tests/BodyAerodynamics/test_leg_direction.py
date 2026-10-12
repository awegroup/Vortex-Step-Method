"""Direction of the attached-trailed legs: along the flow (default), along
the ribs, or perpendicular to the quarter-chord line
(``BodyAerodynamics(leg_direction=...)``).

Perpendicular legs depend only on the quarter-chord line and the
trailing-edge curve, so (a) they coincide with the ribs wherever the ribs are
perpendicular, (b) a straight wing whose ribs are drawn skewed gives the same
vortex system and the same forces as the same wing with straight ribs, and
(c) a kinked swept wing builds without crossing legs, with the root leg on the
symmetry plane. Inviscid polars.
"""

import numpy as np
import pytest

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.core.WingGeometry import Wing
from VSM.core.utils import perpendicular_leg_end_points

_alpha = np.radians(np.linspace(-25, 25, 101))
INVISCID = np.column_stack((_alpha, 2 * np.pi * _alpha, 0 * _alpha, 0 * _alpha))


def _straight(n, skew=0.0, **kw):
    wing = Wing(n_panels=n, spanwise_panel_distribution="uniform")
    for y in np.linspace(-4.0, 4.0, n + 1):
        wing.add_section(np.array([0.0, y, 0.0]), np.array([1.0, y + skew, 0.0]), INVISCID)
    body = BodyAerodynamics([wing], **{"leg_direction": "rib", **kw})
    body.va_initialize(20.0, 5.0, 0.0)
    return body


def _swept(n, sweep_deg, **kw):
    """Kinked swept wing, chord 1, ribs along the freestream."""
    wing = Wing(n_panels=n, spanwise_panel_distribution="uniform")
    for y in np.linspace(-4.0, 4.0, n + 1):
        x = abs(y) * np.tan(np.radians(sweep_deg))
        wing.add_section(np.array([x, y, 0.0]), np.array([x + 1.0, y, 0.0]), INVISCID)
    body = BodyAerodynamics([wing], **{"leg_direction": "rib", **kw})
    body.va_initialize(20.0, 5.0, 0.0)
    return body


def _arc(n, **kw):
    wing = Wing(n_panels=n, spanwise_panel_distribution="uniform")
    for th in np.radians(np.linspace(-55.0, 55.0, n + 1)):
        le = np.array([0.0, 4.0 * np.sin(th), 4.0 * np.cos(th)])
        wing.add_section(le, le + np.array([1.0, 0.0, 0.0]), INVISCID)
    body = BodyAerodynamics([wing], **{"leg_direction": "rib", **kw})
    body.va_initialize(10.0, 8.0, 0.0)
    return body


def _legs(body):
    P = body.panels
    bound = np.array([p.bound_point_1 for p in P] + [P[-1].bound_point_2])
    te = np.array([p.TE_point_1 for p in P] + [P[-1].TE_point_2])
    return bound, te


def _solve(body):
    r = Solver(allowed_error=1e-7, max_iterations=4000).solve(body)
    assert r["gamma_converged"]
    return r


def test_unknown_leg_direction_raises():
    with pytest.raises(ValueError):
        _straight(8, leg_direction="diagonal")


@pytest.mark.parametrize("make", [_straight, _arc])
def test_perpendicular_legs_equal_ribs_where_ribs_are_perpendicular(make):
    rib = make(16)
    perp = make(16, leg_direction="perpendicular")
    assert perp.leg_direction == "perpendicular"
    for p_rib, p_perp in zip(rib.panels, perp.panels):
        assert np.allclose(p_rib.TE_point_1, p_perp.TE_point_1, atol=1e-10)
        assert np.allclose(p_rib.TE_point_2, p_perp.TE_point_2, atol=1e-10)
        assert np.allclose(p_rib.control_point, p_perp.control_point, atol=1e-10)
        assert np.isclose(p_rib.chord, p_perp.chord)


def test_skewed_ribs_give_the_straight_wing():
    """The surface is the same rectangle whichever way the ribs are drawn."""
    reference = _straight(20)
    skewed = _straight(20, skew=0.4, leg_direction="perpendicular")
    bound, te = _legs(skewed)
    # legs along x, ending on the trailing-edge line x = 1 at the node's own y
    assert np.allclose(te[:, 0], 1.0, atol=1e-10)
    assert np.allclose(te[:, 1], bound[:, 1], atol=1e-10)
    r_ref = _solve(reference)
    r_skw = _solve(skewed)
    assert np.isclose(r_skw["cl"], r_ref["cl"], rtol=1e-6)
    assert np.isclose(r_skw["cd"], r_ref["cd"], rtol=1e-5)
    assert np.isclose(r_skw["drag"] / r_skw["drag_induced_trefftz"], 1.0, atol=1e-3)


def test_skewed_rib_legs_are_not_the_straight_wing():
    """The reference behaviour: rib legs carry the skew into the result."""
    reference = _solve(_straight(20))
    skewed = _solve(_straight(20, skew=0.4))
    assert abs(skewed["cd"] / reference["cd"] - 1.0) > 5e-3


def test_kinked_swept_wing_builds_without_crossing_legs():
    body = _swept(40, 30.0, leg_direction="perpendicular")
    bound, te = _legs(body)
    # ends ordered along the trailing edge (panels run from +y to -y)
    assert np.all(np.diff(te[:, 1]) < 0.0)
    # root leg on the symmetry plane, streamwise; away from the root the legs
    # turn towards the perpendicular (aft and inboard), one leg length away
    k = len(bound) // 2
    assert np.isclose(bound[k, 1], 0.0, atol=1e-12)
    assert np.isclose(te[k, 1], 0.0, atol=1e-10)
    leg = te - bound
    j = k - 6  # 1.2 chords from the root at 0.2 panel width, outside the averaging window
    inboard = -np.sign(bound[j, 1]) * leg[j, 1]
    assert np.isclose(inboard / np.linalg.norm(leg[j]), np.sin(np.radians(30.0)), atol=1e-6)
    r = _solve(body)
    assert np.isfinite(r["cl"]) and np.isfinite(r["cd"])
    assert 0.9 < r["drag"] / r["drag_induced_trefftz"] < 1.1


def test_crossing_legs_raise():
    with pytest.raises(ValueError, match="cross|does not reach"):
        _swept(40, 75.0, leg_direction="perpendicular")


def test_leg_end_points_function_on_a_uniformly_swept_line():
    """On a line of constant sweep the legs are the plain perpendiculars."""
    sweep = np.radians(30.0)
    n = 10
    s = np.linspace(0.0, 5.0, n + 1)
    bound = np.column_stack((s * np.sin(sweep), s * np.cos(sweep), 0 * s))
    perp = np.array([np.cos(sweep), -np.sin(sweep), 0.0])
    te = bound + 0.75 * perp + np.array([0.0, 0.0, 0.0])
    chord_dir = np.tile(perp, (n, 1))
    ends = perpendicular_leg_end_points(bound, te, chord_dir, np.ones(n), 0.25)
    assert np.allclose(ends, te, atol=1e-10)


def test_streamwise_legs_equal_ribs_drawn_along_the_flow():
    """On the kinked swept wing the ribs are along x, as are streamwise legs."""
    rib = _swept(40, 30.0)
    stream = _swept(40, 30.0, leg_direction="streamwise")
    for p_rib, p_s in zip(rib.panels, stream.panels):
        assert np.allclose(p_rib.TE_point_1, p_s.TE_point_1, atol=1e-10)
        assert np.allclose(p_rib.control_point, p_s.control_point, atol=1e-10)


def test_streamwise_legs_on_skewed_ribs_give_the_straight_wing():
    reference = _solve(_straight(20))
    skewed = _solve(_straight(20, skew=0.4, leg_direction="streamwise"))
    assert np.isclose(skewed["cl"], reference["cl"], rtol=1e-6)
    assert np.isclose(skewed["cd"], reference["cd"], rtol=1e-5)


def test_streamwise_legs_follow_the_inflow():
    """Setting a sideslip rebuilds the legs along the new apparent velocity."""
    body = _straight(16, skew=0.4, leg_direction="streamwise")
    te_0 = np.array([p.TE_point_1 for p in body.panels])
    body.va_initialize(20.0, 5.0, 15.0)
    te_1 = np.array([p.TE_point_1 for p in body.panels])
    bound = np.array([p.bound_point_1 for p in body.panels])
    legs = te_1 - bound
    # legs lie in the plane of the flat wing and point along the yawed flow
    assert np.allclose(legs[:, 2], 0.0, atol=1e-12)
    va = np.atleast_2d(body.va)[0]
    expected_yaw = np.degrees(np.arctan2(va[1], va[0]))
    yaw = np.degrees(np.arctan2(legs[:, 1], legs[:, 0]))
    assert np.allclose(yaw, expected_yaw, atol=1e-9)
    assert not np.allclose(te_0, te_1)


def test_streamwise_legs_fall_back_to_ribs_for_pure_rotation(caplog):
    """A body that only rotates has no mean inflow direction: the rib legs
    are used, with a warning, and the solve goes through."""
    import logging

    body = _straight(12, skew=0.4, leg_direction="streamwise")
    rib_te = np.array([p._rib_TE_point_1 for p in body.panels])
    centre = np.mean([p.control_point for p in body.panels], axis=0)
    with caplog.at_level(logging.WARNING):
        body.va_initialize(Umag=0.0, angle_of_attack=0.0, side_slip=0.0,
                           body_rates=[0.3, -0.2, 0.1],
                           body_axis=[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                           reference_point=centre)
    assert any("rib legs" in m for m in caplog.messages)
    assert np.allclose(np.array([p.TE_point_1 for p in body.panels]), rib_te)
    # a proper inflow restores the streamwise legs
    body.va_initialize(20.0, 5.0, 0.0)
    assert not np.allclose(np.array([p.TE_point_1 for p in body.panels]), rib_te)
