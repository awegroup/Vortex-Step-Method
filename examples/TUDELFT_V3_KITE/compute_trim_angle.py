from pathlib import Path
from typing import Optional

import numpy as np

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.trim_angle import compute_trim_angle

import math
from typing import Tuple, List, Optional


def circle_circle_intersections(
    c0: Tuple[float, float],
    r0: float,
    c1: Tuple[float, float],
    r1: float,
    *,
    eps: float = 1e-12,
) -> Optional[List[Tuple[float, float]]]:
    """
    Return intersection point(s) of two circles in 2D.

    Args:
        c0, c1: circle centers (x, y)
        r0, r1: radii (>= 0)
        eps: numerical tolerance

    Returns:
        - [] if no intersection
        - [(x, y)] if tangent (one intersection)
        - [(x3, y3), (x4, y4)] if two intersections
        - None if infinite intersections (coincident circles)
    """

    x0, _, y0 = c0
    x1, _, y1 = c1

    if r0 < 0 or r1 < 0:
        raise ValueError("Radii must be non-negative.")

    dx = x1 - x0
    dy = y1 - y0
    d = math.hypot(dx, dy)

    # Coincident centers
    if d <= eps:
        if abs(r0 - r1) <= eps:
            return None  # infinite intersections
        return []  # concentric, no intersections

    # Separate / contained
    if d > r0 + r1 + eps:
        return []
    if d < abs(r0 - r1) - eps:
        return []

    # Solve for intersection(s)
    a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
    h2 = r0 * r0 - a * a

    # Clamp small negative due to floating point
    if h2 < 0 and h2 > -eps:
        h2 = 0.0
    if h2 < 0:
        return []

    h = math.sqrt(h2)

    x2 = x0 + a * dx / d
    y2 = y0 + a * dy / d

    rx = -dy * (h / d)
    ry = dx * (h / d)

    p3 = np.array([x2 + rx, 0, y2 + ry]).T
    p4 = np.array([x2 - rx, 0, y2 - ry]).T

    # Tangent: one point
    if h <= eps:
        return p3

    if p3[2] > y2:
        return p4
    elif p4[2] > y2:
        return p3


def run_v3_geom(
    umag: float = 20.0,
    side_slip: float = 0.0,
    reference_point: Optional[np.ndarray] = None,
    with_bridles: bool = False,
) -> float:
    """Run TUDELFT V3 geometry once and return a single trim alpha [deg]."""
    project_dir = Path(__file__).resolve().parents[2]
    cad_dir = project_dir / "data" / "TUDELFT_V3_KITE" / "CAD_derived_geometry"

    if reference_point is None:
        reference_point = np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        reference_point = np.asarray(reference_point, dtype=float)

    instantiate_kwargs = {
        "n_panels": 50,
        "file_path": cad_dir / "aero_geometry_CAD_CFD_polars.yaml",
        "spanwise_panel_distribution": "uniform",
    }
    if with_bridles:
        instantiate_kwargs["bridle_path"] = (
            cad_dir / "struc_geometry_manually_adjusted.yaml"
        )

    body_aero = BodyAerodynamics.instantiate(**instantiate_kwargs)
    solver = Solver(reference_point=reference_point)

    trim_result = compute_trim_angle(
        body_aero=body_aero,
        solver=solver,
        side_slip=side_slip,
        velocity_magnitude=umag,
        roll_rate=0.0,
        pitch_rate=0.0,
        yaw_rate=0.0,
        alpha_min=-5.0,
        alpha_max=20.0,
        coarse_step=2.0,
        fine_tolerance=1e-3,
        derivative_step=0.25,
        max_bisection_iter=50,
        reference_point=solver.reference_point,
    )

    trim_alpha = float(trim_result["trim_angle"])

    body_aero.va_initialize(
        Umag=umag,
        angle_of_attack=trim_alpha,
        side_slip=side_slip,
        yaw_rate=0.0,
        pitch_rate=0.0,
        roll_rate=0.0,
        reference_point=solver.reference_point,
    )
    solved = solver.solve(body_aero)
    cmy = solved.get("cmy", solved.get("CMy", np.nan))

    print("TUDELFT V3 trim run")
    print(f"trim alpha [deg]: {trim_alpha:.4f}")
    print(f"dCMy/dalpha [1/rad]: {trim_result['dCMy_dalpha']:.6f}")
    print(f"stable: {trim_result['is_stable']}")
    print(f"CMy at trim alpha [-]: {float(cmy):.6f}")

    return trim_alpha


def get_kcu_location():
    mid_span_LE = [-1.1557911914860979, 0.0, 11.00491618799836]
    mid_span_TE = [1.443146003226444, 0.0, 11.004972573823276]
    L_b = np.sqrt((mid_span_LE[0] ** 2 + mid_span_LE[1] ** 2 + mid_span_LE[2] ** 2))
    L_t = np.sqrt((mid_span_TE[0] ** 2 + mid_span_TE[1] ** 2 + mid_span_TE[2] ** 2))

    print(f"L_b: {L_b:.4f} m")
    print(f"L_t: {L_t:.4f} m")

    L_b += 0.036
    delta_lb_pow = 0.036
    delta_lb_dep = 0.66
    L_b_pow = 11.0654 + delta_lb_pow
    L_b_dep = 11.0654 - delta_lb_dep
    L_b = L_b_dep

    print(f"% diff delta pow lb: {(delta_lb_pow) / L_b * 100:.2f}%")
    print(f"% diff delta dep lb: {(delta_lb_dep) / L_b * 100:.2f}%")

    point = circle_circle_intersections(
        c0=mid_span_LE,
        r0=L_b,
        c1=mid_span_TE,
        r1=L_t,
    )
    print(f"point: {point}")

    # run_v3_geom(reference_point=point)


def main() -> None:

    get_kcu_location()


if __name__ == "__main__":
    main()
