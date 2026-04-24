from __future__ import annotations

import copy
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver


@dataclass(frozen=True)
class AxisDefinition:
    """Axis convention for the course frame."""

    course: np.ndarray
    normal: np.ndarray
    radial: np.ndarray


DEFAULT_AXES = AxisDefinition(
    course=np.array([1.0, 0.0, 0.0], dtype=float),
    normal=np.array([0.0, 1.0, 0.0], dtype=float),
    radial=np.array([0.0, 0.0, 1.0], dtype=float),
)


DEFAULT_TRANSFORMATION_C_FROM_VSM = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)

# Bounds for x = [kite_speed, roll_deg, pitch_deg, yaw_deg, course_rate_body].
DEFAULT_BOUNDS_LOWER = np.array([2.0, -5.0, -5.0, -6.0, -2.0], dtype=float)
DEFAULT_BOUNDS_UPPER = np.array([80.0, 5.0, 5.0, 6.0, 2.0], dtype=float)


def _as_3vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != 3:
        raise ValueError(f"Expected a 3-vector, got shape {np.asarray(value).shape}")
    return vector


def _apply_attitude(
    body: BodyAerodynamics,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    axes: AxisDefinition,
    reference_point: np.ndarray,
) -> None:
    combined_rotation = _compose_attitude_rotation(
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        axes=axes,
    )

    origin = _as_3vector(reference_point)

    def rotate_point(point: np.ndarray) -> np.ndarray:
        return origin + combined_rotation @ (_as_3vector(point) - origin)

    for wing in body.wings:
        for section in wing.sections:
            section.LE_point = rotate_point(section.LE_point)
            section.TE_point = rotate_point(section.TE_point)

        rotated_span = combined_rotation @ wing.spanwise_direction
        span_norm = np.linalg.norm(rotated_span)
        if span_norm == 0.0:
            raise ValueError(
                "Combined attitude produced zero spanwise direction vector."
            )
        wing.spanwise_direction = rotated_span / span_norm

    body.geometry_rotation = combined_rotation @ body.geometry_rotation
    body._build_panels()


def _rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    axis_vec = _as_3vector(axis)
    axis_norm = np.linalg.norm(axis_vec)
    if axis_norm == 0.0:
        raise ValueError("Rotation axis must be non-zero.")
    axis_unit = axis_vec / axis_norm
    kx, ky, kz = axis_unit
    skew = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=float,
    )
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def _compose_attitude_rotation(
    *,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    axes: AxisDefinition,
) -> np.ndarray:
    roll_matrix = _rotation_matrix(axes.course, roll_deg)
    pitch_matrix = _rotation_matrix(axes.normal, pitch_deg)
    yaw_matrix = _rotation_matrix(axes.radial, yaw_deg)
    return yaw_matrix @ pitch_matrix @ roll_matrix


def _set_body_attitude_from_baseline(
    body: BodyAerodynamics,
    *,
    baseline_sections: list[list[tuple[np.ndarray, np.ndarray]]],
    baseline_spanwise: list[np.ndarray],
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    axes: AxisDefinition,
    reference_point: np.ndarray,
) -> None:
    combined_rotation = _compose_attitude_rotation(
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        axes=axes,
    )
    origin = _as_3vector(reference_point)

    def rotate_point(point: np.ndarray) -> np.ndarray:
        return origin + combined_rotation @ (_as_3vector(point) - origin)

    for wing, wing_sections, spanwise_base in zip(
        body.wings, baseline_sections, baseline_spanwise
    ):
        for section, (le_base, te_base) in zip(wing.sections, wing_sections):
            section.LE_point = rotate_point(le_base)
            section.TE_point = rotate_point(te_base)

        rotated_span = combined_rotation @ spanwise_base
        span_norm = np.linalg.norm(rotated_span)
        if span_norm == 0.0:
            raise ValueError(
                "Combined attitude produced zero spanwise direction vector."
            )
        wing.spanwise_direction = rotated_span / span_norm

    body.geometry_rotation = combined_rotation
    body._build_panels()


def solve_quasi_steady_state(
    body_aero: BodyAerodynamics,
    center_of_gravity: np.ndarray,
    reference_point: np.ndarray,
    system_model: Any,  # AWETrim class
    x_guess: np.ndarray,
    *,
    solver: Solver | None = None,
    bounds_lower: np.ndarray = DEFAULT_BOUNDS_LOWER,
    bounds_upper: np.ndarray = DEFAULT_BOUNDS_UPPER,
    transformation_C_from_VSM: np.ndarray = DEFAULT_TRANSFORMATION_C_FROM_VSM,
    include_gravity: bool = False,
    axes: AxisDefinition = DEFAULT_AXES,
    moment_tolerance: float = 1e-2,
    return_timing_breakdown: bool = False,
) -> tuple[dict, BodyAerodynamics]:
    """
    Solve a quasi-steady trim problem for x=[kite_speed, roll, pitch, yaw, course_rate_body].

    The evaluator callback must return a mapping with:
    - "va": apparent velocity in course frame (shape (3,))
    - "inertial_force": force in course frame (shape (3,))
    Optional keys used for reporting:
    - "gravity_force", "wind_velocity", "kite_velocity", "apparent_velocity"
    """

    if solver is None:
        solver = Solver(
            reference_point=reference_point, gamma_initial_distribution_type="zero"
        )

    bounds_lower = np.asarray(bounds_lower, dtype=float)
    bounds_upper = np.asarray(bounds_upper, dtype=float)
    transformation_C_from_VSM = np.asarray(transformation_C_from_VSM, dtype=float)
    center_of_gravity = _as_3vector(center_of_gravity)
    reference_point = _as_3vector(reference_point)
    x_guess = np.asarray(x_guess, dtype=float)

    if (
        bounds_lower.shape != (5,)
        or bounds_upper.shape != (5,)
        or x_guess.shape != (5,)
    ):
        raise ValueError(
            "x_guess and bounds must be shape (5,) for [vtau, roll, pitch, yaw, course_rate_body]."
        )

    if transformation_C_from_VSM.shape != (3, 3):
        raise ValueError("transformation_C_from_VSM must be shape (3, 3).")

    def evaluate_kinematics(x: np.ndarray) -> dict[str, np.ndarray]:
        kite_speed, _roll, _pitch, _yaw, course_rate_body = x
        system_model.timeder_angle_course_body = course_rate_body
        system_model.speed_tangential = kite_speed

        inertial_force = -system_model.mass_wing * _as_3vector(
            transformation_C_from_VSM @ system_model.acceleration_course_body
        )
        gravity_force = _as_3vector(
            transformation_C_from_VSM @ system_model.force_gravity
        )
        wind_velocity = _as_3vector(
            transformation_C_from_VSM @ system_model.wind.velocity_wind(system_model)
        )
        kite_velocity = _as_3vector(
            transformation_C_from_VSM @ system_model.velocity_kite
        )
        apparent_velocity = _as_3vector(
            transformation_C_from_VSM @ system_model.velocity_apparent_wind
        )

        return {
            "va": apparent_velocity,
            "inertial_force": inertial_force,
            "gravity_force": gravity_force,
            "wind_velocity": wind_velocity,
            "kite_velocity": kite_velocity,
            "apparent_velocity": apparent_velocity,
        }

    timing_counters = {
        "residual_evaluations": 0,
        "residual_total_s": 0.0,
        "body_copy_rotate_s": 0.0,
        "kinematics_s": 0.0,
        "solver_s": 0.0,
        "postprocess_s": 0.0,
    }

    # Residual-component scaling can be enabled after defining moment_residual.
    residual_scales = np.ones(5, dtype=float)
    warm_start_gamma: np.ndarray | None = None
    cached_eval: dict[str, Any] = {"x": None, "payload": None}
    working_body = copy.deepcopy(body_aero)
    baseline_sections: list[list[tuple[np.ndarray, np.ndarray]]] = []
    baseline_spanwise: list[np.ndarray] = []
    for wing in working_body.wings:
        baseline_sections.append(
            [
                (
                    np.asarray(section.LE_point, dtype=float).copy(),
                    np.asarray(section.TE_point, dtype=float).copy(),
                )
                for section in wing.sections
            ]
        )
        baseline_spanwise.append(
            np.asarray(wing.spanwise_direction, dtype=float).copy()
        )

    def moment_residual(x: np.ndarray) -> np.ndarray:
        nonlocal warm_start_gamma

        x = np.asarray(x, dtype=float)
        cached_x = cached_eval["x"]
        if cached_x is not None and np.array_equal(x, cached_x):
            payload = cached_eval["payload"]
            return np.asarray(payload["residual"], dtype=float)

        eval_t0 = perf_counter()
        kite_speed, roll_deg, pitch_deg, yaw_deg, course_rate_body = x

        t0 = perf_counter()
        body = working_body
        _set_body_attitude_from_baseline(
            body,
            baseline_sections=baseline_sections,
            baseline_spanwise=baseline_spanwise,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            axes=axes,
            reference_point=reference_point,
        )
        timing_counters["body_copy_rotate_s"] += perf_counter() - t0

        t0 = perf_counter()
        kin = evaluate_kinematics(x)
        va = _as_3vector(kin["va"])
        inertial_force = _as_3vector(kin["inertial_force"])
        gravity_force = (
            _as_3vector(kin.get("gravity_force", np.zeros(3, dtype=float)))
            if include_gravity
            else np.zeros(3, dtype=float)
        )
        timing_counters["kinematics_s"] += perf_counter() - t0

        aoa_course_deg = np.rad2deg(np.arctan2(va[2], va[0]))
        beta_course_deg = np.rad2deg(np.arctan2(va[1], np.hypot(va[0], va[2])))
        umag = np.linalg.norm(va)

        body.va_initialize(
            Umag=umag,
            angle_of_attack=aoa_course_deg,
            side_slip=beta_course_deg,
            body_rates=course_rate_body,
            body_axis=axes.radial,
            reference_point=reference_point,
            rates_in_body_frame=False,
        )

        t0 = perf_counter()
        res = solver.solve(body)
        timing_counters["solver_s"] += perf_counter() - t0

        cmx = res.get("cmx")
        cmy = res.get("cmy")
        cmz = res.get("cmz")

        fx = res.get("Fx", np.nan)
        fy = res.get("Fy", np.nan)
        fz = res.get("Fz", np.nan)
        total_aero_force = np.array([fx, fy, fz], dtype=float)

        projected_area = body.wings[0].compute_projected_area()
        q_inf = 0.5 * solver.rho * umag**2
        max_chord = max(panel.chord for panel in body.panels)
        denom = q_inf * projected_area * max_chord if projected_area > 0 else 1.0

        moment_vec = np.cross(center_of_gravity - reference_point, inertial_force)
        if include_gravity:
            moment_vec += np.cross(center_of_gravity - reference_point, gravity_force)

        delta_cm = moment_vec / denom
        cmx += delta_cm[0]
        cmy += delta_cm[1]
        cmz += delta_cm[2]

        net_force = total_aero_force + inertial_force + gravity_force
        cfx = np.dot(net_force, axes.course) / (
            0.5 * solver.rho * umag**2 * projected_area
        )
        cfy = np.dot(net_force, axes.normal) / (
            0.5 * solver.rho * umag**2 * projected_area
        )

        t0 = perf_counter()
        residual = np.array([cmx, cmy, cmz, cfx, cfy])
        residual = residual * residual_scales
        timing_counters["postprocess_s"] += perf_counter() - t0
        timing_counters["residual_evaluations"] += 1
        timing_counters["residual_total_s"] += perf_counter() - eval_t0

        cached_eval["x"] = x.copy()
        cached_eval["payload"] = {
            "residual": residual,
            "kin": kin,
            "va": va,
            "umag": umag,
            "course_rate_body": course_rate_body,
            "res": res,
            "gravity_force": gravity_force,
            "inertial_force": inertial_force,
        }

        return residual

    span_global = bounds_upper - bounds_lower
    local_lower = bounds_lower
    local_upper = bounds_upper
    x_start = np.clip(x_guess, local_lower, local_upper)

    opt = least_squares(
        lambda x: moment_residual(x),
        x_start,
        bounds=(local_lower, local_upper),
    )

    cm_best = moment_residual(opt.x)
    cmx, cmy, cmz, cfx, cfy = cm_best
    kite_speed, roll_deg, pitch_deg, yaw_deg, course_rate_body = opt.x

    physical_success = (
        np.abs(cmx) < moment_tolerance
        and np.abs(cmy) < moment_tolerance
        and np.abs(cmz) < moment_tolerance
    )
    if not physical_success:
        print(f"Moment residual not converged. Full residual: {cm_best}")
    opt_success = bool(opt.success)

    payload = (
        cached_eval["payload"] if np.array_equal(opt.x, cached_eval["x"]) else None
    )
    if payload is None:
        _ = moment_residual(opt.x)
        payload = cached_eval["payload"]

    kin = payload["kin"]
    va = _as_3vector(payload["va"])
    aoa_deg = np.rad2deg(np.arctan2(va[2], va[0]))
    beta_deg = np.rad2deg(np.arctan2(va[1], np.hypot(va[0], va[2])))
    umag = float(payload["umag"])

    res = payload["res"]
    aoa_center_chord_deg = float(res.get("alpha_center_chord_deg", aoa_deg))
    beta_center_chord_deg = float(res.get("beta_center_chord_deg", beta_deg))

    fx = res.get("Fx", np.nan)
    fy = res.get("Fy", np.nan)
    fz = res.get("Fz", np.nan)
    total_aero_force = np.array([fx, fy, fz], dtype=float)

    va_unit = va / np.linalg.norm(va)
    lift_dir = axes.radial - np.dot(axes.radial, va_unit) * va_unit
    side_dir = np.cross(lift_dir, va_unit)

    side_aero_force = np.dot(total_aero_force, side_dir)
    lift_aero_force = np.dot(total_aero_force, lift_dir)
    aero_roll_deg = np.rad2deg(np.arctan2(side_aero_force, lift_aero_force))

    inertial_force = _as_3vector(payload["inertial_force"])
    gravity_force = _as_3vector(payload["gravity_force"])

    trim_residual = None
    trim_jacobian = None

    x_cp = res.get("center_of_pressure", np.nan)
    x_cp_arr = np.asarray(x_cp, dtype=float)
    x_cp_point = (
        x_cp_arr if x_cp_arr.size == 3 else np.array([float(x_cp_arr), 0.0, 0.0])
    )

    result = {
        "opt_x": opt.x,
        "cm": np.array([cmx, cmy, cmz], dtype=float),
        "cfx": float(cfx),
        "cfy": float(cfy),
        "side_slip_deg": beta_center_chord_deg,
        "side_slip_course_deg": beta_deg,
        "aero_roll_deg": aero_roll_deg,
        "aoa_deg": aoa_center_chord_deg,
        "aoa_course_deg": aoa_deg,
        "success": opt_success,
        "success_physical": physical_success,
        "gravity_force": gravity_force,
        "inertial_force": inertial_force,
        "cl": res.get("cl"),
        "cd": res.get("cd"),
        "total_aero_force_vec": total_aero_force,
        "x_cp_point": x_cp_point,
        "wind_vel_world": _as_3vector(
            kin.get("wind_velocity", np.zeros(3, dtype=float))
        ),
        "kite_vel_world": _as_3vector(
            kin.get("kite_velocity", np.zeros(3, dtype=float))
        ),
        "va_vel_world": _as_3vector(kin.get("apparent_velocity", va)),
        "Umag": umag,
        "course_axis": axes.course,
        "radial_axis": axes.radial,
        "normal_axis": axes.normal,
        "F_distribution": res.get("F_distribution"),
        "panel_cp_locations": res.get("panel_cp_locations"),
        "alpha_at_ac": res.get("alpha_at_ac"),
        "gamma_distribution": res.get("gamma_distribution"),
        "cl": res.get("cl"),
        "cd": res.get("cd"),
    }

    if return_timing_breakdown:
        residual_total = float(timing_counters["residual_total_s"])
        if residual_total > 0.0:
            timing_counters["solver_share"] = (
                timing_counters["solver_s"] / residual_total
            )
            timing_counters["copy_rotate_share"] = (
                timing_counters["body_copy_rotate_s"] / residual_total
            )
            timing_counters["kinematics_share"] = (
                timing_counters["kinematics_s"] / residual_total
            )
            timing_counters["postprocess_share"] = (
                timing_counters["postprocess_s"] / residual_total
            )
        result["timing_breakdown"] = timing_counters

    return result, body_aero


def compute_quasi_steady_trim_jacobian(
    body_aero: BodyAerodynamics,
    center_of_gravity: np.ndarray,
    reference_point: np.ndarray,
    system_model: Any,
    x_state: np.ndarray,
    *,
    solver: Solver | None = None,
    transformation_C_from_VSM: np.ndarray = DEFAULT_TRANSFORMATION_C_FROM_VSM,
    include_gravity: bool = False,
    axes: AxisDefinition = DEFAULT_AXES,
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute residual and numerical Jacobian at a given quasi-steady state x."""

    if solver is None:
        solver = Solver(
            reference_point=reference_point, gamma_initial_distribution_type="zero"
        )

    transformation_C_from_VSM = np.asarray(transformation_C_from_VSM, dtype=float)
    center_of_gravity = _as_3vector(center_of_gravity)
    reference_point = _as_3vector(reference_point)
    x_state = np.asarray(x_state, dtype=float)

    if x_state.shape != (5,):
        raise ValueError(
            "x_state must be shape (5,) for [vtau, roll, pitch, yaw, course_rate_body]."
        )
    if transformation_C_from_VSM.shape != (3, 3):
        raise ValueError("transformation_C_from_VSM must be shape (3, 3).")

    def evaluate_kinematics(x: np.ndarray) -> dict[str, np.ndarray]:
        kite_speed, _roll, _pitch, _yaw, course_rate_body = x
        system_model.timeder_angle_course_body = course_rate_body
        system_model.speed_tangential = kite_speed

        inertial_force = -system_model.mass_wing * _as_3vector(
            transformation_C_from_VSM @ system_model.acceleration_course_body
        )
        gravity_force = _as_3vector(
            transformation_C_from_VSM @ system_model.force_gravity
        )
        wind_velocity = _as_3vector(
            transformation_C_from_VSM @ system_model.wind.velocity_wind(system_model)
        )
        kite_velocity = _as_3vector(
            transformation_C_from_VSM @ system_model.velocity_kite
        )
        apparent_velocity = _as_3vector(
            transformation_C_from_VSM @ system_model.velocity_apparent_wind
        )

        return {
            "va": apparent_velocity,
            "inertial_force": inertial_force,
            "gravity_force": gravity_force,
            "wind_velocity": wind_velocity,
            "kite_velocity": kite_velocity,
            "apparent_velocity": apparent_velocity,
        }

    def moment_residual(x: np.ndarray) -> np.ndarray:
        kite_speed, roll_deg, pitch_deg, yaw_deg, course_rate_body = x

        body = copy.deepcopy(body_aero)
        _apply_attitude(body, roll_deg, pitch_deg, yaw_deg, axes, reference_point)

        kin = evaluate_kinematics(x)
        va = _as_3vector(kin["va"])
        inertial_force = _as_3vector(kin["inertial_force"])
        gravity_force = (
            _as_3vector(kin.get("gravity_force", np.zeros(3, dtype=float)))
            if include_gravity
            else np.zeros(3, dtype=float)
        )

        aoa_course_deg = np.rad2deg(np.arctan2(va[2], va[0]))
        beta_course_deg = np.rad2deg(np.arctan2(va[1], np.linalg.norm(va[[0, 2]])))
        umag = np.linalg.norm(va)

        body.va_initialize(
            Umag=umag,
            angle_of_attack=aoa_course_deg,
            side_slip=beta_course_deg,
            body_rates=course_rate_body,
            body_axis=axes.radial,
            reference_point=reference_point,
            rates_in_body_frame=False,
        )

        res = solver.solve(body)
        cmx = res.get("cmx")
        cmy = res.get("cmy")
        cmz = res.get("cmz")

        fx = res.get("Fx", np.nan)
        fy = res.get("Fy", np.nan)
        fz = res.get("Fz", np.nan)
        total_aero_force = np.array([fx, fy, fz], dtype=float)

        projected_area = body.wings[0].compute_projected_area()
        q_inf = 0.5 * solver.rho * umag**2
        max_chord = max(panel.chord for panel in body.panels)
        denom = q_inf * projected_area * max_chord if projected_area > 0 else 1.0

        moment_vec = np.cross(center_of_gravity - reference_point, inertial_force)
        if include_gravity:
            moment_vec += np.cross(center_of_gravity - reference_point, gravity_force)

        delta_cm = moment_vec / denom
        cmx += delta_cm[0]
        cmy += delta_cm[1]
        cmz += delta_cm[2]

        net_force = total_aero_force + inertial_force + gravity_force
        cfx = np.dot(net_force, axes.course) / (
            0.5 * solver.rho * umag**2 * projected_area
        )
        cfy = np.dot(net_force, axes.normal) / (
            0.5 * solver.rho * umag**2 * projected_area
        )

        return np.array([cmx, cmy, cmz, cfx, cfy])

    x = np.asarray(x_state, dtype=float)
    residual = moment_residual(x)
    jac = np.zeros((residual.size, x.size))
    for i in range(x.size):
        step = eps * max(1.0, abs(x[i]))
        xp = x.copy()
        xm = x.copy()
        xp[i] += step
        xm[i] -= step
        jac[:, i] = (moment_residual(xp) - moment_residual(xm)) / (2 * step)

    rad_factor = 180.0 / np.pi
    jac[:, 1] *= rad_factor
    jac[:, 2] *= rad_factor

    return residual, jac


def _as_sequence(value: Sequence[float] | float) -> list[float]:
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def run_quasi_steady_sweep(
    *,
    build_body: Callable[[dict[str, float]], BodyAerodynamics],
    system_model: Any,
    center_of_gravity: np.ndarray,
    reference_point: np.ndarray,
    x_guess: np.ndarray,
    principal_axis: str,
    secondary_axis: str,
    sweep_values: Mapping[str, Sequence[float] | float],
    update_system_model: Callable[[Any, dict[str, float]], None] | None = None,
    solver_factory: Callable[[np.ndarray], Solver] | None = None,
    bounds_lower: np.ndarray = DEFAULT_BOUNDS_LOWER,
    bounds_upper: np.ndarray = DEFAULT_BOUNDS_UPPER,
    transformation_C_from_VSM: np.ndarray = DEFAULT_TRANSFORMATION_C_FROM_VSM,
    include_gravity: bool = False,
    axes: AxisDefinition = DEFAULT_AXES,
    moment_tolerance: float = 1e-4,
    max_nfev: int = 400,
    f_scale: float = 0.15,
    return_timing_breakdown: bool = False,
) -> list[dict[str, Any]]:
    """Run a principal/secondary parameter sweep using solve_quasi_steady_state.

    sweep_values should include scalar or sequence values for each variable of interest.
    principal_axis provides the x-axis sweep and secondary_axis defines curve families.
    """

    if principal_axis not in sweep_values:
        raise KeyError(f"principal_axis '{principal_axis}' missing from sweep_values")
    if secondary_axis not in sweep_values:
        raise KeyError(f"secondary_axis '{secondary_axis}' missing from sweep_values")

    principal_values = _as_sequence(sweep_values[principal_axis])
    secondary_values = _as_sequence(sweep_values[secondary_axis])
    if principal_axis == secondary_axis:
        secondary_values = [secondary_values[0]]

    base_values = {key: _as_sequence(value)[0] for key, value in sweep_values.items()}

    current_guess = np.asarray(x_guess, dtype=float).copy()
    rows: list[dict[str, Any]] = []

    for secondary_value in secondary_values:
        current_guess = np.asarray(x_guess, dtype=float).copy()
        for principal_value in principal_values:
            case_values = dict(base_values)
            case_values[principal_axis] = principal_value
            case_values[secondary_axis] = secondary_value

            if update_system_model is not None:
                update_system_model(system_model, case_values)

            body = build_body(case_values)
            solver = (
                solver_factory(reference_point)
                if solver_factory is not None
                else Solver(
                    reference_point=reference_point,
                    gamma_initial_distribution_type="zero",
                )
            )

            result, _ = solve_quasi_steady_state(
                body_aero=body,
                center_of_gravity=center_of_gravity,
                reference_point=reference_point,
                system_model=system_model,
                x_guess=current_guess,
                solver=solver,
                bounds_lower=bounds_lower,
                bounds_upper=bounds_upper,
                transformation_C_from_VSM=transformation_C_from_VSM,
                include_gravity=include_gravity,
                axes=axes,
                moment_tolerance=moment_tolerance,
                return_timing_breakdown=return_timing_breakdown,
            )

            rows.append(
                {
                    "principal_axis": principal_axis,
                    "secondary_axis": secondary_axis,
                    "principal_value": principal_value,
                    "secondary_value": secondary_value,
                    "case_values": case_values,
                    "result": result,
                }
            )

            if result.get("success", False):
                print(
                    f"Sweep case {principal_axis}={principal_value}, {secondary_axis}={secondary_value} succeeded with opt_x={result['opt_x']}"
                )
                current_guess = np.asarray(result["opt_x"], dtype=float)

    return rows


def quasi_steady_sweep_rows_to_dataframe(sweep_rows: Sequence[Mapping[str, Any]]):
    """Convert run_quasi_steady_sweep output rows into a flat pandas DataFrame."""
    import pandas as pd

    rows = []
    for row in sweep_rows:
        result = row["result"]
        opt_x = np.asarray(result["opt_x"], dtype=float)
        cmx, cmy, cmz = np.asarray(result["cm"], dtype=float)
        rows.append(
            {
                "principal_axis": row["principal_axis"],
                "secondary_axis": row["secondary_axis"],
                "principal_value": float(row["principal_value"]),
                "secondary_value": float(row["secondary_value"]),
                "kite_speed": float(opt_x[0]),
                "roll_deg": float(opt_x[1]),
                "pitch_deg": float(opt_x[2]),
                "yaw_deg": float(opt_x[3]),
                "course_rate_rad_s": float(opt_x[4]),
                "aoa_center_deg": float(result["aoa_deg"]),
                "aoa_course_deg": float(result["aoa_course_deg"]),
                "beta_center_deg": float(result["side_slip_deg"]),
                "beta_course_deg": float(result["side_slip_course_deg"]),
                "aero_roll_deg": float(result["aero_roll_deg"]),
                "cl": float(result["cl"]),
                "cd": float(result["cd"]),
                "cmx": float(cmx),
                "cmy": float(cmy),
                "cmz": float(cmz),
                "norm_cm": float(np.linalg.norm([cmx, cmy, cmz])),
                "success": bool(result["success"]),
            }
        )

    return pd.DataFrame(rows)


def plot_quasi_steady_sweep_dataframe(
    df,
    principal_axis: str,
    secondary_axis: str,
    *,
    show: bool = True,
):
    """Plot standard quasi-steady sweep figures from a dataframe."""
    import matplotlib.pyplot as plt

    if df.empty:
        return

    x_col = "principal_value"
    line_col = "secondary_value"

    fig1, ax1 = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for sec_val in sorted(df[line_col].dropna().unique()):
        sub = df[df[line_col] == sec_val].sort_values(x_col)
        label = f"{secondary_axis}={sec_val:.3f}"
        ax1[0].plot(sub[x_col], sub["course_rate_rad_s"], "o-", label=label)
        ax1[1].plot(sub[x_col], sub["beta_center_deg"], "o-", label=label)
        ax1[2].plot(sub[x_col], sub["aero_roll_deg"], "o-", label=label)

    ax1[0].axhline(0, color="k", linewidth=0.8)
    ax1[0].set_ylabel("course rate [rad/s]")
    ax1[0].legend()
    ax1[1].set_ylabel("Sideslip center [deg]")
    ax1[2].set_xlabel(principal_axis)
    ax1[2].set_ylabel("Aero roll angle [deg]")
    fig1.suptitle(
        f"Quasi-steady sweep (x={principal_axis}, series={secondary_axis})",
        y=0.995,
    )
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for sec_val in sorted(df[line_col].dropna().unique()):
        sub = df[df[line_col] == sec_val].sort_values(x_col)
        label = f"{secondary_axis}={sec_val:.3f}"
        ax2[0].plot(sub[x_col], sub["aoa_center_deg"], "o-", label=label)
        ax2[1].plot(sub[x_col], sub["cl"], "o-", label=label)
        ax2[2].plot(sub[x_col], sub["cd"], "o-", label=label)

    ax2[0].set_ylabel("AoA center [deg]")
    ax2[0].legend()
    ax2[1].set_ylabel("Lift coeff")
    ax2[2].set_ylabel("Drag coeff")
    ax2[2].set_xlabel(principal_axis)
    fig2.tight_layout()

    if show:
        plt.show()


def linearize_fast_dynamics_from_trim_jacobian(
    jacobian: np.ndarray,
    x_state: np.ndarray,
    *,
    mass: float = 15.0,
    Ixx: float = 100.0,
    Iyy: float = 19.43,
    Izz: float = 100.0,
    rho: float = 1.225,
    reference_area: float = 1.0,
    reference_chord: float = 1.0,
    radial_distance: float | None = None,
) -> dict[str, Any]:
    """Build fast-dynamics linearization blocks and timescales from trim Jacobian.

    Notes:
    - The quasi-steady residual Jacobian is ordered as [cmx, cmy, cmz, cfx/scale, cfy/scale]
      wrt [v_tau, roll, pitch, yaw, course_rate_body].
    - Longitudinal block uses states [v_tau, pitch_rad] as a proxy for [v_tau, alpha].
    - Lateral block uses states [yaw_rad, course_rate_body, roll_rad] as a proxy for
      [beta_s, chi_dot_b, phi_a].
    """
    jacobian = np.asarray(jacobian, dtype=float)
    x_state = np.asarray(x_state, dtype=float)

    if jacobian.shape != (5, 5):
        raise ValueError("jacobian must be shape (5,5).")
    if x_state.shape != (5,):
        raise ValueError("x_state must be shape (5,).")
    if mass <= 0.0 or Ixx <= 0.0 or Iyy <= 0.0 or Izz <= 0.0:
        raise ValueError("mass and inertias must be strictly positive.")
    if reference_area <= 0.0 or reference_chord <= 0.0:
        raise ValueError(
            "reference_area and reference_chord must be strictly positive."
        )

    v_tau = float(x_state[0])
    q_ref = 0.5 * rho * v_tau**2
    force_factor = q_ref * reference_area
    moment_factor = force_factor * reference_chord

    rad_factor = 180.0 / np.pi

    dcmx_dx = jacobian[0, :]
    dcmy_dx = jacobian[1, :]
    dcmz_dx = jacobian[2, :]
    dcfx_dx = jacobian[3, :]
    dcfy_dx = jacobian[4, :]

    dcmx_dx_rad = dcmx_dx.copy()
    dcmy_dx_rad = dcmy_dx.copy()
    dcmz_dx_rad = dcmz_dx.copy()
    dcfx_dx_rad = dcfx_dx.copy()
    dcfy_dx_rad = dcfy_dx.copy()
    dcmx_dx_rad[3] *= rad_factor
    dcmy_dx_rad[3] *= rad_factor
    dcmz_dx_rad[3] *= rad_factor
    dcfx_dx_rad[3] *= rad_factor
    dcfy_dx_rad[3] *= rad_factor

    dFchi_dx = force_factor * dcfx_dx_rad
    dFeta_dx = force_factor * dcfy_dx_rad
    dMx_dx = moment_factor * dcmx_dx_rad
    dMy_dx = moment_factor * dcmy_dx_rad
    dMz_dx = moment_factor * dcmz_dx_rad

    idx_long = [0, 2]
    A_long = np.array(
        [
            [-dFchi_dx[idx_long[0]] / mass, dFchi_dx[idx_long[1]] / mass],
            [-dMy_dx[idx_long[0]] / Iyy, dMy_dx[idx_long[1]] / Iyy],
        ],
        dtype=float,
    )

    idx_lat = [3, 4, 1]
    A_lat = np.array(
        [
            [
                dFeta_dx[idx_lat[0]] / mass,
                dFeta_dx[idx_lat[1]] / mass,
                dFeta_dx[idx_lat[2]] / mass,
            ],
            [
                dMz_dx[idx_lat[0]] / Izz,
                dMz_dx[idx_lat[1]] / Izz,
                dMz_dx[idx_lat[2]] / Izz,
            ],
            [
                dMx_dx[idx_lat[0]] / Ixx,
                dMx_dx[idx_lat[1]] / Ixx,
                dMx_dx[idx_lat[2]] / Ixx,
            ],
        ],
        dtype=float,
    )

    eig_long, vec_long = np.linalg.eig(A_long)
    eig_lat, vec_lat = np.linalg.eig(A_lat)

    def _timescales(eigvals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        real_parts = np.real(eigvals)
        T_fast = np.full(real_parts.shape, np.inf, dtype=float)
        nonzero = np.abs(real_parts) > 1e-12
        T_fast[nonzero] = 1.0 / np.abs(real_parts[nonzero])

        if radial_distance is None:
            epsilon = np.full(real_parts.shape, np.nan, dtype=float)
        else:
            T_slow = float(radial_distance) / max(abs(v_tau), 1e-8)
            epsilon = T_fast / T_slow
        return T_fast, epsilon

    T_long, eps_long = _timescales(eig_long)
    T_lat, eps_lat = _timescales(eig_lat)

    return {
        "A_long": A_long,
        "A_lateral": A_lat,
        "eig_long": eig_long,
        "eig_lateral": eig_lat,
        "vec_long": vec_long,
        "vec_lateral": vec_lat,
        "vec_lat": vec_lat,
        "Tfast_long": T_long,
        "Tfast_lateral": T_lat,
        "eps_fast_long": eps_long,
        "eps_fast_lateral": eps_lat,
        "epsilon_long": eps_long,
        "epsilon_lateral": eps_lat,
        "stable_long": bool(np.all(np.real(eig_long) < 0.0)),
        "stable_lateral": bool(np.all(np.real(eig_lat) < 0.0)),
    }


def compute_quasi_steady_fast_timescales(
    body_aero: BodyAerodynamics,
    center_of_gravity: np.ndarray,
    reference_point: np.ndarray,
    system_model: Any,
    x_state: np.ndarray,
    *,
    solver: Solver | None = None,
    transformation_C_from_VSM: np.ndarray = DEFAULT_TRANSFORMATION_C_FROM_VSM,
    include_gravity: bool = False,
    axes: AxisDefinition = DEFAULT_AXES,
    eps: float = 1e-4,
    mass: float = 15.0,
    Ixx: float = 100.0,
    Iyy: float = 19.43,
    Izz: float = 100.0,
    radial_distance: float | None = None,
) -> dict[str, Any]:
    """Compute longitudinal/lateral fast-timescale metrics around a trim state."""
    residual, jacobian = compute_quasi_steady_trim_jacobian(
        body_aero=body_aero,
        center_of_gravity=center_of_gravity,
        reference_point=reference_point,
        system_model=system_model,
        x_state=x_state,
        solver=solver,
        transformation_C_from_VSM=transformation_C_from_VSM,
        include_gravity=include_gravity,
        axes=axes,
        eps=eps,
    )

    reference_area = float(body_aero.wings[0].compute_projected_area())
    reference_chord = float(max(panel.chord for panel in body_aero.panels))

    if radial_distance is None and hasattr(system_model, "distance_radial"):
        try:
            radial_distance = float(system_model.distance_radial)
        except (TypeError, ValueError):
            radial_distance = None

    linearized = linearize_fast_dynamics_from_trim_jacobian(
        jacobian=jacobian,
        x_state=x_state,
        mass=mass,
        Ixx=Ixx,
        Iyy=Iyy,
        Izz=Izz,
        rho=solver.rho if solver is not None else 1.225,
        reference_area=reference_area,
        reference_chord=reference_chord,
        radial_distance=radial_distance,
    )
    linearized["residual"] = residual
    linearized["jacobian"] = jacobian
    return linearized
