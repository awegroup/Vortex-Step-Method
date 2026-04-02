from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

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
    body.rotate(angle_deg=roll_deg, axis=axes.course, point=reference_point)
    body.rotate(angle_deg=pitch_deg, axis=axes.normal, point=reference_point)
    body.rotate(angle_deg=yaw_deg, axis=axes.radial, point=reference_point)


def solve_quasi_steady_state(
    body_aero: BodyAerodynamics,
    center_of_gravity: np.ndarray,
    reference_point: np.ndarray,
    system_model: Any,
    x_guess: np.ndarray,
    *,
    solver: Solver | None = None,
    bounds_lower: np.ndarray = DEFAULT_BOUNDS_LOWER,
    bounds_upper: np.ndarray = DEFAULT_BOUNDS_UPPER,
    transformation_C_from_VSM: np.ndarray = DEFAULT_TRANSFORMATION_C_FROM_VSM,
    include_gravity: bool = False,
    axes: AxisDefinition = DEFAULT_AXES,
    moment_tolerance: float = 1e-3,
    force_residual_scale: float = 1.0,
    window_fraction: float = 0.15,
    max_nfev: int = 400,
    f_scale: float = 0.15,
) -> dict:
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

        return np.array(
            [cmx, cmy, cmz, cfx / force_residual_scale, cfy / force_residual_scale]
        )

    def numerical_jacobian(
        func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-4
    ):
        x = np.asarray(x, dtype=float)
        f0 = func(x)
        jac = np.zeros((f0.size, x.size))
        for i in range(x.size):
            step = eps * max(1.0, abs(x[i]))
            xp = x.copy()
            xm = x.copy()
            xp[i] += step
            xm[i] -= step
            jac[:, i] = (func(xp) - func(xm)) / (2 * step)
        return f0, jac

    span_global = bounds_upper - bounds_lower
    local_lower = np.maximum(bounds_lower, x_guess - window_fraction * span_global)
    local_upper = np.minimum(bounds_upper, x_guess + window_fraction * span_global)
    x_start = np.clip(x_guess, local_lower, local_upper)

    opt = least_squares(
        lambda x: moment_residual(x),
        x_start,
        bounds=(local_lower, local_upper),
        xtol=1e-6,
        ftol=1e-6,
        gtol=1e-6,
        max_nfev=max_nfev,
        loss="soft_l1",
        f_scale=f_scale,
        verbose=0,
    )

    cm_best = moment_residual(opt.x)
    cmx, cmy, cmz, cfx, cfy = cm_best

    physical_success = (
        np.abs(cmx) < moment_tolerance
        and np.abs(cmy) < moment_tolerance
        and np.abs(cmz) < moment_tolerance
    )
    opt_success = bool(opt.success and physical_success)

    kite_speed, roll_deg, pitch_deg, yaw_deg, course_rate_body = opt.x
    body = copy.deepcopy(body_aero)
    _apply_attitude(body, roll_deg, pitch_deg, yaw_deg, axes, reference_point)

    kin = evaluate_kinematics(opt.x)
    va = _as_3vector(kin["va"])
    aoa_deg = np.rad2deg(np.arctan2(va[2], va[0]))
    beta_deg = np.rad2deg(np.arctan2(va[1], np.linalg.norm(va[[0, 2]])))
    umag = np.linalg.norm(va)

    body.va_initialize(
        Umag=umag,
        angle_of_attack=aoa_deg,
        side_slip=beta_deg,
        body_rates=course_rate_body,
        body_axis=axes.radial,
        reference_point=reference_point,
        rates_in_body_frame=False,
    )

    res = solver.solve(body)
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

    inertial_force = _as_3vector(kin["inertial_force"])
    gravity_force = (
        _as_3vector(kin.get("gravity_force", np.zeros(3, dtype=float)))
        if include_gravity
        else np.zeros(3, dtype=float)
    )

    trim_residual = None
    trim_jacobian = None
    if opt_success:
        trim_residual, trim_jacobian = numerical_jacobian(moment_residual, opt.x)
        rad_factor = 180.0 / np.pi
        trim_jacobian[:, 1] *= rad_factor
        trim_jacobian[:, 2] *= rad_factor

    x_cp = res.get("center_of_pressure", np.nan)
    x_cp_arr = np.asarray(x_cp, dtype=float)
    x_cp_point = (
        x_cp_arr if x_cp_arr.size == 3 else np.array([float(x_cp_arr), 0.0, 0.0])
    )

    return {
        "opt_x": opt.x,
        "cm": np.array([cmx, cmy, cmz], dtype=float),
        "side_slip_deg": beta_center_chord_deg,
        "side_slip_course_deg": beta_deg,
        "aero_roll_deg": aero_roll_deg,
        "aoa_deg": aoa_center_chord_deg,
        "aoa_course_deg": aoa_deg,
        "success": opt_success,
        "gravity_force": gravity_force,
        "inertial_force": inertial_force,
        "cl": res.get("cl", np.nan),
        "cd": res.get("cd", np.nan),
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
        "trim_residual": trim_residual,
        "trim_jacobian": trim_jacobian,
    }
