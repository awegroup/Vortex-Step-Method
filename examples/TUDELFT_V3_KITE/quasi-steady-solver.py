"""Sweep tilt and airspeed, optimize AoA/rot/course to zero moments, plot course/rot vs speed."""

from pathlib import Path
import copy
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import least_squares
from VSM.plot_geometry_matplotlib import plot_trim_geometry
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from helpers import transformation_C_from_W, transformation_C_from_CR
from awetrim.kinematics.Kinematics import KiteKinematics
from awetrim.system.system_model import SystemModel

R_C_CR = transformation_C_from_CR()


def as_3vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != 3:
        raise ValueError(f"Expected a 3-vector, got shape {np.asarray(value).shape}")
    return vector


PROJECT_DIR = Path(__file__).resolve().parents[2]

include_gravity = True  # toggle to include gravity in moments and plots

# Bounds and defaults (aoa, sideslip, course_rate_body)
kite_speed_bounds = (2.0, 80.0)  # m/s
aoa_bounds_deg = (2, 12)
beta_bounds_deg = (-10, 10)
course_bounds = (-3, 3)  # rad/s
roll_bounds = (
    -10,
    10,
)  # deg, small roll allowed for numerical reasons; not a physical roll
x0 = np.array([28, 8, 0, 0, 0], dtype=float)

spanwise_panel_distribution = "uniform"
ml_models_dir = PROJECT_DIR / "data" / "ml_models"
cad_derived_geometry_dir = (
    PROJECT_DIR / "data" / "TUDELFT_V3_KITE" / "CAD_derived_geometry"
)
geometry_yaml = cad_derived_geometry_dir / "aero_geometry_CAD_CFD_polars.yaml"
reference_point = np.array([0, 0.0, 0.0])
center_gravity = np.array([0.5, 0.0, 10.0])


def build_base_body(tilt_deg: float, n_panels: int) -> BodyAerodynamics:
    base = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=geometry_yaml,
        spanwise_panel_distribution=spanwise_panel_distribution,
        # ml_models_dir=ml_models_dir,
        bridle_path=(
            cad_derived_geometry_dir / "struc_geometry_manually_adjusted.yaml"
        ),
    )
    base.rotate(
        angle_deg=tilt_deg,
        axis=np.array([1.0, 0.0, 0.0]),
        point=np.array([0.5, 0.0, 7.0]),
    )
    return base


def compute_force_frame(
    aoa_deg: float,
    beta_deg: float,
    kite_speed: float,
    system_model: SystemModel,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return body-aligned apparent wind plus reference axes used for inertial force estimates."""

    aoa_rad = np.deg2rad(aoa_deg)
    beta_rad = np.deg2rad(beta_deg)
    va_unit = np.array(
        [
            np.cos(aoa_rad) * np.cos(beta_rad),
            np.sin(beta_rad),
            np.sin(aoa_rad) * np.cos(beta_rad),
        ],
        dtype=float,
    )

    system_model.speed_tangential = kite_speed

    R_C_CR = transformation_C_from_CR()

    ref_va = as_3vector(R_C_CR @ system_model.velocity_apparent_wind)
    ref_va_unit = ref_va / np.linalg.norm(ref_va)
    ref_z = np.array([0.0, 0.0, 1.0], dtype=float)
    axis = np.cross(ref_va_unit, va_unit)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        R = np.eye(3)
    else:
        axis_unit = axis / axis_norm
        cos_a = np.clip(np.dot(ref_va_unit, va_unit), -1.0, 1.0)
        sin_a = axis_norm
        K = np.array(
            [
                [0, -axis_unit[2], axis_unit[1]],
                [axis_unit[2], 0, -axis_unit[0]],
                [-axis_unit[1], axis_unit[0], 0],
            ],
            dtype=float,
        )
        R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)

    course_axis = R @ ref_z
    return (
        np.linalg.norm(ref_va),
        va_unit,
        course_axis,
        R,
    )


def solve_quasi_steady_state(
    base_body: BodyAerodynamics,
    x_guess: np.ndarray,
    system_model: SystemModel,
):
    """Solve the quasi-steady trim state for a given body and system model."""
    solver = Solver(
        reference_point=reference_point, gamma_initial_distribution_type="zero"
    )

    def moment_residual(x: np.ndarray, system_model: SystemModel) -> np.ndarray:
        kite_speed, aoa_deg, beta_deg, course_rate_body, roll = x
        body = copy.deepcopy(base_body)
        (
            Umag,
            va_unit,
            course_axis_world,
            R,
        ) = compute_force_frame(
            aoa_deg,
            beta_deg,
            kite_speed,
            system_model,
        )

        body.rotate(angle_deg=roll, axis=-va_unit, point=reference_point)
        body.va_initialize(
            Umag=Umag,
            angle_of_attack=aoa_deg,
            side_slip=beta_deg,
            body_rates=course_rate_body,
            body_axis=course_axis_world,
            reference_point=reference_point,
            rates_in_body_frame=False,
        )
        res = solver.solve(body)

        cmx = res.get("cmx")
        cmy = res.get("cmy")
        cmz = res.get("cmz")
        Fx = res.get("Fx", np.nan)
        Fy = res.get("Fy", np.nan)
        Fz = res.get("Fz", np.nan)
        total_aero_force_vec = np.array([Fx, Fy, Fz])

        course_world = np.array([1.0, 0.0, 0.0])
        course_dir = R @ course_world
        projected_area = body.wings[0].compute_projected_area()

        R_C_CR = transformation_C_from_CR()

        system_model.timeder_angle_course_body = course_rate_body
        system_model.speed_tangential = kite_speed

        acc = as_3vector(R_C_CR @ system_model.acceleration_course_body)
        inertial_force = R @ (-system_model.mass_wing * acc)
        r = center_gravity - reference_point
        moment_vec = np.cross(r, inertial_force)

        gravity_force = np.zeros(3, dtype=float)
        if include_gravity:
            gravity_force = R @ as_3vector(R_C_CR @ system_model.force_gravity)
            moment_vec += np.cross(r, gravity_force)

        q_inf = 0.5 * solver.rho * Umag**2

        max_chord = max(panel.chord for panel in body.panels)
        denom = q_inf * projected_area * max_chord if projected_area > 0 else 1.0
        delta_cm = moment_vec / denom
        cmx += delta_cm[0]
        cmy += delta_cm[1]
        cmz += delta_cm[2]
        force_course = np.dot(
            total_aero_force_vec + gravity_force + inertial_force, course_dir
        )
        cfx = (force_course) / (0.5 * solver.rho * Umag**2 * projected_area)

        normal_axis = R @ np.array([0.0, 1.0, 0.0])

        force_normal = np.dot(
            total_aero_force_vec + gravity_force + inertial_force, normal_axis
        )

        cfy = (force_normal) / (0.5 * solver.rho * Umag**2 * projected_area)

        return np.array([cmx, cmy, cmz, cfx / 10, cfy / 10])

    # TODO: Correct jacobian calculation. With vtau the aoa sould change. The apparent velocity is made of kite velocity and wind velocity, so changing kite speed should change the apparent velocity and thus the aoa. This is currently not reflected in the jacobian, which is likely why the eigenvalues are not showing expected trends. The same applies to beta and course_rate_body, which should also affect the apparent velocity direction and thus the aoa and sideslip angle.
    def numerical_jacobian(func, x, eps=1e-4):
        x = np.asarray(x, dtype=float)
        f0 = func(x, x)
        m = f0.size
        n = x.size
        J = np.zeros((m, n))
        for i in range(n):
            step = eps * max(1.0, abs(x[i]))
            xp = x.copy()
            xm = x.copy()
            xp[i] += step
            xm[i] -= step
            fp = func(xp, x)
            fm = func(xm, x)
            J[:, i] = (fp - fm) / (2 * step)
        return f0, J

    def moment_residual_with_va_angles(x: np.ndarray, x_trim: np.ndarray) -> np.ndarray:
        kite_speed_val, aoa_deg_val, beta_deg_val, course_rate_body_val, roll_val = x
        (
            kite_speed_trim,
            aoa_deg_trim,
            beta_deg_trim,
            course_rate_body_trim,
            roll_trim,
        ) = x_trim

        system_model.speed_tangential = kite_speed_val

        va_loc = as_3vector(R_C_CR @ system_model.velocity_apparent_wind)
        _, _, _, R_trim = compute_force_frame(
            aoa_deg_trim,
            beta_deg_trim,
            kite_speed_trim,
            system_model,
        )
        va_body = R_trim @ va_loc
        if kite_speed_val == kite_speed_trim:
            aoa_deg_adj = aoa_deg_val
            beta_deg_adj = beta_deg_val
        else:
            aoa_deg_adj = np.rad2deg(np.arctan2(va_body[2], va_body[0]))
            beta_deg_adj = np.rad2deg(
                np.arctan2(va_body[1], np.linalg.norm(va_body[[0, 2]]))
            )
        x_adjusted = np.array(
            [kite_speed_val, aoa_deg_adj, beta_deg_adj, course_rate_body_val, roll_val]
        )
        residual = moment_residual(x_adjusted, system_model)
        return residual

    bounds_lower = np.array(
        [
            kite_speed_bounds[0],
            aoa_bounds_deg[0],
            beta_bounds_deg[0],
            course_bounds[0],
            roll_bounds[0],
        ]
    )
    bounds_upper = np.array(
        [
            kite_speed_bounds[1],
            aoa_bounds_deg[1],
            beta_bounds_deg[1],
            course_bounds[1],
            roll_bounds[1],
        ]
    )

    moment_tol = 1e-3
    span_global = bounds_upper - bounds_lower

    def run_once(b_lo, b_hi, x_start, max_nfev, f_scale):
        res = least_squares(
            moment_residual,
            x_start,
            bounds=(b_lo, b_hi),
            xtol=1e-6,
            ftol=1e-6,
            gtol=1e-6,
            max_nfev=max_nfev,
            loss="soft_l1",
            f_scale=f_scale,
            verbose=0,
            args=(system_model,),
        )
        cm_vec = moment_residual(res.x, system_model)
        norm_cm = np.linalg.norm(cm_vec)
        return res, cm_vec, norm_cm

    window_frac = 0.15
    local_lower = np.maximum(bounds_lower, x_guess - window_frac * span_global)
    local_upper = np.minimum(bounds_upper, x_guess + window_frac * span_global)
    x_start_local = np.clip(x_guess, local_lower, local_upper)

    res_local, cm_local, norm_local = run_once(
        local_lower, local_upper, x_start_local, max_nfev=400, f_scale=0.15
    )

    opt = res_local
    cm_best = cm_local
    norm_best = norm_local

    cmx, cmy, cmz, cfx, cfy = cm_best
    moment_tol = 1e-3
    physical_success = (
        np.abs(cmx) < moment_tol
        and np.abs(cmy) < moment_tol
        and np.abs(cmz) < moment_tol
    )
    opt_success = bool(opt.success and physical_success)
    if not opt_success:
        print(
            f"Local optimum or failure: |cm|={norm_best:.3e} (cmx={cmx:.3e}, cmy={cmy:.3e}, cmz={cmz:.3e}, cfx={cfx:.3e}, cfy={cfy:.3e}), success_flag={opt.success}"
        )
        print("Best local guess:", opt.x)

    kite_speed, aoa_deg, beta_deg, course_rate_body, roll = opt.x
    body = copy.deepcopy(base_body)
    (
        Umag,
        va_unit,
        course_axis_world,
        R,
    ) = compute_force_frame(
        aoa_deg,
        beta_deg,
        kite_speed,
        system_model,
    )
    course_axis_body = body.geometry_rotation.T @ course_axis_world
    body.va_initialize(
        Umag=Umag,
        angle_of_attack=aoa_deg,
        side_slip=beta_deg,
        body_rates=course_rate_body,
        body_axis=course_axis_body,
        reference_point=reference_point,
        rates_in_body_frame=False,
    )
    print(
        f"Optimized for Umag={Umag} m/s: aoa={aoa_deg:.2f} deg, beta={beta_deg:.2f} deg, course_rate_body={course_rate_body:.2f} rad/s, roll={roll:.2f} deg -> "
        f"cm=({cmx:.4f}, {cmy:.4f}, {cmz:.4f}), {cfx:.4f}, {cfy:.4f},  success={opt_success}"
    )

    def project_onto_plane(vec, normal):
        normal_unit = normal / np.linalg.norm(normal)
        return vec - np.dot(vec, normal_unit) * normal_unit

    # Build lift/side directions in the reference frame and project inertial/gravity components.

    radial_dir = R @ np.array([0, 0.0, 1.0])
    lift_dir = project_onto_plane(radial_dir, va_unit)
    side_dir = np.cross(lift_dir, va_unit)
    res = solver.solve(body)
    cl = res.get("cl", np.nan)
    cs = res.get("cs", np.nan)
    cd = res.get("cd", np.nan)
    Fx = res.get("Fx", np.nan)
    Fy = res.get("Fy", np.nan)
    Fz = res.get("Fz", np.nan)
    x_cp = res.get("center_of_pressure", np.nan)
    total_aero_force_vec = np.array([Fx, Fy, Fz])

    side_aero_force = np.dot(total_aero_force_vec, side_dir)
    lift_aero_force = np.dot(total_aero_force_vec, lift_dir)
    aero_roll = np.arctan2(side_aero_force, lift_aero_force)
    aero_roll_deg = np.rad2deg(aero_roll)

    side_slip_deg = beta_deg

    system_model.speed_tangential = kite_speed

    system_model.timeder_angle_course_body = course_rate_body
    R_C_CR = transformation_C_from_CR()
    inertial_force_plot = R @ (
        -system_model.mass_wing
        * as_3vector(R_C_CR @ system_model.acceleration_course_body)
    )
    gravity_force_plot = R @ as_3vector(R_C_CR @ system_model.force_gravity)
    wind_vel_world = as_3vector(R_C_CR @ system_model.wind.velocity_wind(system_model))
    kite_vel_world = as_3vector(R_C_CR @ system_model.velocity_kite)
    va_vel_world = as_3vector(R_C_CR @ system_model.velocity_apparent_wind)

    course_axis = R @ np.array([1.0, 0.0, 0.0])
    radial_axis = R @ np.array([0.0, 0.0, 1.0])
    normal_axis = -np.cross(course_axis, radial_axis)
    if np.linalg.norm(normal_axis) > 0:
        normal_axis = normal_axis / np.linalg.norm(normal_axis)

    x_cp_arr = np.asarray(x_cp, dtype=float)
    x_cp_point = (
        x_cp_arr if x_cp_arr.size == 3 else np.array([float(x_cp_arr), 0.0, 0.0])
    )

    trim_residual = None
    trim_jacobian = None
    if opt_success:
        trim_residual, trim_jacobian = numerical_jacobian(
            moment_residual_with_va_angles, opt.x
        )
        rad_factor = 180.0 / np.pi
        trim_jacobian[:, 1] *= rad_factor  # aoa deg -> rad
        trim_jacobian[:, 2] *= rad_factor  # beta deg -> rad

    result = {
        "opt_x": opt.x,
        "cm": np.array([cmx, cmy, cmz]),
        "side_slip_deg": side_slip_deg,
        "aero_roll_deg": aero_roll_deg,
        "aoa_deg": aoa_deg,
        "success": opt_success,
        "gravity_force": gravity_force_plot,
        "inertial_force": inertial_force_plot,
        "cl": cl,
        "cd": cd,
        "total_aero_force_vec": total_aero_force_vec,
        "x_cp_point": x_cp_point,
        "wind_vel_world": wind_vel_world,
        "kite_vel_world": kite_vel_world,
        "va_vel_world": va_vel_world,
        "R": R,
        "Umag": Umag,
        "course_axis_world": course_axis_world,
        "course_axis": course_axis,
        "radial_axis": radial_axis,
        "normal_axis": normal_axis,
        "trim_residual": trim_residual,
        "trim_jacobian": trim_jacobian,
    }
    return result


def main():
    tilt = 5  # deg
    system_model = SystemModel()
    system_model.mass_wing = 10.0  # kg
    system_model.angle_elevation = np.deg2rad(0)
    system_model.angle_azimuth = np.deg2rad(0)
    system_model.angle_course = np.deg2rad(90)
    system_model.speed_radial = 0.0
    system_model.distance_radial = 200
    system_model.wind.speed_wind_ref = 4.0
    system_model.timeder_speed_tangential = 0.0
    system_model.timeder_speed_radial = 0.0

    x_guess = np.array([28, 8, 0, 0, 0], dtype=float)

    n_panels = 18

    base_body = build_base_body(tilt, n_panels)

    start_time = time.time()
    opt_result = solve_quasi_steady_state(
        base_body,
        x_guess,
        system_model=system_model,
    )
    end_time = time.time()
    print(f"Optimization took {end_time - start_time:.2f} seconds.")

    jac = opt_result.get("trim_jacobian")
    if jac is not None:
        row_labels = ["cmx", "cmy", "cmz", "cfx", "cfy"]
        col_labels = [
            "d/d_vtau (kite_speed)",
            "d/d_aoa_rad",
            "d/d_beta_rad",
            "d/d_chi_dot (rad/s)",
            "d/d_roll_rad",
        ]
        header = ["residual\\wrt"] + col_labels
        col_widths = [max(len(h), 13) for h in header]

        def fmt_row(label, values):
            cells = [label.ljust(col_widths[0])]
            for j, val in enumerate(values):
                if abs(val) < 1e-8:
                    val = 0.0
                cells.append(f"{val: .3e}".rjust(col_widths[j + 1]))
            return "  ".join(cells)

        header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(header))
        print("Jacobian near trim (central diff):")
        print(header_line)
        for i, label in enumerate(row_labels):
            print(fmt_row(label, jac[i, :]))


if __name__ == "__main__":
    main()
