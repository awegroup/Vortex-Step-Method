"""Solve a single quasi-steady state and animate linearized stability modes.

This script is intentionally single-case (no sweep) to make it easier to inspect
the trim solution and the associated fast-mode eigenstructure.
"""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.quasi_steady_state import (
    DEFAULT_AXES,
    compute_quasi_steady_fast_timescales,
    solve_quasi_steady_state,
)
from awetrim.system.system_model import SystemModel
from awetrim.system.tether import RigidLumpedTether


PROJECT_DIR = Path(__file__).resolve().parents[2]

SPANWISE_PANEL_DISTRIBUTION = "uniform"
GEOMETRY_YAML = (
    PROJECT_DIR
    / "data"
    / "TUDELFT_V3_KITE"
    / "CAD_derived_geometry"
    / "aero_geometry_CAD_CFD_polars.yaml"
)
REFERENCE_POINT = np.array([0.0, 0.0, 0.0], dtype=float)
CENTER_OF_GRAVITY = np.array([0.5, 0.0, 5.0], dtype=float)

MOMENT_TOLERANCE = 1e-6
FORCE_TOLERANCE = MOMENT_TOLERANCE


def build_base_body(tilt_deg: float, n_panels: int) -> BodyAerodynamics:
    body = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=GEOMETRY_YAML,
        spanwise_panel_distribution=SPANWISE_PANEL_DISTRIBUTION,
        bridle_path=None,
    )
    body.rotate(
        angle_deg=tilt_deg,
        axis=DEFAULT_AXES.course,
        point=np.array([0.5, 0.0, 7.0]),
    )
    return body


def build_system_model(case_values: dict[str, float]) -> SystemModel:
    system = SystemModel(tether=RigidLumpedTether(diameter=1e-6))
    system.mass_wing = 0.0
    system.angle_elevation = np.deg2rad(case_values["elevation_deg"])
    system.angle_azimuth = np.deg2rad(case_values["azimuth_deg"])
    system.angle_course = np.deg2rad(case_values["course_deg"])
    system.speed_radial = case_values["radial_speed"]
    system.distance_radial = case_values["distance_radial"]
    system.wind.speed_wind_ref = case_values["wind_speed"]
    system.timeder_speed_tangential = 0.0
    system.timeder_speed_radial = 0.0
    return system


def solve_single_quasi_steady_state(
    case_values: dict[str, float],
    x_guess: np.ndarray,
    *,
    n_panels: int = 18,
    include_gravity: bool = True,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Solve one quasi-steady trim state and return trim and timescale data."""

    bounds_lower = np.array([2.0, -5.0, -5.0, -6.0, -2.0], dtype=float)
    bounds_upper = np.array([80.0, 5.0, 5.0, 6.0, 2.0], dtype=float)

    system_model = build_system_model(case_values)
    body = build_base_body(case_values["tilt_deg"], n_panels=n_panels)

    result, _ = solve_quasi_steady_state(
        body_aero=body,
        center_of_gravity=CENTER_OF_GRAVITY,
        reference_point=REFERENCE_POINT,
        system_model=system_model,
        x_guess=np.asarray(x_guess, dtype=float),
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        include_gravity=include_gravity,
        axes=DEFAULT_AXES,
        moment_tolerance=MOMENT_TOLERANCE,
        max_nfev=400,
        f_scale=0.15,
        window_fraction=1,
    )

    timescale_result = compute_quasi_steady_fast_timescales(
        body_aero=build_base_body(case_values["tilt_deg"], n_panels=n_panels),
        center_of_gravity=CENTER_OF_GRAVITY,
        reference_point=REFERENCE_POINT,
        system_model=system_model,
        x_state=np.asarray(result["opt_x"], dtype=float),
        include_gravity=include_gravity,
        axes=DEFAULT_AXES,
        radial_distance=float(case_values.get("distance_radial", 200.0)),
    )

    return result, timescale_result


def profile_two_quasi_steady_solves(
    case_values: dict[str, float],
    x_guess: np.ndarray,
    *,
    n_panels: int = 18,
    include_gravity: bool = True,
) -> tuple[dict, dict]:
    """Profile two trim solves; second uses first solution as warm-start guess."""

    bounds_lower = np.array([2.0, -5.0, -5.0, -6.0, -2.0], dtype=float)
    bounds_upper = np.array([80.0, 5.0, 5.0, 6.0, 2.0], dtype=float)

    def run_once(x_start: np.ndarray) -> tuple[dict, float]:
        system_model = build_system_model(case_values)
        body = build_base_body(case_values["tilt_deg"], n_panels=n_panels)

        tic = perf_counter()
        result, _ = solve_quasi_steady_state(
            body_aero=body,
            center_of_gravity=CENTER_OF_GRAVITY,
            reference_point=REFERENCE_POINT,
            system_model=system_model,
            x_guess=np.asarray(x_start, dtype=float),
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            include_gravity=include_gravity,
            axes=DEFAULT_AXES,
            moment_tolerance=MOMENT_TOLERANCE,
            max_nfev=3000,
            f_scale=0.05,
            window_fraction=1,
            return_timing_breakdown=True,
        )
        elapsed_s = perf_counter() - tic
        return result, elapsed_s

    first_result, elapsed_first_s = run_once(
        np.asarray(x_guess, dtype=float),
    )
    second_result, elapsed_second_s = run_once(
        np.asarray(first_result["opt_x"], dtype=float),
    )

    timing_first = first_result.get("timing_breakdown", {})
    timing_second = second_result.get("timing_breakdown", {})

    print("\n=== Two-Run Quasi-Steady Profiling ===")
    print("Second run uses first run opt_x as x_guess (better initial guess).")
    print(
        f"run1 elapsed [s]      : {elapsed_first_s:.3f} "
        f"(evals={timing_first.get('residual_evaluations', 'n/a')})"
    )
    print(
        f"run2 elapsed [s]      : {elapsed_second_s:.3f} "
        f"(evals={timing_second.get('residual_evaluations', 'n/a')})"
    )
    if elapsed_first_s > 0.0:
        speedup = elapsed_first_s / max(elapsed_second_s, 1e-12)
        print(f"run2 speedup [x]      : {speedup:.2f}")
    print(f"run1 solver share     : {timing_first.get('solver_share', np.nan):.3f}")
    print(f"run2 solver share     : {timing_second.get('solver_share', np.nan):.3f}")

    opt_x_first = np.asarray(first_result.get("opt_x", np.full(5, np.nan)), dtype=float)
    opt_x_second = np.asarray(
        second_result.get("opt_x", np.full(5, np.nan)), dtype=float
    )
    delta_opt_x = opt_x_second - opt_x_first

    cm_first = np.asarray(first_result.get("cm", np.full(3, np.nan)), dtype=float)
    cm_second = np.asarray(second_result.get("cm", np.full(3, np.nan)), dtype=float)
    cf_first = np.array(
        [
            float(first_result.get("cfx", np.nan)),
            float(first_result.get("cfy", np.nan)),
        ],
        dtype=float,
    )
    cf_second = np.array(
        [
            float(second_result.get("cfx", np.nan)),
            float(second_result.get("cfy", np.nan)),
        ],
        dtype=float,
    )

    print("\n=== Warm-Start vs Cold-Start Solution Comparison ===")
    print("state vector order      : [kite_speed, roll, pitch, yaw, course_rate]")
    print(f"opt_x cold-start        : {np.array2string(opt_x_first, precision=6)}")
    print(f"opt_x warm-start        : {np.array2string(opt_x_second, precision=6)}")
    print(f"delta opt_x (warm-cold) : {np.array2string(delta_opt_x, precision=6)}")
    print(f"||delta opt_x||_2       : {np.linalg.norm(delta_opt_x):.3e}")
    print(f"cm cold-start           : {np.array2string(cm_first, precision=6)}")
    print(f"cm warm-start           : {np.array2string(cm_second, precision=6)}")
    print(
        f"delta cm (warm-cold)    : {np.array2string(cm_second - cm_first, precision=6)}"
    )
    print(f"cf cold-start           : {np.array2string(cf_first, precision=6)}")
    print(f"cf warm-start           : {np.array2string(cf_second, precision=6)}")
    print(
        f"delta cf (warm-cold)    : {np.array2string(cf_second - cf_first, precision=6)}"
    )

    return first_result, second_result


def _mode_time_response(
    eigenvalue: complex,
    eigenvector: np.ndarray,
    time_vector: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    """Return real-valued modal response x(t)=Re(v*exp(lambda*t))."""

    vec = np.asarray(eigenvector, dtype=complex)
    response = amplitude * np.outer(vec, np.exp(eigenvalue * time_vector))
    return np.real(response)


def _rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis_vec = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis_vec)
    if axis_norm == 0.0:
        raise ValueError("Rotation axis must be non-zero.")

    kx, ky, kz = axis_vec / axis_norm
    skew = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=float,
    )
    return (
        np.eye(3) + np.sin(angle_rad) * skew + (1.0 - np.cos(angle_rad)) * (skew @ skew)
    )


def _rotate_points(
    points: np.ndarray, rotation: np.ndarray, origin: np.ndarray
) -> np.ndarray:
    return origin + (points - origin) @ rotation.T


def plot_stability_modes_animation(
    base_body: BodyAerodynamics,
    trim_result: dict,
    timescale_result: dict,
    *,
    lateral_mode_index: int = 0,
    duration_s: float = 8.0,
    fps: int = 30,
    amplitude_rad: float = np.deg2rad(8.0),
    time_scale: float = 400.0,
) -> FuncAnimation:
    """Animate kite geometry from one lateral mode using yaw and roll components."""

    eig_lat = np.asarray(timescale_result["eig_lateral"])
    vec_lat = np.asarray(timescale_result["vec_lateral"])
    if vec_lat.shape[0] != 3:
        raise ValueError(
            "Expected lateral eigenvectors with 3 states [yaw, course_rate, roll]."
        )
    if lateral_mode_index < 0 or lateral_mode_index >= eig_lat.size:
        raise ValueError("lateral_mode_index is out of bounds.")

    n_frames = max(int(duration_s * fps), 2)
    time_vector = np.linspace(0.0, duration_s, n_frames)
    if time_scale <= 0.0:
        raise ValueError("time_scale must be strictly positive.")
    modal_time_vector = time_vector / time_scale
    mode_vec = np.asarray(vec_lat[:, lateral_mode_index], dtype=complex)
    norm_factor = max(np.max(np.abs(mode_vec[[0, 2]])), 1e-12)
    mode_vec = mode_vec / norm_factor
    mode_response = _mode_time_response(
        eig_lat[lateral_mode_index],
        mode_vec,
        modal_time_vector,
        amplitude_rad,
    )

    yaw_perturb_rad = mode_response[0, :]
    course_rate_perturb_rad_s = mode_response[1, :]
    roll_perturb_rad = mode_response[2, :]

    trim_roll_rad = np.deg2rad(float(trim_result["opt_x"][1]))
    trim_yaw_rad = np.deg2rad(float(trim_result["opt_x"][3]))
    trim_course_rate_rad_s = float(trim_result["opt_x"][4])

    panel_corners = np.array(
        [panel.corner_points for panel in base_body.panels], dtype=float
    )

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("First Lateral Mode: Kite Motion (Roll + Yaw)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.grid(True, alpha=0.25)

    all_points = panel_corners.reshape(-1, 3)
    center = np.mean(all_points, axis=0)
    half_range = 0.6 * np.max(np.ptp(all_points, axis=0))
    if half_range <= 0.0:
        half_range = 1.0
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)
    ax.view_init(elev=20, azim=-120)

    panel_lines = []
    for _ in range(panel_corners.shape[0]):
        (line,) = ax.plot([], [], [], color="tab:blue", linewidth=1.1)
        panel_lines.append(line)

    status_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes)

    course_rate_total_rad_s = trim_course_rate_rad_s + course_rate_perturb_rad_s
    ax_rate = fig.add_axes([0.64, 0.08, 0.32, 0.2])
    ax_rate.set_title("Course-Rate Mode Signal", fontsize=9)
    ax_rate.set_xlabel("t_vis [s]", fontsize=8)
    ax_rate.set_ylabel("course_rate [rad/s]", fontsize=8)
    ax_rate.tick_params(labelsize=8)
    ax_rate.grid(True, alpha=0.3)
    ax_rate.plot(
        time_vector,
        course_rate_total_rad_s,
        linestyle="--",
        linewidth=1.0,
        color="0.6",
        label="full signal",
    )
    (course_rate_line,) = ax_rate.plot(
        [],
        [],
        linewidth=2.0,
        color="tab:red",
        label="animated",
    )
    (course_rate_marker,) = ax_rate.plot([], [], marker="o", color="tab:red")

    cr_min = float(np.min(course_rate_total_rad_s))
    cr_max = float(np.max(course_rate_total_rad_s))
    cr_pad = max(0.05 * (cr_max - cr_min), 1e-3)
    ax_rate.set_xlim(time_vector[0], time_vector[-1])
    ax_rate.set_ylim(cr_min - cr_pad, cr_max + cr_pad)
    ax_rate.legend(loc="upper right", fontsize=7)

    def init():
        for line in panel_lines:
            line.set_data([], [])
            line.set_3d_properties([])
        status_text.set_text("")
        course_rate_line.set_data([], [])
        course_rate_marker.set_data([], [])
        return [*panel_lines, status_text, course_rate_line, course_rate_marker]

    def update(frame_idx: int):
        yaw_total = trim_yaw_rad + float(yaw_perturb_rad[frame_idx])
        course_rate_total = trim_course_rate_rad_s + float(
            course_rate_perturb_rad_s[frame_idx]
        )
        roll_total = trim_roll_rad + float(roll_perturb_rad[frame_idx])

        rotation = _rotation_matrix(DEFAULT_AXES.radial, yaw_total) @ _rotation_matrix(
            DEFAULT_AXES.course,
            roll_total,
        )

        corners_rot = _rotate_points(panel_corners, rotation, REFERENCE_POINT)
        for panel_idx, line in enumerate(panel_lines):
            corners = corners_rot[panel_idx]
            closed = np.vstack([corners, corners[0]])
            line.set_data(closed[:, 0], closed[:, 1])
            line.set_3d_properties(closed[:, 2])

        status_text.set_text(
            f"t_vis={time_vector[frame_idx]:.2f} s, t_mode={modal_time_vector[frame_idx]:.4f} s | "
            f"yaw={np.rad2deg(yaw_total):+.2f} deg | "
            f"roll={np.rad2deg(roll_total):+.2f} deg | "
            f"course_rate={course_rate_total:+.3f} rad/s"
        )

        course_rate_line.set_data(
            time_vector[: frame_idx + 1], course_rate_total_rad_s[: frame_idx + 1]
        )
        course_rate_marker.set_data(
            [time_vector[frame_idx]], [course_rate_total_rad_s[frame_idx]]
        )
        return [*panel_lines, status_text, course_rate_line, course_rate_marker]

    animation = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=1000.0 / fps,
        blit=False,
    )
    return animation


def _print_summary(trim_result: dict, timescale_result: dict) -> None:
    opt_x = np.asarray(trim_result["opt_x"], dtype=float)
    cm = np.asarray(trim_result["cm"], dtype=float)
    cfx = float(trim_result.get("cfx", np.nan))
    cfy = float(trim_result.get("cfy", np.nan))
    success_optimizer = bool(
        trim_result.get("success_optimizer", trim_result["success"])
    )
    success_physical = bool(trim_result.get("success_physical", False))

    moments_ok = bool(np.all(np.abs(cm) < MOMENT_TOLERANCE))
    forces_ok = bool(np.abs(cfx) < FORCE_TOLERANCE and np.abs(cfy) < FORCE_TOLERANCE)

    print("\n=== Single-State Quasi-Steady Trim Summary ===")
    print(f"success_optimizer     : {success_optimizer}")
    print(f"success_physical      : {success_physical}")
    print(f"kite_speed [m/s]      : {opt_x[0]: .3f}")
    print(f"roll [deg]            : {opt_x[1]: .3f}")
    print(f"pitch [deg]           : {opt_x[2]: .3f}")
    print(f"yaw [deg]             : {opt_x[3]: .3f}")
    print(f"course_rate [rad/s]   : {opt_x[4]: .4f}")
    print(f"aoa_center [deg]      : {trim_result['aoa_deg']: .3f}")
    print(f"beta_center [deg]     : {trim_result['side_slip_deg']: .3f}")
    print(f"cm = [cmx,cmy,cmz]    : [{cm[0]: .4e}, {cm[1]: .4e}, {cm[2]: .4e}]")
    print(f"cf = [cfx,cfy]        : [{cfx: .4e}, {cfy: .4e}]")
    print(
        "trim check            : "
        f"moments_zero={moments_ok} (tol={MOMENT_TOLERANCE:.1e}), "
        f"forces_zero={forces_ok} (tol={FORCE_TOLERANCE:.1e})"
    )

    print("\nLongitudinal eigenvalues:")
    tfast_long = np.asarray(timescale_result.get("Tfast_long", []), dtype=float)
    eps_long = np.asarray(
        timescale_result.get(
            "epsilon_long",
            timescale_result.get("eps_fast_long", np.full_like(tfast_long, np.nan)),
        ),
        dtype=float,
    )
    for idx, eig in enumerate(np.asarray(timescale_result["eig_long"])):
        t_fast = tfast_long[idx] if idx < tfast_long.size else np.nan
        eps_val = eps_long[idx] if idx < eps_long.size else np.nan
        print(
            f"  lambda_long[{idx}] = {eig.real: .6f} + {eig.imag: .6f}j | "
            f"T_fast={t_fast: .6f} s | eps={eps_val: .3e}"
        )

    print("\nLateral eigenvalues:")
    tfast_lat = np.asarray(timescale_result.get("Tfast_lateral", []), dtype=float)
    eps_lat = np.asarray(
        timescale_result.get(
            "epsilon_lateral",
            timescale_result.get("eps_fast_lateral", np.full_like(tfast_lat, np.nan)),
        ),
        dtype=float,
    )
    for idx, eig in enumerate(np.asarray(timescale_result["eig_lateral"])):
        t_fast = tfast_lat[idx] if idx < tfast_lat.size else np.nan
        eps_val = eps_lat[idx] if idx < eps_lat.size else np.nan
        print(
            f"  lambda_lat[{idx}] = {eig.real: .6f} + {eig.imag: .6f}j | "
            f"T_fast={t_fast: .6f} s | eps={eps_val: .3e}"
        )


def main() -> None:
    case_values = {
        "tilt_deg": 5.0,
        "course_deg": 90.0,
        "wind_speed": 5.0,
        "elevation_deg": 0.0,
        "azimuth_deg": 0.0,
        "radial_speed": 0,
        "distance_radial": 200.0,
    }
    x_guess = np.array([20.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    _, trim_result = profile_two_quasi_steady_solves(
        case_values=case_values,
        x_guess=x_guess,
        n_panels=18,
        include_gravity=True,
    )

    system_model = build_system_model(case_values)
    timescale_result = compute_quasi_steady_fast_timescales(
        body_aero=build_base_body(case_values["tilt_deg"], n_panels=18),
        center_of_gravity=CENTER_OF_GRAVITY,
        reference_point=REFERENCE_POINT,
        system_model=system_model,
        x_state=np.asarray(trim_result["opt_x"], dtype=float),
        include_gravity=True,
        axes=DEFAULT_AXES,
        radial_distance=float(case_values.get("distance_radial", 200.0)),
    )

    _print_summary(trim_result, timescale_result)

    body_for_animation = build_base_body(case_values["tilt_deg"], n_panels=18)
    _ = plot_stability_modes_animation(
        base_body=body_for_animation,
        trim_result=trim_result,
        timescale_result=timescale_result,
        lateral_mode_index=0,
        duration_s=20.0,
        fps=30,
        amplitude_rad=-np.deg2rad(10.0),
        time_scale=400.0,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
