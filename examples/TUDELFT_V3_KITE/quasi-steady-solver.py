"""Sweep tilt and airspeed, optimize AoA/rot/course to zero moments, plot course/rot vs speed."""

from pathlib import Path
import time
import numpy as np
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.quasi_steady_state import (
    compute_quasi_steady_trim_jacobian,
    compute_quasi_steady_fast_timescales,
    DEFAULT_AXES,
    plot_quasi_steady_sweep_dataframe,
    quasi_steady_sweep_rows_to_dataframe,
    run_quasi_steady_sweep,
)
from awetrim.system.system_model import SystemModel


PROJECT_DIR = Path(__file__).resolve().parents[2]

include_gravity = True  # toggle to include gravity in moments and plots

# Bounds and defaults (aoa, sideslip, course_rate_body)
kite_speed_bounds = (2.0, 80.0)  # m/s
pitch_bounds = (-5, 5)  # deg
yaw_bounds = (-6, 6)  # deg
course_bounds = (
    -2,
    2,
)  # rad/s, small course rate allowed for numerical reasons; not a physical course rate
roll_bounds = (
    -5,
    5,
)  # deg, small roll allowed for numerical reasons; not a physical roll

principal_axis = "tilt_deg"
secondary_axis = "course_deg"
sweep_values = {
    "tilt_deg": np.linspace(0.0, 12.0, 5),
    "course_deg": np.array([90]),  # course sweep in radians
    "wind_speed": [4.0],
    "elevation_deg": [0.0],
    "azimuth_deg": [0.0],
    "radial_speed": [0.0],
    "distance_radial": [200.0],
}

is_plot_results = False
is_save_csv = True
compute_jacobian_once = True


spanwise_panel_distribution = "uniform"
cad_derived_geometry_dir = (
    PROJECT_DIR / "data" / "TUDELFT_V3_KITE" / "CAD_derived_geometry"
)
geometry_yaml = cad_derived_geometry_dir / "aero_geometry_CAD_CFD_polars.yaml"
reference_point = np.array([0, 0.0, 0.0])
center_gravity = np.array([0.5, 0.0, 5.0])


def build_base_body(tilt_deg: float, n_panels: int) -> BodyAerodynamics:
    base = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=geometry_yaml,
        spanwise_panel_distribution=spanwise_panel_distribution,
        # ml_models_dir=ml_models_dir,
        # Disabled for this benchmark because current structural YAML uses a
        # schema that is not consumed by BodyAerodynamics.instantiate yet.
        bridle_path=None,
    )
    base.rotate(
        angle_deg=tilt_deg,
        axis=DEFAULT_AXES.course,
        point=np.array([0.5, 0.0, 7.0]),
    )
    return base


def main():
    jacobian_once_remaining = compute_jacobian_once
    from awetrim.system.tether import RigidLumpedTether

    tether = RigidLumpedTether(diameter=0.01)
    system_model = SystemModel(tether=tether)
    # system_model = SystemModel()
    system_model.mass_wing = 30.0  # kg
    system_model.angle_elevation = np.deg2rad(30)
    system_model.angle_azimuth = np.deg2rad(30)
    system_model.angle_course = np.deg2rad(90)
    system_model.speed_radial = 1.5
    system_model.distance_radial = 200
    system_model.wind.speed_wind_ref = 8.0
    system_model.timeder_speed_tangential = 0.0
    system_model.timeder_speed_radial = 0.0

    x_guess = np.array([25, 0, 0, 0, 0], dtype=float)

    n_panels = 18

    bounds_lower = np.array(
        [
            kite_speed_bounds[0],
            roll_bounds[0],
            pitch_bounds[0],
            yaw_bounds[0],
            course_bounds[0],
        ]
    )
    bounds_upper = np.array(
        [
            kite_speed_bounds[1],
            roll_bounds[1],
            pitch_bounds[1],
            yaw_bounds[1],
            course_bounds[1],
        ]
    )

    def update_system_model_for_case(system, case_values: dict[str, float]) -> None:
        system.angle_elevation = np.deg2rad(case_values["elevation_deg"])
        system.angle_azimuth = np.deg2rad(case_values["azimuth_deg"])
        system.angle_course = np.deg2rad(case_values["course_deg"])
        system.speed_radial = case_values["radial_speed"]
        system.distance_radial = case_values["distance_radial"]
        system.wind.speed_wind_ref = case_values["wind_speed"]

    def build_body_for_case(case_values: dict[str, float]) -> BodyAerodynamics:
        return build_base_body(case_values["tilt_deg"], n_panels)

    start_time = time.time()
    sweep_rows = run_quasi_steady_sweep(
        build_body=build_body_for_case,
        system_model=system_model,
        center_of_gravity=center_gravity,
        reference_point=reference_point,
        x_guess=x_guess,
        principal_axis=principal_axis,
        secondary_axis=secondary_axis,
        sweep_values=sweep_values,
        update_system_model=update_system_model_for_case,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        include_gravity=include_gravity,
        axes=DEFAULT_AXES,
        return_timing_breakdown=True,
        # Tighter optimizer tuning to drive course/normal forces to near-zero
        moment_tolerance=1e-4,  # Stricter moment equilibrium
    )

    def fmt_vec(vec: np.ndarray) -> str:
        arr = np.asarray(vec, dtype=float).reshape(3)
        return f"[{arr[0]: .3f}, {arr[1]: .3f}, {arr[2]: .3f}]"

    for row in sweep_rows:
        opt_result = row["result"]
        opt_x = opt_result["opt_x"]
        cmx, cmy, cmz = np.asarray(opt_result["cm"], dtype=float)
        total_aero_force = np.asarray(opt_result["total_aero_force_vec"], dtype=float)
        inertial_force = np.asarray(opt_result["inertial_force"], dtype=float)
        gravity_force = np.asarray(opt_result["gravity_force"], dtype=float)
        net_force = total_aero_force + inertial_force + gravity_force
        wind_vel = np.asarray(opt_result["wind_vel_world"], dtype=float)
        kite_vel = np.asarray(opt_result["kite_vel_world"], dtype=float)
        va_vel = np.asarray(opt_result["va_vel_world"], dtype=float)

        print("\n=== Quasi-Steady Trim Summary ===")
        print(
            f"case ({principal_axis}={row['principal_value']:.3f}, "
            f"{secondary_axis}={row['secondary_value']:.3f})"
        )
        print(f"success               : {opt_result['success']}")
        print(f"kite_speed [m/s]      : {opt_x[0]: .3f}")
        print(f"roll [deg]            : {opt_x[1]: .3f}")
        print(f"pitch [deg]           : {opt_x[2]: .3f}")
        print(f"yaw [deg]             : {opt_x[3]: .3f}")
        print(f"course_rate [rad/s]   : {opt_x[4]: .4f}")
        print(f"Umag [m/s]            : {opt_result['Umag']: .3f}")
        print(f"aoa_center [deg]      : {opt_result['aoa_deg']: .3f}")
        print(f"aoa_course [deg]      : {opt_result['aoa_course_deg']: .3f}")
        print(f"beta_center [deg]     : {opt_result['side_slip_deg']: .3f}")
        print(f"beta_course [deg]     : {opt_result['side_slip_course_deg']: .3f}")
        print(f"cm = [cmx,cmy,cmz]    : [{cmx: .4e}, {cmy: .4e}, {cmz: .4e}]")
        print(
            f"cl, cd                : {opt_result['cl']: .5f}, {opt_result['cd']: .5f}"
        )
        print(f"aero_force [N]        : {fmt_vec(total_aero_force)}")
        print(f"inertial_force [N]    : {fmt_vec(inertial_force)}")
        print(f"gravity_force [N]     : {fmt_vec(gravity_force)}")
        print(f"net_force [N]         : {fmt_vec(net_force)}")
        print(f"wind_velocity [m/s]   : {fmt_vec(wind_vel)}")
        print(f"kite_velocity [m/s]   : {fmt_vec(kite_vel)}")
        print(f"apparent_velocity [m/s]: {fmt_vec(va_vel)}")

        if jacobian_once_remaining and opt_result.get("success", False):
            _, jac = compute_quasi_steady_trim_jacobian(
                body_aero=build_body_for_case(row["case_values"]),
                center_of_gravity=center_gravity,
                reference_point=reference_point,
                system_model=system_model,
                x_state=opt_result["opt_x"],
                include_gravity=include_gravity,
                axes=DEFAULT_AXES,
            )

            # Compute timescale analysis
            timescale_result = compute_quasi_steady_fast_timescales(
                body_aero=build_body_for_case(row["case_values"]),
                center_of_gravity=center_gravity,
                reference_point=reference_point,
                system_model=system_model,
                x_state=opt_result["opt_x"],
                include_gravity=include_gravity,
                axes=DEFAULT_AXES,
                radial_distance=row["case_values"].get("distance_radial", 200.0),
            )

            row_labels = ["cmx", "cmy", "cmz", "cfx", "cfy"]
            col_labels = [
                "d/d_vtau (kite_speed)",
                "d/d_roll_rad",
                "d/d_pitch_rad",
                "d/d_yaw_rad",
                "d/d_course_rate_body",
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

            header_line = "  ".join(
                h.ljust(col_widths[i]) for i, h in enumerate(header)
            )
            print("Jacobian near trim (central diff):")
            print(header_line)
            for i, label in enumerate(row_labels):
                print(fmt_row(label, jac[i, :]))

            # Print timescale analysis
            print("\n=== Fast Timescale Analysis ===\n")

            eig_long = timescale_result["eig_long"]
            vec_long = timescale_result["vec_long"]
            T_fast_long = timescale_result["Tfast_long"]
            eps_long = timescale_result["epsilon_long"]
            stable_long = timescale_result["stable_long"]

            long_var_names = ["v_tau", "pitch"]

            print("Longitudinal (v_tau, pitch):")
            for i, (lam, T_f, eps) in enumerate(zip(eig_long, T_fast_long, eps_long)):
                mode_vec = vec_long[:, i]
                mag = np.abs(mode_vec)
                dominant_idx = np.argmax(mag)
                dominant_var = long_var_names[dominant_idx]
                print(f"  λ[{i}] = {lam.real: .6f} + {lam.imag: .6f}j")
                print(f"    T_fast = {T_f: .6f} s,  ε = {eps: .6e}")
                print(
                    f"    Mode composition: {', '.join(f'{long_var_names[j]}={mode_vec[j].real: .4f}' for j in range(len(long_var_names)))}"
                )
                print(f"    Dominant: {dominant_var}")
            print(f"  Stable: {stable_long}")

            eig_lat = timescale_result["eig_lateral"]
            vec_lat = timescale_result["vec_lateral"]
            T_fast_lat = timescale_result["Tfast_lateral"]
            eps_lat = timescale_result["epsilon_lateral"]
            stable_lat = timescale_result["stable_lateral"]

            lat_var_names = ["yaw", "course_rate", "roll"]

            print("\nLateral-directional (yaw, course_rate, roll):")
            for i, (lam, T_f, eps) in enumerate(zip(eig_lat, T_fast_lat, eps_lat)):
                mode_vec = vec_lat[:, i]
                mag = np.abs(mode_vec)
                dominant_idx = np.argmax(mag)
                dominant_var = lat_var_names[dominant_idx]
                print(f"  λ[{i}] = {lam.real: .6f} + {lam.imag: .6f}j")
                print(f"    T_fast = {T_f: .6f} s,  ε = {eps: .6e}")
                print(
                    f"    Mode composition: {', '.join(f'{lat_var_names[j]}={mode_vec[j].real: .4f}' for j in range(len(lat_var_names)))}"
                )
                print(f"    Dominant: {dominant_var}")
            print(f"  Stable: {stable_lat}")

            # jacobian_once_remaining = False

    end_time = time.time()
    print(f"Optimization took {end_time - start_time:.2f} seconds.")

    timing_keys = [
        "residual_evaluations",
        "residual_total_s",
        "solver_s",
        "body_copy_rotate_s",
        "kinematics_s",
        "postprocess_s",
    ]
    timing_totals = {k: 0.0 for k in timing_keys}
    timing_count = 0
    for row in sweep_rows:
        timing = row["result"].get("timing_breakdown")
        if timing is None:
            continue
        timing_count += 1
        for key in timing_keys:
            timing_totals[key] += float(timing.get(key, 0.0))

    if timing_count > 0 and timing_totals["residual_total_s"] > 0:
        residual_total = timing_totals["residual_total_s"]
        print("\n=== Quasi-steady Timing Breakdown (aggregate over sweep) ===")
        print(f"Solved cases                    : {timing_count}")
        print(
            f"Residual evaluations            : {int(timing_totals['residual_evaluations'])}"
        )
        print(f"Residual total [s]              : {residual_total:.3f}")
        print(
            f"solver.solve [s]                : {timing_totals['solver_s']:.3f} "
            f"({100.0 * timing_totals['solver_s'] / residual_total:.1f}%)"
        )
        print(
            f"copy+rotate body [s]            : {timing_totals['body_copy_rotate_s']:.3f} "
            f"({100.0 * timing_totals['body_copy_rotate_s'] / residual_total:.1f}%)"
        )
        print(
            f"kinematics eval [s]             : {timing_totals['kinematics_s']:.3f} "
            f"({100.0 * timing_totals['kinematics_s'] / residual_total:.1f}%)"
        )
        print(
            f"residual postprocess [s]        : {timing_totals['postprocess_s']:.3f} "
            f"({100.0 * timing_totals['postprocess_s'] / residual_total:.1f}%)"
        )

    df = quasi_steady_sweep_rows_to_dataframe(sweep_rows)
    if df.empty:
        return

    if is_save_csv:
        save_dir = PROJECT_DIR / "results" / "TUDELFT_V3_KITE"
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / "quasi_steady_sweep.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved sweep results to {out_path}")

    if is_plot_results:
        plot_quasi_steady_sweep_dataframe(
            df,
            principal_axis=principal_axis,
            secondary_axis=secondary_axis,
            show=True,
        )


if __name__ == "__main__":
    main()
