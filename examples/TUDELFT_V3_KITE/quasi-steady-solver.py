"""Sweep tilt and airspeed, optimize AoA/rot/course to zero moments, plot course/rot vs speed."""

from pathlib import Path
import time
import numpy as np
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.quasi_steady_state import DEFAULT_AXES, solve_quasi_steady_state
from awetrim.system.system_model import SystemModel


PROJECT_DIR = Path(__file__).resolve().parents[2]

include_gravity = False  # toggle to include gravity in moments and plots

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


spanwise_panel_distribution = "uniform"
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
        axis=DEFAULT_AXES.course,
        point=np.array([0.5, 0.0, 7.0]),
    )
    return base


def main():
    tilt = 2  # deg
    system_model = SystemModel()
    system_model.mass_wing = 10.0  # kg
    system_model.angle_elevation = np.deg2rad(0)
    system_model.angle_azimuth = np.deg2rad(0)
    system_model.angle_course = np.deg2rad(0)
    system_model.speed_radial = 0.0
    system_model.distance_radial = 200
    system_model.wind.speed_wind_ref = 4.0
    system_model.timeder_speed_tangential = 0.0
    system_model.timeder_speed_radial = 0.0

    x_guess = np.array([25, 0, 0, 0, 0], dtype=float)

    n_panels = 18

    base_body = build_base_body(tilt, n_panels)

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

    start_time = time.time()
    opt_result = solve_quasi_steady_state(
        body_aero=base_body,
        center_of_gravity=center_gravity,
        reference_point=reference_point,
        system_model=system_model,
        x_guess=x_guess,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        include_gravity=include_gravity,
        axes=DEFAULT_AXES,
    )

    def fmt_vec(vec: np.ndarray) -> str:
        arr = np.asarray(vec, dtype=float).reshape(3)
        return f"[{arr[0]: .3f}, {arr[1]: .3f}, {arr[2]: .3f}]"

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
    print(f"cl, cd                : {opt_result['cl']: .5f}, {opt_result['cd']: .5f}")
    print(f"aero_force [N]        : {fmt_vec(total_aero_force)}")
    print(f"inertial_force [N]    : {fmt_vec(inertial_force)}")
    print(f"gravity_force [N]     : {fmt_vec(gravity_force)}")
    print(f"net_force [N]         : {fmt_vec(net_force)}")
    print(f"wind_velocity [m/s]   : {fmt_vec(wind_vel)}")
    print(f"kite_velocity [m/s]   : {fmt_vec(kite_vel)}")
    print(f"apparent_velocity [m/s]: {fmt_vec(va_vel)}")

    end_time = time.time()
    print(f"Optimization took {end_time - start_time:.2f} seconds.")

    jac = opt_result.get("trim_jacobian")
    if jac is not None:
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

        header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(header))
        print("Jacobian near trim (central diff):")
        print(header_line)
        for i, label in enumerate(row_labels):
            print(fmt_row(label, jac[i, :]))


if __name__ == "__main__":
    main()
