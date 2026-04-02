"""Sweep tilt and airspeed, optimize AoA/rot/course to zero moments, plot course/rot vs speed."""

from pathlib import Path
import time
import numpy as np
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.quasi_steady_state import DEFAULT_AXES, solve_quasi_steady_state
from awetrim.system.system_model import SystemModel

transformation_C_from_CR = np.array(
    [
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ],
    dtype=float,
)


def as_3vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != 3:
        raise ValueError(f"Expected a 3-vector, got shape {np.asarray(value).shape}")
    return vector


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
    tilt = 0  # deg
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
    solver = Solver(
        reference_point=reference_point, gamma_initial_distribution_type="zero"
    )

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

    def evaluate_kinematics(x: np.ndarray) -> dict:
        kite_speed, _roll, _pitch, _yaw, course_rate_body = x
        system_model.timeder_angle_course_body = course_rate_body
        system_model.speed_tangential = kite_speed

        inertial_force = -system_model.mass_wing * as_3vector(
            transformation_C_from_CR @ system_model.acceleration_course_body
        )
        gravity_force = as_3vector(
            transformation_C_from_CR @ system_model.force_gravity
        )
        wind_velocity = as_3vector(
            transformation_C_from_CR @ system_model.wind.velocity_wind(system_model)
        )
        kite_velocity = as_3vector(
            transformation_C_from_CR @ system_model.velocity_kite
        )
        apparent_velocity = as_3vector(
            transformation_C_from_CR @ system_model.velocity_apparent_wind
        )

        return {
            "va": apparent_velocity,
            "inertial_force": inertial_force,
            "gravity_force": gravity_force,
            "wind_velocity": wind_velocity,
            "kite_velocity": kite_velocity,
            "apparent_velocity": apparent_velocity,
        }

    start_time = time.time()
    opt_result = solve_quasi_steady_state(
        base_body,
        solver,
        x_guess,
        evaluate_kinematics,
        bounds_lower,
        bounds_upper,
        reference_point=reference_point,
        center_of_gravity=center_gravity,
        include_gravity=include_gravity,
        axes=DEFAULT_AXES,
    )
    print("Optimization result:")
    print(opt_result["opt_x"])
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
