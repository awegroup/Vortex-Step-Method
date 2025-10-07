from pathlib import Path
from VSM import trim_angle
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.stability_derivatives import compute_rigid_body_stability_derivatives
from VSM.trim_angle import compute_trim_angle

# Default step sizes for finite differences


PROJECT_DIR = Path(__file__).resolve().parents[2]
n_panels = 150
spanwise_panel_distribution = "uniform"
solver_base_version = Solver()

cad_derived_geometry_dir = (
    Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "CAD_derived_geometry"
)
body_aero_CAD_CFD_polars = BodyAerodynamics.instantiate(
    n_panels=n_panels,
    file_path=(cad_derived_geometry_dir / "config_kite_CAD_CFD_polars.yaml"),
    spanwise_panel_distribution=spanwise_panel_distribution,
)

### inputs for stability derivatives
alpha_initial_guess = 8.0
side_slip = 0.0
velocity_magnitude = 10.0
roll_rate = 0.0
pitch_rate = 0.0
yaw_rate = 0.0
step_sizes = {
    "u": 0.1,
    "v": 0.1,
    "w": 0.1,
    "alpha": 1.0,
    "beta": 1.0,
    "p": 0.1,
    "q": 0.1,
    "r": 0.1,
}

## Compute trim-angle
results = compute_trim_angle(
    body_aero=body_aero_CAD_CFD_polars,
    solver=solver_base_version,
    alpha_initial_guess=alpha_initial_guess,
    side_slip=side_slip,
    velocity_magnitude=velocity_magnitude,
    roll_rate=roll_rate,
    pitch_rate=pitch_rate,
    yaw_rate=yaw_rate,
    alpha_min=-2.0,
    alpha_max=13.0,
    coarse_step=2.0,
    fine_tolerance=0.1,
    derivative_step=1.0,
    max_bisection_iter=40,
)

trim_angle = results["trim_angle"]
dCMy_dalpha = results["dCMy_dalpha"]
print(f"Trim angle found at {trim_angle:.3f} degrees.")
print(f"Pitching moment derivative at trim angle: {dCMy_dalpha:.3f} per radian.")

## Compute stability derivatives
derivatives = compute_rigid_body_stability_derivatives(
    body_aero=body_aero_CAD_CFD_polars,
    solver=solver_base_version,
    angle_of_attack=trim_angle,
    side_slip=side_slip,
    velocity_magnitude=velocity_magnitude,
    roll_rate=roll_rate,
    pitch_rate=pitch_rate,
    yaw_rate=yaw_rate,
    step_sizes=step_sizes,
)

print("Computed stability derivatives (in rad):")
for key, value in derivatives.items():
    print(f"  {key}: {value:.4f}")
