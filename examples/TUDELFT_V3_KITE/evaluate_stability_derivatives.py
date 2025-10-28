from pathlib import Path
import numpy as np
from VSM import trim_angle
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.stability_derivatives import (
    compute_rigid_body_stability_derivatives,
    map_derivatives_to_aircraft_frame,
)
from VSM.trim_angle import compute_trim_angle


# Default step sizes for finite differences
PROJECT_DIR = Path(__file__).resolve().parents[2]
n_panels = 100
spanwise_panel_distribution = "uniform"
solver_base_version = Solver(reference_point=np.array([0.0, 0.0, 0.0]))

cad_derived_geometry_dir = (
    Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "CAD_derived_geometry"
)
body_aero_CAD_CFD_polars = BodyAerodynamics.instantiate(
    n_panels=n_panels,
    file_path=(cad_derived_geometry_dir / "aero_geometry_CAD_CFD_polars.yaml"),
    spanwise_panel_distribution=spanwise_panel_distribution,
)

### inputs for stability derivatives
side_slip = 0.0
velocity_magnitude = 10.0
roll_rate = 0.0
pitch_rate = 0.0
yaw_rate = 0.0
step_sizes = {
    "alpha": 1.0,  # degrees
    "beta": 1.0,  # degrees
    "p": 0.1,  # rad/s
    "q": 0.1,  # rad/s
    "r": 0.1,  # rad/s
}

# Get reference point from solver for physically correct moment calculations
# v_rot(r) = omega × (r - r_ref)
reference_point = solver_base_version.reference_point

## Compute trim-angle
results = compute_trim_angle(
    body_aero=body_aero_CAD_CFD_polars,
    solver=solver_base_version,
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
    reference_point=reference_point,
)
print(f"results: {results}")

trim_angle = results["trim_angle"]
dCMy_dalpha = results["dCMy_dalpha"]
print(f"Trim angle found at {trim_angle:.3f} degrees.")
print(f"Pitching moment derivative at trim angle: {dCMy_dalpha:.3f} per radian.")

## Compute stability derivatives (non-dimensionalized)
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
    reference_point=reference_point,
    nondimensionalize_rates=True,  # Convert rate derivatives to per hat-rate
)

print("\nComputed stability derivatives (VSM frame, x rearward, y right, z up):")
print("=" * 60)
print("Angle derivatives (per radian):")
for key, value in derivatives.items():
    if "alpha" in key or "beta" in key:
        print(f"  {key}: {value:+.6f}")

print("\nRate derivatives (per hat-rate, dimensionless):")
for key, value in derivatives.items():
    if any(rate in key for rate in ["_dp", "_dq", "_dr"]):
        print(f"  {key}: {value:+.6f}")

# Apply reference frame transformation using the VSM module function

derivatives_aircraft = map_derivatives_to_aircraft_frame(derivatives)

coeffs = ["Cx", "Cy", "Cz", "CMx", "CMy", "CMz"]
angle_keys = [f"d{coeff}_dalpha" for coeff in coeffs] + [
    f"d{coeff}_dbeta" for coeff in coeffs
]
rate_keys = [f"d{coeff}_dp" for coeff in coeffs] + [
    f"d{coeff}_dq" for coeff in coeffs
] + [f"d{coeff}_dr" for coeff in coeffs]


def print_combined(title, keys) -> None:
    print(f"\n{title}")
    header = f"  {'Derivative':<18}{'VSM':>14}{'Aircraft':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in keys:
        if key not in derivatives and key not in derivatives_aircraft:
            continue
        vsm_val = (
            f"{derivatives[key]:+.6f}" if key in derivatives else "     n/a"
        )
        ac_val = (
            f"{derivatives_aircraft[key]:+.6f}"
            if key in derivatives_aircraft
            else "     n/a"
        )
        print(f"  {key:<18}{vsm_val:>14}{ac_val:>14}")


print_combined("Angle derivatives (per radian)", angle_keys)
print_combined(
    "Rate derivatives (per hat-rate, dimensionless)",
    rate_keys,
)
