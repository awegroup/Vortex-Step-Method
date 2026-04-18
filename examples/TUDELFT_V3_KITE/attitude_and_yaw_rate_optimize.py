"""Solve for roll/pitch/yaw (body rotations) and yaw_rate to zero moments.
Adjust fixed aoa/beta as needed.
"""

from pathlib import Path
import copy
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Sweep settings
speeds = [20]
tilt_angles_deg = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Vehicle properties
mass_kite = 0.0  # kg

# Fixed aerodynamic incidence (adjust if needed)
AOA_FIXED_DEG = 14
BETA_FIXED_DEG = 0.0

# Bounds: pitch/yaw in degrees, yaw_rate in rad/s. For tilt=0 we lock yaw/yaw_rate to 0; for tilt>0 they are free.
pitch_bounds_deg = (-20, 20)
yaw_bounds_deg = (-10, 10)
yaw_rate_bounds = (-5, 5)

x0 = np.array([-6.8, 0.0, 0.0], dtype=float)  # pitch, yaw, yaw_rate

# Geometry config
n_panels = 20
spanwise_panel_distribution = "uniform"
ml_models_dir = PROJECT_DIR / "data" / "ml_models"
cad_derived_geometry_dir = (
    PROJECT_DIR / "data" / "TUDELFT_V3_KITE" / "CAD_derived_geometry"
)
geometry_yaml = cad_derived_geometry_dir / "aero_geometry_CAD_CFD_polars.yaml"
reference_point = np.array([0.0, 0.0, 0.0])


def build_base_body(tilt_deg: float) -> BodyAerodynamics:
    base = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=geometry_yaml,
        spanwise_panel_distribution=spanwise_panel_distribution,
        ml_models_dir=ml_models_dir,
        scale=1.0,
    )
    if abs(tilt_deg) > 1e-6:
        base.rotate(
            angle_deg=tilt_deg,
            axis=np.array([1.0, 0.0, 0.0]),
            point=np.array([0.0, 0.0, 9.0]),
        )
    return base


def apply_attitude(body: BodyAerodynamics, pitch_deg: float, yaw_deg: float):
    # Yaw then pitch about the reference point.
    if abs(yaw_deg) > 1e-9:
        body.rotate(
            angle_deg=yaw_deg, axis=np.array([0.0, 0.0, 1.0]), point=reference_point
        )
    if abs(pitch_deg) > 1e-9:
        body.rotate(
            angle_deg=pitch_deg, axis=np.array([0.0, 1.0, 0.0]), point=reference_point
        )
    return body


def optimize_for(
    base_body: BodyAerodynamics,
    Umag: float,
    x_guess: np.ndarray,
    neighbor_x=None,
    tilt_current: float | None = None,
):
    solver = Solver(
        reference_point=reference_point, gamma_initial_distribution_type="zero"
    )

    def moment_residual(x: np.ndarray) -> np.ndarray:
        pitch_deg, yaw_deg, yaw_rate = x
        # For zero tilt, force yaw and yaw_rate to zero to recover the known solution; for tilt>0 allow them to vary.
        if tilt_current is not None and abs(tilt_current) < 1e-9:
            yaw_deg = 0.0
            yaw_rate = 0.0
        body = copy.deepcopy(base_body)
        body = apply_attitude(body, pitch_deg, yaw_deg)
        body.va_initialize(
            Umag=Umag,
            angle_of_attack=AOA_FIXED_DEG,
            side_slip=BETA_FIXED_DEG,
            yaw_rate=yaw_rate,
            pitch_rate=0.0,
            roll_rate=0.0,
            reference_point=reference_point,
            rates_in_body_frame=False,
        )
        res = solver.solve(body)
        cmx = res.get("cmx")
        cmy = res.get("cmy")
        cmz = res.get("cmz")

        # Additional inertial-like moment: force at (0,0,7) scaled by yaw_rate and Umag.
        if mass_kite != 0.0:
            aoa_rad = np.deg2rad(AOA_FIXED_DEG)
            beta_rad = np.deg2rad(BETA_FIXED_DEG)
            va_unit = np.array(
                [
                    np.cos(aoa_rad) * np.cos(beta_rad),
                    np.sin(beta_rad),
                    np.sin(aoa_rad) * np.cos(beta_rad),
                ],
                dtype=float,
            )
            # horiz_perp = np.array([-va_unit[1], va_unit[0], 0.0], dtype=float)
            # norm_hp = np.linalg.norm(horiz_perp)
            horiz_unit = np.array([0.0, 1.0, 0.0])

            mag_force = mass_kite * yaw_rate * Umag * np.cos(aoa_rad)
            force_vec = mag_force * horiz_unit
            r = np.array([0.0, 0.0, 7.0])
            moment_vec = np.cross(r, force_vec)

            q_inf = 0.5 * solver.rho * Umag**2
            projected_area = body.wings[0].compute_projected_area()
            max_chord = max(panel.chord for panel in body.panels)
            denom = q_inf * projected_area * max_chord if projected_area > 0 else 1.0
            delta_cm = moment_vec / denom
            cmx += delta_cm[0]
            cmy += delta_cm[1]
            cmz += delta_cm[2]

        return np.array([cmx, cmy, cmz])

    bounds_lower = np.array(
        [pitch_bounds_deg[0], yaw_bounds_deg[0], yaw_rate_bounds[0]]
    )
    bounds_upper = np.array(
        [pitch_bounds_deg[1], yaw_bounds_deg[1], yaw_rate_bounds[1]]
    )

    moment_tol = 1e-3

    def run_once(x_start, max_nfev, f_scale):
        res = least_squares(
            moment_residual,
            x_start,
            bounds=(bounds_lower, bounds_upper),
            xtol=1e-6,
            ftol=1e-6,
            gtol=1e-6,
            max_nfev=max_nfev,
            loss="soft_l1",
            f_scale=f_scale,
            x_scale="jac",
            diff_step=5e-2,
            verbose=0,
        )
        cm_vec = moment_residual(res.x)
        norm_cm = np.linalg.norm(cm_vec)
        return res, cm_vec, norm_cm

    attempts = []
    # Baseline start (clipped to bounds)
    attempts.append(np.clip(x_guess, bounds_lower, bounds_upper))
    # Neighbor warm start if available
    if neighbor_x is not None:
        attempts.append(np.clip(neighbor_x, bounds_lower, bounds_upper))
    # Jittered starts around pitch/yaw/yaw_rate to escape flat gradients (yaw/yaw_rate zeroed when tilt=0 inside residual)
    for delta_p in (-4.0, -2.0, 2.0, 4.0):
        attempts.append(
            np.clip(x_guess + np.array([delta_p, 0.0, 0.0]), bounds_lower, bounds_upper)
        )
    for delta_y in (-4.0, 4.0):
        attempts.append(
            np.clip(x_guess + np.array([0.0, delta_y, 0.0]), bounds_lower, bounds_upper)
        )
    for delta_r in (-1.0, 1.0):
        attempts.append(
            np.clip(x_guess + np.array([0.0, 0.0, delta_r]), bounds_lower, bounds_upper)
        )

    best_res = None
    best_cm = None
    best_norm = np.inf

    for idx, start in enumerate(attempts):
        res_try, cm_try, norm_try = run_once(start, max_nfev=900, f_scale=0.25)
        if norm_try < moment_tol:
            print(
                f"Early success with start {idx} at x={start}: |cm|={norm_try:.3e} (cmx={cm_try[0]:.3e}, cmy={cm_try[1]:.3e}, cmz={cm_try[2]:.3e}), success_flag={res_try.success}"
            )
            best_norm = norm_try
            best_res = res_try
            best_cm = cm_try
            break
        if norm_try < best_norm:
            best_norm = norm_try
            best_res = res_try
            best_cm = cm_try

    cmx, cmy, cmz = best_cm
    physical_success = (
        np.abs(cmx) < moment_tol
        and np.abs(cmy) < moment_tol
        and np.abs(cmz) < moment_tol
    )
    success = bool(best_res.success and physical_success)
    if not success:
        print(
            f"Local optimum or failure: |cm|={best_norm:.3e} (cmx={cmx:.3e}, cmy={cmy:.3e}, cmz={cmz:.3e}), success_flag={best_res.success}, starts={len(attempts)}"
        )
    print(
        f"Optimized for Umag={Umag} m/s: pitch={best_res.x[0]:.2f} deg, yaw={best_res.x[1]:.2f} deg, yaw_rate={best_res.x[2]:.2f} rad/s, cm=({cmx:.4f}, {cmy:.4f}, {cmz:.4f}), success={success}"
    )
    return best_res.x, np.array([cmx, cmy, cmz]), success


rows = []
x_guess = x0.copy()
last_success_x = None
for tilt in tilt_angles_deg:
    base_body = build_base_body(tilt)
    x_guess = last_success_x if last_success_x is not None else x0.copy()
    for Umag in speeds:
        print(
            f"\nOptimizing for tilt={tilt} deg, Umag={Umag} m/s with initial guess {x_guess}..."
        )
        opt_x, opt_cm, success = optimize_for(
            base_body,
            Umag,
            x_guess,
            neighbor_x=last_success_x,
            tilt_current=tilt,
        )
        rows.append(
            {
                "tilt_deg": tilt,
                "Umag": Umag,
                "pitch_deg": opt_x[0],
                "yaw_deg": opt_x[1] if abs(tilt) > 1e-9 else 0.0,
                "yaw_rate_rad_s": opt_x[2] if abs(tilt) > 1e-9 else 0.0,
                "cmx": opt_cm[0],
                "cmy": opt_cm[1],
                "cmz": opt_cm[2],
                "norm_cm": np.linalg.norm(opt_cm),
                "success": success,
            }
        )
        x_guess = opt_x if success else x0
        if success:
            last_success_x = opt_x

# Save results
save_dir = PROJECT_DIR / "results" / "TUDELFT_V3_KITE"
save_dir.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
out_path = save_dir / "cm_attitude_yawrate_optimum.csv"
df.to_csv(out_path, index=False)
print(f"Saved results to {out_path}")
