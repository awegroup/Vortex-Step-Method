import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.fitting import fit_and_evaluate_model
from scipy.interpolate import interp1d


def main():
    """
    This script demonstrates how to use the VSM library to perform a 3D aerodynamic analysis of the TUDELFT_V3_KITE.

    The example covers the following steps:
    1. Define file paths for the kite geometry, 2D polars, and bridle geometry.
    2. Load the kite geometry from a CSV file.
    3. Create three BodyAerodynamics objects:
       - One using the baseline Breukels input.
       - One with corrected polar data.
       - One with corrected polar data and bridles.
    4. Initialize the aerodynamic model with a specific wind speed, angle of attack, side slip angle, and yaw rate.
    5. Plot the kite geometry using Matplotlib.
    6. Generate an interactive plot using Plotly.
    7. Plot and save polar curves (both angle of attack and side slip sweeps) for different settings, comparing them to literature data.
    """

    ### 1. defining paths
    PROJECT_DIR = Path(__file__).resolve().parents[2]

    file_path = (
        Path(PROJECT_DIR)
        / "data"
        / "TUDELFT_V3_KITE"
        / "CAD_derived_geometry"
        / "config_kite_CAD_CFD_polars.yaml"
    )
    bridle_path = (
        Path(PROJECT_DIR)
        / "data"
        / "TUDELFT_V3_KITE"
        / "CAD_derived_geometry"
        / "struc_geometry.yaml"
    )

    ### 2. defining settings
    n_panels = 40
    spanwise_panel_distribution = "uniform"
    solver = Solver(reference_point=[0, 0, 0])

    ### 3. Loading kite geometry from CSV file and instantiating BodyAerodynamics
    print(f"\nCreating corrected polar input with bridles")
    body_aero_polar_with_bridles = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=file_path,
        spanwise_panel_distribution=spanwise_panel_distribution,
        bridle_path=bridle_path,
    )

    ### 4. Setting va
    Umag = 20
    body_aero_polar_with_bridles.va_initialize(Umag, 8, 5)

    x_tow_point = np.arange(-1.25, 0.5, 0.25)
    alpha_range = np.linspace(0, 20, 11)  # Angle of attack sweep
    ss_range = np.linspace(0, 10, 11)  # Sideslip sweep
    # Sweep over angle of attack for each reference point and collect all curves
    cmz_alpha_all = []
    cmx_alpha_all = []
    cmy_alpha_all = []
    trim_aoa_all = []
    for x in x_tow_point:
        solver = Solver(reference_point=[x, 0, 0])
        cmz_alpha = []
        cmx_alpha = []
        cmy_alpha = []
        for alpha in alpha_range:
            body_aero_polar_with_bridles.va_initialize(Umag, alpha, 0)
            results = solver.solve(body_aero_polar_with_bridles)
            cmz_alpha.append(results.get("cmz", np.nan))
            cmx_alpha.append(results.get("cmx", np.nan))
            cmy_alpha.append(results.get("cmy", np.nan))
        # Interpolate to find trim aoa where cmy crosses zero

        cmy_alpha_np = np.array(cmy_alpha)
        # Only interpolate if there is a sign change
        if np.any(np.diff(np.sign(cmy_alpha_np))):
            f_trim = interp1d(
                cmy_alpha_np,
                alpha_range,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            trim_aoa = f_trim(0.0)
        else:
            trim_aoa = np.nan
        trim_aoa_all.append(trim_aoa)

        cmz_alpha_all.append(cmz_alpha)
        cmx_alpha_all.append(cmx_alpha)
        cmy_alpha_all.append(cmy_alpha)

    # Triple plot for angle of attack
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for i, x in enumerate(x_tow_point):
        axs[0].plot(alpha_range, cmz_alpha_all[i], label=f"x_tow_point={x:.2f}")
        axs[1].plot(alpha_range, cmx_alpha_all[i], label=f"x_tow_point={x:.2f}")
        axs[2].plot(alpha_range, cmy_alpha_all[i], label=f"x_tow_point={x:.2f}")
    axs[0].set_xlabel("Angle of Attack (deg)")
    axs[0].set_ylabel("Yaw Moment Coefficient (cmz)")
    axs[0].set_title("cmz vs Angle of Attack")
    axs[0].grid()
    axs[0].legend()
    axs[1].set_xlabel("Angle of Attack (deg)")
    axs[1].set_ylabel("Roll Moment Coefficient (cmx)")
    axs[1].set_title("cmx vs Angle of Attack")
    axs[1].grid()
    axs[1].legend()
    axs[2].set_xlabel("Angle of Attack (deg)")
    axs[2].set_ylabel("Pitch Moment Coefficient (cmy)")
    axs[2].set_title("cmy vs Angle of Attack")
    axs[2].grid()
    axs[2].legend()
    plt.suptitle("Dependency on Angle of Attack for Different Reference Points")
    plt.tight_layout()
    # plt.show()

    # Sweep over sideslip for each reference point and collect all curves
    cmz_ss_all = []
    cmx_ss_all = []
    cmy_ss_all = []
    for i, x in enumerate(x_tow_point):
        solver = Solver(reference_point=[x, 0, 0])
        cmz_ss = []
        cmx_ss = []
        cmy_ss = []
        for ss in ss_range:
            body_aero_polar_with_bridles.va_initialize(Umag, trim_aoa_all[i], ss)
            results = solver.solve(body_aero_polar_with_bridles)
            cmz_ss.append(results.get("cmz", np.nan))
            cmx_ss.append(results.get("cmx", np.nan))
            cmy_ss.append(results.get("cmy", np.nan))
        cmz_ss_all.append(cmz_ss)
        cmx_ss_all.append(cmx_ss)
        cmy_ss_all.append(cmy_ss)

    # Triple plot for sideslip
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for i, x in enumerate(x_tow_point):
        axs[0].plot(ss_range, cmz_ss_all[i], label=f"x_tow_point={x:.2f}")
        axs[1].plot(ss_range, cmx_ss_all[i], label=f"x_tow_point={x:.2f}")
        axs[2].plot(ss_range, cmy_ss_all[i], label=f"x_tow_point={x:.2f}")
    axs[0].set_xlabel("Sideslip Angle (deg)")
    axs[0].set_ylabel("Yaw Moment Coefficient (cmz)")
    axs[0].set_title("cmz vs Sideslip Angle")
    axs[0].grid()
    axs[0].legend()
    axs[1].set_xlabel("Sideslip Angle (deg)")
    axs[1].set_ylabel("Roll Moment Coefficient (cmx)")
    axs[1].set_title("cmx vs Sideslip Angle")
    axs[1].grid()
    axs[1].legend()
    axs[2].set_xlabel("Sideslip Angle (deg)")
    axs[2].set_ylabel("Pitch Moment Coefficient (cmy)")
    axs[2].set_title("cmy vs Sideslip Angle")
    axs[2].grid()
    axs[2].legend()
    plt.suptitle("Dependency on Sideslip for Different Reference Points")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
