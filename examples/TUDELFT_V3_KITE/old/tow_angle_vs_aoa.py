import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.fitting import fit_and_evaluate_model
from scipy.interpolate import interp1d
from VSM.plot_styling import set_plot_style, PALETTE

import math


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
    set_plot_style()
    ### 2. defining settings
    n_panels = 40
    mass_kite = 12
    spanwise_panel_distribution = "uniform"
    reference_point = np.array([0.0, 0, 0])
    solver = Solver(reference_point=reference_point)

    ### 3. Loading kite geometry from CSV file and instantiating BodyAerodynamics
    print(f"\nCreating corrected polar input with bridles")
    body_aero_polar_with_bridles = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=file_path,
        spanwise_panel_distribution=spanwise_panel_distribution,
        bridle_path=None,
    )

    ### 4. Setting va
    Umag = 20
    yaw_rate = 0
    body_aero_polar_with_bridles.va_initialize(Umag, 5, 0, yaw_rate)

    ### 7. Plotting the polar curves for different angles of attack and side slip angles
    # and saving in results with literature
    save_folder = Path(PROJECT_DIR) / "results" / "TUDELFT_V3_KITE"

    angle_of_attack_range = np.linspace(-5, 10, 61)
    gamma = None
    center_of_pressure = np.zeros((len(angle_of_attack_range), 3))
    total_force = np.zeros((len(angle_of_attack_range), 3))
    total_moment = np.zeros((len(angle_of_attack_range), 3))
    cl = np.zeros((len(angle_of_attack_range)))
    cd = np.zeros((len(angle_of_attack_range)))
    aero_roll = np.zeros((len(angle_of_attack_range)))
    tow_angle = np.zeros((len(angle_of_attack_range)))
    corner_points = np.array(
        [panel.corner_points for panel in body_aero_polar_with_bridles.panels]
    )
    begin_time = time.time()
    N = len(corner_points)
    x_corner = corner_points[N // 2, 0, :]
    fbridle_length = np.linalg.norm(x_corner - reference_point)
    trim_angle = np.zeros((len(angle_of_attack_range)))
    print("Front bridle length:", fbridle_length)
    elevation_angle = np.radians(30)
    course_angle = np.radians(90)
    for i, angle_i in enumerate(angle_of_attack_range):

        body_aero_polar_with_bridles.va_initialize(Umag, angle_i, 0, yaw_rate)

        results = solver.solve(body_aero_polar_with_bridles, gamma_distribution=gamma)
        center_of_pressure[i, :] = results["center_of_pressure"]

        x_tow_point = x_corner[0] - center_of_pressure[i, 0]
        tow_angle[i] = np.arctan(abs(x_tow_point) / fbridle_length)

        # tow_angle =
        total_force[i, :] = np.array(
            [
                results["Fx"],
                results["Fy"],
                results["Fz"],
            ]
        )
        total_moment[i, :] = np.array(
            [
                results["Mx"],
                results["My"]
                + mass_kite
                * 9.81
                * np.cos(elevation_angle)
                * np.cos(course_angle)
                * 10,
                results["Mz"],
            ]
        )

        cl[i] = np.sqrt(results["cl"] ** 2 + results["cs"] ** 2)
        cd[i] = results["cd"]
        aero_roll[i] = np.arctan2(results["cs"], results["cl"]) * 180 / np.pi

    end_time = time.time()
    print(f"Time taken for calculations: {end_time - begin_time} seconds")
    # Interpolate the angle where My crosses 0

    # Find indices where My changes sign
    sign_change_indices = np.where(np.diff(np.sign(total_moment[:, 1])))[0]
    trim_angles = []
    for idx in sign_change_indices:
        # Interpolate between angle_of_attack_range[idx] and angle_of_attack_range[idx+1]
        f_interp = interp1d(
            total_moment[idx : idx + 2, 1],
            angle_of_attack_range[idx : idx + 2],
            kind="linear",
        )
        trim_angle_zero = f_interp(0)
        trim_angles.append(trim_angle_zero)
    print("Trim angle(s) where My crosses zero:", trim_angles)
    dependencies = [
        "np.ones(len(alpha))",
        "alpha",
        "alpha**2",
    ]
    # Fit lift coeffcients
    fit_cl = fit_and_evaluate_model(
        cl,
        dependencies=dependencies,
        alpha=angle_of_attack_range / 180 * np.pi,
    )
    print("Fitted coefficients for lift coefficient:")
    print(fit_cl["coeffs"])
    dependencies = [
        "np.ones(len(alpha))",
        "alpha",
        "alpha**2",
    ]
    fit_cd = fit_and_evaluate_model(
        cd,
        dependencies=dependencies,
        alpha=angle_of_attack_range / 180 * np.pi,
    )
    print("Fitted coefficients for drag coefficient:")
    print(fit_cd["coeffs"])
    mean_aoa_exp_ro = 6.37
    std_aoa_exp_ro = 1.93
    mean_aoa_exp_ri = 3  # rough estimate
    std_aoa_exp_ri = 0.9

    plt.figure(figsize=(5, 4))
    plt.plot(angle_of_attack_range, tow_angle * 180 / np.pi)
    # Fill between mean ± std for experimental outer and inner
    plt.axvspan(
        mean_aoa_exp_ro - std_aoa_exp_ro,
        mean_aoa_exp_ro + std_aoa_exp_ro,
        alpha=0.3,
        color=PALETTE["Orange"],
        label="Reel-out range",
    )

    plt.axvspan(
        mean_aoa_exp_ri - std_aoa_exp_ri,
        mean_aoa_exp_ri + std_aoa_exp_ri,
        color=PALETTE["Sky Blue"],
        alpha=0.3,
        label="Reel-in range",
    )

    plt.xlabel(r"Angle of attack, $\alpha_\mathrm{w}$ [$^\circ$]")
    plt.ylabel(r"Tow angle, $\lambda_{\mathrm{b}}$ [$^\circ$]")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_folder / "tow_angle_vs_aoa.pdf")
    plt.show()
    # #### 5. Plotting the kite geometry using Matplotlib
    # fig,ax = creating_geometry_plot(
    #         body_aero_polar_with_bridles,
    #         title="Center of Pressure",
    #         view_elevation=15,
    #         view_azimuth=-120,
    #     )
    # # Plot points in the 3d plot

    # plt.show()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    print("number of panels: ", len(body_aero_polar_with_bridles.panels))

    for i, panel in enumerate(body_aero_polar_with_bridles.panels):
        # Get the corner points of the current panel and close the loop by adding the first point again
        x_corners = np.append(corner_points[i, :, 0], corner_points[i, 0, 0])
        y_corners = np.append(corner_points[i, :, 1], corner_points[i, 0, 1])
        z_corners = np.append(corner_points[i, :, 2], corner_points[i, 0, 2])

        # Plot the panel edges
        ax.plot(
            x_corners,
            y_corners,
            z_corners,
            color="grey",
            label="Panel Edges" if i == 0 else "",
            linewidth=1,
        )
    for i in range(center_of_pressure.shape[0]):
        # center of pressure point
        x_cp = center_of_pressure[i, :]
        ax.scatter(
            x_cp[0],
            x_cp[1],
            x_cp[2],
            s=100,  # size of the point
            label=f"CP {i+1}",
        )
        # Add angle of attack as text near the scatter point
        ax.text(
            x_cp[0],
            x_cp[1] + 0.1,
            x_cp[2],
            f"{angle_of_attack_range[i]:.1f}°",
            color="black",
            fontsize=8,
        )

    print("Center of pressure: ", center_of_pressure)
    max_diff_x = np.max(center_of_pressure[:, 0]) - np.min(center_of_pressure[:, 0])
    print("max_diff_x: ", max_diff_x / 2.6 * 100, "%")

    # --- Compare designs with different reference point heights ---
    ref_heights = np.linspace(0, 8, 8)  # Example heights, adjust as needed
    # ref_x = np.linspace(0, -0.5, 8)  # Example x-coordinates, adjust as needed

    trim_angles = []
    fbridle_lengths = []
    T_half = np.zeros(len(ref_heights))
    T = np.zeros(len(ref_heights))
    t_zero = np.zeros(len(ref_heights))

    plt.figure()
    for i, z in enumerate(ref_heights):
        reference_point = np.array([0, 0, z])
        solver = Solver(reference_point=reference_point)
        body_aero_polar_with_bridles.va_initialize(Umag, 5, 0, yaw_rate)
        corner_points = np.array(
            [panel.corner_points for panel in body_aero_polar_with_bridles.panels]
        )
        N = len(corner_points)
        x_corner = corner_points[N // 2, 0, :]
        fbridle_length = np.linalg.norm(x_corner - reference_point)
        fbridle_lengths.append(fbridle_length)

        my_vs_alpha = []
        for angle_i in angle_of_attack_range:
            body_aero_polar_with_bridles.va_initialize(Umag, angle_i, 0, yaw_rate)
            results = solver.solve(
                body_aero_polar_with_bridles, gamma_distribution=gamma
            )
            my_vs_alpha.append(results["My"])
        dMy_dalpha_field = np.gradient(
            my_vs_alpha, angle_of_attack_range
        )  # local (central) slope at each grid point
        dMy_dalpha = np.interp(
            trim_angle[i], angle_of_attack_range, dMy_dalpha_field
        )  # slope evaluated at α*

        # α ≈ θ locally ⇒ ∂α/∂θ = 1 ⇒ m_theta = dMy/dα
        k = -dMy_dalpha  # restoring stiffness [N·m/rad]
        print(f"z={z:.2f}, bridle={fbridle_length:.2f}, k={k:.2f}")

        if k <= 0:
            T_half[:] = T[:] = t_zero[:] = np.nan  # unstable around trim
        else:
            L = x_corner[2] - z
            Iy = mass_kite * (L**2) / 12.0  # slender-rod about CG (adjust if needed)
            print(f"Iy={Iy:.2f}")
            if Iy <= 0:
                T_half[i] = T[i] = t_zero[i] = np.nan
                continue
            wn = np.sqrt(k / Iy)
            T_half[i] = np.pi / wn
            T[i] = 2.0 * np.pi / wn
            t_zero[i] = np.pi / (2.0 * wn)
            print(f"T_half={T_half[i]:.2f}, T={T[i]:.2f}, t_zero={t_zero[i]:.2f}")
        plt.plot(
            angle_of_attack_range,
            my_vs_alpha,
            label=f"z={z:.2f}, bridle={fbridle_length:.2f}",
        )

        # Find trim angle (where My crosses zero)
        my_vs_alpha = np.array(my_vs_alpha)
        sign_change_indices = np.where(np.diff(np.sign(my_vs_alpha)))[0]
        if len(sign_change_indices) > 0:
            idx = sign_change_indices[0]
            f_interp = interp1d(
                my_vs_alpha[idx : idx + 2],
                angle_of_attack_range[idx : idx + 2],
                kind="linear",
            )
            trim_angle_zero = f_interp(0)
            trim_angles.append(trim_angle_zero)
        else:
            trim_angles.append(np.nan)

    plt.xlabel("Angle of Attack (degrees)")
    plt.ylabel("Moment My")
    plt.title("Moment My vs Angle of Attack for Different Reference Point Heights")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot trim angle vs bridle length
    plt.figure()
    plt.plot(fbridle_lengths, trim_angles, "o-")
    plt.xlabel("Bridle Line Length")
    plt.ylabel("Trim Angle (deg)")
    plt.title("Trim Angle vs Bridle Line Length")
    plt.grid()
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(angle_of_attack_range, cl, label="CL")
    axs[0].plot(angle_of_attack_range, fit_cl["data_est"], label="Fitted CL")
    axs[0].set_xlabel("Angle of Attack (degrees)")
    axs[0].set_ylabel("Lift Coefficient (CL)")

    axs[0].legend()
    axs[0].grid()
    axs[1].plot(angle_of_attack_range, cd, label="CD")
    axs[1].plot(angle_of_attack_range, fit_cd["data_est"], label="Fitted CD")
    axs[1].set_xlabel("Angle of Attack (degrees)")
    axs[1].set_ylabel("Drag Coefficient (CD)")

    axs[1].legend()
    axs[1].grid()
    plt.show()

    plt.figure()
    plt.plot(fbridle_lengths, T_half, "o-", label="Half Period")
    plt.plot(fbridle_lengths, T, "o-", label="Full Period")
    # plt.yscale("log")
    plt.xlabel("Bridle Length")
    plt.ylabel("Period")
    plt.title("Period vs Bridle Length")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
