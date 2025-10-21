import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.fitting import fit_and_evaluate_model


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
        / "aero_geometry_CAD_CFD_polars.yaml"
    )

    # bridle_path = (
    #     Path(PROJECT_DIR)s
    #     / "data"
    #     / "TUDELFT_V3_KITE"
    #     / "CAD_derived_geometry"
    #     / "struc_geometry.yaml"
    # )

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
        # bridle_path=None,
    )

    ### 4. Setting va
    Umag = 20
    body_aero_polar_with_bridles.va_initialize(Umag, 5, 0)

    ### 7. Plotting the polar curves for different angles of attack and side slip angles
    # and saving in results with literature
    save_folder = Path(PROJECT_DIR) / "results" / "V9_KITE"

    angle_of_attack_range = np.linspace(0, 20, 21)
    gamma = None
    center_of_pressure = np.zeros((len(angle_of_attack_range), 3))
    total_force = np.zeros((len(angle_of_attack_range), 3))
    cl = np.zeros((len(angle_of_attack_range)))
    cd = np.zeros((len(angle_of_attack_range)))
    aero_roll = np.zeros((len(angle_of_attack_range)))
    begin_time = time.time()
    for i, angle_i in enumerate(angle_of_attack_range):

        body_aero_polar_with_bridles.va_initialize(Umag, angle_i, 0)

        results = solver.solve(body_aero_polar_with_bridles, gamma_distribution=gamma)
        center_of_pressure[i, :] = results["center_of_pressure"]
        total_force[i, :] = np.array(
            [
                results["Fx"],
                results["Fy"],
                results["Fz"],
            ]
        )
        # print(f"Center of pressure: {x_cp}")
        # print(results["cl"], results["cd"], results["cs"])
        # print(results["cl"]**2 + results["cs"]**2)
        cl[i] = np.sqrt(results["cl"] ** 2 + results["cs"] ** 2)
        cd[i] = results["cd"]
        aero_roll[i] = np.arctan2(results["cs"], results["cl"]) * 180 / np.pi
    end_time = time.time()
    print(f"Time taken for calculations: {end_time - begin_time} seconds")
    # angle_of_attack_range = angle_of_attack_range - 1
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
    corner_points = np.array(
        [panel.corner_points for panel in body_aero_polar_with_bridles.panels]
    )
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

    # --- Save polars to CSV ---
    polars_df = pd.DataFrame(
        {
            "angle_of_attack_deg": angle_of_attack_range,
            "CL": cl,
            "CD": cd,
            "CL_fit": fit_cl["data_est"],
            "CD_fit": fit_cd["data_est"],
        }
    )
    save_folder.mkdir(parents=True, exist_ok=True)
    polars_csv_path = save_folder / "polars_VSM.csv"
    polars_df.to_csv(polars_csv_path, index=False)
    print(f"Polars saved to {polars_csv_path}")

    # --- Save fit to JSON (according to your structure) ---
    import json

    fit_json = {
        "model": "coeffs",
        "params": {
            "CD0": float(fit_cd["coeffs"][0]),
            "CL0": float(fit_cl["coeffs"][0]),
            "angle_pitch_depower_0": -0.04,
            "delta_pitch_depower": -0.32,
        },
        "coefficients": {
            "CL": [
                {"var": "alpha", "power": 1, "coef": float(fit_cl["coeffs"][1])},
                {"var": "alpha", "power": 2, "coef": float(fit_cl["coeffs"][2])},
                {"var": "u_s", "power": 1, "coef": 0.0},
                {"var": "u_p", "power": 1, "coef": -0.0},
            ],
            "CD": [
                {"var": "alpha", "power": 1, "coef": float(fit_cd["coeffs"][1])},
                {"var": "alpha", "power": 2, "coef": float(fit_cd["coeffs"][2])},
                {"var": "u_s", "power": 1, "coef": 0.0},
                {"var": "u_p", "power": 1, "coef": -0.04},
            ],
            "CS": [{"var": "u_s", "power": 1, "coef": -0.25}],
        },
    }
    fit_json_path = save_folder / "fit_coeffs.json"
    with open(fit_json_path, "w") as f:
        json.dump(fit_json, f, indent=2)
    print(f"Fit coefficients saved to {fit_json_path}")

    # --- Simulate at alpha=8° for multiple yaw rates and fit a line ---
    alpha_fixed = 8  # degrees
    yaw_rates = np.linspace(-2, 2, 9)  # example yaw rates, adjust as needed
    cs_yaw = np.zeros(len(yaw_rates))

    for i, yaw in enumerate(yaw_rates):
        body_aero_polar_with_bridles.va_initialize(Umag, alpha_fixed, 0, yaw)
        results = solver.solve(body_aero_polar_with_bridles, gamma_distribution=gamma)
        cs_yaw[i] = results["cs"]

    # Fit a line: CS = a * yaw_rate + b
    coeffs = np.polyfit(yaw_rates, cs_yaw, 1)
    cs_fit = np.polyval(coeffs, yaw_rates)
    print(f"Fitted CS vs yaw_rate: slope={coeffs[0]:.4f}, intercept={coeffs[1]:.4f}")

    # Plot
    plt.figure()
    plt.plot(yaw_rates, cs_yaw, "o", label="Simulated CS")
    plt.plot(yaw_rates, cs_fit, "-", label="Linear fit")
    plt.xlabel("Yaw rate")
    plt.ylabel("CS")
    plt.title(f"CS vs Yaw Rate at alpha={alpha_fixed}°")
    plt.legend()
    plt.grid()
    plt.show()

    # Save to CSV
    cs_yaw_df = pd.DataFrame(
        {
            "yaw_rate": yaw_rates,
            "CS": cs_yaw,
            "CS_fit": cs_fit,
        }
    )
    cs_yaw_csv_path = save_folder / "cs_vs_yawrate.csv"
    cs_yaw_df.to_csv(cs_yaw_csv_path, index=False)
    print(f"CS vs yaw rate saved to {cs_yaw_csv_path}")

    # --- Fit Mz vs yaw_rate at alpha=8° ---
    mz_yaw = np.zeros(len(yaw_rates))
    for i, yaw in enumerate(yaw_rates):
        body_aero_polar_with_bridles.va_initialize(Umag, alpha_fixed, 0, yaw)
        results = solver.solve(body_aero_polar_with_bridles, gamma_distribution=gamma)
        mz_yaw[i] = results.get("cmz", np.nan)  # Use .get in case "cmz" is missing

    cmz_coeffs = np.polyfit(yaw_rates, mz_yaw, 1)
    cmz_fit = np.polyval(cmz_coeffs, yaw_rates)
    print(
        f"Fitted Mz vs yaw_rate: slope={cmz_coeffs[0]:.4f}, intercept={cmz_coeffs[1]:.4f}"
    )

    plt.figure()
    plt.plot(yaw_rates, mz_yaw, "o", label="Simulated Mz")
    plt.plot(yaw_rates, cmz_fit, "-", label="Linear fit")
    plt.xlabel("Yaw rate")
    plt.ylabel("Mz")
    plt.title(f"Mz vs Yaw Rate at alpha={alpha_fixed}°")
    plt.legend()
    plt.grid()
    plt.show()

    mz_yaw_df = pd.DataFrame(
        {
            "yaw_rate": yaw_rates,
            "Mz": mz_yaw,
            "Mz_fit": cmz_fit,
        }
    )
    mz_yaw_csv_path = save_folder / "mz_vs_yawrate.csv"
    mz_yaw_df.to_csv(mz_yaw_csv_path, index=False)
    print(f"Mz vs yaw rate saved to {mz_yaw_csv_path}")

    # --- Sideslip sweep at alpha=8° and fit CS vs sideslip ---
    beta_range = np.linspace(-15, 15, 11)  # degrees
    cs_beta = np.zeros(len(beta_range))
    for i, beta in enumerate(beta_range):
        body_aero_polar_with_bridles.va_initialize(Umag, alpha_fixed, beta, 0)
        results = solver.solve(body_aero_polar_with_bridles, gamma_distribution=gamma)
        cs_beta[i] = results["cs"]

    # Fit a line: CS = a * beta + b
    cs_beta_coeffs = np.polyfit(np.deg2rad(beta_range), cs_beta, 1)
    cs_beta_fit = np.polyval(cs_beta_coeffs, np.deg2rad(beta_range))
    print(
        f"Fitted CS vs sideslip: slope={cs_beta_coeffs[0]:.4f}, intercept={cs_beta_coeffs[1]:.4f}"
    )

    plt.figure()
    plt.plot(beta_range, cs_beta, "o", label="Simulated CS")
    plt.plot(beta_range, cs_beta_fit, "-", label="Linear fit")
    plt.xlabel("Sideslip angle (deg)")
    plt.ylabel("CS")
    plt.title(f"CS vs Sideslip at alpha={alpha_fixed}°")
    plt.legend()
    plt.grid()
    plt.show()

    cs_beta_df = pd.DataFrame(
        {
            "beta_deg": beta_range,
            "CS": cs_beta,
            "CS_fit": cs_beta_fit,
        }
    )
    cs_beta_csv_path = save_folder / "cs_vs_sideslip.csv"
    cs_beta_df.to_csv(cs_beta_csv_path, index=False)
    print(f"CS vs sideslip saved to {cs_beta_csv_path}")

    # --- Fit cmz vs sideslip at alpha=8° ---
    cmz_beta = np.zeros(len(beta_range))
    for i, beta in enumerate(beta_range):
        body_aero_polar_with_bridles.va_initialize(Umag, alpha_fixed, beta, 0)
        results = solver.solve(body_aero_polar_with_bridles, gamma_distribution=gamma)
        cmz_beta[i] = results.get("cmz", np.nan)

    cmz_beta_coeffs = np.polyfit(np.deg2rad(beta_range), cmz_beta, 1)
    cmz_beta_fit = np.polyval(cmz_beta_coeffs, np.deg2rad(beta_range))
    print(
        f"Fitted cmz vs sideslip: slope={cmz_beta_coeffs[0]:.4f}, intercept={cmz_beta_coeffs[1]:.4f}"
    )

    plt.figure()
    plt.plot(beta_range, cmz_beta, "o", label="Simulated cmz")
    plt.plot(beta_range, cmz_beta_fit, "-", label="Linear fit")
    plt.xlabel("Sideslip angle (deg)")
    plt.ylabel("cmz")
    plt.title(f"cmz vs Sideslip at alpha={alpha_fixed}°")
    plt.legend()
    plt.grid()
    plt.show()

    cmz_beta_df = pd.DataFrame(
        {
            "beta_deg": beta_range,
            "cmz": cmz_beta,
            "cmz_fit": cmz_beta_fit,
        }
    )
    cmz_beta_csv_path = save_folder / "cmz_vs_sideslip.csv"
    cmz_beta_df.to_csv(cmz_beta_csv_path, index=False)
    print(f"cmz vs sideslip saved to {cmz_beta_csv_path}")


if __name__ == "__main__":
    main()
