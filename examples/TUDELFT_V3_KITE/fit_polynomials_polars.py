import numpy as np
import logging
import time
import matplotlib.pyplot as plt

import pandas as pd
from pathlib import Path
from VSM.core.WingGeometry import Wing
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.plotting import (
    plot_polars,
    plot_geometry,
    creating_geometry_plot,
)
from VSM.interactive import interactive_plot
from VSM.fitting import fit_and_evaluate_model


def main():
    """
    Fit and evaluate polynomials for polars using YAML config input.
    """

    ### 1. defining paths
    PROJECT_DIR = Path(__file__).resolve().parents[2]

    yaml_config_path = (
        Path(PROJECT_DIR)
        / "data"
        / "TUDELFT_V3_KITE"
        / "config_kite_CAD_CFD_polars.yaml"
    )
    ### 2. defining settings
    n_panels = 40
    spanwise_panel_distribution = "uniform"
    solver = Solver(reference_point=[0, 0, 0])

    ### 3. Loading kite geometry from CSV file and instantiating BodyAerodynamics
    print(f"\nCreating polar input with bridles from YAML")
    body_aero_polar_with_bridles = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=yaml_config_path,
        spanwise_panel_distribution=spanwise_panel_distribution,
        is_with_bridles=True,
    )

    ### 4. Setting va
    Umag = 20
    yaw_rate = 0
    body_aero_polar_with_bridles.va_initialize(Umag, 5, 0, yaw_rate)

    ### 7. Plotting the polar curves for different angles of attack and side slip angles
    # and saving in results with literature
    save_folder = Path(PROJECT_DIR) / "results" / "V9_KITE"

    begin_time = time.time()
    angle_of_attack_range = np.linspace(0, 15, 12)
    gamma = None
    center_of_pressure = np.zeros((len(angle_of_attack_range), 3))
    total_force = np.zeros((len(angle_of_attack_range), 3))
    cl = np.zeros((len(angle_of_attack_range)))
    cd = np.zeros((len(angle_of_attack_range)))
    aero_roll = np.zeros((len(angle_of_attack_range)))
    for i, angle_i in enumerate(angle_of_attack_range):
        body_aero_polar_with_bridles.va_initialize(Umag, angle_i, 0, yaw_rate)

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


if __name__ == "__main__":
    main()
