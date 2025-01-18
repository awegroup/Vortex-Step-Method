import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from VSM.WingGeometry import Wing
from VSM.WingAerodynamics import WingAerodynamics
from VSM.Solver import Solver
from VSM.plotting import (
    plot_polars,
    plot_distribution,
    plot_geometry,
    plot_panel_coefficients,
    process_panel_coefficients,
)
from VSM.plot_styling import set_plot_style
from VSM.interactive import interactive_plot

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent


def create_wing_aero(
    file_path,
    n_panels,
    spanwise_panel_distribution,
    is_with_corrected_polar=False,
    path_polar_data_dir="",
):
    df = pd.read_csv(file_path, delimiter=",")  # , skiprows=1)
    LE_x_array = df["LE_x"].values
    LE_y_array = df["LE_y"].values
    LE_z_array = df["LE_z"].values
    TE_x_array = df["TE_x"].values
    TE_y_array = df["TE_y"].values
    TE_z_array = df["TE_z"].values
    d_tube_array = df["d_tube"].values
    camber_array = df["camber"].values

    ## populating this list
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs = []

    for i in range(len(LE_x_array)):
        LE = np.array([LE_x_array[i], LE_y_array[i], LE_z_array[i]])
        TE = np.array([TE_x_array[i], TE_y_array[i], TE_z_array[i]])
        rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs.append(
            [LE, TE, ["lei_airfoil_breukels", [d_tube_array[i], camber_array[i]]]]
        )
    CAD_wing = Wing(n_panels, spanwise_panel_distribution)

    for i, CAD_rib_i in enumerate(
        rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
    ):
        CAD_rib_i_0 = CAD_rib_i[0]
        CAD_rib_i_1 = CAD_rib_i[1]

        if is_with_corrected_polar:
            ### using corrected polar
            df_polar_data = pd.read_csv(
                Path(path_polar_data_dir) / f"corrected_polar_{i}.csv"
            )
            alpha = df_polar_data["alpha"].values
            cl = df_polar_data["cl"].values
            cd = df_polar_data["cd"].values
            cm = df_polar_data["cm"].values
            polar_data = ["polar_data", np.array([alpha, cl, cd, cm])]
            CAD_wing.add_section(CAD_rib_i_0, CAD_rib_i_1, polar_data)
        else:
            ### using breukels
            CAD_wing.add_section(CAD_rib_i_0, CAD_rib_i_1, CAD_rib_i[2])

    wing_aero = WingAerodynamics([CAD_wing])

    return wing_aero


def run_solver(wing_aero, Umag, alpha, beta, yaw_rate):
    wing_aero.va_initialize(Umag, alpha, beta, yaw_rate)
    solver = Solver()
    results = solver.solve(wing_aero)
    cl = results["cl"]
    cd = results["cd"]
    cs = results["cs"]
    cmx = results["cmx"]
    cmy = results["cmy"]
    cmz = results["cmz"]
    return cl, cd, cs, cmx, cmy, cmz


def run_batch(Umag, alpha, beta, file_path, n_ribs, is_with_corrected_polar=True):
    results = {}
    spanwise_panel_distributions = [
        # "unchanged",
        "split_provided",
        "linear",
        # "cosine",
        # "cosine_van_garrel",
    ]  # Add more distributions as needed

    n_panels = n_ribs - 1

    for distribution in spanwise_panel_distributions:
        print(f"\n{distribution}")
        results[distribution] = {}
        if distribution == "unchanged":
            n_panels_list = [n_panels]
        elif distribution == "split_provided":
            n_panels_list = np.arange(n_panels, 150, n_panels).astype(int)
        elif distribution == "linear":
            # n_panels_list = np.arange(n_panels, 125, n_panels).astype(int)
            n_panels_list = [
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                110,
                120,
                130,
                140,
                150,
            ]
        elif distribution == "cosine":
            n_panels_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100]
        elif distribution == "cosine_van_garrel":
            n_panels_list = [10, 15, 20, 25, 30, 35]
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        for n_panels in n_panels_list:
            print(f"n_panels:{n_panels}")
            wing_aero = create_wing_aero(
                file_path,
                n_panels,
                distribution,
                is_with_corrected_polar,
                (
                    Path(PROJECT_DIR)
                    / "examples"
                    / "TUDELFT_V3_LEI_KITE"
                    / "polar_engineering"
                    / "csv_files"
                ),
            )
            yaw_rate = 0

            cl, cd, cs, cmx, cmy, cmz = run_solver(
                wing_aero, Umag, alpha, beta, yaw_rate
            )

            results[distribution][n_panels] = {
                "cl": cl,
                "cd": cd,
                "cs": cs,
                "cmx": cmx,
                "cmy": cmy,
                "cmz": cmz,
            }

    return results


def plot_results(results_breukels, results_polars, Umag, alpha, beta, file_path):
    set_plot_style()

    # Define the aerodynamic coefficients to plot
    outputs = ["cl", "cd", "cs", "cmx", "cmy", "cmz"]
    output_titles = {
        "cl": r"$C_L$",
        "cd": r"$C_D$",
        "cs": r"$C_S$",
        "cmx": r"$CM_X$ (roll)",
        "cmy": r"$CM_Y$ (pitch)",
        "cmz": r"$CM_Z$ (yaw)",
    }

    # Set up the subplot grid (3 rows x 2 columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()  # Flatten to easily iterate

    max_value = 5
    for idx, output in enumerate(outputs):
        ax = axes[idx]
        for distribution, panels_data in results_breukels.items():
            # Extract sorted n_panels and corresponding output values
            n_panels_sorted = sorted(panels_data.keys())
            output_values = np.array([panels_data[n][output] for n in n_panels_sorted])

            ax.plot(
                n_panels_sorted,
                output_values,
                marker="o",
                label="breukel-" + distribution,
            )

        for distribution, panels_data in results_polars.items():
            # Extract sorted n_panels and corresponding output values
            n_panels_sorted = sorted(panels_data.keys())
            output_values = np.array([panels_data[n][output] for n in n_panels_sorted])
            ax.plot(
                n_panels_sorted,
                output_values,
                marker="o",
                label="polar-" + distribution,
            )

        # Set labels and title
        ax.set_xlabel("Number of Panels")
        ax.set_ylabel(output_titles.get(output, output))
        ax.grid(True)

    axes[1].legend()

    # Remove any unused subplots if the grid is larger than needed
    total_subplots = 3 * 2
    if len(axes) > len(outputs):
        for idx in range(len(outputs), len(axes)):
            fig.delaxes(axes[idx])

    # adjusting y limits
    ### Ensuring that a value does not ruin the naturally zooomed in ylim
    for i, ax in enumerate(axes.flat):
        if i == 4:  # CMy
            y_min_allowed, y_max_allowed = -2, 2
        else:
            y_min_allowed, y_max_allowed = -1.5, 1.5

        # Collect all y-data from the lines in the current axis
        y_data = np.concatenate([line.get_ydata() for line in ax.get_lines()])

        # Identify data within the allowed range
        in_range = y_data[(y_data >= y_min_allowed) & (y_data <= y_max_allowed)]

        if in_range.size > 0:
            # Optionally add some padding to the y-limits
            padding = 0.05 * (in_range.max() - in_range.min())
            ax.set_ylim(in_range.min() - padding, in_range.max() + padding)
        else:
            # If no data is within the range, you might choose to set default limits or skip
            pass  # Or set default limits, e.g., ax.set_ylim(y_min_allowed, y_max_allowed)

    plt.tight_layout()
    plt.savefig(
        Path(PROJECT_DIR)
        / "examples"
        / "TUDELFT_V3_LEI_KITE"
        / "convergence_study"
        / f"convergence_n_panels_vw_{int(Umag)}_alpha_{alpha:.1f}_beta_{beta:.1f}_filename_{file_path.stem}.pdf"
    )


if __name__ == "__main__":
    # user input

    file_path_geometry = (
        Path(PROJECT_DIR) / "data" / "TUDELFT_V3_LEI_KITE" / "geometry.csv"
    )
    file_path_19ribs = (
        Path(PROJECT_DIR)
        / "processed_data"
        / "TUDELFT_V3_LEI_KITE"
        / "rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs.csv"
    )
    file_paths = [file_path_geometry, file_path_19ribs]
    file_path_geometry_corrected = (
        Path(PROJECT_DIR) / "data" / "TUDELFT_V3_LEI_KITE" / "geometry_corrected.csv"
    )
    file_paths = [file_path_geometry_corrected]
    n_ribs_list = [36]
    Umag_list = [3.15]
    alpha_list = [6.8, 11.9]
    beta_list = [0, 10, 20]

    # cl_list, cd_list, cs_list, cmx_list, cmy_list, cmz_list = [], [], [], [], [], []
    for file_path, n_ribs in zip(file_paths, n_ribs_list):
        for Umag in Umag_list:
            print(f"\n--- Umag:{Umag} Re:{1.2*Umag*2.628/(1.79):.2f}e5 ---")
            for alpha in alpha_list:
                print(f"\n--- alpha:{alpha} ---")
                for beta in beta_list:
                    print(f"\n--- beta:{beta} ---")
                    results_breukels = run_batch(
                        Umag=Umag,
                        alpha=alpha,
                        beta=beta,
                        file_path=file_path,
                        n_ribs=n_ribs,
                        is_with_corrected_polar=False,
                    )
                    results_polars = run_batch(
                        Umag=Umag,
                        alpha=alpha,
                        beta=beta,
                        file_path=file_path,
                        n_ribs=n_ribs,
                        is_with_corrected_polar=True,
                    )

                    plot_results(
                        results_breukels,
                        results_polars,
                        Umag=Umag,
                        alpha=alpha,
                        beta=beta,
                        file_path=file_path,
                    )
