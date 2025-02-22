import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math
import json
from VSM.WingGeometry import Wing
from VSM.BodyAerodynamics import WingAerodynamics
from VSM.Solver import Solver
from VSM.plotting import (
    plot_polars,
    plot_distribution,
    plot_geometry,
    plot_panel_coefficients,
    process_panel_coefficients,
)
from VSM.plot_styling import set_plot_style, plot_on_ax
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


def run_solver(
    wing_aero, Umag, alpha, beta, yaw_rate, is_with_stall, smoothness_factor=0.08
):
    wing_aero.va_initialize(Umag, alpha, beta, yaw_rate)
    solver = Solver(
        is_with_artificial_damping=is_with_stall, smoothness_factor=smoothness_factor
    )
    results = solver.solve(wing_aero)
    cl = results["cl"]
    cd = results["cd"]
    cs = results["cs"]
    cmx = results["cmx"]
    cmy = results["cmy"]
    cmz = results["cmz"]
    return cl, cd, cs, cmx, cmy, cmz


def plot_3x3_special_new(csv_file_dir, alpha_list, beta_list):

    set_plot_style()

    n_rows = 3
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 7), sharex=True)

    def get_coeff_and_label(row_idx, col_idx, n_rows=3, n_cols=3):
        """
        Determine which coefficient to plot in a given cell and
        whether to attach X or Y labels.
        row 0 => beta_list[0], row 1 => beta_list[1], row 2 => beta_list[2]
        col 0 => 'cl', col 1 => 'cd', col 2 => if row=0 => 'cmy' else => 'cs'
        """
        # Decide which subplots get axis labels:
        is_with_ylabel = True  # Only first column gets y-label
        is_with_x_label = row_idx == n_rows - 1  # Only bottom row gets x-label

        if col_idx == 0:
            return "cl", r"$C_{\mathrm{L}}$", is_with_ylabel, is_with_x_label
        elif col_idx == 1:
            return "cd", r"$C_{\mathrm{D}}$", is_with_ylabel, is_with_x_label
        else:
            # Third column: row=0 => 'cmy', row>0 => 'cs'
            if row_idx == 0 or row_idx == 1 or row_idx == 2:
                return "cmy", r"$C_{\mathrm{M,y}}$", is_with_ylabel, is_with_x_label
            else:
                return "cs", r"$C_{\mathrm{s}}$", is_with_ylabel, is_with_x_label

    # Keep track of all lines/labels for a figure-level legend
    all_handles = []
    all_labels = []

    # --- Main loop over rows (beta values) ---
    for row_idx in range(n_rows):
        if row_idx >= len(beta_list):
            # Hide this row if we have fewer betas than rows
            for col_idx in range(n_cols):
                axes[row_idx, col_idx].set_visible(False)
            continue

        alpha = alpha_list[row_idx]
        beta = beta_list[row_idx]

        # Read all CSV files matching beta in their name
        for file in Path(csv_file_dir).rglob(f"*alpha_{alpha}_beta_{beta}*"):
            filename_parts = file.stem.split("_")

            # Safely parse the required parts
            try:
                # Adjust indices if your filename pattern is different
                alpha = filename_parts[2]
                beta = filename_parts[4]
                distribution = filename_parts[6]
                is_corrected_str = filename_parts[8]
                is_stall_str = filename_parts[10]

                is_corrected = is_corrected_str.lower() == "true"
                is_stall = is_stall_str.lower() == "true"

                print(
                    f"File: {file}, \n alpha: {alpha}, beta: {beta}, distribution: {distribution}, is_corrected: {is_corrected}, is_stall: {is_stall}"
                )
            except IndexError:
                # Fallback if the file name doesn't match the pattern
                distribution = "unknown_dist"
                is_corrected = False
                is_stall = False

            # Read CSV
            df = pd.read_csv(file, index_col="coeff")
            n_panels_sorted = df.columns.astype(int).tolist()

            # Loop over columns in this row
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                coeff_key, coeff_label, is_with_ylabel, is_with_x_label = (
                    get_coeff_and_label(row_idx, col_idx)
                )

                # if col_idx == 0:
                #     # Option A: Vertical text, matching y-label orientation
                #     ax.text(
                #         -0.25,  # move left of the axis (experiment with how negative you need)
                #         0.5,  # halfway up the axis
                #         rf"$\alpha =$ {alpha}"
                #         + r"$^{\circ}$"
                #         + f"\n"
                #         + rf"$\beta =$ {beta}"
                #         + r"$^{\circ}$",
                #         rotation=0,
                #         va="center",
                #         ha="center",
                #         transform=ax.transAxes,
                #         # fontsize=11,  # Match the fontsize of other labels
                #         # fontweight="normal",  # Ensure normal weight
                #         # fontstyle="normal",  # Ensure normal style
                #         # fontfamily="sans-serif",  # Match the font family
                #     )
                # ax.annotate(
                #     "",
                #     xy=(0.5, 1.05),
                #     xycoords="axes fraction",
                #     xytext=(0.5, 0.95),
                #     textcoords="axes fraction",
                #     arrowprops=dict(arrowstyle="<|-|>", lw=1.5),
                # )

                # Set axis labels if needed
                ax.tick_params(labelbottom=False)
                if is_with_ylabel:
                    if col_idx == 0:
                        label = (
                            rf"$\alpha =$ {alpha}"
                            + r"$^{\circ}$"
                            + f"\n"
                            + rf"$\beta =$ {beta}"
                            + r"$^{\circ}$"
                            + f"\n"
                            + f"\n"
                            + coeff_label
                        )
                    else:
                        label = coeff_label
                    ax.set_ylabel(label)  # , fontsize=11)
                if is_with_x_label:
                    ax.set_xlabel(r"$N_{\mathrm{p}}$")  # , fontsize=11)
                    ax.tick_params(labelbottom=True)

                # Extract y-values for the chosen coefficient
                y_vals = [df[str(n_panels)][coeff_key] for n_panels in n_panels_sorted]

                # Choose marker style based on distribution
                if distribution == "splitprovided":
                    lbl_dist = " (strut split)"
                    markerstyle = "o"
                else:
                    lbl_dist = ""
                    markerstyle = ""

                # Determine color/linestyle/label
                if is_corrected:
                    color = "red"
                    if is_stall:
                        linestyle = "solid"
                        lbl = "Corrected + Smoothening"
                    else:
                        linestyle = "dashed"
                        lbl = "Corrected"
                else:
                    color = "blue"
                    if is_stall:
                        linestyle = "dashdot"
                        lbl = "Breukels + Smoothening"
                    else:
                        linestyle = "dotted"
                        lbl = "Breukels"

                # Combine label with distribution
                final_label = f"{lbl}{lbl_dist}"

                (line,) = ax.plot(
                    n_panels_sorted,
                    y_vals,
                    marker=markerstyle,
                    markersize=3.5,
                    linestyle=linestyle,
                    label=final_label,
                    color=color,
                )

                # Collect all handles/labels for a global legend
                all_handles.append(line)
                all_labels.append(final_label)

    # --- After plotting all subplots, adjust y-limits & create global legend ---

    # 1) Adjust the y-limits to exclude extreme outliers
    y_min_allowed, y_max_allowed = -1.5, 1.5
    for i, ax in enumerate(axes.flat):
        if not ax.lines:
            # Skip axes with no data
            continue
        if i in [2, 5, 8]:  # CMy
            y_min_allowed, y_max_allowed = -2, 1.5
        y_data = np.concatenate([line.get_ydata() for line in ax.lines])
        # Identify data within an allowed range
        in_range = y_data[(y_data >= y_min_allowed) & (y_data <= y_max_allowed)]
        if in_range.size > 0:
            y_min, y_max = in_range.min(), in_range.max()
            # Add small padding
            padding = 0.05 * (y_max - y_min if (y_max - y_min) != 0 else 1.0)
            ax.set_ylim(y_min - padding, y_max + padding)

    # 2) Remove duplicate legend entries
    unique_labels_dict = {}
    for handle, label in zip(all_handles, all_labels):
        unique_labels_dict[label] = handle
    final_labels = list(unique_labels_dict.keys())
    final_handles = list(unique_labels_dict.values())

    # 3) Create a figure-level legend below the plots
    # Adjust ncol to fit your needs
    # adjusting the ordering of the legend
    # # swap 4 and 7
    # final_handles[4], final_handles[7] = final_handles[7], final_handles[4]
    # final_labels[4], final_labels[7] = final_labels[7], final_labels[4]

    # # swap 4 and 5
    # final_handles[4], final_handles[5] = final_handles[5], final_handles[4]
    # final_labels[4], final_labels[5] = final_labels[5], final_labels[4]

    # # swap 6 and 7
    # final_handles[6], final_handles[7] = final_handles[7], final_handles[6]
    # final_labels[6], final_labels[7] = final_labels[7], final_labels[6]

    # swap 2 and 4
    final_handles[2], final_handles[4] = final_handles[4], final_handles[2]
    final_labels[2], final_labels[4] = final_labels[4], final_labels[2]

    # swap 3 and 5
    final_handles[3], final_handles[5] = final_handles[5], final_handles[3]
    final_labels[3], final_labels[5] = final_labels[5], final_labels[3]

    fig.legend(
        handles=final_handles,
        labels=final_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=True,
        # title="Legends",
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.17)  # Make space for legend below
    return fig, axes


def run_batch_new(
    Umag,
    alpha,
    beta,
    file_path,
    n_ribs,
    distribution,
    is_with_corrected_polar=True,
    is_with_stall=True,
    smoothness_factor=0.08,
    path_polar_data_dir=Path(PROJECT_DIR)
    / "examples"
    / "TUDELFT_V3_LEI_KITE"
    / "polar_engineering"
    / "csv_files",
):

    n_panels = n_ribs - 1

    print(f"\n{distribution}")
    results = {}
    if distribution == "unchanged":
        n_panels_list = [n_panels]
    elif distribution == "splitprovided":
        n_panels_list = np.arange(n_panels, 250, n_panels).astype(int)
        # n_panels_list = [35, 70]
    elif distribution == "linear":
        # n_panels_list = np.arange(n_panels, 125, n_panels).astype(int)
        n_panels_list = [
            5,
            10,
            20,
            25,
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
            175,
            200,
            225,
            250,
        ]
        # n_panels_list = [10, 50]
    elif distribution == "cosine":
        n_panels_list = [20, 25, 30, 35, 40, 45, 50, 75, 100]
        # n_panels_list = [30, 100]
    elif distribution == "cosine_van_garrel":
        n_panels_list = [10, 15, 20, 25, 30, 35]
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    ##TODO: remove
    # n_panels_list = [35, 70, 105]
    for n_panels in n_panels_list:
        print(f"n_panels:{n_panels}")
        if distribution == "splitprovided":
            distribution_for_VSM = "split_provided"
        else:
            distribution_for_VSM = distribution
        wing_aero = create_wing_aero(
            file_path,
            n_panels,
            distribution_for_VSM,
            is_with_corrected_polar,
            path_polar_data_dir,
        )
        yaw_rate = 0

        cl, cd, cs, cmx, cmy, cmz = run_solver(
            wing_aero, Umag, alpha, beta, yaw_rate, is_with_stall, smoothness_factor
        )

        results[n_panels] = {
            "cl": cl,
            "cd": cd,
            "cs": cs,
            "cmx": cmx,
            "cmy": cmy,
            "cmz": cmz,
        }

    return results


def save_results(
    alpha_list,
    beta_list,
    file_path,
    convergence_data_dir,
    n_ribs,
    Umag,
    path_polar_data_dir,
):
    for i, (alpha, beta) in enumerate(zip(alpha_list, beta_list)):
        # if i in [0, 1, 2]:
        #     distribution_list = ["splitprovided", "linear"]
        #     is_stall_list = [False]
        # else:
        #     distribution_list = ["splitprovided", "linear"]
        is_stall_list = [False, True]
        is_polar_list = [False, True]

        for is_polar in is_polar_list:
            if is_polar:
                distribution_list = ["splitprovided", "linear"]
            else:
                distribution_list = ["linear"]
            for distribution in distribution_list:
                for is_stall in is_stall_list:
                    results = run_batch_new(
                        Umag,
                        alpha,
                        beta,
                        file_path,
                        n_ribs,
                        distribution=distribution,
                        is_with_corrected_polar=is_polar,
                        is_with_stall=is_stall,
                        smoothness_factor=0.08,
                        path_polar_data_dir=path_polar_data_dir,
                    )
                    # converting dict to df
                    df = pd.DataFrame(results)
                    df.to_csv(
                        Path(convergence_data_dir)
                        / f"results_alpha_{alpha}_beta_{beta}_dist_{distribution}_corrected_{is_polar}_stall_{is_stall}.csv",
                        index_label="coeff",
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
    convergence_data_dir = (
        Path(PROJECT_DIR)
        / "examples"
        / "TUDELFT_V3_LEI_KITE"
        / "convergence_study"
        / "csv_files"
    )
    file_path = (
        Path(PROJECT_DIR) / "data" / "TUDELFT_V3_LEI_KITE" / "geometry_corrected.csv"
    )
    path_polar_data_dir = (
        Path(PROJECT_DIR)
        / "examples"
        / "TUDELFT_V3_LEI_KITE"
        / "polar_engineering"
        / "csv_files"
    )

    n_ribs = 36
    Umag = 3.15
    alpha_list = [6.8, 20, 20]
    beta_list = [10, 10, 20]

    save_results(
        alpha_list,
        beta_list,
        file_path,
        convergence_data_dir,
        n_ribs,
        Umag,
        path_polar_data_dir,
    )
    fig, axes = plot_3x3_special_new(convergence_data_dir, alpha_list, beta_list)
    fig.savefig(
        Path(PROJECT_DIR)
        / "examples"
        / "TUDELFT_V3_LEI_KITE"
        / "convergence_study"
        / "results"
        / "convergence_n_panels_new.pdf"
    )

    ##TODO: left off trying to understand the data flow exactly.
    # wnat to make the point that different distributions dont matter.
    # and that smoothening slows convergence
    # and that breukels < corrected

    #################################
    # ## trying a no billow version
    #################################
    # convergence_data_dir = (
    #     Path(PROJECT_DIR)
    #     / "examples"
    #     / "TUDELFT_V3_LEI_KITE"
    #     / "convergence_study"
    #     / "csv_files_no_billow"
    # )
    # file_path = (
    #     Path(PROJECT_DIR)
    #     / "data"
    #     / "TUDELFT_V3_LEI_KITE"
    #     / "rib_list_height_scaled.csv"
    # )
    # path_polar_data_dir = (
    #     Path(PROJECT_DIR)
    #     / "examples"
    #     / "TUDELFT_V3_LEI_KITE"
    #     / "polar_engineering"
    #     / "csv_files_no_billow"
    # )

    # n_ribs = 19
    # Umag = 3.15
    # alpha_list = [6.8, 20, 20]
    # beta_list = [10, 10, 20]

    # save_results(
    #     alpha_list,
    #     beta_list,
    #     file_path,
    #     convergence_data_dir,
    #     n_ribs,
    #     Umag,
    #     path_polar_data_dir,
    # )
    # fig, axes = plot_3x3_special_new(convergence_data_dir, alpha_list, beta_list)
    # fig.savefig(
    #     Path(PROJECT_DIR)
    #     / "examples"
    #     / "TUDELFT_V3_LEI_KITE"
    #     / "convergence_study"
    #     / "convergence_n_panels_no_billow.pdf"
    # )
