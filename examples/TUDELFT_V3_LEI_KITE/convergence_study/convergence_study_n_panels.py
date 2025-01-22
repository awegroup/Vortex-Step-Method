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


def run_batch(
    Umag,
    alpha,
    beta,
    file_path,
    n_ribs,
    is_with_corrected_polar=True,
    is_with_stall=True,
    spanwise_panel_distribution_list=["linear"],
    smoothness_factor=0.08,
):
    results = {}

    n_panels = n_ribs - 1

    for distribution in spanwise_panel_distribution_list:
        print(f"\n{distribution}")
        results[distribution] = {}
        if distribution == "unchanged":
            n_panels_list = [n_panels]
        elif distribution == "split_provided":
            n_panels_list = np.arange(n_panels, 181, n_panels).astype(int)
            n_panels_list = [35, 140]
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
                160,
                170,
            ]
            n_panels_list = [20, 100, 170]
        elif distribution == "cosine":
            n_panels_list = [20, 25, 30, 35, 40, 45, 50, 75, 100]
            n_panels_list = [30, 100]
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
                wing_aero, Umag, alpha, beta, yaw_rate, is_with_stall, smoothness_factor
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


def plot_results(results_list, label_list, Umag, alpha, beta, file_path):
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

    for idx, output in enumerate(outputs):
        ax = axes[idx]
        for result, label in zip(results_list, label_list):
            for distribution, panels_data in result.items():
                # Extract sorted n_panels and corresponding output values
                n_panels_sorted = sorted(panels_data.keys())
                output_values = np.array(
                    [panels_data[n][output] for n in n_panels_sorted]
                )

                ax.plot(
                    n_panels_sorted,
                    output_values,
                    marker="o",
                    label=label + "-" + distribution,
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
        / f"convergence_n_panels_vw_{int(Umag)}_alpha_{alpha:.1f}_beta_{beta:.1f}_filename_{file_path.stem}_stall_effect.pdf"
    )


def plot_3x3_special(result_list_list, label_list, beta_list):
    """
    Plots a 3x3 figure:
      - Rows: beta_list[0], beta_list[1], beta_list[2] (top to bottom).
      - Columns:
          0 => C_L
          1 => C_D
          2 => (if row=0 => C_my) else => C_S
      - Overlays lines from each element of result_list_list[row].
        The function automatically detects distributions by checking the keys of each
        single_result in result_list_list[row] (e.g., "split_provided", "linear", etc.).
      - result_list_list has shape [#betas][#result_sets], e.g. 3 rows if 3 betas,
        each row having one or more result sets.
      - label_list: same length as #result_sets (e.g., ["Breukels", "Polars"]).
      - beta_list: list of betas in the same order (up to 3).

    Each entry of result_list_list[row][res_idx] is a dict like:
        {
           "<some_distribution>": {
               n_panels: { 'cl': ..., 'cd': ..., 'cs': ..., 'cmy': ... }
           },
           "<another_distribution>": {
               n_panels: {...}
           }
        }
    """

    # If you use a custom style function, call it here (optional).
    # set_plot_style()

    # Force a 3x3 figure. Unused rows will be hidden if beta_list < 3.
    n_rows = 3
    n_cols = 3
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(15, 8), sharex=False, sharey=False
    )

    def get_coeff_and_label(row_idx, col_idx):
        """
        Determine which coefficient to plot in a given cell and
        whether to attach X or Y labels.
        row 0 => beta_list[0], row 1 => beta_list[1], row 2 => beta_list[2]
        col 0 => 'cl', col 1 => 'cd', col 2 => if row=0 => 'cmy' else => 'cs'
        """
        # Decide which subplots get axis labels:
        is_with_ylabel = True  # col_idx == 0  # Y-label in first column
        is_with_x_label = row_idx == n_rows - 1  # X-label on bottom row

        if col_idx == 0:
            return "cl", r"$C_L$", is_with_ylabel, is_with_x_label
        elif col_idx == 1:
            return "cd", r"$C_D$", is_with_ylabel, is_with_x_label
        else:
            # Third column: row=0 => 'cmy', row>0 => 'cs'
            if row_idx == 0:
                return "cmy", r"$C_{m_y}$", is_with_ylabel, is_with_x_label
            else:
                return "cs", r"$C_S$", is_with_ylabel, is_with_x_label

    # Iterate over each row (up to 3) based on beta_list
    for row_idx in range(n_rows):
        if row_idx >= len(beta_list):
            # Hide this row if we have fewer betas than 3
            for col_idx in range(n_cols):
                axes[row_idx, col_idx].set_visible(False)
            continue

        beta = beta_list[row_idx]
        row_results = result_list_list[row_idx]  # e.g., [res_breukels, res_polars, ...]

        # Collect legend handles/labels for this row
        row_handles = []
        row_labels = []

        # Combine all distribution keys found in this row’s results
        distribution_keys = set()
        for single_result in row_results:
            distribution_keys.update(single_result.keys())
        # Convert to a sorted list for consistent plotting order
        distribution_keys = sorted(distribution_keys)

        # Plot each column
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]

            coeff_key, coeff_label, is_with_ylabel, is_with_x_label = (
                get_coeff_and_label(row_idx, col_idx)
            )

            if is_with_ylabel:
                ax.set_ylabel(coeff_label)  # , fontsize=12)
            if is_with_x_label:
                ax.set_xlabel(r"$N_{\mathrm{p}}$")  # , fontsize=12)

            # For each result set in row_results
            for single_result, lbl in zip(row_results, label_list):
                # Plot each distribution present in distribution_keys if available in single_result
                for dist in distribution_keys:
                    if dist not in single_result:
                        # This single_result doesn't have dist => skip
                        continue

                    dist_data = single_result[
                        dist
                    ]  # e.g., { n_panels: {cl, cd, cs, cmy, ...} }
                    n_panels_sorted = sorted(dist_data.keys())
                    y_vals = [dist_data[np][coeff_key] for np in n_panels_sorted]

                    if "Polar" in lbl:
                        linestyle = "--"
                    elif "Breukels" in lbl:
                        linestyle = "-"
                    else:
                        linestyle = "-."

                    if "linear" and "stall" and "010" in dist:
                        markerstyle = "d"
                    elif "linear" and "stall" in dist:
                        markerstyle = "x"
                    elif "split_provided" in dist:
                        markerstyle = "s"
                    elif "cosine" in dist:
                        markerstyle = "*"
                    elif "linear" in dist:
                        markerstyle = "o"

                    (line,) = ax.plot(
                        n_panels_sorted,
                        y_vals,
                        marker=markerstyle,
                        markersize=5,
                        linestyle=linestyle,
                        label=f"{lbl} - {dist.replace('_', ' ').title()}",
                    )
                    row_handles.append(line)
                    row_labels.append(f"{lbl} - {dist.replace('_', ' ').title()}")

            ax.grid(True)

        # Legend for this row on the left of the first subplot in the row
        if row_handles and row_labels:
            # Remove duplicates in labels (use a dict to preserve only last handle for each label)
            unique_labels_dict = dict(zip(row_labels, row_handles))
            final_labels = list(unique_labels_dict.keys())
            final_handles = list(unique_labels_dict.values())

            axes[row_idx, 0].legend(
                final_handles,
                final_labels,
                loc="center left",
                bbox_to_anchor=(-1.4, 0.5),  # Adjust horizontal offset as needed
                borderaxespad=0.0,
                title=f"Distributions (β = {beta}°)",
                # fontsize=10,
            )

    ### Ensuring that a value does not ruin the naturally zooomed in ylim
    for i, ax in enumerate(axes.flat):
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
    return fig, axes


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
    alpha_list = [6.8]
    beta_list = [0, 10, 20]

    # 1) Define the 'cases' dictionary for special beta = 0, beta = 10, and "default" for everything else.
    cases = {
        0: {
            "spanwise_panel_distributions": ["split_provided", "linear", "cosine"],
            "params": [
                dict(
                    is_with_corrected_polar=False,
                    is_with_stall=False,
                    smoothness_factor=0.08,
                ),
                dict(
                    is_with_corrected_polar=True,
                    is_with_stall=False,
                    smoothness_factor=0.08,
                ),
            ],
        },
        10: {
            "spanwise_panel_distributions": ["split_provided", "linear"],
            "params": [
                dict(
                    is_with_corrected_polar=False,
                    is_with_stall=False,
                    smoothness_factor=0.08,
                ),
                dict(
                    is_with_corrected_polar=True,
                    is_with_stall=False,
                    smoothness_factor=0.08,
                ),
                dict(
                    is_with_corrected_polar=False,
                    is_with_stall=True,
                    smoothness_factor=0.08,
                ),
                dict(
                    is_with_corrected_polar=True,
                    is_with_stall=True,
                    smoothness_factor=0.08,
                ),
            ],
        },
        "default": {
            "spanwise_panel_distributions": ["linear"],
            "params": [
                dict(
                    is_with_corrected_polar=False,
                    is_with_stall=False,
                    smoothness_factor=0.08,
                ),
                dict(
                    is_with_corrected_polar=True,
                    is_with_stall=False,
                    smoothness_factor=0.08,
                ),
                dict(
                    is_with_corrected_polar=False,
                    is_with_stall=True,
                    smoothness_factor=0.08,
                ),
                dict(
                    is_with_corrected_polar=True,
                    is_with_stall=True,
                    smoothness_factor=0.08,
                ),
            ],
        },
    }

    # 2) A helper function that runs the correct set of run_batch calls for a given beta.
    def run_batch_suite(Umag, alpha, beta, file_path, n_ribs):
        """
        Returns a list of results for the combination of (Umag, alpha, beta, file_path, n_ribs)
        according to the definition in `cases`.
        """
        # Check if beta is special or default
        config = cases[beta] if beta in cases else cases["default"]
        dist_list = config["spanwise_panel_distributions"]

        suite_results = []
        for p in config["params"]:
            # Build kwargs for run_batch
            batch_kwargs = dict(
                Umag=Umag,
                alpha=alpha,
                beta=beta,
                file_path=file_path,
                n_ribs=n_ribs,
                is_with_corrected_polar=p["is_with_corrected_polar"],
                is_with_stall=p["is_with_stall"],
                smoothness_factor=p.get("smoothness_factor", None),
                spanwise_panel_distribution_list=dist_list,
            )
            # Call run_batch
            result = run_batch(**batch_kwargs)
            suite_results.append(result)
        return suite_results

    result_list_list = []

    # 3) Nested loops over file, n_ribs, Umag, alpha, beta
    for file_path, n_ribs in zip(file_paths, n_ribs_list):
        for Umag in Umag_list:
            # Example Reynolds number (replace if your formula differs)
            Re = 1.2 * Umag * 2.628 / 1.79
            print(f"\n--- Umag: {Umag}  -->  Re ~ {Re:.2f}e5 ---")
            for alpha in alpha_list:
                print(f"    alpha: {alpha}")
                for beta in beta_list:
                    print(f"      beta: {beta}")

                    # 4) Run the suite for each (Umag, alpha, beta)
                    batch_results = run_batch_suite(
                        Umag, alpha, beta, file_path, n_ribs
                    )
                    # Append the entire list (one entry per param set) to the master list
                    result_list_list.append(batch_results)

    # 5) Convert to DataFrame and save to CSV
    #    This will give you a DataFrame with each row a list of run_batch() outputs.
    #    If run_batch() itself returns dictionaries, you might want to flatten them first.
    df_results = pd.DataFrame(result_list_list)
    output_path = (
        Path(PROJECT_DIR)
        / "examples"
        / "TUDELFT_V3_LEI_KITE"
        / "convergence_study"
        / "convergence_n_panels_results.csv"
    )
    df_results.to_csv(output_path, index=False)
    print(f"\nSaved convergence study results to: {output_path}")

    # 6) Optional: call your plot function
    label_list = [
        "Breukels",
        "Polar",
        "Breukels + Stall",
        "Polar + Stall",
    ]
    fig, axes = plot_3x3_special(result_list_list, label_list, beta_list)
    fig.savefig(
        Path(PROJECT_DIR)
        / "examples"
        / "TUDELFT_V3_LEI_KITE"
        / "convergence_study"
        / "convergence_n_panels_special.pdf"
    )
    # fig.show() or plt.show() if you want an interactive window

    # cl_list, cd_list, cs_list, cmx_list, cmy_list, cmz_list = [], [], [], [], [], []
    # result_list_list = []
    # for file_path, n_ribs in zip(file_paths, n_ribs_list):
    #     for Umag in Umag_list:
    #         print(f"\n--- Umag:{Umag} Re:{1.2*Umag*2.628/(1.79):.2f}e5 ---")
    #         for alpha in alpha_list:
    #             print(f"\n--- alpha:{alpha} ---")
    #             for beta in beta_list:
    #                 print(f"\n--- beta:{beta} ---")

    #                 if beta == 0:
    #                     spanwise_panel_distributions = [
    #                         "split_provided",
    #                         "linear",
    #                         "cosine",
    #                     ]
    #                     results_breukels = run_batch(
    #                         Umag=Umag,
    #                         alpha=alpha,
    #                         beta=beta,
    #                         file_path=file_path,
    #                         n_ribs=n_ribs,
    #                         is_with_corrected_polar=False,
    #                         is_with_stall=False,
    #                         spanwise_panel_distribution_list=spanwise_panel_distributions,
    #                     )
    #                     results_polars = run_batch(
    #                         Umag=Umag,
    #                         alpha=alpha,
    #                         beta=beta,
    #                         file_path=file_path,
    #                         n_ribs=n_ribs,
    #                         is_with_corrected_polar=True,
    #                         is_with_stall=False,
    #                         spanwise_panel_distribution_list=spanwise_panel_distributions,
    #                     )
    #                     result_list_list.append(
    #                         [
    #                             results_breukels,
    #                             results_polars,
    #                         ]
    #                     )
    #                 elif beta == 10:
    #                     spanwise_panel_distributions = [
    #                         "split_provided",
    #                         "linear",
    #                     ]
    #                     results_breukels = run_batch(
    #                         Umag=Umag,
    #                         alpha=alpha,
    #                         beta=beta,
    #                         file_path=file_path,
    #                         n_ribs=n_ribs,
    #                         is_with_corrected_polar=False,
    #                         is_with_stall=False,
    #                         spanwise_panel_distribution_list=spanwise_panel_distributions,
    #                     )
    #                     results_polars = run_batch(
    #                         Umag=Umag,
    #                         alpha=alpha,
    #                         beta=beta,
    #                         file_path=file_path,
    #                         n_ribs=n_ribs,
    #                         is_with_corrected_polar=True,
    #                         is_with_stall=False,
    #                         spanwise_panel_distribution_list=spanwise_panel_distributions,
    #                     )
    #                     results_breukels_stall = run_batch(
    #                         Umag=Umag,
    #                         alpha=alpha,
    #                         beta=beta,
    #                         file_path=file_path,
    #                         n_ribs=n_ribs,
    #                         is_with_corrected_polar=False,
    #                         is_with_stall=True,
    #                         spanwise_panel_distribution_list=spanwise_panel_distributions,
    #                     )
    #                     results_polars_stall = run_batch(
    #                         Umag=Umag,
    #                         alpha=alpha,
    #                         beta=beta,
    #                         file_path=file_path,
    #                         n_ribs=n_ribs,
    #                         is_with_corrected_polar=True,
    #                         is_with_stall=True,
    #                         spanwise_panel_distribution_list=spanwise_panel_distributions,
    #                     )
    #                     result_list_list.append(
    #                         [
    #                             results_breukels,
    #                             results_polars,
    #                             results_breukels_stall,
    #                             results_polars_stall,
    #                         ]
    #                     )
    #                 else:
    #                     spanwise_panel_distributions = [
    #                         # "split_provided",
    #                         "linear",
    #                     ]
    #                     results_breukels = run_batch(
    #                         Umag=Umag,
    #                         alpha=alpha,
    #                         beta=beta,
    #                         file_path=file_path,
    #                         n_ribs=n_ribs,
    #                         is_with_corrected_polar=False,
    #                         is_with_stall=False,
    #                         spanwise_panel_distribution_list=spanwise_panel_distributions,
    #                     )
    #                     results_polars = run_batch(
    #                         Umag=Umag,
    #                         alpha=alpha,
    #                         beta=beta,
    #                         file_path=file_path,
    #                         n_ribs=n_ribs,
    #                         is_with_corrected_polar=True,
    #                         is_with_stall=False,
    #                         spanwise_panel_distribution_list=spanwise_panel_distributions,
    #                     )
    #                     results_breukels_stall = run_batch(
    #                         Umag=Umag,
    #                         alpha=alpha,
    #                         beta=beta,
    #                         file_path=file_path,
    #                         n_ribs=n_ribs,
    #                         is_with_corrected_polar=False,
    #                         is_with_stall=True,
    #                         spanwise_panel_distribution_list=spanwise_panel_distributions,
    #                     )
    #                     results_polars_stall = run_batch(
    #                         Umag=Umag,
    #                         alpha=alpha,
    #                         beta=beta,
    #                         file_path=file_path,
    #                         n_ribs=n_ribs,
    #                         is_with_corrected_polar=True,
    #                         is_with_stall=True,
    #                         spanwise_panel_distribution_list=spanwise_panel_distributions,
    #                     )
    #                     results_breukels_stall_005 = run_batch(
    #                         Umag=Umag,
    #                         alpha=alpha,
    #                         beta=beta,
    #                         file_path=file_path,
    #                         n_ribs=n_ribs,
    #                         is_with_corrected_polar=False,
    #                         is_with_stall=True,
    #                         smoothness_factor=0.10,
    #                         spanwise_panel_distribution_list=spanwise_panel_distributions,
    #                     )
    #                     results_polars_stall_005 = run_batch(
    #                         Umag=Umag,
    #                         alpha=alpha,
    #                         beta=beta,
    #                         file_path=file_path,
    #                         n_ribs=n_ribs,
    #                         is_with_corrected_polar=True,
    #                         is_with_stall=True,
    #                         smoothness_factor=0.10,
    #                         spanwise_panel_distribution_list=spanwise_panel_distributions,
    #                     )
    #                     result_list_list.append(
    #                         [
    #                             results_breukels,
    #                             results_polars,
    #                             results_breukels_stall,
    #                             results_polars_stall,
    #                             results_breukels_stall_005,
    #                             results_polars_stall_005,
    #                         ]
    #                     )

    # ### Save the results
    # label_list = [
    #     "Breukels",
    #     "Polar",
    #     "Breukels + Stall (0.08)",
    #     "Polar + Stall (0.08)",
    #     "Breukels + Stall (0.10)",
    #     "Polar + Stall (0.10)",
    # ]
    # df_results = pd.DataFrame(result_list_list)
    # df_results.to_csv(
    #     Path(PROJECT_DIR)
    #     / "examples"
    #     / "TUDELFT_V3_LEI_KITE"
    #     / "convergence_study"
    #     / "convergence_n_panels_results.csv"
    # )

    # fig, axes = plot_3x3_special(result_list_list, label_list, beta_list)
    # fig.savefig(
    #     Path(PROJECT_DIR)
    #     / "examples"
    #     / "TUDELFT_V3_LEI_KITE"
    #     / "convergence_study"
    #     / "convergence_n_panels_special.pdf"
    # )
