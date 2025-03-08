import numpy as np
import logging
import matplotlib.pyplot as plt

import pandas as pd
from pathlib import Path
from VSM.WingGeometry import Wing
from VSM.BodyAerodynamics import BodyAerodynamics
from VSM.Solver import Solver
from VSM.plotting import plot_polars, plot_distribution
from VSM.interactive import interactive_plot
from VSM.plot_styling import set_plot_style, plot_on_ax
import time as time
import math

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent

file_path = Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "wing_geometry.csv"
path_polar_data_dir = (
    Path(PROJECT_DIR)
    / "examples"
    / "TUDELFT_V3_KITE"
    / "polar_engineering"
    / "csv_files"
)
save_folder = (
    Path(PROJECT_DIR)
    / "examples"
    / "TUDELFT_V3_KITE"
    / "stall_model_testing"
    / "results"
    / "testing_allowed_error"
)
# Operating Conditions
Umag = 3.15
angle_of_attack = 6.5
side_slip = 0
yaw_rate = 0

# Wing and Aerodynamic Setup
n_panels = 50
spanwise_panel_distribution = "uniform"

# Initialize body_aero
wing_instance = Wing(n_panels, spanwise_panel_distribution)
body_aero_uniform = BodyAerodynamics.from_file(
    wing_instance,
    file_path,
    is_with_corrected_polar=True,
    path_polar_data_dir=path_polar_data_dir,
)
body_aero_uniform.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)

save_dir = Path(PROJECT_DIR) / "examples" / "TUDELFT_V3_KITE" / ""


def testing_solver_setting(
    save_folder,
    body_aero_list,
    parameter="allowed_error",
    value_list=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    max_iterations=5e4,
    relaxation_factor=1e-6,
    gamma_loop_type="base",
    core_radius_fraction=1e-20,
    alpha_range=np.linspace(0, 25, 20),
    alpha_range_distribution=[19, 20, 21, 22, 23],
    beta_range=[0, 3, 6, 9, 12],
    beta_range_distribution=[0, 3, 6],
    Umag=3.15,
    angle_of_attack=6.5,
    side_slip=0,
    yaw_rate=0,
):
    solver_list = []
    label_list = []
    y_coords_list = []
    for value, body_aero in zip(value_list, body_aero_list):
        solver_list.append(
            Solver(
                allowed_error=value,
                max_iterations=max_iterations,
                relaxation_factor=relaxation_factor,
                gamma_loop_type=gamma_loop_type,
                core_radius_fraction=core_radius_fraction,
            )
        )
        label_list.append(f"{parameter}={value}")
        y_coords_list.append([panel.control_point[1] for panel in body_aero.panels])

    # plotting alpha-polar
    plot_polars(
        solver_list=solver_list,
        body_aero_list=body_aero_list,
        label_list=label_list,
        angle_range=alpha_range,
        angle_type="angle_of_attack",
        angle_of_attack=angle_of_attack,
        side_slip=side_slip,
        yaw_rate=yaw_rate,
        Umag=Umag,
        title=f"alphasweep_{parameter}",
        data_type=".pdf",
        save_path=Path(save_folder),
        is_save=True,
        is_show=False,
    )
    # plotting beta-polar
    plot_polars(
        solver_list=solver_list,
        body_aero_list=body_aero_list,
        label_list=label_list,
        angle_range=beta_range,
        angle_type="side_slip",
        angle_of_attack=angle_of_attack,
        side_slip=side_slip,
        yaw_rate=yaw_rate,
        Umag=Umag,
        title=f"betasweep_{parameter}",
        data_type=".pdf",
        save_path=Path(save_folder),
        is_save=True,
        is_show=False,
    )
    # plotting distributions
    for side_slip in beta_range_distribution:
        for alpha in alpha_range_distribution:
            print(f"\nalpha: {alpha}")
            results_list = []
            y_coords_list = []
            for solver, body_aero in zip(solver_list, body_aero_list):
                print(f"\n{parameter}={solver.allowed_error}")
                body_aero.va_initialize(Umag, alpha, side_slip, yaw_rate)
                results_list.append(solver.solve(body_aero, gamma_distribution=None))

            plot_distribution(
                y_coordinates_list=y_coords_list,
                results_list=results_list,
                label_list=label_list,
                title=f"spanwise_distribution_{parameter}_{side_slip}_alpha_{alpha}",
                data_type=".pdf",
                save_path=save_folder,
                is_save=True,
                is_show=False,
            )


#     #### plotting distributions
#     gamma = None
#     alpha_range = [19, 20, 21, 22, 23]
#     y_coords = [panel.control_point[1] for panel in body_aero_uniform.panels]
#     for side_slip in [0, 5]:
#         for alpha in alpha_range:
#             print(f"\nalpha: {alpha}")
#             body_aero_uniform.va_initialize(Umag, alpha, side_slip, yaw_rate)
#             results_base_e1 = solver_base_e1.solve(
#                 body_aero_uniform, gamma_distribution=None
#             )
#             results_base_e2 = solver_base_e2.solve(
#                 body_aero_uniform, gamma_distribution=None
#             )
#             results_base_e3 = solver_base_e3.solve(
#                 body_aero_uniform, gamma_distribution=None
#             )
#             results_base_e4 = solver_base_e4.solve(
#                 body_aero_uniform, gamma_distribution=None
#             )
#             results_base_e5 = solver_base_e5.solve(
#                 body_aero_uniform, gamma_distribution=None
#             )
#             results_base_e6 = solver_base_e6.solve(
#                 body_aero_uniform, gamma_distribution=None
#             )
#             results_base_e7 = solver_base_e7.solve(
#                 body_aero_uniform, gamma_distribution=None
#             )
#             results_base_e8 = solver_base_e8.solve(
#                 body_aero_uniform, gamma_distribution=None
#             )

#             plot_distribution(
#                 y_coordinates_list=[
#                     y_coords,
#                     y_coords,
#                     y_coords,
#                     y_coords,
#                     y_coords,
#                     y_coords,
#                     y_coords,
#                     y_coords,
#                 ],
#                 results_list=[
#                     results_base_e1,
#                     results_base_e2,
#                     results_base_e3,
#                     results_base_e4,
#                     results_base_e5,
#                     results_base_e6,
#                     results_base_e7,
#                     results_base_e8,
#                 ],
#                 label_list=[
#                     "base 1e-1",
#                     "base 1e-2",
#                     "base 1e-3",
#                     "base 1e-4",
#                     "base 1e-5",
#                     "base 1e-6",
#                     "base 1e-7",
#                     "base 1e-8",
#                 ],
#                 title=f"spanwise_distribution_beta_{side_slip}_alpha_{alpha}",
#                 data_type=".pdf",
#                 save_path=save_folder,
#                 is_save=True,
#                 is_show=False,
#             )


# # Solver Setup
# solver_base_e1 = Solver(allowed_error=1e-1)
# solver_base_e2 = Solver(allowed_error=1e-2)
# solver_base_e3 = Solver(allowed_error=1e-3)
# solver_base_e4 = Solver(allowed_error=1e-4)
# solver_base_e5 = Solver(allowed_error=1e-5)
# solver_base_e6 = Solver(allowed_error=1e-6)
# solver_base_e7 = Solver(allowed_error=1e-7)
# solver_base_e8 = Solver(allowed_error=1e-8)

# ### plotting alpha-polar
# alpha_range = np.linspace(0, 25, 20)
# plot_polars(
#     solver_list=[
#         solver_base_e1,
#         solver_base_e2,
#         solver_base_e3,
#         solver_base_e4,
#         solver_base_e5,
#         solver_base_e6,
#         solver_base_e7,
#         solver_base_e8,
#     ],
#     body_aero_list=[
#         body_aero_uniform,
#         body_aero_uniform,
#         body_aero_uniform,
#         body_aero_uniform,
#         body_aero_uniform,
#         body_aero_uniform,
#         body_aero_uniform,
#         body_aero_uniform,
#     ],
#     label_list=[
#         "base 1e-1",
#         "base 1e-2",
#         "base 1e-3",
#         "base 1e-4",
#         "base 1e-5",
#         "base 1e-6",
#         "base 1e-7",
#         "base 1e-8",
#         "Lebesque CFD",
#     ],
#     literature_path_list=[
#         Path(PROJECT_DIR)
#         / "data"
#         / "TUDELFT_V3_KITE"
#         / "literature_results"
#         / "V3_CL_CD_RANS_Lebesque_2024_Rey_300e4.csv"
#     ],
#     angle_range=alpha_range,  # np.linspace(-10, 25, 10),
#     angle_type="angle_of_attack",
#     angle_of_attack=0,
#     side_slip=0,
#     yaw_rate=0,
#     Umag=Umag,
#     title=f"alphasweep",
#     data_type=".pdf",
#     save_path=Path(save_folder),
#     is_save=True,
#     is_show=False,
# )

# ### plotting beta sweep
# plot_polars(
#     solver_list=[
#         solver_base_e1,
#         solver_base_e2,
#         solver_base_e3,
#         solver_base_e4,
#         solver_base_e5,
#         solver_base_e6,
#         solver_base_e7,
#         solver_base_e8,
#     ],
#     body_aero_list=[
#         body_aero_uniform,
#         body_aero_uniform,
#         body_aero_uniform,
#         body_aero_uniform,
#         body_aero_uniform,
#         body_aero_uniform,
#         body_aero_uniform,
#         body_aero_uniform,
#     ],
#     label_list=[
#         "base 1e-1",
#         "base 1e-2",
#         "base 1e-3",
#         "base 1e-4",
#         "base 1e-5",
#         "base 1e-6",
#         "base 1e-7",
#         "base 1e-8",
#         "Lebesque CFD",
#     ],
#     literature_path_list=[
#         Path(PROJECT_DIR)
#         / "data"
#         / "TUDELFT_V3_KITE"
#         / "literature_results"
#         / "V3_CL_CD_RANS_Lebesque_2024_Rey_300e4.csv"
#     ],
#     angle_range=[0, 3, 6, 9, 12],  # np.linspace(-10, 25, 10),
#     angle_type="side_slip",
#     angle_of_attack=6.8,
#     side_slip=0,
#     yaw_rate=0,
#     Umag=Umag,
#     title=f"betasweep",
#     data_type=".pdf",
#     save_path=Path(save_folder),
#     is_save=True,
#     is_show=False,
# )

# #### plotting distributions
# gamma = None
# alpha_range = [19, 20, 21, 22, 23]
# y_coords = [panel.control_point[1] for panel in body_aero_uniform.panels]
# for side_slip in [0, 5]:
#     for alpha in alpha_range:
#         print(f"\nalpha: {alpha}")
#         body_aero_uniform.va_initialize(Umag, alpha, side_slip, yaw_rate)
#         results_base_e1 = solver_base_e1.solve(
#             body_aero_uniform, gamma_distribution=None
#         )
#         results_base_e2 = solver_base_e2.solve(
#             body_aero_uniform, gamma_distribution=None
#         )
#         results_base_e3 = solver_base_e3.solve(
#             body_aero_uniform, gamma_distribution=None
#         )
#         results_base_e4 = solver_base_e4.solve(
#             body_aero_uniform, gamma_distribution=None
#         )
#         results_base_e5 = solver_base_e5.solve(
#             body_aero_uniform, gamma_distribution=None
#         )
#         results_base_e6 = solver_base_e6.solve(
#             body_aero_uniform, gamma_distribution=None
#         )
#         results_base_e7 = solver_base_e7.solve(
#             body_aero_uniform, gamma_distribution=None
#         )
#         results_base_e8 = solver_base_e8.solve(
#             body_aero_uniform, gamma_distribution=None
#         )

#         plot_distribution(
#             y_coordinates_list=[
#                 y_coords,
#                 y_coords,
#                 y_coords,
#                 y_coords,
#                 y_coords,
#                 y_coords,
#                 y_coords,
#                 y_coords,
#             ],
#             results_list=[
#                 results_base_e1,
#                 results_base_e2,
#                 results_base_e3,
#                 results_base_e4,
#                 results_base_e5,
#                 results_base_e6,
#                 results_base_e7,
#                 results_base_e8,
#             ],
#             label_list=[
#                 "base 1e-1",
#                 "base 1e-2",
#                 "base 1e-3",
#                 "base 1e-4",
#                 "base 1e-5",
#                 "base 1e-6",
#                 "base 1e-7",
#                 "base 1e-8",
#             ],
#             title=f"spanwise_distribution_beta_{side_slip}_alpha_{alpha}",
#             data_type=".pdf",
#             save_path=save_folder,
#             is_save=True,
#             is_show=False,
#         )


# def plot_subplots_for_fva(save_folder, body_aero_uniform, y_coordinates):
#     """
#     For a set of fva values and a given alpha_range, solve the aerodynamic model
#     and plot the gamma distribution on only one half of the wing (e.g., positive y)
#     in subplots arranged in 3 columns. Each subplot corresponds to a different fva.

#     Args:
#         case_params (list): Case parameters (same format as in get_elliptical_case_params).
#         save_folder (Path): Folder to save the figure.
#         body_aero_uniform: BodyAerodynamics object.
#         y_coordinates (np.ndarray): The spanwise coordinates for the panels.
#     """
#     # Define the alpha range and the list of fva values
#     alpha_range = [16, 17, 18, 19, 20, 21, 22, 23]  # angles in degrees
#     fva_list = [0]
#     num_fva = len(fva_list)

#     # Compute number of rows required (3 columns)
#     ncols = 3
#     nrows = math.ceil(num_fva / ncols)

#     # Create the figure and subplots
#     fig, axes = plt.subplots(
#         nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
#     )

#     # We'll plot only the "right half" of the wing.
#     # Assume y_coordinates is a 1D array and that panels are arranged from left to right.
#     half_index = len(y_coordinates) // 2
#     y_half = y_coordinates[half_index:]

#     # Loop over fva values and plot in each subplot.
#     for idx, fva in enumerate(fva_list):
#         print(f"\nfva: {fva:.1e}")
#         # Determine row and column indices for subplot:
#         row = idx // ncols
#         col = idx % ncols
#         ax = axes[row, col]

#         # For fva==0, disable artificial viscosity.
#         is_with_damp = False if fva == 0 else True

#         # Create a solver with the given fva value.
#         solver = Solver(
#             allowed_error=1e-6,
#             relaxation_factor=1e-4,
#             max_iterations=1e4,
#             gamma_loop_type="non_linear_simonet_stall",
#             is_with_simonet_artificial_viscosity=is_with_damp,
#             simonet_artificial_viscosity_fva=fva,
#         )

#         # For each angle in alpha_range, solve and plot the gamma distribution.
#         for i, alpha in enumerate(alpha_range):
#             # Set the inflow for this angle.
#             body_aero_uniform.va_initialize(Umag, alpha, side_slip=0, yaw_rate=0)
#             results = solver.solve(body_aero_uniform, gamma_distribution=None)
#             gamma_distribution = results["gamma_distribution"]
#             # Take only one half of the gamma distribution (assume same indexing as y_coordinates)
#             gamma_half = gamma_distribution[half_index:]

#             # Construct a label (you can adjust as needed)
#             label = r"$\alpha$=" + f"{alpha}° + cl={results['cl']:.3f}"
#             ax.plot(y_half, gamma_half, label=label)

#         ax.grid(True)
#         ax.set_xlabel(r"$y$ [m]")
#         ax.set_ylabel(r"$\Gamma$")
#         ax.set_title(f"fva = {fva:.1e}")
#         ax.legend()

#     # Hide any empty subplots.
#     total_subplots = nrows * ncols
#     for idx in range(num_fva, total_subplots):
#         row = idx // ncols
#         col = idx % ncols
#         axes[row, col].axis("off")

#     plt.tight_layout()
#     fig.savefig(Path(save_folder) / f"Gamma_vs_alpha_uniform_all_fva_new_activated.pdf")


# # set_plot_style()
# y_coordinates = [panel.control_point[1] for panel in body_aero_uniform.panels]
# plot_subplots_for_fva(save_folder, body_aero_uniform, y_coordinates)
