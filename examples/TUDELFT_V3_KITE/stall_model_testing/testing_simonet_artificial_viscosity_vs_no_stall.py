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
    / "testing_simonet_artificial_viscosity_vs_no_stall"
)
# Operating Conditions
Umag = 3.15
angle_of_attack = 6.5
side_slip = 0
yaw_rate = 0
alpha_range = [
    # 14,
    # 15,
    # 16,
    # 17,
    # 18,
    19,
    # 20,
    21,
    # 22,
    # 23,
    # 24,
]

# Wing and Aerodynamic Setup
n_panels = 50
spanwise_panel_distribution = "uniform"

# Initialize solver
# solver_base = Solver()
# solver_base_simonet_stall = Solver(
#     gamma_loop_type="simonet_stall",
#     is_with_simonet_artificial_viscosity=True,
#     simonet_artificial_viscosity_fva=1e2,
# )
# solver_non_linear = Solver(
#     gamma_loop_type="non_linear",
#     is_with_simonet_artificial_viscosity=False,
# )
# solver_non_linear_simonet_stall = Solver(
#     gamma_loop_type="non_linear_simonet_stall",
#     is_with_simonet_artificial_viscosity=True,
#     simonet_artificial_viscosity_fva=1e2,
# )

# Initialize body_aero
wing_instance = Wing(n_panels, spanwise_panel_distribution)
body_aero_uniform = BodyAerodynamics.from_file(
    wing_instance,
    file_path,
    is_with_corrected_polar=True,
    path_polar_data_dir=path_polar_data_dir,
)
body_aero_uniform.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)


# ==========================
# Plotting CL-alpha fva (Fig. 12)
# ==========================

alpha_range = np.linspace(0, 25, 10)
fva_range = [0]  # [0, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
set_plot_style()
fig, ax = plt.subplots(figsize=(8, 6))
for fva in fva_range:
    # alpha_range = [5, 23]
    # for fva in [1e1]:
    if fva == 0:
        is_with_damp = False
    else:
        is_with_damp = True
    solver_base = Solver(
        allowed_error=1e-6,
        relaxation_factor=1e-4,
        gamma_loop_type="base",
        is_with_simonet_artificial_viscosity=is_with_damp,
        simonet_artificial_viscosity_fva=fva,
    )
    cl_list = []
    for alpha in alpha_range:
        body_aero_uniform.va_initialize(Umag, alpha, side_slip=0, yaw_rate=0)
        results_VSM_uniform = solver_base.solve(
            body_aero_uniform, gamma_distribution=None
        )
        cl_list.append(results_VSM_uniform["cl"])

    plot_on_ax(ax, alpha_range, cl_list, label=f"VSM fva:{fva}")


# Loading the 3D results
df_polar_3D = pd.read_csv(
    Path(PROJECT_DIR)
    / "data"
    / "TUDELFT_V3_KITE"
    / "literature_results"
    / "V3_CL_CD_RANS_Lebesque_2024_Rey_300e4.csv"
)
plot_on_ax(
    ax,
    df_polar_3D["aoa"],
    df_polar_3D["CL"],
    label="CFD Lebesque",
    color="black",
    marker="x",
    linestyle="None",
)

ax.grid(True)
ax.legend()
ax.set_xlabel(r"$\alpha$ [$^{\circ}$]")
ax.set_ylabel(r"$C_L$")
ax.set_title(r"$C_L$ vs $\alpha$")
plt.tight_layout()
plt.savefig(Path(save_folder) / f"CL_vs_alpha_fva_new_activation.pdf")


# ==========================
# # plotting alpha-polar
# plot_polars(
#     solver_list=[
#         solver_base,
#         solver_base_simonet_stall,
#         solver_non_linear,
#         solver_non_linear_simonet_stall,
#         # solver_newton_krylov,
#         # solver_newton_raphson,
#         # solver_newton_raphson_simonet_1,
#         # solver_newton_raphson_simonet_01,
#         # solver_newton_raphson_simonet_001,
#     ],
#     body_aero_list=[
#         body_aero_polar,
#         body_aero_polar,
#         body_aero_polar,
#         body_aero_polar,
#         # body_aero_polar,
#         # body_aero_polar,
#         # body_aero_polar,
#         # body_aero_polar,
#         # body_aero_polar,
#     ],
#     label_list=[
#         "base",
#         "base_simonet_stall",
#         "non_linear",
#         "non_linear_simonet_stall",
#         # "newton_krylov",
#         # "newton_raphson",
#         # "newton_raphson_simonet fva = 1",
#         # "newton_raphson_simonet fva = 0.1",
#         # "newton_raphson_simonet fva = 0.01",
#     ],
#     literature_path_list=[],
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

# # generate results
# y_coordinates = [panels.aerodynamic_center[1] for panels in body_aero_polar.panels]

# gamma = None
# alpha_range = [17, 19, 21, 23]
# for alpha in alpha_range:
#     print(f"\nalpha: {alpha}")
#     body_aero_polar.va_initialize(Umag, alpha, side_slip, yaw_rate)
#     results_base = solver_base.solve(body_aero_polar, gamma_distribution=None)
#     results_base_simonet_stall = solver_base_simonet_stall.solve(
#         body_aero_polar, gamma_distribution=None
#     )
#     # results_non_linear = solver_non_linear.solve(
#     #     body_aero_polar, gamma_distribution=None
#     # )
#     results_non_linear_simonet_stall = solver_non_linear_simonet_stall.solve(
#         body_aero_polar, gamma_distribution=None
#     )

#     plot_distribution(
#         y_coordinates_list=[y_coordinates, y_coordinates, y_coordinates],
#         results_list=[
#             results_base,
#             results_base_simonet_stall,
#             # results_non_linear,
#             results_non_linear_simonet_stall,
#         ],
#         label_list=[
#             "base",
#             "base_simonet_stall",
#             # "non_linear",
#             "non_linear_simonet_stall",
#         ],
#         title=f"spanwise_distribution_alpha_{alpha}",
#         data_type=".pdf",
#         save_path=save_folder,
#         is_save=True,
#         is_show=False,
#     )


def plot_subplots_for_fva(save_folder, body_aero_uniform, y_coordinates):
    """
    For a set of fva values and a given alpha_range, solve the aerodynamic model
    and plot the gamma distribution on only one half of the wing (e.g., positive y)
    in subplots arranged in 3 columns. Each subplot corresponds to a different fva.

    Args:
        case_params (list): Case parameters (same format as in get_elliptical_case_params).
        save_folder (Path): Folder to save the figure.
        body_aero_uniform: BodyAerodynamics object.
        y_coordinates (np.ndarray): The spanwise coordinates for the panels.
    """
    # Define the alpha range and the list of fva values
    alpha_range = [16, 17, 18, 19, 20, 21, 22, 23]  # angles in degrees
    fva_list = fva_range
    num_fva = len(fva_list)

    # Compute number of rows required (3 columns)
    ncols = 3
    nrows = math.ceil(num_fva / ncols)

    # Create the figure and subplots
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )

    # We'll plot only the "right half" of the wing.
    # Assume y_coordinates is a 1D array and that panels are arranged from left to right.
    half_index = len(y_coordinates) // 2
    y_half = y_coordinates[half_index:]

    # Loop over fva values and plot in each subplot.
    for idx, fva in enumerate(fva_list):
        print(f"\nfva: {fva:.1e}")
        # Determine row and column indices for subplot:
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        # For fva==0, disable artificial viscosity.
        is_with_damp = False if fva == 0 else True

        # Create a solver with the given fva value.
        solver = Solver(
            allowed_error=1e-6,
            relaxation_factor=1e-4,
            max_iterations=1e4,
            gamma_loop_type="non_linear_simonet_stall",
            is_with_simonet_artificial_viscosity=is_with_damp,
            simonet_artificial_viscosity_fva=fva,
        )

        # For each angle in alpha_range, solve and plot the gamma distribution.
        for i, alpha in enumerate(alpha_range):
            # Set the inflow for this angle.
            body_aero_uniform.va_initialize(Umag, alpha, side_slip=0, yaw_rate=0)
            results = solver.solve(body_aero_uniform, gamma_distribution=None)
            gamma_distribution = results["gamma_distribution"]
            # Take only one half of the gamma distribution (assume same indexing as y_coordinates)
            gamma_half = gamma_distribution[half_index:]

            # Construct a label (you can adjust as needed)
            label = r"$\alpha$=" + f"{alpha}° + cl={results['cl']:.3f}"
            ax.plot(y_half, gamma_half, label=label)

        ax.grid(True)
        ax.set_xlabel(r"$y$ [m]")
        ax.set_ylabel(r"$\Gamma$")
        ax.set_title(f"fva = {fva:.1e}")
        ax.legend()

    # Hide any empty subplots.
    total_subplots = nrows * ncols
    for idx in range(num_fva, total_subplots):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis("off")

    plt.tight_layout()
    fig.savefig(Path(save_folder) / f"Gamma_vs_alpha_uniform_all_fva_new_activated.pdf")


set_plot_style()
y_coordinates = [panel.control_point[1] for panel in body_aero_uniform.panels]
plot_subplots_for_fva(save_folder, body_aero_uniform, y_coordinates)


# ### plot beta sweep
# plot_polars(
#     solver_list=[
#         solver_base_version,
#         solver_base_version,
#     ],
#     body_aero_list=[
#         body_aero_breukels,
#         body_aero_polar,
#     ],
#     label_list=[
#         "VSM Breukels",
#         "VSM Corrected",
#     ],
#     literature_path_list=[],
#     angle_range=[0, 3, 6, 9, 12],
#     angle_type="side_slip",
#     angle_of_attack=6.8,
#     side_slip=0,
#     yaw_rate=0,
#     Umag=3.15,
#     title=f"betasweep",
#     data_type=".pdf",
#     save_path=Path(save_folder),
#     is_save=True,
#     is_show=True,
# )
