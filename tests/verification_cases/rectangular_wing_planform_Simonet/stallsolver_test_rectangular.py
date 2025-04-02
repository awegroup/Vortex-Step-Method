"""
This is the same test as described in p.10 and 11
of Simonet et al. (2024) in Ch.4.2
URL: https://doi.org/10.21203/rs.3.rs-3955527/v1
building on: Chierighin et al. 2017,2020
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
from copy import deepcopy
import math

import pandas as pd
from pathlib import Path
from scipy.optimize import root

from VSM.WingGeometry import Wing
from VSM.BodyAerodynamics import BodyAerodynamics
from VSM.Solver import Solver
from VSM.plotting import plot_polars, plot_distribution
from VSM.interactive import interactive_plot
from VSM.plot_styling import set_plot_style, plot_on_ax


def cosine_spacing(min, max, n_points):
    """
    Create an array with cosine spacing, from min to max values, with n points

    """
    mean = (max + min) / 2
    amp = (max - min) / 2

    return mean + amp * np.cos(np.linspace(np.pi, 0, n_points))


def offcenter_cosine_space(a, b, n):
    """
    Generate n points in [a, b] using an off-center 'cosine-like' distribution,
    often used so that the control points (or boundaries) are shifted from the endpoints.

    Formula (for k = 0..n-1):
        y_k = 0.5 * [ (b + a) - (b - a) * cos( (2k + 1)*pi / (2n) ) ]

    This ensures that the points are not exactly at a or b, but 'off-center.'
    """
    k = np.arange(n)
    return 0.5 * ((b + a) - (b - a) * np.cos((2 * k + 1) * np.pi / (2 * n)))


def flip_created_coord_in_pairs(coord):
    # Reshape the array into pairs
    reshaped = coord.reshape(-1, 2, 3)

    # Reverse the order of the pairs
    flipped = np.flip(reshaped, axis=0)

    # Flatten back to the original shape
    return flipped.reshape(-1, 3)


def generate_coordinates_rect_wing(max_chord, span, N, dist):
    """
    Generate 2D coordinates (leading edge and trailing edge) for N spanwise sections
    of a rectangular wing. Here, the chord is constant along the span (i.e. rectangular).

    Args:
        max_chord (float): Constant chord for the wing.
        span (float): Total wing span.
        N (int): Number of spanwise sections (results in 2*N points).
        dist (str): Spanwise distribution type: 'uniform', 'cosine', or 'off_center_cosine'.

    Returns:
        coord (np.ndarray of shape (2*N, 3)):
            For each i in [0, N-1]:
                - coord[2*i]   is the leading edge point,
                - coord[2*i+1] is the trailing edge point.
    """
    coord = np.empty((2 * N, 3))
    start = span * 1e-5  # tiny offset to avoid exact endpoints

    # Choose the y-distribution as before:
    if dist == "uniform":
        y_arr = np.linspace(-span / 2 + start, span / 2 - start, N)
    elif dist == "cosine":
        y_arr = cosine_spacing(-span / 2 + start, span / 2 - start, N)
    elif dist == "off_center_cosine":
        y_arr = offcenter_cosine_space(-span / 2 + start, span / 2 - start, N)
    else:
        raise ValueError(
            f"Invalid dist='{dist}'. Must be 'uniform', 'cosine', or 'off_center_cosine'."
        )

    # For a rectangular wing, the chord is constant at max_chord for every station:
    c_arr = np.full_like(y_arr, max_chord)

    # Define the leading and trailing edge points.
    # Here we position the wing such that the LE is at -0.25*max_chord and the TE at +0.75*max_chord.
    for i in range(N):
        coord[2 * i, :] = [-0.25 * max_chord, y_arr[i], 0.0]  # Leading edge
        coord[2 * i + 1, :] = [0.75 * max_chord, y_arr[i], 0.0]  # Trailing edge

    return coord


def build_rectangular_wing_instance(
    max_chord: float, span: float, N: int, dist: str, airfoil_data=None
) -> Wing:
    """
    Build a Wing instance with a rectangular planform using the
    'generate_coordinates_rect_wing' function. Each spanwise section is added
    to the Wing with the specified airfoil data.

    Args:
        max_chord (float): Constant chord (e.g. wing root chord).
        span (float): Total wing span.
        N (int): Number of spanwise sections (the Wing will have N panels).
        dist (str): Spanwise distribution, e.g. "cosine" or "uniform".
        airfoil_data (list or None): Airfoil specification for each section.

    Returns:
        Wing: A Wing object representing the rectangular planform.
    """

    # 1) Generate the LE/TE coordinates for the rectangular wing.
    coord = generate_coordinates_rect_wing(max_chord, span, N, dist)

    # 2) Optionally flip the order of the coordinate pairs (if needed by your workflow)
    coord_left_to_right = flip_created_coord_in_pairs(deepcopy(coord))

    # 3) Create the Wing object.
    wing_instance = Wing(N, "unchanged")

    # 4) Add each panel to the Wing.
    for idx in range(N):
        LE_point = coord_left_to_right[2 * idx]
        TE_point = coord_left_to_right[2 * idx + 1]
        wing_instance.add_section(LE_point, TE_point, airfoil_data)

    logging.info(
        f"Built rectangular wing with chord={max_chord}, span={span}, N={N}, dist={dist}"
    )
    return wing_instance


# ==========================
# Settings
# AR_simonet = 10
# n_simonet_low = 50
# n_simonet_high = 200
# Re = 20e3
# ==========================

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent

save_folder = (
    Path(PROJECT_DIR)
    / "tests"
    / "verification_cases"
    / "rectangular_wing_planform"
    / "results"
)

span = 1
chord = 0.1
AR = span / chord
print(f"Simonet AR=10 and our AR={AR:.2f} ")
density = 1
mu = 2e-5
Umag = 4
print(f"Simonet Re=2e4 and our Re={density * Umag * chord / mu/1e4:.1f}e4")
n_panels = 50
alpha_range = np.linspace(0, 20, 21)
print(f"alpha_range: {alpha_range}")

df_polar = pd.read_csv(
    Path(PROJECT_DIR)
    / "tests"
    / "verification_cases"
    / "rectangular_wing_planform"
    / "polar_2D_WT_test_Chiereghin2017.csv"
)
polar_data = [
    "polar_data",
    np.column_stack(
        (
            np.deg2rad(df_polar["alpha"].values),
            df_polar["cl"].values,
            df_polar["cd"].values,
            df_polar["cm"].values,
        )
    ),
]
# ==========================
# Building the wing instance
# ==========================
body_aero_uniform = BodyAerodynamics(
    [
        build_rectangular_wing_instance(
            max_chord=chord,
            span=span,
            N=n_panels,
            dist="uniform",
            airfoil_data=polar_data,
        )
    ]
)
# body_aero_cosine = BodyAerodynamics(
#     [
#         build_rectangular_wing_instance(
#             max_chord=chord,
#             span=span,
#             N=n_panels,
#             dist="cosine",
#             airfoil_data=polar_data,
#         )
#     ]
# )
# body_aero_off_center_cosine = BodyAerodynamics(
#     [
#         build_rectangular_wing_instance(
#             max_chord=chord,
#             span=span,
#             N=n_panels,
#             dist="off_center_cosine",
#             airfoil_data=polar_data,
#         )
#     ]
# )
# ==========================
#   INTERACTIVE PLOT
# ==========================
# angle_of_attack = 10
# side_slip = 0
# yaw_rate = 0
# interactive_plot(
#     body_aero_uniform,
#     vel=Umag,
#     angle_of_attack=angle_of_attack,
#     side_slip=side_slip,
#     yaw_rate=yaw_rate,
#     is_with_aerodynamic_details=True,
# )
# interactive_plot(
#     body_aero_cosine,
#     vel=Umag,
#     angle_of_attack=angle_of_attack,
#     side_slip=side_slip,
#     yaw_rate=yaw_rate,
#     is_with_aerodynamic_details=True,
# )
# interactive_plot(
#     body_aero_off_center_cosine,
#     vel=Umag,
#     angle_of_attack=angle_of_attack,
#     side_slip=side_slip,
#     yaw_rate=yaw_rate,
#     is_with_aerodynamic_details=True,
# )


# # ==========================
# # Plotting CL-alpha Distributions (Fig. 12)
# # ==========================
# solver_base = Solver(
#     allowed_error=1e-6,
#     gamma_loop_type="non_linear",
# )


# df = pd.DataFrame(
#     columns=[
#         "alpha",
#         "CL_VSM_uniform",
#         "CL_VSM_cosine",
#         "CL_VSM_off_center_cosine",
#     ]
# )

# for alpha in alpha_range:
#     body_aero_uniform.va_initialize(Umag, alpha, side_slip=0, yaw_rate=0)
#     results_VSM_uniform = solver_base.solve(body_aero_uniform, gamma_distribution=None)
#     CL_VSM_uniform = results_VSM_uniform["cl"]

#     body_aero_cosine.va_initialize(Umag, alpha, side_slip=0, yaw_rate=0)
#     results_VSM_cosine = solver_base.solve(body_aero_cosine, gamma_distribution=None)
#     CL_VSM_cosine = results_VSM_cosine["cl"]

#     body_aero_off_center_cosine.va_initialize(Umag, alpha, side_slip=0, yaw_rate=0)
#     results_VSM_off_center_cosine = solver_base.solve(
#         body_aero_off_center_cosine, gamma_distribution=None
#     )
#     CL_VSM_off_center_cosine = results_VSM_off_center_cosine["cl"]

#     df = pd.concat(
#         [
#             df,
#             pd.DataFrame(
#                 {
#                     "alpha": [alpha],
#                     "CL_VSM_uniform": [CL_VSM_uniform],
#                     "CL_VSM_cosine": [CL_VSM_cosine],
#                     "CL_VSM_off_center_cosine": [CL_VSM_off_center_cosine],
#                 }
#             ),
#         ],
#         ignore_index=True,
#     )

# # Loading the 3D results
# df_polar_3D = pd.read_csv(
#     Path(PROJECT_DIR)
#     / "tests"
#     / "verification_cases"
#     / "rectangular_wing_planform"
#     / "polar_3D_WT_test_Chiereghin2020.csv"
# )

# set_plot_style()
# fig, ax = plt.subplots(figsize=(8, 8))

# plot_on_ax(
#     ax,
#     df_polar["alpha"],
#     df_polar["cl"],
#     label="2D Experimental",
#     color="black",
#     linestyle="dashed",
# )
# plot_on_ax(
#     ax,
#     df_polar_3D["alpha"],
#     df_polar_3D["cl"],
#     label="3D Experimental",
#     color="black",
#     marker="x",
#     linestyle="None",
# )
# plot_on_ax(ax, df["alpha"], df["CL_VSM_uniform"], label="VSM uniform")
# plot_on_ax(ax, df["alpha"], df["CL_VSM_cosine"], label="VSM cosine")
# plot_on_ax(
#     ax,
#     df["alpha"],
#     df["CL_VSM_off_center_cosine"],
#     label="VSM off-center_cosine",
#     is_with_grid=True,
# )

# ax.grid(True)
# ax.legend()
# ax.set_xlabel(r"$\alpha$ [$^{\circ}$]")
# ax.set_ylabel(r"$C_L$")
# ax.set_title(r"$C_L$ vs $\alpha$")
# plt.tight_layout()
# plt.savefig(Path(save_folder) / f"CL_vs_alpha_distributions.pdf")

# ==========================
# Plotting CL-alpha fva (Fig. 12)
# ==========================

set_plot_style()
fig, ax = plt.subplots(figsize=(8, 6))

# alpha_range = [15]
fva_list = [
    0,
    1e-4,
    5e-4,
    1e-4,
    1e-2,
    1e-1,
    1e0,
]
for fva in fva_list:

    if fva == 0:
        is_with_damp = False
    else:
        is_with_damp = True
    solver_base = Solver(
        allowed_error=1e-6,
        relaxation_factor=1e-4,
        gamma_loop_type="non_linear_simonet_stall",
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
    / "tests"
    / "verification_cases"
    / "rectangular_wing_planform"
    / "polar_3D_WT_test_Chiereghin2020.csv"
)

plot_on_ax(
    ax,
    df_polar["alpha"],
    df_polar["cl"],
    label="2D Experimental",
    color="black",
    linestyle="dashed",
)
plot_on_ax(
    ax,
    df_polar_3D["alpha"],
    df_polar_3D["cl"],
    label="3D Experimental",
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
# Plotting Circulation (Fig. 14)
# ==========================


# set_plot_style()
# y_coordinates = [panel.control_point[1] for panel in body_aero_uniform.panels]
# # alpha_range = [16]  # , 15, 16]
# alpha_range = [12, 13, 14, 15]
# cl_fva_0 = np.zeros(len(alpha_range))
# for fva in [0, 1e-4, 1e-3, 1e-2]:
#     if fva == 0:
#         is_with_damp = False
#     else:
#         is_with_damp = True
#     solver_base = Solver(
#         allowed_error=1e-2,
#         relaxation_factor=1e-3,
#         gamma_loop_type="non_linear_simonet_stall_newton_raphson",
#         is_with_simonet_artificial_viscosity=is_with_damp,
#         simonet_artificial_viscosity_fva=fva,
#     )
#     fig, ax = plt.subplots(figsize=(8, 6))

#     for i, alpha in enumerate(alpha_range):
#         print(f"\nfva: {fva}, alpha: {alpha}")
#         body_aero_uniform.va_initialize(Umag, alpha, side_slip=0, yaw_rate=0)
#         results_VSM_uniform = solver_base.solve(
#             body_aero_uniform, gamma_distribution=None
#         )
#         gamma_VSM_uniform = results_VSM_uniform["gamma_distribution"]
#         plot_on_ax(
#             ax,
#             y_coordinates,
#             gamma_VSM_uniform,
#             label=r"$\alpha$"
#             + f"={alpha:.1f}"
#             + r" $\Delta$"
#             + f'cl={results_VSM_uniform["cl"] - cl_fva_0[i]:.2f}'
#             + f"cl={results_VSM_uniform['cl']:.2f}, cl_fva0 ={cl_fva_0[i]:.2f}",
#         )

#         if fva == 0:
#             cl_fva_0[i] = results_VSM_uniform["cl"]

#     ax.grid(True)
#     ax.legend()
#     ax.set_xlabel(r"$y$ [m]")
#     ax.set_ylabel(r"$\Gamma$")
#     ax.set_title(r"$\Gamma$ vs $y$")
#     plt.tight_layout()
#     plt.savefig(Path(save_folder) / f"Gamma_vs_alpha_uniform_fva_{fva}.pdf")


def plot_subplots_for_fva(save_folder, body_aero_uniform):
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

    alpha_range = [11, 13, 15, 17, 19]  # angles in degrees
    fva_list = [
        0,
        1e-4,
        5e-4,
        1e-4,
        1e-2,
        1e-1,
        1e0,
    ]  # , 1e-1, 1, 10, 100]  # , 1e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]

    num_fva = len(fva_list)
    y_coordinates = [panel.control_point[1] for panel in body_aero_uniform.panels]
    set_plot_style()

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
            allowed_error=1e-3,
            relaxation_factor=1e-4,
            max_iterations=1e4,
            gamma_loop_type="non_linear_simonet_stall_newton_raphson",
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

            # ax.plot(
            #     y_half,
            #     smooth_array(np.array(gamma_half), 10),
            #     label=label + "smoothened",
            # )

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
    fig.savefig(Path(save_folder) / f"Gamma_vs_alpha_uniform_all_fva_J_analytical.pdf")


plot_subplots_for_fva(save_folder, body_aero_uniform)
