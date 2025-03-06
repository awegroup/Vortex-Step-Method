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
import time as time

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent

file_path = Path(PROJECT_DIR) / "data" / "TUDELFT_V3_LEI_KITE" / "wing_geometry.csv"
path_polar_data_dir = (
    Path(PROJECT_DIR)
    / "examples"
    / "TUDELFT_V3_LEI_KITE"
    / "polar_engineering"
    / "csv_files"
)
save_folder = (
    Path(PROJECT_DIR)
    / "examples"
    / "TUDELFT_V3_LEI_KITE"
    / "stall_model_testing"
    / "results"
    / "testing_elliptical_vs_previous_gamma_initialisation"
)
# Operating Conditions
Umag = 3.15
angle_of_attack = 6.5
side_slip = 0
yaw_rate = 0
alpha_range = [
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    24,
    23,
    22,
    21,
    20,
    19,
    18,
    17,
]

# Wing and Aerodynamic Setup
n_panels = 50
spanwise_panel_distribution = "linear"
wing_instance = Wing(n_panels, spanwise_panel_distribution)
body_aero_polar = BodyAerodynamics.from_file(
    wing_instance,
    file_path,
    is_with_corrected_polar=True,
    path_polar_data_dir=path_polar_data_dir,
)

body_aero_polar.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)

solver_non_linear_Claude_elliptical = Solver(
    gamma_loop_type="gamma_loop_non_linear_Claude",
    gamma_initial_distribution_type="elliptical",
    is_with_gamma_feedback=False,
)
solver_non_linear_Claude_elliptical_with_feedback = Solver(
    gamma_loop_type="gamma_loop_non_linear_Claude",
    gamma_initial_distribution_type="elliptical",
    is_with_gamma_feedback=True,
)

# solver_gamma_loop = Solver(gamma_loop_type="gamma_loop")
# solver_base_version = Solver()
# solver_smooth_circulation_04 = Solver(
#     is_smooth_circulation=True, smoothness_factor=0.04
# )
# solver_smooth_circulation_08 = Solver(
#     is_smooth_circulation=True, smoothness_factor=0.08
# )
# solver_smooth_circulation_12 = Solver(
#     is_smooth_circulation=True, smoothness_factor=0.08
# )
# solver_smooth_circulation_16 = Solver(
#     is_smooth_circulation=True, smoothness_factor=0.16
# )
# solver_smooth_circulation_70 = Solver(is_smooth_circulation=True, smoothness_factor=0.7)

# solver_gamma_loop = Solver(gamma_loop_type="gamma_loop")
# solver_gamma_loop_non_linear_Claude = Solver(
#     gamma_loop_type="gamma_loop_non_linear_Claude",
# )
# # solver_gamma_loop_non_linear_Grok = Solver(gamma_loop_type="gamma_loop_non_linear_Grok")
# solver_gamma_loop_non_linear_ChatGPT = Solver(
#     gamma_loop_type="gamma_loop_non_linear_ChatGPT"
# )


# solver_gamma_loop_zero = Solver(
#     gamma_loop_type="gamma_loop", gamma_initial_distribution_type="zero"
# )
# solver_gamma_loop_non_linear_Claude_zero = Solver(
#     gamma_loop_type="gamma_loop_non_linear_Claude",
#     gamma_initial_distribution_type="zero",
# )
# solver_gamma_loop_non_linear_ChatGPT_zero = Solver(
#     gamma_loop_type="gamma_loop_non_linear_ChatGPT",
#     gamma_initial_distribution_type="zero",
# )
# solver_gamma_loop_elliptical = Solver(
#     gamma_loop_type="gamma_loop", gamma_initial_distribution_type="elliptical"
# )
# solver_gamma_loop_non_linear_Claude_elliptical = Solver(
#     gamma_loop_type="gamma_loop_non_linear_Claude",
#     gamma_initial_distribution_type="elliptical",
# )
# solver_gamma_loop_non_linear_ChatGPT_elliptical = Solver(
#     gamma_loop_type="gamma_loop_non_linear_ChatGPT",
#     gamma_initial_distribution_type="elliptical",
# )
# solver_gamma_loop_cosine = Solver(
#     gamma_loop_type="gamma_loop", gamma_initial_distribution_type="cosine"
# )
# solver_gamma_loop_non_linear_Claude_cosine = Solver(
#     gamma_loop_type="gamma_loop_non_linear_Claude",
#     gamma_initial_distribution_type="cosine",
# )
# solver_gamma_loop_non_linear_ChatGPT_cosine = Solver(
#     gamma_loop_type="gamma_loop_non_linear_ChatGPT",
#     gamma_initial_distribution_type="cosine",
# )


# ## plotting alpha-polar
# plot_polars(
#     solver_list=[
#         solver_non_linear_Claude_elliptical,
#         solver_non_linear_Claude_elliptical_with_feedback,
#     ],
#     body_aero_list=[
#         body_aero_polar,
#         body_aero_polar,
#     ],
#     label_list=[
#         "non_linear_Claude_elliptical",
#         "non_linear_Claude_elliptical_with_feedback",
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

# generate results
y_coordinates = [panels.aerodynamic_center[1] for panels in body_aero_polar.panels]

gamma = None
alpha_range = [17, 18, 19, 20, 21, 22, 23, 24, 25]
for alpha in alpha_range:
    body_aero_polar.va_initialize(Umag, alpha, side_slip, yaw_rate)
    results_non_linear_Claude_elliptical = solver_non_linear_Claude_elliptical.solve(
        body_aero_polar, gamma_distribution=None
    )
    results_non_linear_Claude_elliptical_with_feedback = (
        solver_non_linear_Claude_elliptical_with_feedback.solve(
            body_aero_polar, gamma_distribution=gamma
        )
    )
    gamma = results_non_linear_Claude_elliptical_with_feedback["gamma_distribution"]
    print(f"\nalpha: {alpha}, gamma: {gamma}")
    plot_distribution(
        y_coordinates_list=[
            y_coordinates,
            y_coordinates,
        ],
        results_list=[
            results_non_linear_Claude_elliptical,
            results_non_linear_Claude_elliptical_with_feedback,
        ],
        label_list=[
            "non_linear_Claude_elliptical",
            "non_linear_Claude_elliptical_with_feedback",
        ],
        title=f"spanwise_distribution_alpha_{alpha}",
        data_type=".pdf",
        save_path=save_folder,
        is_save=True,
        is_show=False,
    )

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
