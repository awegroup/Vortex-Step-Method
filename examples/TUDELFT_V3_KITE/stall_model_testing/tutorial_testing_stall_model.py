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

n_panels = 50
spanwise_panel_distribution = "linear"
wing_instance = Wing(n_panels, spanwise_panel_distribution)
print(f"Creating breukels input")
body_aero_breukels = BodyAerodynamics.from_file(
    wing_instance, file_path, is_with_corrected_polar=False
)
print(f"Creating corrected polar input")
wing_instance = Wing(n_panels, spanwise_panel_distribution)
body_aero_polar = BodyAerodynamics.from_file(
    wing_instance,
    file_path,
    is_with_corrected_polar=True,
    path_polar_data_dir=path_polar_data_dir,
)

Umag = 3.15
angle_of_attack = 6.5
side_slip = 0
yaw_rate = 0
alpha_range = [15, 16, 17, 18, 19, 20]


body_aero_breukels.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)
body_aero_polar.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)

solver_base_version = Solver()
solver_smooth_circulation_04 = Solver(
    is_smooth_circulation=True, smoothness_factor=0.04
)
solver_smooth_circulation_08 = Solver(
    is_smooth_circulation=True, smoothness_factor=0.08
)
solver_smooth_circulation_12 = Solver(
    is_smooth_circulation=True, smoothness_factor=0.08
)
solver_smooth_circulation_16 = Solver(
    is_smooth_circulation=True, smoothness_factor=0.16
)
solver_smooth_circulation_70 = Solver(is_smooth_circulation=True, smoothness_factor=0.7)

solver_gamma_loop = Solver(gamma_loop_type="gamma_loop")
solver_gamma_loop_non_linear_Claude = Solver(
    gamma_loop_type="gamma_loop_non_linear_Claude",
)
# solver_gamma_loop_non_linear_Grok = Solver(gamma_loop_type="gamma_loop_non_linear_Grok")
solver_gamma_loop_non_linear_ChatGPT = Solver(
    gamma_loop_type="gamma_loop_non_linear_ChatGPT"
)


solver_gamma_loop_zero = Solver(
    gamma_loop_type="gamma_loop", gamma_initial_distribution_type="zero"
)
solver_gamma_loop_non_linear_Claude_zero = Solver(
    gamma_loop_type="gamma_loop_non_linear_Claude",
    gamma_initial_distribution_type="zero",
)
solver_gamma_loop_non_linear_ChatGPT_zero = Solver(
    gamma_loop_type="gamma_loop_non_linear_ChatGPT",
    gamma_initial_distribution_type="zero",
)
solver_gamma_loop_elliptical = Solver(
    gamma_loop_type="gamma_loop", gamma_initial_distribution_type="elliptical"
)
solver_gamma_loop_non_linear_Claude_elliptical = Solver(
    gamma_loop_type="gamma_loop_non_linear_Claude",
    gamma_initial_distribution_type="elliptical",
)
solver_gamma_loop_non_linear_ChatGPT_elliptical = Solver(
    gamma_loop_type="gamma_loop_non_linear_ChatGPT",
    gamma_initial_distribution_type="elliptical",
)
solver_gamma_loop_cosine = Solver(
    gamma_loop_type="gamma_loop", gamma_initial_distribution_type="cosine"
)
solver_gamma_loop_non_linear_Claude_cosine = Solver(
    gamma_loop_type="gamma_loop_non_linear_Claude",
    gamma_initial_distribution_type="cosine",
)
solver_gamma_loop_non_linear_ChatGPT_cosine = Solver(
    gamma_loop_type="gamma_loop_non_linear_ChatGPT",
    gamma_initial_distribution_type="cosine",
)

save_folder = (
    Path(PROJECT_DIR)
    / "examples"
    / "TUDELFT_V3_LEI_KITE"
    / "stall_model_testing"
    / "results"
    / "gamma_loop_and_gamma_initialisation"
)

## plotting alpha-polar
# path_cfd_lebesque = (
#     Path(PROJECT_DIR)
#     / "data"
#     / "TUDELFT_V3_LEI_KITE"
#     / "literature_results"
#     / "V3_CL_CD_RANS_Lebesque_2024_Rey_300e4.csv"
# )
plot_polars(
    solver_list=[
        # solver_base_version,
        # solver_smooth_circulation_08,
        # solver_base_version,
        # solver_smooth_circulation_08,
        # solver_smooth_circulation_12,
        # solver_smooth_circulation_16,
        # solver_smooth_circulation_70,
        # solver_gamma_loop,
        # solver_gamma_loop_non_linear_Claude,
        # solver_gamma_loop_non_linear_Grok,
        # solver_gamma_loop_non_linear_ChatGPT,
        solver_gamma_loop_zero,
        solver_gamma_loop_elliptical,
        solver_gamma_loop_cosine,
        solver_gamma_loop_non_linear_Claude_zero,
        solver_gamma_loop_non_linear_Claude_elliptical,
        solver_gamma_loop_non_linear_Claude_cosine,
        solver_gamma_loop_non_linear_ChatGPT_zero,
        solver_gamma_loop_non_linear_ChatGPT_elliptical,
        solver_gamma_loop_non_linear_ChatGPT_cosine,
    ],
    body_aero_list=[
        # body_aero_breukels,
        # body_aero_breukels,
        # body_aero_polar,
        # body_aero_polar,
        body_aero_polar,
        body_aero_polar,
        body_aero_polar,
        body_aero_polar,
        body_aero_polar,
        body_aero_polar,
        body_aero_polar,
        body_aero_polar,
        body_aero_polar,
    ],
    label_list=[
        # "Breukels",
        # "Breukels + smooth 0.08",
        # "Polar",
        # "Polar + smooth 0.08",
        # "Polar + smooth 0.12",
        # "Polar + smooth 0.16",
        # "Polar + smooth 0.70",
        # "solver_gamma_loop",
        # "solver_gamma_loop_non_linear_Claude",
        # # "solver_gamma_loop_non_linear_Grok",
        # "solver_gamma_loop_non_linear_ChatGPT",
        "solver_gamma_loop_zero",
        "solver_gamma_loop_elliptical",
        "solver_gamma_loop_cosine",
        "solver_gamma_loop_non_linear_Claude_zero",
        "solver_gamma_loop_non_linear_Claude_elliptical",
        "solver_gamma_loop_non_linear_Claude_cosine",
        "solver_gamma_loop_non_linear_ChatGPT_zero",
        "solver_gamma_loop_non_linear_ChatGPT_elliptical",
        "solver_gamma_loop_non_linear_ChatGPT_cosine",
    ],
    literature_path_list=[],
    angle_range=alpha_range,  # np.linspace(-10, 25, 10),
    angle_type="angle_of_attack",
    angle_of_attack=0,
    side_slip=0,
    yaw_rate=0,
    Umag=Umag,
    title=f"alphasweep",
    data_type=".pdf",
    save_path=Path(save_folder),
    is_save=True,
    is_show=False,
)

# # generate results
# y_coordinates = [panels.aerodynamic_center[1] for panels in body_aero_breukels.panels]

# for alpha in alpha_range:
#     # results_base_breukels = solver_base_version.solve(body_aero_breukels)
#     # result_stall_08_breukels = solver_smooth_circulation_08.solve(body_aero_breukels)
#     # results_base_polar = solver_base_version.solve(body_aero_polar)
#     # result_stall_08_polar = solver_smooth_circulation_08.solve(body_aero_polar)
#     # results_stall_12_polar = solver_smooth_circulation_12.solve(body_aero_polar)
#     # results_stall_16_polar = solver_smooth_circulation_16.solve(body_aero_polar)
#     # results_stall_70_polar = solver_smooth_circulation_70.solve(body_aero_polar)


#     begin_time = time.time()
#     results_gamma_loop = solver_gamma_loop.solve(body_aero_polar)
#     body_aero_polar.gamma = None
#     print(f"Time for gamma_loop: {time.time() - begin_time}")

#     begin_time = time.time()
#     results_gamma_loop_non_linear_Claude = solver_gamma_loop_non_linear_Claude.solve(
#         body_aero_polar
#     )
#     body_aero_polar.gamma = None
#     print(f"Time for gamma_loop_non_linear_Claude: {time.time() - begin_time}")

#     begin_time = time.time()
#     results_gamma_loop_non_linear_ChatGPT = solver_gamma_loop_non_linear_ChatGPT.solve(
#         body_aero_polar
#     )
#     body_aero_polar.gamma = None
#     print(f"Time for gamma_loop_non_linear_ChatGPT: {time.time() - begin_time}")

#     plot_distribution(
#         y_coordinates_list=[
#             y_coordinates,
#             y_coordinates,
#             y_coordinates,
#             # y_coordinates,
#             # y_coordinates,
#             # y_coordinates,
#             # y_coordinates,
#         ],
#         results_list=[
#             # results_base_breukels,
#             # result_stall_08_breukels,
#             # results_base_polar,
#             # result_stall_08_polar,
#             # results_stall_12_polar,
#             # results_stall_16_polar,
#             # results_stall_70_polar,
#             results_gamma_loop,
#             results_gamma_loop_non_linear_Claude,
#             results_gamma_loop_non_linear_ChatGPT,
#         ],
#         label_list=[
#             # "Breukels",
#             # "Breukels + smooth 0.08",
#             # "Polar",
#             # "Polar + smooth 0.08",
#             # "Polar + smooth 0.12",
#             # "Polar + smooth 0.16",
#             # "Polar + smooth 0.70",
#             "results_gamma_loop",
#             "results_gamma_loop_non_linear_Claude",
#             "results_gamma_loop_non_linear_ChatGPT",
#         ],
#         title=f"spanwise_distribution_alpha_{alpha}",
#         data_type=".pdf",
#         save_path=save_folder,
#         is_save=True,
#         is_show=False,
#     )

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
