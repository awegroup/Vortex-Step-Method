import numpy as np
import logging
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
from pathlib import Path
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
from VSM.interactive import interactive_plot

# Find the root directory of the repository
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(PROJECT_DIR, ".gitignore")):
    PROJECT_DIR = os.path.abspath(os.path.join(PROJECT_DIR, ".."))
    if PROJECT_DIR == "/":
        raise FileNotFoundError("Could not find the root directory of the repository.")
save_folder = Path(PROJECT_DIR) / "results" / "TUDELFT_V3_LEI_KITE"

### rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
csv_file_path = (
    Path(PROJECT_DIR)
    / "processed_data"
    / "TUDELFT_V3_LEI_KITE"
    / "rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs.csv"
)
(
    LE_x_array,
    LE_y_array,
    LE_z_array,
    TE_x_array,
    TE_y_array,
    TE_z_array,
    d_tube_array,
    camber_array,
) = np.loadtxt(csv_file_path, delimiter=",", skiprows=1, unpack=True)
rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs = []
for i in range(len(LE_x_array)):
    LE = np.array([LE_x_array[i], LE_y_array[i], LE_z_array[i]])
    TE = np.array([TE_x_array[i], TE_y_array[i], TE_z_array[i]])
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs.append(
        [LE, TE, ["lei_airfoil_breukels", [d_tube_array[i], camber_array[i]]]]
    )

#############################################
#############################################
#############################################
### Processing panel coefficients
n_panels_breukels = 18
CAD_wing_breukels = Wing(n_panels_breukels, "split_provided")
for i, CAD_rib_i in enumerate(
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
):
    CAD_wing_breukels.add_section(CAD_rib_i[0], CAD_rib_i[1], CAD_rib_i[2])
wing_aero_CAD_19ribs_breukels = WingAerodynamics([CAD_wing_breukels])
process_panel_coefficients(
    wing_aero_CAD_19ribs_breukels,
    PROJECT_DIR,
    n_panels_breukels,
    alpha_range=[-40, 40],
)
import testing_neuralfoil as testing_neuralfoil

for i in range(18):
    testing_neuralfoil.main(n_i=i, PROJECT_DIR=PROJECT_DIR)
#############################################
#############################################
#############################################


#### NORMAL OPERATION ####
# Defining discretisation
n_panels = 54
spanwise_panel_distribution = "split_provided"
CAD_wing = Wing(n_panels, spanwise_panel_distribution)
CAD_wing_breukels = Wing(n_panels, spanwise_panel_distribution)
csv_folder_path = Path(
    PROJECT_DIR, "examples", "TUDELFT_V3_LEI_KITE", "polar_engineering", "csv_files"
)
for i, CAD_rib_i in enumerate(
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
):
    # CAD_wing.add_section(CAD_rib_i[0], CAD_rib_i[1], CAD_rib_i[2])

    df_polar_data = pd.read_csv(Path(csv_folder_path, f"corrected_polar_{i}.csv"))
    alpha = df_polar_data["alpha"].values
    cl = df_polar_data["cl"].values
    cd = df_polar_data["cd"].values
    cm = df_polar_data["cm"].values
    polar_data = ["polar_data", np.array([alpha, cl, cd, cm])]
    CAD_wing.add_section(CAD_rib_i[0], CAD_rib_i[1], polar_data)
    CAD_wing_breukels.add_section(CAD_rib_i[0], CAD_rib_i[1], CAD_rib_i[2])

wing_aero_CAD_19ribs = WingAerodynamics([CAD_wing])
wing_aero_CAD_19ribs_breukels = WingAerodynamics([CAD_wing_breukels])

angle_of_attack = 6.8
side_slip = 0
yaw_rate = 0
Umag = 3.15

wing_aero_CAD_19ribs.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)
wing_aero_CAD_19ribs_breukels.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)

#### Solvers
VSM_base = Solver(
    aerodynamic_model_type="VSM",
    is_with_artificial_damping=False,
    is_new_vector_definition=False,
)
VSM_with_stall_correction = Solver(
    aerodynamic_model_type="VSM",
    is_with_artificial_damping=True,
    is_new_vector_definition=False,
)
# VSM_no_gamma_feedback = Solver(
#     aerodynamic_model_type="VSM",
#     is_with_artificial_damping=False,
#     is_with_gamma_feedback=False,
#     is_new_vector_definition=False,
# )
# VSM_no_gamma_feedback_stall = Solver(
#     aerodynamic_model_type="VSM",
#     is_with_artificial_damping=True,
#     is_with_gamma_feedback=False,
#     is_new_vector_definition=False,
# )
# VSM_new_vector_diff = Solver(
#     aerodynamic_model_type="VSM",
#     is_new_vector_definition=True,
#     is_with_artificial_damping=False,
# )
# VSM_new_vector_diff_stall = Solver(
#     aerodynamic_model_type="VSM",
#     is_new_vector_definition=True,
#     is_with_artificial_damping=True,
# )

# ### Plotting GEOMETRY
# # plot_geometry(
# #     wing_aero_CAD_19ribs,
# #     title=" ",
# #     data_type=".svg",
# #     save_path=" ",
# #     is_save=False,
# #     is_show=True,
# #     view_elevation=15,
# #     view_azimuth=-120,
# # )


# #### INTERACTIVE PLOT
# interactive_plot(
#     wing_aero_CAD_19ribs,
#     vel=Umag,
#     angle_of_attack=angle_of_attack,
#     side_slip=side_slip,
#     yaw_rate=yaw_rate,
#     is_with_aerodynamic_details=True,
# )

# interactive_plot(
#     wing_aero_CAD_19ribs,
#     vel=Umag,
#     angle_of_attack=angle_of_attack,
#     side_slip=10,
#     yaw_rate=yaw_rate,
#     is_with_aerodynamic_details=True,
# )


# ## plotting distributions
# results = VSM_base.solve(wing_aero_CAD_19ribs)
# results_with_stall_correction = VSM_with_stall_correction.solve(wing_aero_CAD_19ribs)
# CAD_y_coordinates = [
#     panels.aerodynamic_center[1] for panels in wing_aero_CAD_19ribs.panels
# ]
# results_no_gamma_feedback = VSM_no_gamma_feedback.solve(wing_aero_CAD_19ribs)
# results_no_gamma_feedback_stall = VSM_no_gamma_feedback_stall.solve(
#     wing_aero_CAD_19ribs
# )

# plot_distribution(
#     y_coordinates_list=[
#         CAD_y_coordinates,
#         CAD_y_coordinates,
#         # CAD_y_coordinates,
#         # CAD_y_coordinates,
#     ],
#     results_list=[
#         results,
#         results_with_stall_correction,
#         # results_no_gamma_feedback,
#         # results_no_gamma_feedback_stall,
#     ],
#     label_list=[
#         "VSM",
#         "VSM with stall correction",
#         # "VSM_no_gamma_feedback",
#         # "VSM_no_gamma_feedback_stall",
#     ],
#     title=f"CAD_spanwise_distributions_alpha_{angle_of_attack:.1f}_beta_{side_slip:.1f}_yaw_{yaw_rate:.1f}_Umag_{Umag:.1f}",
#     data_type=".pdf",
#     save_path=Path(save_folder) / "spanwise_distributions",
#     is_save=False,
#     is_show=True,
# )

## plotting alpha-polar
save_path = Path(PROJECT_DIR) / "results" / "TUD_V3_LEI_KITE"
path_cfd_lebesque = (
    Path(PROJECT_DIR)
    / "data"
    / "TUDELFT_V3_LEI_KITE"
    / "literature_results"
    / "V3_CL_CD_RANS_Lebesque_2024_Rey_300e4.csv"
)

plot_polars(
    solver_list=[
        VSM_base,
        VSM_base,
        VSM_with_stall_correction,
        VSM_with_stall_correction,
        # VSM_no_gamma_feedback,
        # VSM_no_gamma_feedback_stall,
        # VSM_new_vector_diff,
        # VSM_new_vector_diff_stall,
    ],
    wing_aero_list=[
        wing_aero_CAD_19ribs_breukels,
        wing_aero_CAD_19ribs,
        wing_aero_CAD_19ribs_breukels,
        wing_aero_CAD_19ribs,
        # wing_aero_CAD_19ribs,
        # wing_aero_CAD_19ribs,
        # wing_aero_CAD_19ribs,
        # wing_aero_CAD_19ribs,
    ],
    label_list=[
        "VSM Breukels",
        "VSM Corrected",
        "VSM Breukels (+stall)",
        "VSM Corrected (+stall)",
        # "VSM CAD 19ribs , no gamma feedback",
        # "VSM CAD 19ribs , no gamma feedback, with stall correction",
        # "VSM new vector diff",
        # "VSM new vector diff, with stall correction",
        "CFD_Lebesque Rey 30e5",
    ],
    literature_path_list=[path_cfd_lebesque],
    angle_range=np.linspace(-5, 25, 30),
    angle_type="angle_of_attack",
    angle_of_attack=0,
    side_slip=0,
    yaw_rate=0,
    Umag=Umag,
    title=f"alphasweep",
    data_type=".pdf",
    save_path=Path(save_folder) / "polars",
    is_save=True,
    is_show=True,
)
#### plot beta sweep
plot_polars(
    solver_list=[
        # VSM_base,
        VSM_base,
        # VSM_with_stall_correction,
        VSM_with_stall_correction,
        # VSM_no_gamma_feedback,
        # VSM_no_gamma_feedback_stall,
        # VSM_new_vector_diff,
        # VSM_new_vector_diff_stall,
    ],
    wing_aero_list=[
        # wing_aero_CAD_19ribs_breukels,
        wing_aero_CAD_19ribs,
        # wing_aero_CAD_19ribs_breukels,
        wing_aero_CAD_19ribs,
        # wing_aero_CAD_19ribs,
        # wing_aero_CAD_19ribs,
        # wing_aero_CAD_19ribs,
        # wing_aero_CAD_19ribs,
    ],
    label_list=[
        # "VSM Breukels",
        "VSM Corrected",
        # "VSM Breukels (+stall)",
        "VSM Corrected (+stall)",
        # "VSM CAD 19ribs , no gamma feedback",
        # "VSM CAD 19ribs , no gamma feedback, with stall correction",
        # "VSM new vector diff",
        # "VSM new vector diff, with stall correction",
        # "CFD_Lebesque Rey 30e5",
    ],
    literature_path_list=[],
    angle_range=np.linspace(-20, 20, 40),
    angle_type="side_slip",
    angle_of_attack=6.8,
    side_slip=0,
    yaw_rate=0,
    Umag=3.15,
    title=f"betasweep",
    data_type=".pdf",
    save_path=Path(save_folder) / "polars",
    is_save=True,
    is_show=True,
)
