import numpy as np
import logging
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
from pathlib import Path
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
from VSM.interactive import interactive_plot

PROJECT_DIR = Path(__file__).resolve().parents[3]


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


# ############################################
# ############################################
# ############################################
# ### Processing panel coefficients
# file_path = (
#     Path(PROJECT_DIR) / "data" / "TUDELFT_V3_LEI_KITE" / "rib_list_height_scaled.csv"
# )
# n_panels = 18
# spanwise_panel_distribution = "unchanged"
# wing_aero_breukels = create_wing_aero(
#     file_path, n_panels, spanwise_panel_distribution, is_with_corrected_polar=False
# )
# process_panel_coefficients(
#     wing_aero_breukels,
#     PROJECT_DIR,
#     n_panels,
#     polar_folder_path=Path(
#         PROJECT_DIR, "examples", "TUDELFT_V3_LEI_KITE", "polar_engineering", "no_billow"
#     ),
#     alpha_range=[-40, 40],
# )
# # import testing_neuralfoil as testing_neuralfoil

# # # Plot all profiles in the profiles folder
# # for i in range(n_panels):
# #     testing_neuralfoil.main(n_i=i, PROJECT_DIR=PROJECT_DIR)
# ############################################
# ############################################
# ############################################

#### NORMAL OPERATION ####
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
n_panels = 150
angle_of_attack = 20
side_slip = 0
yaw_rate = 0
Umag = 3.15
spanwise_panel_distribution = "linear"
wing_aero_breukels = create_wing_aero(
    file_path,
    n_panels,
    spanwise_panel_distribution,
    is_with_corrected_polar=False,
    path_polar_data_dir=path_polar_data_dir,
)
wing_aero_polar = create_wing_aero(
    file_path,
    n_panels,
    spanwise_panel_distribution,
    is_with_corrected_polar=True,
    path_polar_data_dir=path_polar_data_dir,
)
wing_aero_breukels.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)
wing_aero_polar.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)

import time

# solving wing_aero_polar
# circulation smoothening: 4.32269287109375 sec
# Kostia Roncins model: 393.7673542499542 sec
# no stall 3.019317865371704
# stall 3.070854663848877
print(
    f"% increase roncin model: {100*(393.7673542499542-4.32269287109375)/4.32269287109375}%"
)
print(
    f"% increase circulation smoothening: {100*(3.019317865371704-3.070854663848877)/3.070854663848877}%"
)


VSM_base = Solver(aerodynamic_model_type="VSM", is_with_artificial_damping=False)
VSM_with_circulation_smoothing = Solver(
    aerodynamic_model_type="VSM", is_with_artificial_damping=True
)
results = VSM_base.solve(wing_aero_polar)
results = VSM_with_circulation_smoothing.solve(wing_aero_polar)
start_time = time.time()
results = VSM_base.solve(wing_aero_polar)
end_time = time.time()
print(f"Time taken to solve wing_aero_polar without stall: {end_time-start_time}")

start_time = time.time()
results = VSM_with_circulation_smoothing.solve(wing_aero_polar)
end_time = time.time()
print(f"Time taken to solve wing_aero_polar with stall: {end_time-start_time}")
# circulation smoothening: 4.32269287109375 sec
# Kostia Roncins model: 393.7673542499542 sec
# 3.019317865371704
# 3.070854663848877
