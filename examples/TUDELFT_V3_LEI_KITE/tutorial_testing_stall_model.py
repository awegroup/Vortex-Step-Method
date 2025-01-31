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

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


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
### Processing panel coefficients
file_path = (
    Path(PROJECT_DIR) / "data" / "TUDELFT_V3_LEI_KITE" / "geometry_corrected.csv"
)
n_panels = 35
spanwise_panel_distribution = "unchanged"
wing_aero_breukels = create_wing_aero(
    file_path, n_panels, spanwise_panel_distribution, is_with_corrected_polar=False
)
process_panel_coefficients(
    wing_aero_breukels,
    PROJECT_DIR,
    n_panels,
    polar_folder_path=Path(
        PROJECT_DIR, "examples", "TUDELFT_V3_LEI_KITE", "polar_engineering"
    ),
    alpha_range=[-40, 40],
)
# import testing_neuralfoil as testing_neuralfoil

# # Plot all profiles in the profiles folder
# for i in range(n_panels):
#     testing_neuralfoil.main(n_i=i, PROJECT_DIR=PROJECT_DIR)

breakpoint()
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
angle_of_attack = 10
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
# breakpoint()
wing_aero_breukels.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)
wing_aero_polar.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)

wing_aero_polar_35 = create_wing_aero(
    file_path,
    35,
    spanwise_panel_distribution,
    is_with_corrected_polar=True,
    path_polar_data_dir=path_polar_data_dir,
)
wing_aero_polar_70 = create_wing_aero(
    file_path,
    70,
    spanwise_panel_distribution,
    is_with_corrected_polar=True,
    path_polar_data_dir=path_polar_data_dir,
)
wing_aero_polar_105 = create_wing_aero(
    file_path,
    105,
    spanwise_panel_distribution,
    is_with_corrected_polar=True,
    path_polar_data_dir=path_polar_data_dir,
)
wing_aero_polar_140 = create_wing_aero(
    file_path,
    140,
    spanwise_panel_distribution,
    is_with_corrected_polar=True,
    path_polar_data_dir=path_polar_data_dir,
)

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

# # ## Plotting GEOMETRY
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
#     wing_aero_breukels,
#     vel=Umag,
#     angle_of_attack=angle_of_attack,
#     side_slip=side_slip,
#     yaw_rate=yaw_rate,
#     is_with_aerodynamic_details=True,
# )
# breakpoint()
# # interactive_plot(
# #     wing_aero_CAD_19ribs,
# #     vel=Umag,
# #     angle_of_attack=angle_of_attack,
# #     side_slip=10,
# #     yaw_rate=yaw_rate,
# #     is_with_aerodynamic_details=True,
# # )

save_folder = Path(PROJECT_DIR) / "results" / "TUDELFT_V3_LEI_KITE"

# ## plotting distributions
for angle_of_attack in [6.8]:
    for side_slip in [10, 20]:
        print(f"\nangle_of_attack: {angle_of_attack}, side_slip: {side_slip}")
        wing_aero_breukels.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)
        wing_aero_polar.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)

        plot_distribution(
            y_coordinates_list=[
                [panels.aerodynamic_center[1] for panels in wing_aero_breukels.panels],
                [panels.aerodynamic_center[1] for panels in wing_aero_breukels.panels],
                [panels.aerodynamic_center[1] for panels in wing_aero_polar.panels],
                [panels.aerodynamic_center[1] for panels in wing_aero_polar.panels],
            ],
            results_list=[
                VSM_base.solve(wing_aero_breukels),
                VSM_with_stall_correction.solve(wing_aero_breukels),
                VSM_base.solve(wing_aero_polar),
                VSM_with_stall_correction.solve(wing_aero_polar),
            ],
            label_list=[
                "VSM Breukels",
                "VSM Breukels stall",
                "VSM Corrected",
                "VSM Corrected stall",
            ],
            title=f"spanwise_distribution_effects_alpha_{angle_of_attack:.1f}_beta_{side_slip:.1f}_smoothing",
            data_type=".pdf",
            save_path=Path(save_folder) / "spanwise_distributions",
            is_save=True,
            is_show=False,
        )

# ## plotting alpha-polar
# path_cfd_lebesque = (
#     Path(PROJECT_DIR)
#     / "data"
#     / "TUDELFT_V3_LEI_KITE"
#     / "literature_results"
#     / "V3_CL_CD_RANS_Lebesque_2024_Rey_300e4.csv"
# )
# plot_polars(
#     solver_list=[
#         # VSM_base,
#         # VSM_base,
#         VSM_with_stall_correction,
#         VSM_with_stall_correction,
#         VSM_with_stall_correction,
#         VSM_with_stall_correction,
#     ],
#     wing_aero_list=[
#         # wing_aero_breukels,
#         # wing_aero_polar,
#         # wing_aero_breukels,
#         wing_aero_polar_35,
#         wing_aero_polar_70,
#         wing_aero_polar_105,
#         wing_aero_polar_140,
#     ],
#     label_list=[
#         # "VSM Breukels",
#         # "VSM Corrected",
#         # "VSM Breukels (+stall)",
#         "35 VSM Corrected (+stall)",
#         "70 VSM Corrected (+stall)",
#         "105 VSM Corrected (+stall)",
#         "140 VSM Corrected (+stall)",
#         "CFD_Lebesque Rey 30e5",
#     ],
#     literature_path_list=[path_cfd_lebesque],
#     angle_range=np.linspace(-10, 25, 20),
#     angle_type="angle_of_attack",
#     angle_of_attack=0,
#     side_slip=0,
#     yaw_rate=0,
#     Umag=Umag,
#     title=f"alphasweep_n_panels",
#     data_type=".pdf",
#     save_path=Path(save_folder) / "polars",
#     is_save=True,
#     is_show=True,
# )
# ### plot beta sweep
# plot_polars(
#     solver_list=[
#         VSM_base,
#         VSM_base,
#         VSM_with_stall_correction,
#         VSM_with_stall_correction,
#         # VSM_with_stall_correction,
#         # VSM_with_stall_correction,
#     ],
#     wing_aero_list=[
#         wing_aero_breukels,
#         wing_aero_polar,
#         wing_aero_breukels,
#         wing_aero_polar,
#         # wing_aero_polar_35,
#         # wing_aero_polar_70,
#         # wing_aero_polar_105,
#         # wing_aero_polar_140,
#     ],
#     label_list=[
#         "VSM Breukels",
#         "VSM Corrected",
#         "VSM Breukels (+stall)",
#         "VSM Corrected (+stall)",
#         # "35 VSM Corrected (+stall)",
#         # "70 VSM Corrected (+stall)",
#         # "105 VSM Corrected (+stall)",
#         # "140 VSM Corrected (+stall)",
#     ],
#     literature_path_list=[],
#     angle_range=np.linspace(-20, 20, 20),
#     angle_type="side_slip",
#     angle_of_attack=6.8,
#     side_slip=0,
#     yaw_rate=0,
#     Umag=3.15,
#     title=f"betasweep_n_panels_130_linear",
#     data_type=".pdf",
#     save_path=Path(save_folder) / "polars",
#     is_save=True,
#     is_show=True,
# ),


solver_list = (
    [
        # VSM_base,
        VSM_base,
        VSM_with_stall_correction,
        VSM_base,
        VSM_with_stall_correction,
        # VSM_with_stall_correction,
        # VSM_with_stall_correction,
    ],
)
wing_aero_list = (
    [
        wing_aero_breukels,
        wing_aero_breukels,
        wing_aero_polar,
        wing_aero_polar,
    ],
)
#         wing_aero_polar_35,
#         wing_aero_polar_70,
#         wing_aero_polar_105,
#         wing_aero_polar_140,
#     ],
label_list = (
    [
        "VSM_Breukels",
        "VSM_Breukels_stall",
        "VSM_Corrected",
        "VSM_Corrected_stall",
        # "35 VSM Corrected (+stall)",
        # "70 VSM Corrected (+stall)",
        # "105 VSM Corrected (+stall)",
        # "140 VSM Corrected (+stall)",
        # "CFD_Lebesque Rey 30e5",
    ],
)
angle_range = np.linspace(-10, 25, 20)
angle_type = "angle_of_attack"
angle_of_attack = 0
side_slip = 0
yaw_rate = 0
Umag = Umag

from VSM.plotting import generate_polar_data

save_folder = Path(PROJECT_DIR) / "results" / "TUDELFT_V3_LEI_KITE"

polar_data_list = []
for i, (solver, wing_aero, label) in enumerate(
    zip(solver_list, wing_aero_list, label_list)
):
    polar_data, reynolds_number = generate_polar_data(
        solver=solver,
        wing_aero=wing_aero,
        angle_range=angle_range,
        angle_type=angle_type,
        angle_of_attack=angle_of_attack,
        side_slip=side_slip,
        yaw_rate=yaw_rate,
        Umag=Umag,
    )
    df = pd.DataFrame(polar_data, columns=["alpha", "cl", "cd", "cm"])
    df.to_csv(Path(save_folder) / f"_{label}.csv")

    polar_data_list.append(polar_data)
    # Appending Reynolds numbers to the labels of the solvers
    label_list[i] += f" Re = {1e-5*reynolds_number:.1f}e5"
