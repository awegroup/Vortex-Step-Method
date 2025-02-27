import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from VSM.plot_styling import set_plot_style
from VSM.interactive import interactive_plot

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
file_path = Path(PROJECT_DIR, "data", "TUDELFT_V3_LEI_KITE", "geometry.csv")
# file_path = "/home/jellepoland/ownCloud/phd/code/Vortex-Step-Method/data/TUDELFT_V3_LEI_KITE/rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs.csv"
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

n_ribs = len(LE_x_array)
for i in range(n_ribs):
    LE_x = LE_x_array[i]
    LE_y = LE_y_array[i]
    LE_z = LE_z_array[i]
    TE_x = TE_x_array[i]
    TE_y = TE_y_array[i]
    TE_z = TE_z_array[i]

    if "geometry" in str(file_path):

        if i < 4 or i > (n_ribs - 5):
            print(f"i:{i}")
            print(f"\nTE_y: {TE_y}\nLE_y: {LE_y}")
            if i in [0, n_ribs - 1]:
                factor = 0.4
            elif i in [1, n_ribs - 2]:
                factor = 0.2
            elif i in [2, n_ribs - 3]:
                factor = 0.1
            else:
                factor = 0.05

        LE = np.array([LE_x, LE_y, LE_z])
        TE = np.array([TE_x, TE_y, TE_z])
        if LE_y < 0:
            LE_y = LE_y - factor * d_tube_array[i] * np.linalg.norm(LE - TE)
        else:
            LE_y = LE_y + factor * d_tube_array[i] * np.linalg.norm(LE - TE)
            print(f"LE_y: {LE_y} (new)")

        LE_y_array[i] = LE_y

    # if np.abs(TE_y) > np.abs(LE_y):
    #     larger_value = TE_y
    #     smaller_value = LE_y
    #     LE_y = larger_value
    #     TE_y = smaller_value
    # else:
    #     height_ratio = 2.628 / 3.688
    #     LE_z *= height_ratio
    #     TE_z *= height_ratio

    LE = np.array([LE_x, LE_y, LE_z])
    TE = np.array([TE_x, TE_y, TE_z])
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs.append(
        [LE, TE, ["lei_airfoil_breukels", [d_tube_array[i], camber_array[i]]]]
    )

# if "geometry" in str(file_path):
#     df["LE_y"] = LE_y_array
#     df.to_csv(
#         Path(PROJECT_DIR)
#         / "data"
#         / "TUDELFT_V3_LEI_KITE"
#         / "geometry_surfplan_billow_wind_tunnel_scale.csv",
#         index=False,
#     )
# else:
#     df["TE_z"] = TE_y_array
#     df["LE_z"] = LE_y_array
#     df.to_csv(
#         Path(PROJECT_DIR)
#         / "data"
#         / "TUDELFT_V3_LEI_KITE"
#         / "geometry_no_billow_wind_tunnel_scale.csv",
#         index=False,
#     )


n_panels = 2 * (n_ribs - 1)
spanwise_panel_distribution = "split_provided"
# n_panels = 40
# spanwise_panel_distribution = "linear"
CAD_wing = Wing(n_panels, spanwise_panel_distribution)
for i, CAD_rib_i in enumerate(
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
):
    CAD_wing.add_section(CAD_rib_i[0], CAD_rib_i[1], CAD_rib_i[2])

wing_aero = WingAerodynamics([CAD_wing])

angle_of_attack = 6.8
side_slip = 20
yaw_rate = 0
Umag = 3.15


#### INTERACTIVE PLOT
interactive_plot(
    wing_aero,
    vel=Umag,
    angle_of_attack=angle_of_attack,
    side_slip=side_slip,
    yaw_rate=yaw_rate,
    is_with_aerodynamic_details=True,
)

## Saving geometry
df = pd.DataFrame(
    {
        "LE_x": [
            section[0][0]
            for section in rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
        ],
        "LE_y": [
            section[0][1]
            for section in rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
        ],
        "LE_z": [
            section[0][2]
            for section in rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
        ],
        "TE_x": [
            section[1][0]
            for section in rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
        ],
        "TE_y": [
            section[1][1]
            for section in rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
        ],
        "TE_z": [
            section[1][2]
            for section in rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
        ],
        "d_tube": d_tube_array,
        "camber": camber_array,
    }
)
df.to_csv(
    Path(PROJECT_DIR) / "data" / "TUDELFT_V3_LEI_KITE" / "geometry_corrected.csv",
    index=False,
)

### Findings
# - with too many panels, under sideslip it does not always find smooth result.
# - with high side slips the sensitivy to the number of panels increases drastically.
