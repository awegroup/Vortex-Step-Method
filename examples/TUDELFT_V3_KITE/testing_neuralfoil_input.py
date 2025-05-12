import numpy as np
import logging
import matplotlib.pyplot as plt

import pandas as pd
from pathlib import Path
from VSM.WingGeometry import Wing
from VSM.BodyAerodynamics import BodyAerodynamics
from VSM.Solver import Solver
from VSM.plotting import (
    plot_polars,
    plot_geometry,
    plot_distribution,
)
from VSM.interactive import interactive_plot


def main():
    """
    This script demonstrates how to use the VSM library to perform a 3D aerodynamic analysis of the TUDELFT_V3_KITE.

    The example covers the following steps:
    1. Define file paths for the kite geometry, 2D polars, and bridle geometry.
    2. Load the kite geometry from a CSV file.
    3. Create three BodyAerodynamics objects:
       - One using the baseline Breukels input.
       - One with corrected polar data.
       - One with corrected polar data and bridles.
    4. Initialize the aerodynamic model with a specific wind speed, angle of attack, side slip angle, and yaw rate.
    5. Plot the kite geometry using Matplotlib.
    6. Generate an interactive plot using Plotly.
    7. Plot and save polar curves (both angle of attack and side slip sweeps) for different settings, comparing them to literature data.
    """

    ### 1. defining paths
    PROJECT_DIR = Path(__file__).resolve().parents[2]

    file_path = (
        Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "wing_geometry_from_CAD.csv"
    )
    polar_data_dir = Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "2D_polars_CFD"
    airfoil_data_dir = (
        Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "airfoils_sliced_from_CAD"
    )

    ### 2. defining settings
    n_panels = 40
    spanwise_panel_distribution = "uniform"
    solver_base_version = Solver(reference_point=[0, 0, 0])

    print(f"\nCreating corrected polar input")
    body_aero_polar_CFD_CAD = BodyAerodynamics.from_file(
        file_path,
        n_panels,
        spanwise_panel_distribution,
        is_with_corrected_polar=True,
        polar_data_dir=polar_data_dir,
        is_half_wing=True,
    )
    print(f"\nCreating Neuralfoil input")
    body_aero_polar_neuralfoil = BodyAerodynamics.from_file(
        file_path,
        n_panels,
        spanwise_panel_distribution,
        is_half_wing=True,
        is_neuralfoil=True,
        nf_airfoil_data_dir=airfoil_data_dir,
        nf_reynolds_number=5e5,
        nf_xtr_lower=0.000001,
        nf_xtr_upper=0.000001,
        nf_n_crit=9,
        nf_is_with_save_polar=True,
        nf_alpha_range=[-30, 30, 61],
    )

    ### 4. Setting va
    Umag = 2.83
    angle_of_attack = 6.8
    side_slip = 0
    yaw_rate = 0
    body_aero_polar_CFD_CAD.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)
    body_aero_polar_neuralfoil.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)

    # #### 6. Creating an interactive plot using Plotly
    # interactive_plot(
    #     body_aero_polar_CFD_CAD,
    #     vel=Umag,
    #     angle_of_attack=angle_of_attack,
    #     side_slip=side_slip,
    #     yaw_rate=yaw_rate,
    #     is_with_aerodynamic_details=True,
    #     title="TUDELFT_V3_KITE",
    # )
    ### 7. Plotting the polar curves for different angles of attack and side slip angles
    # and saving in results with literature
    save_folder = Path(PROJECT_DIR) / "results" / "TUDELFT_V3_KITE"

    ### plotting alpha-polar
    path_cfd_lebesque_alpha_sweep = (
        Path(PROJECT_DIR)
        / "data"
        / "TUDELFT_V3_KITE"
        / "3D_polars_literature"
        / "CFD_RANS_Rey_10e5_Poland2025_alpha_sweep_beta_0.csv"
    )
    path_wt_alpha_sweep = (
        Path(PROJECT_DIR)
        / "data"
        / "TUDELFT_V3_KITE"
        / "3D_polars_literature"
        / "V3_CL_CD_CS_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4.csv"
    )
    plot_polars(
        solver_list=[
            solver_base_version,
            solver_base_version,
        ],
        body_aero_list=[
            body_aero_polar_CFD_CAD,
            body_aero_polar_neuralfoil,
        ],
        label_list=[
            "VSM Polar CFD CAD",
            "VSM Polar neuralfoil",
            "CFD Rey 10e5",
            "WT Rey 5e5",
        ],
        literature_path_list=[path_cfd_lebesque_alpha_sweep, path_wt_alpha_sweep],
        angle_range=np.linspace(-5, 25, 10),
        angle_type="angle_of_attack",
        angle_of_attack=0,
        side_slip=0,
        yaw_rate=0,
        Umag=Umag,
        title=f"alphasweep_testing_neuralfoil",
        data_type=".pdf",
        save_path=Path(save_folder),
        is_save=True,
        is_show=False,
    )
    # ### plot beta sweep
    # path_cfd_lebesque_beta_sweep = (
    #     Path(PROJECT_DIR)
    #     / "data"
    #     / "TUDELFT_V3_KITE"
    #     / "3D_polars_literature"
    #     / "CFD_RANS_Rey_10e5_Poland2025_beta_sweep_alpha_13_02.csv"
    # )
    # plot_polars(
    #     solver_list=[
    #         solver_base_version,
    #         solver_base_version,
    #         # solver_base_version,
    #         # solver_base_version,
    #     ],
    #     body_aero_list=[
    #         body_aero_polar_CFD_CAD,
    #         body_aero_CFD_CAD_smooth,
    #         # body_aero_polar_surfplan,
    #         # body_aero_breukels_surfplan,
    #     ],
    #     label_list=[
    #         "VSM Polar CFD CAD",
    #         "VSM Polar CFD CAD smooth",
    #         # "VSM Polar Breukels-neuralfoil Surfplan",
    #         # "VSM Breukels Surfplan",
    #         "CFD Rey 10e5",
    #     ],
    #     literature_path_list=[path_cfd_lebesque_beta_sweep],
    #     angle_range=[0, 4, 8, 12],
    #     angle_type="side_slip",
    #     angle_of_attack=13.02,
    #     side_slip=0,
    #     yaw_rate=0,
    #     Umag=Umag,
    #     title=f"betasweep",
    #     data_type=".pdf",
    #     save_path=Path(save_folder),
    #     is_save=True,
    #     is_show=False,
    # )
    # ### plot distributions
    # plot_distribution(
    #     alpha_list=[-10, -5, 0, 6.8, 11.9, 15, 20, 25],
    #     Umag=Umag,
    #     side_slip=0,
    #     yaw_rate=0,
    #     solver_list=[
    #         solver_base_version,
    #         solver_base_version,
    #         # solver_base_version,
    #         # solver_base_version,
    #     ],
    #     body_aero_list=[
    #         body_aero_polar_CFD_CAD,
    #         body_aero_CFD_CAD_smooth,
    #         # body_aero_polar_surfplan,
    #         # body_aero_breukels_surfplan,
    #     ],
    #     label_list=[
    #         "VSM Polar CFD CAD",
    #         "VSM Polar CFD CAD smooth",
    #         # "VSM Polar Breukels-neuralfoil Surfplan",
    #         # "VSM Breukels Surfplan",
    #     ],
    #     title=f"spanwise_distribution",
    #     data_type=".pdf",
    #     save_path=Path(save_folder) / "spanwise_distribution",
    #     is_save=True,
    #     is_show=False,
    # )


if __name__ == "__main__":
    main()
