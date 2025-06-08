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
)
from VSM.interactive import interactive_plot


def main():
    """
    This script demonstrates how to use the VSM library to perform a 3D aerodynamic analysis of the TUDELFT_V3_KITE.

    The example covers the following steps:
    1. Define file paths for the kite geometry, 2D polars, and bridle geometry.
    2. Load the kite geometry and airfoil/polar data from a YAML config file.
    3. Create BodyAerodynamics objects:
       - One using the baseline Breukels input (from YAML).
       - One with corrected polar data (from YAML).
       - One with corrected polar data and bridles (from YAML).
    4. Initialize the aerodynamic model with a specific wind speed, angle of attack, side slip angle, and yaw rate.
    5. Plot the kite geometry using Matplotlib.
    6. Generate an interactive plot using Plotly.
    7. Plot and save polar curves (both angle of attack and side slip sweeps) for different settings, comparing them to literature data.
    """

    ### 1. defining paths
    PROJECT_DIR = Path(__file__).resolve().parents[2]

    yaml_config_path = (
        Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "config_kite_surfplan.yaml"
    )

    ### 2. defining settings
    n_panels = 36
    spanwise_panel_distribution = "uniform"
    solver_base_version = Solver()

    ### 3. Loading kite geometry and polars from YAML config and instantiating BodyAerodynamics
    print(f"\nCreating breukels input (from YAML)")
    body_aero_breukels = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=yaml_config_path,
        spanwise_panel_distribution=spanwise_panel_distribution,
        is_with_bridles=False,
    )
    print(f"\nCreating corrected polar input (from YAML)")
    body_aero_polar = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=yaml_config_path,
        spanwise_panel_distribution=spanwise_panel_distribution,
        is_with_bridles=False,
    )
    print(f"\nCreating corrected polar input with bridles (from YAML)")
    body_aero_polar_with_bridles = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=yaml_config_path,
        spanwise_panel_distribution=spanwise_panel_distribution,
        is_with_bridles=True,
    )

    ### 4. Setting va
    Umag = 3.15
    angle_of_attack = 6.8
    side_slip = 0
    yaw_rate = 0
    body_aero_breukels.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)
    body_aero_polar.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)
    body_aero_polar_with_bridles.va_initialize(
        Umag, angle_of_attack, side_slip, yaw_rate
    )

    #### 5. Plotting the kite geometry using Matplotlib
    plot_geometry(
        body_aero_polar,
        title="TUDELFT_V3_KITE",
        data_type=".pdf",
        save_path=".",
        is_save=False,
        is_show=True,
    )

    #### 6. Creating an interactive plot using Plotly
    interactive_plot(
        body_aero_breukels,
        vel=Umag,
        angle_of_attack=angle_of_attack,
        side_slip=side_slip,
        yaw_rate=yaw_rate,
        is_with_aerodynamic_details=True,
        title="TUDELFT_V3_KITE",
    )

    ### 7. Plotting the polar curves for different angles of attack and side slip angles
    # and saving in results with literature
    save_folder = Path(PROJECT_DIR) / "results" / "TUDELFT_V3_KITE"

    ### plotting alpha-polar
    path_cfd_lebesque = (
        Path(PROJECT_DIR)
        / "data"
        / "TUDELFT_V3_KITE"
        / "3D_polars_literature"
        / "CFD_RANS_Rey_10e5_Poland2025_alpha_sweep_beta_0.csv"
    )
    plot_polars(
        solver_list=[solver_base_version, solver_base_version, solver_base_version],
        body_aero_list=[
            body_aero_breukels,
            body_aero_polar,
            body_aero_polar_with_bridles,
        ],
        label_list=[
            "VSM Breukels",
            "VSM Polar",
            "VSM Polar with Bridles",
            "CFD_RANS_Rey_10e5_Poland2025_alpha_sweep_beta_0",
        ],
        literature_path_list=[path_cfd_lebesque],
        angle_range=[5, 15, 20, 25],  # np.linspace(-10, 25, 10),
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
    ### plot beta sweep
    plot_polars(
        solver_list=[
            solver_base_version,
            solver_base_version,
        ],
        body_aero_list=[
            body_aero_breukels,
            body_aero_polar,
        ],
        label_list=[
            "VSM Breukels",
            "VSM Corrected",
        ],
        literature_path_list=[],
        angle_range=[0, 3, 6, 9, 12],
        angle_type="side_slip",
        angle_of_attack=6.8,
        side_slip=0,
        yaw_rate=0,
        Umag=3.15,
        title=f"betasweep",
        data_type=".pdf",
        save_path=Path(save_folder),
        is_save=True,
        is_show=False,
    )


if __name__ == "__main__":
    main()
