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
)
from VSM.interactive import interactive_plot

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

file_path = Path(PROJECT_DIR) / "data" / "TUDELFT_V3_LEI_KITE" / "wing_geometry.csv"
path_polar_data_dir = (
    Path(PROJECT_DIR)
    / "examples"
    / "TUDELFT_V3_LEI_KITE"
    / "polar_engineering"
    / "csv_files"
)
path_bridle_data = (
    Path(PROJECT_DIR) / "data" / "TUDELFT_V3_LEI_KITE" / "bridle_lines.csv"
)
n_panels = 40
spanwise_panel_distribution = "linear"
wing_instance = Wing(n_panels, spanwise_panel_distribution)
print(f"\nCreating breukels input")
body_aero_breukels = BodyAerodynamics.from_file(
    wing_instance, file_path, is_with_corrected_polar=False
)
print(f"\nCreating corrected polar input")
wing_instance = Wing(n_panels, spanwise_panel_distribution)
body_aero_polar = BodyAerodynamics.from_file(
    wing_instance,
    file_path,
    is_with_corrected_polar=True,
    path_polar_data_dir=path_polar_data_dir,
)
print(f"\nCreating corrected polar input with bridles")
wing_instance = Wing(n_panels, spanwise_panel_distribution)
body_aero_polar_with_bridles = BodyAerodynamics.from_file(
    wing_instance,
    file_path,
    is_with_corrected_polar=True,
    path_polar_data_dir=path_polar_data_dir,
    is_with_bridles=True,
    path_bridle_data=path_bridle_data,
)

Umag = 3.15
angle_of_attack = 6.8
side_slip = 0
yaw_rate = 0
body_aero_breukels.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)
body_aero_polar.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)

solver_base_version = Solver()

#### INTERACTIVE PLOT
interactive_plot(
    body_aero_breukels,
    vel=Umag,
    angle_of_attack=angle_of_attack,
    side_slip=side_slip,
    yaw_rate=yaw_rate,
    is_with_aerodynamic_details=True,
)

save_folder = Path(PROJECT_DIR) / "examples" / "TUDELFT_V3_LEI_KITE"

## plotting alpha-polar
path_cfd_lebesque = (
    Path(PROJECT_DIR)
    / "data"
    / "TUDELFT_V3_LEI_KITE"
    / "literature_results"
    / "V3_CL_CD_RANS_Lebesque_2024_Rey_300e4.csv"
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
        "CFD_Lebesque Rey 30e5",
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
    is_show=True,
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
    is_show=True,
)
