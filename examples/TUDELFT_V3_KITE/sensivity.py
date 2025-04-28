import numpy as np
import time as time
from pathlib import Path
from VSM.WingGeometry import Wing
from VSM.BodyAerodynamics import BodyAerodynamics
from VSM.sensitivity_analysis import (
    testing_all_solver_settings,
)


def main():
    """
    This script demonstrates how to perform a sensitivity analysis on the TUDELFT_V3_KITE.
    It includes the following steps:
    1. Define and resolve project directories for geometry, polar data, and results storage.
    2. Load literature CFD data for angle-of-attack sweeps.
    3. Specify kite geometry and corrected polar data file paths.
    4. Set aerodynamic parameters such as wind speed, angle of attack, and side slip.
    5. Execute a sensitivity analysis using various solver settings by calling
       the 'testing_all_solver_settings' function with multiple configurations.
    """

    PROJECT_DIR = Path(__file__).resolve().parents[2]
    print(f"PROJECT_DIR: {PROJECT_DIR}")

    # literature results
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

    literature_path_list_alpha = [
        path_cfd_lebesque_alpha_sweep,
        path_wt_alpha_sweep,
    ]
    literature_label_list_alpha = [
        "RANS CFD Re = 10e5",
        "Wind Tunnel Re = 5e5",
    ]
    # literature_path_list_beta = [
    #     Path(path_to_lit) / "CFD_V3_CL_CD_CS_RANS_Vire2022_Rey_10e5_beta_sweep.csv"
    # ]
    # literature_label_list_beta = [
    #     "RANS CFD Re = 10e5 (Vire et al. 2022)",
    # ]

    # kite geometry
    geometry_path = Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "wing_geometry.csv"
    geometry_path = (
        Path(PROJECT_DIR)
        / "data"
        / "TUDELFT_V3_KITE"
        / "wing_geometry_from_CAD_orderded_tip_to_mid.csv"
    )
    is_with_corrected_polar = True
    polar_data_dir = (
        Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "2D_polars_corrected"
    )
    polar_data_dir = Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "2D_polars_CFD"
    # sensitivity results
    sensitivity_results_dir = (
        Path(PROJECT_DIR) / "results" / "TUDELFT_V3_KITE" / "sensitivity_analysis"
    )

    testing_all_solver_settings(
        sensitivity_results_dir=sensitivity_results_dir,
        geometry_path=geometry_path,
        is_with_corrected_polar=is_with_corrected_polar,
        polar_data_dir=polar_data_dir,
        n_panels=20,
        spanwise_panel_distribution="uniform",
        Umag=3.15,
        angle_of_attack=6.5,
        side_slip=0,
        yaw_rate=0,
        aerodynamic_model_type_list=["VSM", "LLT"],
        allowed_error_list=[1e-2, 1e-5],
        core_radius_fraction_list=[1e-5, 1e-10],
        gamma_initial_distribution_type_list=[
            "previous",
            "elliptical",
        ],
        gamma_loop_type_list=["base"],
        max_iterations_list=[1e3, 5e3],
        n_panels_list=[20, 40],
        relaxation_factor_list=[0.1, 0.01],
        alpha_range=[5, 10],
        alpha_range_distribution=[22, 23],
        beta_range=[0, 3],
        beta_range_distribution=[0],
        literature_path_list_alpha=literature_path_list_alpha,
        literature_label_list_alpha=literature_label_list_alpha,
        # literature_path_list_beta=literature_path_list_beta,
        # literature_label_list_beta=literature_label_list_beta,
    )


if __name__ == "__main__":
    main()
