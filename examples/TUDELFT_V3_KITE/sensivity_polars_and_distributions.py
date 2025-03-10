import numpy as np
import time as time
from pathlib import Path
from VSM.WingGeometry import Wing
from VSM.BodyAerodynamics import BodyAerodynamics
from VSM.sensitivity_analysis import (
    testing_all_solver_settings,
    testing_n_panels_effect,
    testing_spanwise_distribution_effect,
)

if __name__ == "__main__":
    # Example usage
    PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
    file_path = Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "wing_geometry.csv"
    polar_data_dir = (
        Path(PROJECT_DIR)
        / "examples"
        / "TUDELFT_V3_KITE"
        / "polar_engineering"
        / "csv_files"
    )
    sensitivity_results_dir = (
        Path(PROJECT_DIR) / "results" / "TUDELFT_V3_KITE" / "sensitivity_analysis"
    )

    # # EXTENSIVE VERSION - for parameter sweeps
    # wing_instance = Wing(n_panels=50, spanwise_panel_distribution="uniform")
    # body_aero_uniform = BodyAerodynamics.from_file(
    #     wing_instance,
    #     file_path=Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "wing_geometry.csv",
    #     is_with_corrected_polar=True,
    #     path_polar_data_dir=(
    #         Path(PROJECT_DIR)
    #         / "examples"
    #         / "TUDELFT_V3_KITE"
    #         / "polar_engineering"
    #         / "csv_files"
    #     ),
    # )
    # testing_all_solver_settings(
    #     aerodynamic_model_type_list=["VSM", "LLT"],
    #     max_iterations_list=[1e2, 1e3, 5e3, 1e4],
    #     allowed_error_list=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
    #     relaxation_factor_list=[0.2, 0.1, 0.05, 0.03, 0.01, 0.005, 0.001],
    #     core_radius_fraction_list=[1e-5, 1e-10, 1e-20],
    #     gamma_loop_type_list=["base", "non_linear"],
    #     is_with_gamma_feedback_list=[False, True],
    #     gamma_initial_distribution_type_list=["elliptical", "cosine", "zero"],
    #     sensitivity_results_dir=sensitivity_results_dir,
    #     body_aero_uniform=body_aero_uniform,
    #     alpha_range=np.linspace(0, 25, 20),
    #     alpha_range_distribution=[18, 19, 20, 21, 22, 23],
    #     beta_range=[0, 3, 6, 9, 12],
    #     beta_range_distribution=[0, 3, 6],
    #     Umag=3.15,
    #     angle_of_attack=6.5,
    #     side_slip=0,
    #     yaw_rate=0,
    # )

    n_panels_list = [10, 30, 60, 100, 150, 200]
    testing_n_panels_effect(
        sensitivity_results_dir=sensitivity_results_dir,
        file_path=file_path,
        polar_data_dir=polar_data_dir,
        n_panels_list=n_panels_list,
        Umag=3.15,
        angle_of_attack=6.5,
        side_slip=0,
        yaw_rate=0,
    )
    spanwise_panel_distribution_list = [
        # "unchanged",
        "uniform",
        "cosine",
        # "split_provided",
        # "cosine_van_garrel",
    ]
    n_panels_for_testing_spanwise_distributions = 50
    testing_spanwise_distribution_effect(
        sensitivity_results_dir=sensitivity_results_dir,
        file_path=file_path,
        polar_data_dir=polar_data_dir,
        spanwise_panel_distribution_list=spanwise_panel_distribution_list,
        n_panels=n_panels_for_testing_spanwise_distributions,
        Umag=3.15,
        angle_of_attack=6.5,
        side_slip=0,
        yaw_rate=0,
    )
