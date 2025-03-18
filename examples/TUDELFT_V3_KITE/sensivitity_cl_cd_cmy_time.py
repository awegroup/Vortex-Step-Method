import time
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Assuming these are your project modules:
from VSM.Solver import Solver
from VSM.sensitivity_analysis import (
    generate_sensitivity_csv,
    plot_param_variation_from_csv_list,
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
    sensivity_save_dir = (
        Path(PROJECT_DIR) / "results" / "TUDELFT_V3_KITE" / "sensitivity_analysis"
    )
    n_panels_list = [10, 20, 30, 40, 50, 70, 90, 110, 140, 180, 220, 260, 300, 400, 500]
    generate_sensitivity_csv(
        file_path,
        polar_data_dir,
        save_csv_path=Path(sensivity_save_dir)
        / "sensivitivy_results_uniform_error_1e4.csv",
        parameter_list=["n_panels"],
        allowed_error_list=None,
        core_radius_fraction_list=[1e-20],
        relaxation_factor_list=None,
        n_panels_list=n_panels_list,
        spanwise_panel_distribution="uniform",
        n_panels=50,
        Umag=3.15,
        angle_of_attack=6.5,
        side_slip=0,
        yaw_rate=0,
        solver_instance=Solver(allowed_error=1e-4),
    )
    generate_sensitivity_csv(
        file_path,
        polar_data_dir,
        save_csv_path=Path(sensivity_save_dir)
        / "sensivitivy_results_uniform_error_1e5.csv",
        parameter_list=["n_panels"],
        allowed_error_list=None,
        core_radius_fraction_list=[1e-20],
        relaxation_factor_list=None,
        n_panels_list=n_panels_list,
        spanwise_panel_distribution="uniform",
        n_panels=50,
        Umag=3.15,
        angle_of_attack=6.5,
        side_slip=0,
        yaw_rate=0,
        solver_instance=Solver(allowed_error=1e-5),
    )
    generate_sensitivity_csv(
        file_path,
        polar_data_dir,
        save_csv_path=Path(sensivity_save_dir)
        / "sensivitivy_results_uniform_error_1e6.csv",
        parameter_list=["n_panels"],
        allowed_error_list=None,
        core_radius_fraction_list=[1e-20],
        relaxation_factor_list=None,
        n_panels_list=n_panels_list,
        spanwise_panel_distribution="uniform",
        n_panels=50,
        Umag=3.15,
        angle_of_attack=6.5,
        side_slip=0,
        yaw_rate=0,
        solver_instance=Solver(allowed_error=1e-6),
    )
    generate_sensitivity_csv(
        file_path,
        polar_data_dir,
        save_csv_path=Path(sensivity_save_dir) / "sensivitivy_results_cosine.csv",
        parameter_list=["n_panels"],
        allowed_error_list=None,
        core_radius_fraction_list=None,
        relaxation_factor_list=None,
        n_panels_list=n_panels_list,
        spanwise_panel_distribution="cosine",
        n_panels=50,
        Umag=3.15,
        angle_of_attack=6.5,
        side_slip=0,
        yaw_rate=0,
    )

    plot_param_variation_from_csv_list(
        csv_paths=[
            Path(sensivity_save_dir) / "sensivitivy_results_uniform_error_1e4.csv",
            Path(sensivity_save_dir) / "sensivitivy_results_uniform_error_1e5.csv",
            Path(sensivity_save_dir) / "sensivitivy_results_uniform_error_1e6.csv",
            Path(sensivity_save_dir) / "sensivitivy_results_cosine.csv",
        ],
        labels=["Uniform_1e4", "Uniform_1e5", "Uniform_1e6", "Cosine"],
        save_path=Path(sensivity_save_dir) / "sensitivity_plot.pdf",
    )
