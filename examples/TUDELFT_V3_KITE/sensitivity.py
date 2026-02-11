import numpy as np
from pathlib import Path
from VSM.sensitivity_analysis import testing_all_solver_settings


def main():
    """
    Perform sensitivity analysis on TUDELFT_V3_KITE using YAML config input.
    """

    PROJECT_DIR = Path(__file__).resolve().parents[2]
    yaml_config_path = (
        Path(PROJECT_DIR)
        / "data"
        / "TUDELFT_V3_KITE"
        / "CAD_derived_geometry"
        / "aero_geometry_CAD_CFD_polars.yaml"
    )
    sensitivity_results_dir = (
        Path(PROJECT_DIR) / "results" / "TUDELFT_V3_KITE" / "sensitivity_analysis"
    )

    # Literature data (optional, can be extended as needed)
    literature_path_list_alpha = [
        Path(PROJECT_DIR)
        / "data"
        / "TUDELFT_V3_KITE"
        / "3D_polars_literature"
        / "CFD_RANS_Rey_10e5_Poland2025_alpha_sweep_beta_0.csv",
        Path(PROJECT_DIR)
        / "data"
        / "TUDELFT_V3_KITE"
        / "3D_polars_literature"
        / "V3_CL_CD_CS_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4.csv",
    ]
    literature_label_list_alpha = [
        "RANS CFD Re = 10e5",
        "Wind Tunnel Re = 5e5",
    ]

    testing_all_solver_settings(
        sensitivity_results_dir=sensitivity_results_dir,
        geometry_path=yaml_config_path,
        is_with_corrected_polar=True,
        polar_data_dir=None,
        n_panels=20,
        spanwise_panel_distribution="uniform",
        Umag=3.15,
        angle_of_attack=6.5,
        side_slip=0,
        yaw_rate=0,
        aerodynamic_model_type_list=["VSM", "LLT"],
        allowed_error_list=[1e-2, 1e-5],
        core_radius_fraction_list=[1e-5, 1e-10],
        gamma_initial_distribution_type_list=["previous", "elliptical"],
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
    )


if __name__ == "__main__":
    main()
