from pathlib import Path
from VSM.convergence_analysis import generate_csv_files, plot_convergence


def main():
    """
    Run convergence analysis for TUDELFT_V3_KITE using YAML config input.
    """

    PROJECT_DIR = Path(__file__).resolve().parents[2]
    config_path = (
        Path(PROJECT_DIR)
        / "data"
        / "TUDELFT_V3_KITE"
        / "config_kite_CAD_CFD_polars.yaml"
    )
    convergence_analysis_dir = (
        Path(PROJECT_DIR) / "results" / "TUDELFT_V3_KITE" / "convergence_analysis"
    )
    n_panels_list = [20, 40, 50]
    spanwise_panel_distribution = "uniform"
    Umag = 10
    angle_of_attack = 10
    side_slip = 10
    yaw_rate = 0

    # First convergence analysis (varying n_panels and model type)
    convergence_results_dir = generate_csv_files(
        config_path,
        convergence_analysis_dir,
        spanwise_panel_distribution,
        Umag,
        angle_of_attack,
        side_slip,
        yaw_rate,
        n_panels_list,
        aerodynamic_model_type_list=["VSM", "LLT"],
        allowed_error_list=None,
        core_radius_fraction_list=None,
        gamma_initial_distribution_type_list=None,
        gamma_loop_type_list=None,
        max_iterations_list=None,
        relaxation_factor_list=None,
        spanwise_panel_distribution_list=None,
    )
    plot_convergence(
        convergence_results_dir,
        name="convergence",
    )

    # Second convergence analysis (varying solver parameters)
    convergence_results_dir = generate_csv_files(
        config_path,
        convergence_analysis_dir,
        spanwise_panel_distribution,
        Umag,
        angle_of_attack,
        side_slip,
        yaw_rate,
        n_panels_list,
        aerodynamic_model_type_list=None,
        allowed_error_list=[1e-2, 1e-5],
        core_radius_fraction_list=None,
        gamma_initial_distribution_type_list=["zero", "elliptical", "previous"],
        gamma_loop_type_list=None,
        max_iterations_list=[1000, 3000],
        relaxation_factor_list=[0.01, 0.05, 0.1],
        spanwise_panel_distribution_list=None,
    )
    plot_convergence(
        convergence_results_dir,
        name="convergence_solver_parameters",
    )


if __name__ == "__main__":
    main()
