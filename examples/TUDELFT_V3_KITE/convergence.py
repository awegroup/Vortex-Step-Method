from pathlib import Path
from VSM.Solver import Solver
from VSM.convergence_analysis import (
    generate_csv_files,
    plot_convergence,
)

def main():
    """
    Main function for performing convergence analysis of aerodynamic simulations.
    This script:
    - Determines key file paths based on the project directory, including:
        - Wing geometry CSV file.
        - Directory for 2D polar data.
        - Output folder for convergence analysis results.
    - Initializes a list of panel counts for the study of convergence behavior.
    - Sets testing parameters such as:
        - Whether to use corrected polar data.
        - Panel distribution across the wing.
        - Flow conditions (e.g., velocity, angle of attack, side slip, yaw rate).
    - Calls a function to generate CSV files that store convergence analysis results:
        - The analysis is performed using specified aerodynamic models ("VSM" and "LLT") with various configurations.
    - Plots the convergence results to visualize performance and accuracy trends.
    The function does not return any value, but it produces output files and plots that facilitate the evaluation of simulation convergence as the number of panels changes.   
    """

    # Example usage
    PROJECT_DIR = Path(__file__).resolve().parents[2]
    geometry_path = Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "wing_geometry.csv"
    polar_data_dir = (
        Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "2D_polars_corrected"
    )
    convergence_analysis_dir = (
        Path(PROJECT_DIR) / "results" / "TUDELFT_V3_KITE" / "convergence_analysis"
    )
    n_panels_list = [
        10,
        20,
        30,
        40,
        50,
        70,
        90,
        110,
        140,
        180,
        220,
        260,
        300,
        400,
    ]  # , 500]

    ### Testing aerodynamic model type and is with corrected polar
    is_with_corrected_polar = True
    spanwise_panel_distribution = "uniform"
    Umag = 10
    angle_of_attack = 10
    side_slip = 10
    yaw_rate = 0
    n_panels_list = [20, 40, 50]

    convergence_results_dir = generate_csv_files(
        convergence_analysis_dir,
        geometry_path,
        is_with_corrected_polar,
        polar_data_dir,
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
        is_with_corrected_polar_list=[True, False],
        polar_data_dir_list=None,
        spanwise_panel_distribution_list=None,
    )
    plot_convergence(
        convergence_results_dir,
        name="convergence",
    )

    ## Testing allowed error, gamma_initial_distribution_type, and max_iterations
    convergence_results_dir = generate_csv_files(
        convergence_analysis_dir,
        geometry_path,
        is_with_corrected_polar,
        polar_data_dir,
        spanwise_panel_distribution,
        Umag,
        angle_of_attack,
        side_slip,
        yaw_rate,
        n_panels_list,
        aerodynamic_model_type_list=None,
        allowed_error_list=[1e-2, 1e-5],
        core_radius_fraction_list=None,
        gamma_initial_distribution_type_list=["zero", "elliptical","previous"],
        gamma_loop_type_list=None,
        max_iterations_list=[1000, 3000],
        relaxation_factor_list=[0.01, 0.05,0.1],
        is_with_corrected_polar_list=None,
        polar_data_dir_list=None,
        spanwise_panel_distribution_list=None,
    )
    plot_convergence(
        convergence_results_dir,
        name="convergence",
    )


if __name__ == "__main__":
    main()
