from pathlib import Path

# Assuming these are your project modules:
from VSM.Solver import Solver
from VSM.convergence_analysis import (
    generate_csv_files,
    plot_convergence,
)

if __name__ == "__main__":
    # Example usage
    PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
    geometry_path = Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "wing_geometry.csv"
    polar_data_dir = (
        Path(PROJECT_DIR)
        / "examples"
        / "TUDELFT_V3_KITE"
        / "polar_engineering"
        / "csv_files"
    )
    convergence_results_dir = (
        Path(PROJECT_DIR) / "results" / "TUDELFT_V3_KITE" / "convergence_analysis_2"
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

    ### Testing version
    is_with_corrected_polar = True
    spanwise_panel_distribution = "uniform"
    Umag = 10
    angle_of_attack = 10
    side_slip = 10
    yaw_rate = 0
    # n_panels_list = [40, 50]
    # convergence_results_dir = Path(
    #     convergence_results_dir,
    #     "Umag_{:.1f}alpha_{:.1f}_beta_{:.1f}".format(Umag, angle_of_attack, side_slip),
    # )

    # generate_csv_files(
    #     convergence_results_dir,
    #     geometry_path,
    #     is_with_corrected_polar,
    #     polar_data_dir,
    #     spanwise_panel_distribution,
    #     Umag,
    #     angle_of_attack,
    #     side_slip,
    #     yaw_rate,
    #     n_panels_list,
    #     aerodynamic_model_type_list=None,
    #     allowed_error_list=None,
    #     core_radius_fraction_list=None,
    #     gamma_initial_distribution_type_list=None,
    #     gamma_loop_type_list=None,
    #     max_iterations_list=None,
    #     relaxation_factor_list=None,
    #     is_with_corrected_polar_list=[True, False],
    #     polar_data_dir_list=None,
    #     spanwise_panel_distribution_list=None,
    # )
    plot_convergence(
        convergence_results_dir,
        name="convergence",
    )
