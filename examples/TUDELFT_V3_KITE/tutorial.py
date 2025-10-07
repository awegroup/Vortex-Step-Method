from pathlib import Path
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.plotting import (
    plot_polars,
)
from VSM.plot_geometry_matplotlib import plot_geometry
from VSM.plot_geometry_plotly import interactive_plot


def main():
    """
    Example: 3D Aerodynamic Analysis of TUDELFT_V3_KITE using VSM

    This script demonstrates the workflow for performing a 3D aerodynamic analysis of the TUDELFT_V3_KITE
    using the Vortex Step Method (VSM) library. The workflow is structured as follows:

    Step 1: Instantiate BodyAerodynamics objects from different YAML configuration files.
        - Each YAML config defines the geometry and airfoil/polar data for a specific modeling approach.
        - Supported approaches include:
            - Breukels regression (empirical model)
            - CFD-based polars
            - NeuralFoil-based polars
            - Masure regression (machine learning model)
            - Inviscid theory

    Step 2: Set inflow conditions for each aerodynamic object.
        - Specify wind speed (Umag), angle of attack, side slip, and yaw rate.
        - Initialize the apparent wind for each BodyAerodynamics object.

    Step 3: Plot the kite geometry using Matplotlib.
        - Visualize the panel mesh, control points, and aerodynamic centers.

    Step 4: Create an interactive 3D plot using Plotly.
        - Allows for interactive exploration of the geometry and panel arrangement.

    Step 5: Plot and save polar curves for different angles of attack and side slip angles.
        - Compare the results of different aerodynamic models.
        - Optionally include literature/CFD data for validation.

    Step 5a: Plot alpha sweep (angle of attack variation).
    Step 5b: Plot beta sweep (side slip variation).

    Returns:
        None
    """
    ### 1. defining paths
    PROJECT_DIR = Path(__file__).resolve().parents[2]

    ### 2. defining settings
    n_panels = 100
    spanwise_panel_distribution = "uniform"
    solver_base_version = Solver()

    # Step 1: Instantiate BodyAerodynamics objects from different YAML configs
    cad_derived_geometry_dir = (
        Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "CAD_derived_geometry"
    )
    body_aero_CAD_CFD_polars = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=(cad_derived_geometry_dir / "config_kite_CAD_CFD_polars.yaml"),
        spanwise_panel_distribution=spanwise_panel_distribution,
    )
    body_aero_CAD_CFD_polars_with_bridles = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        # file_path=(cad_derived_geometry_dir / "config_kite_CAD_CFD_polars.yaml"),
        file_path=(cad_derived_geometry_dir / "aero_geometry.yaml"),
        spanwise_panel_distribution=spanwise_panel_distribution,
        bridle_path=(cad_derived_geometry_dir / "struc_geometry.yaml"),
        ml_models_dir=(Path(PROJECT_DIR) / "data" / "ml_models"),
    )
    body_aero_CAD_neuralfoil = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=(cad_derived_geometry_dir / "config_kite_CAD_neuralfoil.yaml"),
        spanwise_panel_distribution=spanwise_panel_distribution,
    )
    body_aero_masure_regression = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=(cad_derived_geometry_dir / "config_kite_CAD_masure_regression.yaml"),
        ml_models_dir=(Path(PROJECT_DIR) / "data" / "ml_models"),
        spanwise_panel_distribution=spanwise_panel_distribution,
    )

    # Step 2: Set inflow conditions for each aerodynamic object
    """
    Set the wind speed, angle of attack, side slip, and yaw rate for each BodyAerodynamics object.
    This initializes the apparent wind vector and prepares the objects for analysis.
    """
    Umag = 3.15
    angle_of_attack = 6.8
    side_slip = 0
    yaw_rate = 0
    body_aero_CAD_CFD_polars.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)
    body_aero_CAD_CFD_polars_with_bridles.va_initialize(
        Umag, angle_of_attack, side_slip, yaw_rate
    )
    body_aero_CAD_neuralfoil.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)
    body_aero_masure_regression.va_initialize(
        Umag, angle_of_attack, side_slip, yaw_rate
    )

    # Step 3: Plot the kite geometry using Matplotlib
    """
    Visualize the panel mesh, control points, and aerodynamic centers for the selected BodyAerodynamics object.
    """
    # plot_geometry(
    #     body_aero_CAD_CFD_polars,
    #     title="TUDELFT_V3_KITE",
    #     data_type=".pdf",
    #     save_path=".",
    #     is_save=False,
    #     is_show=True,
    # )

    # Step 4: Create an interactive plot using Plotly
    """
    Generate an interactive 3D plot for the selected BodyAerodynamics object.
    This allows for interactive exploration of the geometry and panel arrangement.
    """
    interactive_plot(
        body_aero_CAD_CFD_polars_with_bridles,
        vel=Umag,
        angle_of_attack=angle_of_attack,
        side_slip=side_slip,
        yaw_rate=yaw_rate,
        is_with_aerodynamic_details=True,
        title="TUDELFT_V3_KITE",
        is_with_bridles=True,
        ### uncomment the lines below to save
        # is_save=True,
        # save_path=Path(PROJECT_DIR)
        # / "results"
        # / "TUDELFT_V3_KITE"
        # / "interactive_plot.html",
    )

    breakpoint()

    # Step 5: Plot polar curves for different angles of attack and side slip angles, and save results
    """
    Compare the aerodynamic performance of different models by plotting lift, drag, and side force coefficients
    as a function of angle of attack (alpha sweep) and side slip (beta sweep).
    Literature/CFD data can be included for validation.
    """
    save_folder = Path(PROJECT_DIR) / "results" / "TUDELFT_V3_KITE"

    # Step 5a: Plot alpha sweep (angle of attack)
    path_cfd_lebesque_alpha = (
        Path(PROJECT_DIR)
        / "data"
        / "TUDELFT_V3_KITE"
        / "3D_polars_literature"
        / "CFD_RANS_Rey_10e5_Poland2025_alpha_sweep_beta_0.csv"
    )
    plot_polars(
        solver_list=[
            solver_base_version,
            solver_base_version,
            solver_base_version,
            solver_base_version,
        ],
        body_aero_list=[
            body_aero_CAD_CFD_polars,
            body_aero_CAD_CFD_polars_with_bridles,
            body_aero_CAD_neuralfoil,
            body_aero_masure_regression,
        ],
        label_list=[
            "VSM CAD CFD Polars",
            "VSM CAD CFD Polars with Bridles",
            "VSM CAD NeuralFoil",
            "VSM CAD Masure Regression",
            "CFD_RANS_Rey_10e5_Poland2025_alpha_sweep_beta_0",
        ],
        literature_path_list=[path_cfd_lebesque_alpha],
        angle_range=[0, 5, 8, 10, 12, 15, 20, 25],
        angle_type="angle_of_attack",
        angle_of_attack=0,
        side_slip=0,
        yaw_rate=0,
        Umag=Umag,
        title="alphasweep",
        data_type=".pdf",
        save_path=Path(save_folder),
        is_save=True,
        is_show=False,
    )

    # Step 5b: Plot beta sweep (side slip)
    path_cfd_lebesque_beta = (
        Path(PROJECT_DIR)
        / "data"
        / "TUDELFT_V3_KITE"
        / "3D_polars_literature"
        / "CFD_RANS_Rey_10e5_Poland2025_beta_sweep_alpha_13_02.csv"
    )
    plot_polars(
        solver_list=[
            solver_base_version,
            solver_base_version,
            solver_base_version,
            solver_base_version,
        ],
        body_aero_list=[
            body_aero_CAD_CFD_polars,
            body_aero_CAD_CFD_polars_with_bridles,
            body_aero_CAD_neuralfoil,
            body_aero_masure_regression,
        ],
        label_list=[
            "VSM CAD CFD Polars",
            "VSM CAD CFD Polars with Bridles",
            "VSM CAD NeuralFoil",
            "VSM CAD Masure Regression",
            "CFD_RANS_Rey_10e5_Poland2025_beta_sweep_alpha_13_02",
        ],
        literature_path_list=[path_cfd_lebesque_beta],
        angle_range=[0, 3, 6, 9, 12],
        angle_type="side_slip",
        angle_of_attack=6.8,
        side_slip=0,
        yaw_rate=0,
        Umag=3.15,
        title="betasweep",
        data_type=".pdf",
        save_path=Path(save_folder),
        is_save=True,
        is_show=True,
    )


if __name__ == "__main__":
    main()
