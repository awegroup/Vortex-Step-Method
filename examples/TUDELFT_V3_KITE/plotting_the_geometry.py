from pathlib import Path
import numpy as np
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
    n_panels = 50
    spanwise_panel_distribution = "uniform"
    solver_base_version = Solver(reference_point=np.array([0.0, 0.0, 0.0]))

    # Step 1: Instantiate BodyAerodynamics objects from different YAML configs
    cad_derived_geometry_dir = (
        Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "CAD_derived_geometry"
    )
    body_aero_CAD_CFD_polars = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=(cad_derived_geometry_dir / "aero_geometry_CAD_CFD_polars.yaml"),
        spanwise_panel_distribution=spanwise_panel_distribution,
    )
    body_aero_CAD_CFD_polars_with_bridles = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=(cad_derived_geometry_dir / "aero_geometry_CAD_CFD_polars.yaml"),
        spanwise_panel_distribution=spanwise_panel_distribution,
        bridle_path=(
            cad_derived_geometry_dir / "struc_geometry_manually_adjusted.yaml"
        ),
        ml_models_dir=(Path(PROJECT_DIR) / "data" / "ml_models"),
    )
    body_aero_CAD_neuralfoil = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=(cad_derived_geometry_dir / "aero_geometry_CAD_neuralfoil.yaml"),
        spanwise_panel_distribution=spanwise_panel_distribution,
    )
    body_aero_masure_regression = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=(
            cad_derived_geometry_dir / "aero_geometry_CAD_masure_regression.yaml"
        ),
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
    plot_geometry(
        body_aero_CAD_CFD_polars,
        title="TUDELFT_V3_KITE",
        data_type=".pdf",
        save_path=".",
        is_save=False,
        is_show=True,
    )

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


if __name__ == "__main__":
    main()
