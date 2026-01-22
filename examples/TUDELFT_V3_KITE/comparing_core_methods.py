from pathlib import Path
import numpy as np
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.plotting import (
    plot_polars,
)
from VSM.core2.BodyAerodynamics2 import BodyAerodynamics as BodyAerodynamics2
from VSM.core2.Solver2 import Solver as Solver2

from VSM.plot_geometry_matplotlib import plot_geometry
from VSM.plot_geometry_plotly import interactive_plot


def main():

    ### defining paths
    PROJECT_DIR = Path(__file__).resolve().parents[2]

    ### defining settings
    n_panels = 50
    spanwise_panel_distribution = "uniform"
    solver_base_version = Solver(reference_point=np.array([0.0, 0.0, 0.0]))
    solver_new_version = Solver2(reference_point=np.array([0.0, 0.0, 0.0]))

    # Step 1: Instantiate BodyAerodynamics objects from different YAML configs
    cad_derived_geometry_dir = (
        Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "CAD_derived_geometry"
    )
    body_aero_CAD_CFD_polars = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=(
            cad_derived_geometry_dir
            / "config_kite_CAD_CFD_polars_converged_fitted_pchip.yaml"
        ),
        spanwise_panel_distribution=spanwise_panel_distribution,
    )
    body_aero_CAD_CFD_polars2 = BodyAerodynamics2.instantiate(
        n_panels=n_panels,
        file_path=(
            cad_derived_geometry_dir
            / "config_kite_CAD_CFD_polars_converged_fitted_pchip.yaml"
        ),
        spanwise_panel_distribution=spanwise_panel_distribution,
        use_jointed_wake=False,
    )
    body_aero_CAD_CFD_polars2_jointed_wake = BodyAerodynamics2.instantiate(
        n_panels=n_panels,
        file_path=(
            cad_derived_geometry_dir
            / "config_kite_CAD_CFD_polars_converged_fitted_pchip.yaml"
        ),
        spanwise_panel_distribution=spanwise_panel_distribution,
        use_jointed_wake=True,
    )

    # Set inflow conditions for each aerodynamic object
    """
    Set the wind speed, angle of attack, side slip, and yaw rate for each BodyAerodynamics object.
    This initializes the apparent wind vector and prepares the objects for analysis.
    """
    Umag = 3.15
    angle_of_attack = 6.8
    side_slip = 0
    yaw_rate = 0
    body_aero_CAD_CFD_polars.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)
    body_aero_CAD_CFD_polars2.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)
    body_aero_CAD_CFD_polars2_jointed_wake.va_initialize(
        Umag, angle_of_attack, side_slip, yaw_rate
    )

    #  Plot polar curves for different angles of attack and side slip angles, and save results
    """
    Compare the aerodynamic performance of different models by plotting lift, drag, and side force coefficients
    as a function of angle of attack (alpha sweep) and side slip (beta sweep).
    Literature/CFD data can be included for validation.
    """
    save_folder = Path(PROJECT_DIR) / "results" / "TUDELFT_V3_KITE"

    # Step 5a: Plot alpha sweep (angle of attack)
    path_cfd_alpha = (
        Path(PROJECT_DIR)
        / "data"
        / "TUDELFT_V3_KITE"
        / "3D_polars_literature"
        / "CFD_RANS_Rey_10e5_Poland2025_alpha_sweep_beta_0.csv"
    )
    path_windtunnel_alpha = (
        Path(PROJECT_DIR)
        / "data"
        / "TUDELFT_V3_KITE"
        / "3D_polars_literature"
        / "WindTunnel_Re5e5_alpha_sweep_beta_0_Poland2025.csv"
    )
    plot_polars(
        solver_list=[
            solver_base_version,
            solver_new_version,
            solver_new_version,
        ],
        body_aero_list=[
            body_aero_CAD_CFD_polars,
            body_aero_CAD_CFD_polars2,
            body_aero_CAD_CFD_polars2_jointed_wake,
        ],
        label_list=[
            "VSM CAD CFD Polars",
            "VSM2 CAD CFD Polars",
            "VSM2 CAD CFD Polars Jointed Wake",
            path_cfd_alpha.stem,
            path_windtunnel_alpha.stem,
        ],
        literature_path_list=[path_cfd_alpha, path_windtunnel_alpha],
        angle_range=[0, 5, 8, 10, 12, 15, 20, 25],
        angle_type="angle_of_attack",
        angle_of_attack=0,
        side_slip=0,
        yaw_rate=0,
        Umag=Umag,
        title="alphasweep",
        data_type=".pdf",
        save_path=Path(save_folder),
        is_save=False,
        is_show=True,
    )

    # Step 5b: Plot beta sweep (side slip)
    path_cfd_beta = (
        Path(PROJECT_DIR)
        / "data"
        / "TUDELFT_V3_KITE"
        / "3D_polars_literature"
        / "CFD_RANS_Re1e6_beta_sweep_alpha_13_Vire2022_CorrectedByPoland2025.csv"
    )
    path_windtunnel_beta = (
        Path(PROJECT_DIR)
        / "data"
        / "TUDELFT_V3_KITE"
        / "3D_polars_literature"
        / "WindTunnel_Re5e5_beta_sweep_alpha_13_Poland2025.csv"
    )
    plot_polars(
        solver_list=[
            solver_base_version,
            solver_new_version,
            solver_new_version,
        ],
        body_aero_list=[
            body_aero_CAD_CFD_polars,
            body_aero_CAD_CFD_polars2,
            body_aero_CAD_CFD_polars2_jointed_wake,
        ],
        label_list=[
            "VSM CAD CFD Polars",
            "VSM2 CAD CFD Polars",
            "VSM2 CAD CFD Polars Jointed Wake",
            path_cfd_beta.stem,
            path_windtunnel_beta.stem,
        ],
        literature_path_list=[path_cfd_beta, path_windtunnel_beta],
        angle_range=[0, 3, 6, 9, 12],
        angle_type="side_slip",
        angle_of_attack=12.5,
        side_slip=0,
        yaw_rate=0,
        Umag=3.15,
        title="betasweep",
        data_type=".pdf",
        save_path=Path(save_folder),
        is_save=False,
        is_show=True,
    )


if __name__ == "__main__":
    main()
