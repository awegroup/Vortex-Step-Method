import numpy as np
import time as time
from pathlib import Path
from VSM.WingGeometry import Wing
from VSM.BodyAerodynamics import BodyAerodynamics
from VSM.Solver import Solver
from VSM.plotting import plot_polars, plot_distribution


def testing_single_solver_setting(
    save_folder,
    body_aero_list,
    parameter,
    value_list,
    aerodynamic_model_type="VSM",
    max_iterations=5000,
    allowed_error=1e-6,
    relaxation_factor=0.01,
    core_radius_fraction=1e-20,
    gamma_loop_type="base",
    is_with_gamma_feedback=False,
    gamma_initial_distribution_type="elliptical",
    alpha_range=np.linspace(0, 25, 20),
    alpha_range_distribution=[19, 20, 21, 22, 23],
    beta_range=[0, 3, 6, 9, 12],
    beta_range_distribution=[0, 3, 6],
    Umag=3.15,
    angle_of_attack=6.5,
    side_slip=0,
    yaw_rate=0,
):
    if parameter not in [
        "aerodynamic_model_type",
        "max_iterations",
        "allowed_error",
        "relaxation_factor",
        "core_radius_fraction",
        "gamma_loop_type",
        "is_with_gamma_feedback",
        "gamma_initial_distribution_type",
    ]:
        raise ValueError(
            f"Parameter {parameter} not recognized. Please choose from ['aerodynamic_model_type', 'max_iterations', 'allowed_error', 'relaxation_factor', 'core_radius_fraction', 'gamma_loop_type', 'is_with_gamma_feedback', 'gamma_initial_distribution_type']"
        )

    solver_list = []
    label_list = []
    y_coords_list = []
    for value, body_aero in zip(value_list, body_aero_list):
        if "aerodynamic_model_type" == parameter:
            solver_list.append(
                Solver(
                    aerodynamic_model_type=value,
                    max_iterations=max_iterations,
                    allowed_error=allowed_error,
                    relaxation_factor=relaxation_factor,
                    core_radius_fraction=core_radius_fraction,
                    gamma_loop_type=gamma_loop_type,
                    is_with_gamma_feedback=is_with_gamma_feedback,
                    gamma_initial_distribution_type=gamma_initial_distribution_type,
                )
            )
        elif "max_iterations" == parameter:
            solver_list.append(
                Solver(
                    aerodynamic_model_type=aerodynamic_model_type,
                    max_iterations=value,
                    allowed_error=allowed_error,
                    relaxation_factor=relaxation_factor,
                    core_radius_fraction=core_radius_fraction,
                    gamma_loop_type=gamma_loop_type,
                    is_with_gamma_feedback=is_with_gamma_feedback,
                    gamma_initial_distribution_type=gamma_initial_distribution_type,
                )
            )
        elif "allowed_error" == parameter:
            solver_list.append(
                Solver(
                    aerodynamic_model_type=aerodynamic_model_type,
                    max_iterations=max_iterations,
                    allowed_error=value,
                    relaxation_factor=relaxation_factor,
                    core_radius_fraction=core_radius_fraction,
                    gamma_loop_type=gamma_loop_type,
                    is_with_gamma_feedback=is_with_gamma_feedback,
                    gamma_initial_distribution_type=gamma_initial_distribution_type,
                )
            )
        elif "relaxation_factor" == parameter:
            solver_list.append(
                Solver(
                    aerodynamic_model_type=aerodynamic_model_type,
                    max_iterations=max_iterations,
                    allowed_error=allowed_error,
                    relaxation_factor=value,
                    core_radius_fraction=core_radius_fraction,
                    gamma_loop_type=gamma_loop_type,
                    is_with_gamma_feedback=is_with_gamma_feedback,
                    gamma_initial_distribution_type=gamma_initial_distribution_type,
                )
            )
        elif "core_radius_fraction" == parameter:
            solver_list.append(
                Solver(
                    aerodynamic_model_type=aerodynamic_model_type,
                    max_iterations=max_iterations,
                    allowed_error=allowed_error,
                    relaxation_factor=relaxation_factor,
                    core_radius_fraction=value,
                    gamma_loop_type=gamma_loop_type,
                    is_with_gamma_feedback=is_with_gamma_feedback,
                    gamma_initial_distribution_type=gamma_initial_distribution_type,
                )
            )
        elif "gamma_loop_type" == parameter:
            solver_list.append(
                Solver(
                    aerodynamic_model_type=aerodynamic_model_type,
                    max_iterations=max_iterations,
                    allowed_error=allowed_error,
                    relaxation_factor=relaxation_factor,
                    core_radius_fraction=core_radius_fraction,
                    gamma_loop_type=value,
                    is_with_gamma_feedback=is_with_gamma_feedback,
                    gamma_initial_distribution_type=gamma_initial_distribution_type,
                )
            )
        elif "is_with_gamma_feedback" == parameter:
            solver_list.append(
                Solver(
                    aerodynamic_model_type=aerodynamic_model_type,
                    max_iterations=max_iterations,
                    allowed_error=allowed_error,
                    relaxation_factor=relaxation_factor,
                    core_radius_fraction=core_radius_fraction,
                    gamma_loop_type=gamma_loop_type,
                    is_with_gamma_feedback=value,
                    gamma_initial_distribution_type=gamma_initial_distribution_type,
                )
            )
        elif "gamma_initial_distribution_type" == parameter:
            solver_list.append(
                Solver(
                    aerodynamic_model_type=aerodynamic_model_type,
                    max_iterations=max_iterations,
                    allowed_error=allowed_error,
                    relaxation_factor=relaxation_factor,
                    core_radius_fraction=core_radius_fraction,
                    gamma_loop_type=gamma_loop_type,
                    is_with_gamma_feedback=is_with_gamma_feedback,
                    gamma_initial_distribution_type=value,
                )
            )
        label_list.append(f"{parameter} = {value}")
        y_coords_list.append([panel.control_point[1] for panel in body_aero.panels])

    # plotting alpha-polar
    plot_polars(
        solver_list=solver_list,
        body_aero_list=body_aero_list,
        label_list=label_list,
        literature_path_list=[],
        angle_range=alpha_range,
        angle_type="angle_of_attack",
        angle_of_attack=angle_of_attack,
        side_slip=side_slip,
        yaw_rate=yaw_rate,
        Umag=Umag,
        title=f"alphasweep_{parameter}",
        data_type=".pdf",
        save_path=Path(save_folder),
        is_save=True,
        is_show=False,
    )
    # plotting beta-polar
    plot_polars(
        solver_list=solver_list,
        body_aero_list=body_aero_list,
        label_list=label_list,
        literature_path_list=[],
        angle_range=beta_range,
        angle_type="side_slip",
        angle_of_attack=angle_of_attack,
        side_slip=side_slip,
        yaw_rate=yaw_rate,
        Umag=Umag,
        title=f"betasweep_{parameter}",
        data_type=".pdf",
        save_path=Path(save_folder),
        is_save=True,
        is_show=False,
    )
    # plotting distributions
    for side_slip in beta_range_distribution:
        for alpha in alpha_range_distribution:
            print(f"\nalpha: {alpha}")
            results_list = []
            run_time_list = []
            for solver, body_aero in zip(solver_list, body_aero_list):
                print(f"\n{parameter}={getattr(solver, parameter)}")
                body_aero.va_initialize(Umag, alpha, side_slip, yaw_rate)
                begin_time = time.time()
                results_list.append(solver.solve(body_aero, gamma_distribution=None))
                run_time_list.append(time.time() - begin_time)

            plot_distribution(
                y_coordinates_list=y_coords_list,
                results_list=results_list,
                label_list=label_list,
                title=f"spanwise_distribution_{parameter}_{side_slip}_alpha_{alpha}",
                data_type=".pdf",
                save_path=save_folder,
                is_save=True,
                is_show=False,
                run_time_list=run_time_list,
            )


def testing_all_solver_settings(
    aerodynamic_model_type_list,
    max_iterations_list,
    allowed_error_list,
    relaxation_factor_list,
    core_radius_fraction_list,
    gamma_loop_type_list,
    is_with_gamma_feedback_list,
    gamma_initial_distribution_type_list,
    save_folder,
    body_aero_uniform,
    aerodynamic_model_type="VSM",
    max_iterations=5000,
    allowed_error=1e-6,
    relaxation_factor=0.01,
    core_radius_fraction=1e-20,
    gamma_loop_type="base",
    is_with_gamma_feedback=False,
    gamma_initial_distribution_type="elliptical",
    alpha_range=np.linspace(0, 25, 20),
    alpha_range_distribution=[19, 20, 21, 22, 23],
    beta_range=[0, 3, 6, 9, 12],
    beta_range_distribution=[0, 3, 6],
    Umag=3.15,
    angle_of_attack=6.5,
    side_slip=0,
    yaw_rate=0,
):

    parameter_list = [
        "aerodynamic_model_type",
        "max_iterations",
        "allowed_error",
        "relaxation_factor",
        "core_radius_fraction",
        "gamma_loop_type",
        "is_with_gamma_feedback",
        "gamma_initial_distribution_type",
    ]
    value_list_list = [
        aerodynamic_model_type_list,
        max_iterations_list,
        allowed_error_list,
        relaxation_factor_list,
        core_radius_fraction_list,
        gamma_loop_type_list,
        is_with_gamma_feedback_list,
        gamma_initial_distribution_type_list,
    ]
    for parameter, value_list in zip(parameter_list, value_list_list):
        body_aero_list = [body_aero_uniform] * len(value_list)
        save_dir = Path(save_folder) / parameter
        save_dir.mkdir(parents=True, exist_ok=True)
        testing_single_solver_setting(
            save_dir,
            body_aero_list,
            parameter,
            value_list,
            aerodynamic_model_type,
            max_iterations,
            allowed_error,
            relaxation_factor,
            core_radius_fraction,
            gamma_loop_type,
            is_with_gamma_feedback,
            gamma_initial_distribution_type,
            alpha_range,
            alpha_range_distribution,
            beta_range,
            beta_range_distribution,
            Umag,
            angle_of_attack,
            side_slip,
            yaw_rate,
        )


### Settings
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
print(f"\nPROJECT_DIR: {PROJECT_DIR}")
save_folder = (
    Path(PROJECT_DIR) / "examples" / "TUDELFT_V3_KITE" / "solver_setting_effects"
)
wing_instance = Wing(n_panels=50, spanwise_panel_distribution="uniform")
body_aero_uniform = BodyAerodynamics.from_file(
    wing_instance,
    file_path=Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "wing_geometry.csv",
    is_with_corrected_polar=True,
    path_polar_data_dir=(
        Path(PROJECT_DIR)
        / "examples"
        / "TUDELFT_V3_KITE"
        / "polar_engineering"
        / "csv_files"
    ),
)

# ## QUICK TEST VERSION - just to see if things work
# testing_all_solver_settings(
#     aerodynamic_model_type_list=["VSM", "LLT"],
#     max_iterations_list=[1e2],
#     allowed_error_list=[1e-1],
#     relaxation_factor_list=[0.01],
#     core_radius_fraction_list=[1e-20],
#     gamma_loop_type_list=["base"],
#     is_with_gamma_feedback_list=[False],
#     gamma_initial_distribution_type_list=["elliptical"],
#     save_folder=save_folder,
#     body_aero_uniform=body_aero_uniform,
#     aerodynamic_model_type="VSM",
#     max_iterations=5000,
#     allowed_error=1e-6,
#     relaxation_factor=0.01,
#     core_radius_fraction=1e-20,
#     gamma_loop_type="base",
#     is_with_gamma_feedback=False,
#     gamma_initial_distribution_type="elliptical",
#     alpha_range=[5, 10],
#     alpha_range_distribution=[22, 23],
#     beta_range=[0, 6],
#     beta_range_distribution=[0],
#     Umag=3.15,
#     angle_of_attack=6.5,
#     side_slip=0,
#     yaw_rate=0,
# )

# EXTENSIVE VERSION - for parameter sweeps
testing_all_solver_settings(
    aerodynamic_model_type_list=["VSM", "LLT"],
    max_iterations_list=[1e2, 1e3, 5e3, 1e4],
    allowed_error_list=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
    relaxation_factor_list=[0.2, 0.1, 0.05, 0.03, 0.01, 0.005, 0.001],
    core_radius_fraction_list=[1e-5, 1e-10, 1e-20],
    gamma_loop_type_list=["base", "non_linear"],
    is_with_gamma_feedback_list=[False, True],
    gamma_initial_distribution_type_list=["elliptical", "cosine", "zero"],
    save_folder=save_folder,
    body_aero_uniform=body_aero_uniform,
    aerodynamic_model_type="VSM",
    max_iterations=5000,
    allowed_error=1e-6,
    relaxation_factor=0.01,
    core_radius_fraction=1e-20,
    gamma_loop_type="base",
    is_with_gamma_feedback=False,
    gamma_initial_distribution_type="elliptical",
    alpha_range=np.linspace(0, 25, 20),
    alpha_range_distribution=[18, 19, 20, 21, 22, 23],
    beta_range=[0, 3, 6, 9, 12],
    beta_range_distribution=[0, 3, 6],
    Umag=3.15,
    angle_of_attack=6.5,
    side_slip=0,
    yaw_rate=0,
)
