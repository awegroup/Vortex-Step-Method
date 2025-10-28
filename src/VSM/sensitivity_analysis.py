import numpy as np
import time as time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from VSM.core.WingGeometry import Wing
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.plotting import plot_polars, plot_distribution
from VSM.plot_styling import set_plot_style


def testing_single_solver_setting(
    save_folder,
    body_aero_list,
    parameter,
    value_list,
    alpha_range,
    alpha_range_distribution,
    beta_range,
    beta_range_distribution,
    Umag,
    angle_of_attack,
    side_slip,
    yaw_rate,
    pitch_rate,
    roll_rate,
    literature_path_list_alpha,
    literature_label_list_alpha,
    literature_path_list_beta,
    literature_label_list_beta,
):

    solver_list = []
    label_list = []
    y_coords_list = []
    for value, body_aero in zip(value_list, body_aero_list):
        if parameter in [
            "aerodynamic_model_type",
            "max_iterations",
            "allowed_error",
            "relaxation_factor",
            "core_radius_fraction",
            "gamma_loop_type",
            "gamma_initial_distribution_type",
        ]:
            solver_list.append(Solver(**{parameter: value}))
        else:
            raise ValueError(
                f"Parameter {parameter} not recognized. \nPlease choose from ['aerodynamic_model_type', 'max_iterations', 'allowed_error', 'relaxation_factor', 'core_radius_fraction', 'gamma_loop_type', 'gamma_initial_distribution_type']"
            )

        label_list.append(f"{parameter} = {value}")
        y_coords_list.append([panel.control_point[1] for panel in body_aero.panels])

    label_list_alpha = label_list.copy()
    if len(literature_label_list_alpha) > 0:
        for literature_label in enumerate(literature_label_list_alpha):
            label_list_alpha.append(literature_label[1])

    label_list_beta = label_list.copy()
    if len(literature_label_list_beta) > 0:
        for literature_label in enumerate(literature_label_list_beta):
            label_list_beta.append(literature_label[1])

    # plotting alpha-polar
    plot_polars(
        solver_list=solver_list,
        body_aero_list=body_aero_list,
        label_list=label_list_alpha,
        literature_path_list=literature_path_list_alpha,
        angle_range=alpha_range,
        angle_type="angle_of_attack",
        angle_of_attack=angle_of_attack,
        side_slip=side_slip,
        yaw_rate=yaw_rate,
        pitch_rate=pitch_rate,
        roll_rate=roll_rate,
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
        label_list=label_list_beta,
        literature_path_list=literature_path_list_beta,
        angle_range=beta_range,
        angle_type="side_slip",
        angle_of_attack=angle_of_attack,
        side_slip=side_slip,
        yaw_rate=yaw_rate,
        pitch_rate=pitch_rate,
        roll_rate=roll_rate,
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
                body_aero.va_initialize(
                    Umag,
                    alpha,
                    side_slip,
                    yaw_rate,
                    pitch_rate,
                    roll_rate,
                )
                begin_time = time.time()
                results_list.append(solver.solve(body_aero))
                run_time_list.append(time.time() - begin_time)
            # Prepare inputs for plot_distribution
            alpha_list = [alpha]
            solver_list_for_plot = solver_list
            body_aero_list_for_plot = body_aero_list
            # plot_distribution expects:
            # (alpha_list, Umag, side_slip, yaw_rate, solver_list, body_aero_list, label_list, ...)
            plot_distribution(
                alpha_list=alpha_list,
                Umag=Umag,
                side_slip=side_slip,
                yaw_rate=yaw_rate,
                pitch_rate=pitch_rate,
                roll_rate=roll_rate,
                solver_list=solver_list_for_plot,
                body_aero_list=body_aero_list_for_plot,
                label_list=label_list,
                title=f"spanwise_distribution_{parameter}_{side_slip}_alpha_{alpha}",
                data_type=".pdf",
                save_path=save_folder,
                is_save=True,
                is_show=False,
                run_time_list=run_time_list,
            )


def testing_n_panels_effect(
    save_dir,
    file_path,
    is_with_corrected_polar,
    polar_data_dir,
    spanwise_panel_distribution,
    n_panels_list,
    solver,
    alpha_range,
    alpha_range_distribution,
    beta_range,
    beta_range_distribution,
    Umag,
    angle_of_attack,
    side_slip,
    yaw_rate,
    pitch_rate,
    roll_rate,
    literature_path_list_alpha,
    literature_label_list_alpha,
    literature_path_list_beta,
    literature_label_list_beta,
):

    # Create default solver if not provided
    if solver is None:
        solver = Solver()

    # Process n_panels parameter
    body_aero_list = []
    label_list = []
    y_coords_list = []

    for n_panels in n_panels_list:
        # Use the new instantiate method
        body_aero = BodyAerodynamics.instantiate(
            n_panels=n_panels,
            file_path=file_path,
            spanwise_panel_distribution=spanwise_panel_distribution,
        )
        body_aero_list.append(body_aero)
        label_list.append(f"n_panels = {n_panels}")
        y_coords_list.append([panel.control_point[1] for panel in body_aero.panels])

    label_list_alpha = label_list.copy()
    if len(literature_label_list_alpha) > 0:
        for literature_label in enumerate(literature_label_list_alpha):
            label_list_alpha.append(literature_label)

    label_list_beta = label_list.copy()
    if len(literature_label_list_beta) > 0:
        for literature_label in enumerate(literature_label_list_beta):
            label_list_beta.append(literature_label)

    # Create a list of solvers (same solver for each wing)
    solver_list = [solver] * len(n_panels_list)

    # Plotting alpha-polar
    plot_polars(
        solver_list=solver_list,
        body_aero_list=body_aero_list,
        label_list=label_list_alpha,
        literature_path_list=literature_path_list_alpha,
        angle_range=alpha_range,
        angle_type="angle_of_attack",
        angle_of_attack=angle_of_attack,
        side_slip=side_slip,
        yaw_rate=yaw_rate,
        pitch_rate=pitch_rate,
        roll_rate=roll_rate,
        Umag=Umag,
        title="alphasweep_n_panels",
        data_type=".pdf",
        save_path=save_dir,
        is_save=True,
        is_show=False,
    )

    # Plotting beta-polar
    plot_polars(
        solver_list=solver_list,
        body_aero_list=body_aero_list,
        label_list=label_list_beta,
        literature_path_list=literature_path_list_beta,
        angle_range=beta_range,
        angle_type="side_slip",
        angle_of_attack=angle_of_attack,
        side_slip=side_slip,
        yaw_rate=yaw_rate,
        pitch_rate=pitch_rate,
        roll_rate=roll_rate,
        Umag=Umag,
        title="betasweep_n_panels",
        data_type=".pdf",
        save_path=save_dir,
        is_save=True,
        is_show=False,
    )

    # Plotting distributions
    for side_slip in beta_range_distribution:
        for alpha in alpha_range_distribution:
            print(f"\nalpha: {alpha}")
            results_list = []
            run_time_list = []

            for i, body_aero in enumerate(body_aero_list):
                print(f"\nn_panels={n_panels_list[i]}")

                body_aero.va_initialize(
                    Umag,
                    alpha,
                    side_slip,
                    yaw_rate,
                    pitch_rate,
                    roll_rate,
                )
                begin_time = time.time()
                results_list.append(solver.solve(body_aero))
                run_time_list.append(time.time() - begin_time)

            # Prepare inputs for plot_distribution
            alpha_list = [alpha]
            solver_list_for_plot = solver_list
            body_aero_list_for_plot = body_aero_list
            # plot_distribution expects:
            # (alpha_list, Umag, side_slip, yaw_rate, solver_list, body_aero_list, label_list, ...)
            plot_distribution(
                alpha_list=alpha_list,
                Umag=Umag,
                side_slip=side_slip,
                yaw_rate=yaw_rate,
                pitch_rate=pitch_rate,
                roll_rate=roll_rate,
                solver_list=solver_list_for_plot,
                body_aero_list=body_aero_list_for_plot,
                label_list=label_list,
                title=f"spanwise_distribution_n_panels_beta_{side_slip}_alpha_{alpha}",
                data_type=".pdf",
                save_path=save_dir,
                is_save=True,
                is_show=False,
                run_time_list=run_time_list,
            )

    print("\nN_panels effect testing completed.")


def testing_all_solver_settings(
    sensitivity_results_dir,
    geometry_path,
    is_with_corrected_polar=False,
    polar_data_dir=None,
    n_panels=50,
    spanwise_panel_distribution="uniform",
    solver=None,
    Umag=5,
    angle_of_attack=5,
    side_slip=0,
    yaw_rate=0,
    pitch_rate=0,
    roll_rate=0,
    aerodynamic_model_type_list=None,
    allowed_error_list=None,
    core_radius_fraction_list=None,
    gamma_initial_distribution_type_list=None,
    gamma_loop_type_list=None,
    max_iterations_list=None,
    n_panels_list=None,
    relaxation_factor_list=None,
    alpha_range=np.linspace(0, 25, 20),
    alpha_range_distribution=[19, 20, 21, 22, 23],
    beta_range=[0, 3, 6, 9, 12],
    beta_range_distribution=[0, 3, 6],
    literature_path_list_alpha=[],
    literature_label_list_alpha=[],
    literature_path_list_beta=[],
    literature_label_list_beta=[],
):

    parameter_list = [
        "aerodynamic_model_type",
        "allowed_error",
        "core_radius_fraction",
        "gamma_initial_distribution_type",
        "gamma_loop_type",
        "max_iterations",
        "n_panels",
        "relaxation_factor",
    ]
    value_list_list = [
        aerodynamic_model_type_list,
        allowed_error_list,
        core_radius_fraction_list,
        gamma_initial_distribution_type_list,
        gamma_loop_type_list,
        max_iterations_list,
        n_panels_list,
        relaxation_factor_list,
    ]
    # Use instantiate for the base object
    body_aero = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=geometry_path,
        spanwise_panel_distribution=spanwise_panel_distribution,
        is_with_bridles=False,
    )

    for parameter, value_list in zip(parameter_list, value_list_list):
        if value_list is None:
            continue

        body_aero_list = [body_aero] * len(value_list)
        save_dir = Path(sensitivity_results_dir) / parameter
        save_dir.mkdir(parents=True, exist_ok=True)
        if parameter == "n_panels":
            testing_n_panels_effect(
                save_dir,
                geometry_path,
                is_with_corrected_polar,
                polar_data_dir,
                spanwise_panel_distribution,
                n_panels_list,
                solver,
                alpha_range,
                alpha_range_distribution,
                beta_range,
                beta_range_distribution,
                Umag,
                angle_of_attack,
                side_slip,
                yaw_rate,
                pitch_rate,
                roll_rate,
                literature_path_list_alpha,
                literature_label_list_alpha,
                literature_path_list_beta,
                literature_label_list_beta,
            )
        else:
            testing_single_solver_setting(
                save_dir,
                body_aero_list,
                parameter,
                value_list,
                alpha_range,
                alpha_range_distribution,
                beta_range,
                beta_range_distribution,
                Umag,
                angle_of_attack,
                side_slip,
                yaw_rate,
                pitch_rate,
                roll_rate,
                literature_path_list_alpha,
                literature_label_list_alpha,
                literature_path_list_beta,
                literature_label_list_beta,
            )
