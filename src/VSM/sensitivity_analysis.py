import numpy as np
import time as time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from VSM.WingGeometry import Wing
from VSM.BodyAerodynamics import BodyAerodynamics
from VSM.Solver import Solver
from VSM.plotting import plot_polars, plot_distribution
from VSM.plot_styling import set_plot_style


def testing_single_solver_setting(
    save_folder,
    body_aero_list,
    parameter,
    value_list,
    alpha_range=np.linspace(0, 25, 20),
    alpha_range_distribution=[19, 20, 21, 22, 23],
    beta_range=[0, 3, 6, 9, 12],
    beta_range_distribution=[0, 3, 6],
    Umag=3.15,
    angle_of_attack=6.5,
    side_slip=0,
    yaw_rate=0,
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
            "is_with_gamma_feedback",
            "gamma_initial_distribution_type",
        ]:
            solver_list.append(Solver(**{parameter: value}))
        else:
            raise ValueError(
                f"Parameter {parameter} not recognized. \nPlease choose from ['aerodynamic_model_type', 'max_iterations', 'allowed_error', 'relaxation_factor', 'core_radius_fraction', 'gamma_loop_type', 'is_with_gamma_feedback', 'gamma_initial_distribution_type']"
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
    sensitivity_results_dir,
    body_aero_uniform,
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
        save_dir = Path(sensitivity_results_dir) / parameter
        save_dir.mkdir(parents=True, exist_ok=True)
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
        )


def testing_n_panels_effect(
    sensitivity_results_dir,
    file_path,
    polar_data_dir,
    n_panels_list,
    solver=None,
    alpha_range=np.linspace(0, 25, 20),
    alpha_range_distribution=[19, 20, 21, 22, 23],
    beta_range=[0, 3, 6, 9, 12],
    beta_range_distribution=[0, 3, 6],
    Umag=3.15,
    angle_of_attack=6.5,
    side_slip=0,
    yaw_rate=0,
):
    """
    Test the effects of different n_panels values on aerodynamic performance.

    Parameters:
    -----------
    save_folder : Path or str
        Directory to save the results
    file_path : Path or str
        Path to the wing geometry file
    polar_data_dir : Path or str
        Directory containing polar data files
    n_panels_list : list
        List of n_panels values to test
    solver : Solver, optional
        Solver instance to use. If None, a default solver will be created
    alpha_range : array-like, default=np.linspace(0, 25, 20)
        Range of angles of attack to test
    alpha_range_distribution : list, default=[19, 20, 21, 22, 23]
        Specific angles of attack to use for distribution plots
    beta_range : list, default=[0, 3, 6, 9, 12]
        Range of side slip angles to test
    beta_range_distribution : list, default=[0, 3, 6]
        Specific side slip angles to use for distribution plots
    Umag : float, default=3.15
        Magnitude of the freestream velocity
    angle_of_attack : float, default=6.5
        Default angle of attack
    side_slip : float, default=0
        Default side slip angle
    yaw_rate : float, default=0
        Default yaw rate
    """

    save_folder = Path(Path(sensitivity_results_dir) / "n_panels")
    save_folder.mkdir(parents=True, exist_ok=True)

    # Create default solver if not provided
    if solver is None:
        solver = Solver()

    # Process n_panels parameter
    body_aero_list = []
    label_list = []
    y_coords_list = []

    for n_panels in n_panels_list:
        wing = Wing(n_panels=n_panels, spanwise_panel_distribution="uniform")
        body_aero = BodyAerodynamics.from_file(
            wing,
            file_path=file_path,
            is_with_corrected_polar=True,
            path_polar_data_dir=polar_data_dir,
        )

        body_aero_list.append(body_aero)
        label_list.append(f"n_panels = {n_panels}")
        y_coords_list.append([panel.control_point[1] for panel in body_aero.panels])

    # Create a list of solvers (same solver for each wing)
    solver_list = [solver] * len(n_panels_list)

    # Plotting alpha-polar
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
        title="alphasweep_n_panels",
        data_type=".pdf",
        save_path=save_folder,
        is_save=True,
        is_show=False,
    )

    # Plotting beta-polar
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
        title="betasweep_n_panels",
        data_type=".pdf",
        save_path=save_folder,
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

                body_aero.va_initialize(Umag, alpha, side_slip, yaw_rate)
                begin_time = time.time()
                results_list.append(solver.solve(body_aero, gamma_distribution=None))
                run_time_list.append(time.time() - begin_time)

            plot_distribution(
                y_coordinates_list=y_coords_list,
                results_list=results_list,
                label_list=label_list,
                title=f"spanwise_distribution_n_panels_beta_{side_slip}_alpha_{alpha}",
                data_type=".pdf",
                save_path=save_folder,
                is_save=True,
                is_show=False,
                run_time_list=run_time_list,
            )

    print("\nN_panels effect testing completed.")


def testing_spanwise_distribution_effect(
    sensitivity_results_dir,
    file_path,
    polar_data_dir,
    spanwise_panel_distribution_list,
    n_panels,
    solver=None,
    alpha_range=np.linspace(0, 25, 20),
    alpha_range_distribution=[19, 20, 21, 22, 23],
    beta_range=[0, 3, 6, 9, 12],
    beta_range_distribution=[0, 3, 6],
    Umag=3.15,
    angle_of_attack=6.5,
    side_slip=0,
    yaw_rate=0,
):
    """
    Test the effects of different spanwise panel distributions on aerodynamic performance.

    Parameters:
    -----------
    save_folder : Path or str
        Directory to save the results
    file_path : Path or str
        Path to the wing geometry file
    polar_data_dir : Path or str
        Directory containing polar data files
    spanwise_panel_distribution_list : list
        List of spanwise panel distribution types to test
    n_panels : int, default=50
        Number of panels to use for all distribution tests
    solver : Solver, optional
        Solver instance to use. If None, a default solver will be created
    alpha_range : array-like, default=np.linspace(0, 25, 20)
        Range of angles of attack to test
    alpha_range_distribution : list, default=[19, 20, 21, 22, 23]
        Specific angles of attack to use for distribution plots
    beta_range : list, default=[0, 3, 6, 9, 12]
        Range of side slip angles to test
    beta_range_distribution : list, default=[0, 3, 6]
        Specific side slip angles to use for distribution plots
    Umag : float, default=3.15
        Magnitude of the freestream velocity
    angle_of_attack : float, default=6.5
        Default angle of attack
    side_slip : float, default=0
        Default side slip angle
    yaw_rate : float, default=0
        Default yaw rate
    """
    save_folder = Path(Path(sensitivity_results_dir) / "spanwise_distribution")
    save_folder.mkdir(parents=True, exist_ok=True)

    # Create default solver if not provided
    if solver is None:
        solver = Solver()

    # Process spanwise distribution parameter
    body_aero_list = []
    label_list = []
    y_coords_list = []

    for distribution in spanwise_panel_distribution_list:
        wing = Wing(n_panels=n_panels, spanwise_panel_distribution=distribution)
        body_aero = BodyAerodynamics.from_file(
            wing,
            file_path=file_path,
            is_with_corrected_polar=True,
            path_polar_data_dir=polar_data_dir,
        )

        body_aero_list.append(body_aero)
        label_list.append(f"distribution = {distribution}")
        y_coords_list.append([panel.control_point[1] for panel in body_aero.panels])

    label_list_copy = label_list.copy()

    # Create a list of solvers (same solver for each wing)
    solver_list = [solver] * len(spanwise_panel_distribution_list)

    # Plotting alpha-polar
    plot_polars(
        solver_list=solver_list,
        body_aero_list=body_aero_list,
        label_list=label_list_copy,
        literature_path_list=[],
        angle_range=alpha_range,
        angle_type="angle_of_attack",
        angle_of_attack=angle_of_attack,
        side_slip=side_slip,
        yaw_rate=yaw_rate,
        Umag=Umag,
        title="alphasweep_spanwise_distribution",
        data_type=".pdf",
        save_path=save_folder,
        is_save=True,
        is_show=False,
    )

    # Plotting beta-polar
    plot_polars(
        solver_list=solver_list,
        body_aero_list=body_aero_list,
        label_list=label_list_copy,
        literature_path_list=[],
        angle_range=beta_range,
        angle_type="side_slip",
        angle_of_attack=angle_of_attack,
        side_slip=side_slip,
        yaw_rate=yaw_rate,
        Umag=Umag,
        title="betasweep_spanwise_distribution",
        data_type=".pdf",
        save_path=save_folder,
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
                print(f"\nspanwise_distribution={spanwise_panel_distribution_list[i]}")

                body_aero.va_initialize(Umag, alpha, side_slip, yaw_rate)
                begin_time = time.time()
                results_list.append(solver.solve(body_aero, gamma_distribution=None))
                run_time_list.append(time.time() - begin_time)

            plot_distribution(
                y_coordinates_list=y_coords_list,
                results_list=results_list,
                label_list=label_list_copy,
                title=f"spanwise_distribution_panel_distribution_beta_{side_slip}_alpha_{alpha}",
                data_type=".pdf",
                save_path=save_folder,
                is_save=True,
                is_show=False,
                run_time_list=run_time_list,
            )

    print("\nSpanwise distribution effect testing completed.")


def generate_solver_setting_sensivity_df(
    parameter,
    value_list,
    n_panels,
    Umag=3.15,
    angle_of_attack=6.5,
    side_slip=0,
    yaw_rate=0,
    file_path=None,
    polar_data_dir=None,
    spanwise_panel_distribution="uniform",
):

    # Check required inputs for geometry
    if file_path is None or polar_data_dir is None:
        raise ValueError(
            "For solver settings sensitivity, 'file_path' and 'polar_data_dir' must be provided."
        )

    # Create a uniform BodyAerodynamics instance using a fixed Wing
    wing_instance = Wing(
        n_panels=n_panels, spanwise_panel_distribution=spanwise_panel_distribution
    )
    body_aero_uniform = BodyAerodynamics.from_file(
        wing_instance,
        file_path=file_path,
        is_with_corrected_polar=True,
        path_polar_data_dir=polar_data_dir,
    )
    # Create a list of identical aerodynamic bodies for each test value.
    body_aero_list = [body_aero_uniform] * len(value_list)

    results_list = []
    for value, body_aero in zip(value_list, body_aero_list):
        # Create a solver instance with the given parameter set to the test value.
        solver = Solver(**{parameter: value})
        # Initialize the aerodynamic model with the test conditions.
        body_aero.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)

        # Run the solver and measure runtime.
        start_time = time.time()
        results = solver.solve(body_aero, gamma_distribution=None)
        runtime = time.time() - start_time

        results_list.append(
            {
                "parameter": parameter,
                "value": value,
                "alpha": angle_of_attack,
                "beta": side_slip,
                "cl": results.get("cl", None),
                "cd": results.get("cd", None),
                "cs": results.get("cs", None),
                "cmx": results.get("cmx", None),
                "cmy": results.get("cmy", None),
                "cmz": results.get("cmz", None),
                "runtime": runtime,
            }
        )

    df = pd.DataFrame(results_list)
    return df


def generate_n_panel_sensivitity_df(
    value_list,
    Umag=3.15,
    angle_of_attack=6.5,
    side_slip=0,
    yaw_rate=0,
    file_path=None,
    polar_data_dir=None,
    spanwise_panel_distribution="uniform",
    solver_instance=None,
):
    if file_path is None or polar_data_dir is None:
        raise ValueError(
            "For n_panels sensitivity, 'file_path' and 'polar_data_dir' must be provided."
        )
    if solver_instance is None:
        solver_instance = Solver()

    results_list = []
    for value in value_list:
        # Create a Wing instance for the current n_panels value.
        wing = Wing(
            n_panels=value, spanwise_panel_distribution=spanwise_panel_distribution
        )
        body_aero = BodyAerodynamics.from_file(
            wing,
            file_path=file_path,
            is_with_corrected_polar=True,
            path_polar_data_dir=polar_data_dir,
        )
        # Initialize the aerodynamic model with the test conditions.
        body_aero.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)

        # Run the solver and measure runtime.
        start_time = time.time()
        results = solver_instance.solve(body_aero, gamma_distribution=None)
        runtime = time.time() - start_time

        results_list.append(
            {
                "parameter": "n_panels",
                "value": value,
                "alpha": angle_of_attack,
                "beta": side_slip,
                "cl": results.get("cl", None),
                "cd": results.get("cd", None),
                "cs": results.get("cs", None),
                "cmx": results.get("cmx", None),
                "cmy": results.get("cmy", None),
                "cmz": results.get("cmz", None),
                "runtime": runtime,
            }
        )

    df = pd.DataFrame(results_list)
    return df


def generate_sensitivity_csv(
    file_path,
    polar_data_dir,
    save_csv_path,
    parameter_list,
    allowed_error_list=None,
    core_radius_fraction_list=None,
    relaxation_factor_list=None,
    n_panels_list=None,
    spanwise_panel_distribution="uniform",
    n_panels=50,
    Umag=3.15,
    angle_of_attack=6.5,
    side_slip=0,
    yaw_rate=0,
    solver_instance=None,
):

    results_list = []

    for parameter in parameter_list:
        # Get the corresponding value list
        value_list = locals().get(f"{parameter}_list")
        if value_list is None:
            raise ValueError(f"Value list for parameter '{parameter}' is required.")

        if parameter in ["allowed_error", "core_radius_fraction", "relaxation_factor"]:
            # we are dealing with a solver setting
            df = generate_solver_setting_sensivity_df(
                parameter,
                value_list,
                n_panels,
                Umag=Umag,
                angle_of_attack=angle_of_attack,
                side_slip=side_slip,
                yaw_rate=yaw_rate,
                file_path=file_path,
                polar_data_dir=polar_data_dir,
                spanwise_panel_distribution=spanwise_panel_distribution,
            )

        elif parameter == "n_panels":
            # we are dealing with a wing setting
            df = generate_n_panel_sensivitity_df(
                value_list,
                Umag=Umag,
                angle_of_attack=angle_of_attack,
                side_slip=side_slip,
                yaw_rate=yaw_rate,
                file_path=file_path,
                polar_data_dir=polar_data_dir,
                spanwise_panel_distribution=spanwise_panel_distribution,
                solver_instance=solver_instance,
            )
        else:
            raise ValueError(f"Unknown parameter '{parameter}'")
        results_list.append(df)

    # concatenate the results
    results_df = pd.concat(results_list, ignore_index=True)
    results_df.to_csv(save_csv_path, index=False)
    print(f"CSV file saved to {save_csv_path}")


def plot_param_variation_from_csv_list(csv_paths, labels, save_path):
    """
    Create a subplot grid for convergence/sensitivity plots from a list of CSV files.
    Each CSV file is expected to have the following columns:
      parameter, value, alpha, beta, cl, cd, cmy, runtime.
    It is assumed that each CSV file contains results for one series (with an accompanying label).
    The unique parameter values (from the 'parameter' column) are used to form the rows.
    For each parameter group, curves from all CSV files are plotted on the same axes:
      - Column 0: CL
      - Column 1: CD
      - Column 2: CMy
      - Column 3: runtime
    A legend is added for each row (using the provided labels) without appending runtime info.

    Parameters
    ----------
    csv_paths : list of str or Path
        List of paths to CSV files.
    labels : list of str
        List of labels corresponding to each CSV file.
    save_path : str or Path
        Path to save the resulting figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.
    axes : 2D array of Axes
        Array of Axes objects (n_rows x 4).
    """
    set_plot_style()
    # Read CSV files into a list of DataFrames
    dfs = [pd.read_csv(path) for path in csv_paths]

    # Check that each DataFrame has the required columns
    required_columns = {
        "parameter",
        "value",
        "alpha",
        "beta",
        "cl",
        "cd",
        "cmy",
        "runtime",
    }
    for i, df in enumerate(dfs):
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"CSV file {csv_paths[i]} must contain columns: {required_columns}"
            )

    # Assume that all CSVs have the same set of unique parameters; use the first file to define order.
    unique_parameters = sorted(dfs[0]["parameter"].unique())
    n_rows = len(unique_parameters)
    n_cols = 4  # For CL, CD, CMy, and runtime

    # Create subplots; adjust figsize as needed.
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), squeeze=False
    )

    # Loop over each parameter (each row)
    for row_idx, param_name in enumerate(unique_parameters):
        # Prepare a legend for the current row.
        legend_handles = []
        legend_labels = []

        # Extract data from the first CSV to obtain alpha and beta (assumed constant across CSVs)
        group0 = dfs[0][dfs[0]["parameter"] == param_name].sort_values("value")
        if group0.empty:
            continue
        alpha_val = group0["alpha"].iloc[0]
        beta_val = group0["beta"].iloc[0]

        # For each CSV file (each series), plot the data for the current parameter.
        for i, df in enumerate(dfs):
            group = df[df["parameter"] == param_name].sort_values("value")
            if group.empty:
                continue
            x_values = group["value"].values
            cl_values = group["cl"].values
            cd_values = group["cd"].values
            cmy_values = group["cmy"].values
            runtime_values = group["runtime"].values

            # Plot on each column; let matplotlib assign a color.
            (h,) = axes[row_idx, 0].plot(x_values, cl_values, marker="o", linestyle="-")
            axes[row_idx, 1].plot(
                x_values, cd_values, marker="o", linestyle="-", color=h.get_color()
            )
            axes[row_idx, 2].plot(
                x_values, cmy_values, marker="o", linestyle="-", color=h.get_color()
            )
            axes[row_idx, 3].plot(
                x_values, runtime_values, marker="o", linestyle="-", color=h.get_color()
            )

            # Append to legend: use the provided label (runtime info removed)
            legend_handles.append(h)
            legend_labels.append(f"{labels[i]}")

        # Set the y-label for each column.
        y_label_left = (
            f"$\\alpha = {alpha_val}^\\circ$\n$\\beta = {beta_val}^\\circ$\n\n$C_L$"
        )
        axes[row_idx, 0].set_ylabel(y_label_left)
        axes[row_idx, 1].set_ylabel("$C_D$")
        axes[row_idx, 2].set_ylabel("$C_{M,y}$")
        axes[row_idx, 3].set_ylabel("t [s]")

        # Set the common x-axis label for each subplot in this row.
        for col in range(n_cols):
            axes[row_idx, col].set_xlabel(param_name)

        # Add legend to the first subplot in this row.
        axes[row_idx, 2].legend(legend_handles, legend_labels, loc="best")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")
    return fig, axes
