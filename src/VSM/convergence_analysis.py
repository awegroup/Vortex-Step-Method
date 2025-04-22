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


def generate_n_panel_sensivitity_df(
    n_panels_list,
    Umag,
    angle_of_attack,
    side_slip,
    yaw_rate,
    geometry_path,
    is_with_corrected_polar,
    polar_data_dir,
    spanwise_panel_distribution,
    solver_instance,
):
    if geometry_path is None or polar_data_dir is None:
        raise ValueError(
            "For n_panels sensitivity, 'geometry_path' and 'polar_data_dir' must be provided."
        )
    if solver_instance is None:
        solver_instance = Solver()

    results_list = []
    for n_panels in n_panels_list:
        body_aero = BodyAerodynamics.from_file(
            n_panels=n_panels,
            spanwise_panel_distribution=spanwise_panel_distribution,
            file_path=geometry_path,
            is_with_corrected_polar=is_with_corrected_polar,
            polar_data_dir=polar_data_dir,
            is_half_wing=True,
        )
        # Initialize the aerodynamic model with the test conditions.
        body_aero.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)

        # Run the solver and measure runtime.
        start_time = time.time()
        results = solver_instance.solve(body_aero)
        runtime = time.time() - start_time

        results_list.append(
            {
                "n_panels": n_panels,
                "va_mag": Umag,
                "alpha": angle_of_attack,
                "beta": side_slip,
                "yaw_rate": yaw_rate,
                "cl": results.get("cl", None),
                "cd": results.get("cd", None),
                "cs": results.get("cs", None),
                "cmx": results.get("cmx", None),
                "cmy": results.get("cmy", None),
                "cmz": results.get("cmz", None),
                "runtime": runtime,
            }
        )

    return pd.DataFrame(results_list)


def generate_csv_files(
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
    aerodynamic_model_type_list=["VSM"],
    allowed_error_list=None,
    core_radius_fraction_list=None,
    gamma_initial_distribution_type_list=None,
    gamma_loop_type_list=None,
    max_iterations_list=None,
    relaxation_factor_list=None,
    is_with_corrected_polar_list=None,
    polar_data_dir_list=None,
    spanwise_panel_distribution_list=None,
):
    """
    Generate CSV files for convergence analysis of the aerodynamic solver.

    This function performs a sensitivity analysis by varying both the number of panels and several solver parameters.
    It runs the aerodynamic solver for each combination of a particular parameter value and number of panels, collects
    the resulting aerodynamic coefficients and runtime data, and saves the results as CSV files. This allows the user
    to investigate how convergence is affected by changes in model settings and discretization resolution.

    Possible options for solver parameters include:
        - aerodynamic_model_type (default: "VSM"): e.g., "VSM", "LLT", indicating the aerodynamic model to use.
        - allowed_error (default: 1e-6): e.g., 1e-6, representing the normalized error tolerance for convergence.
        - core_radius_fraction (default: 1e-20): e.g., 1e-20, a small fraction to prevent singularities in vortex core calculations.
        - gamma_initial_distribution_type (default: "elliptical"): e.g., "previous", "elliptical", "cosine", "zero",
          which sets the initial distribution of circulation.
        - gamma_loop_type (default: "base"): e.g., "base", "non_linear", or other methods for stall-related adjustments.
        - max_iterations (default: 5000): e.g., 5000, the maximum number of iterations allowed for solver convergence.
        - relaxation_factor (default: 0.01): e.g., 0.01, the factor by which the solution is relaxed in each iteration.

    Args:
        convergence_analysis_dir (str): Directory to save the CSV files for convergence analysis.
        geometry_path (str): Path to the geometry file defining the body.
        is_with_corrected_polar (bool): Flag to indicate whether to use corrected polar data.
        polar_data_dir (str): Directory containing polar data files.
        spanwise_panel_distribution (str): Type of panel distribution along the span.
        Umag (float): Magnitude of the freestream velocity (m/s).
        angle_of_attack (float): Angle of attack (in degrees).
        side_slip (float): Side slip angle (in degrees).
        yaw_rate (float): Yaw rate for the body (in appropriate units, e.g., rad/s).
        n_panels_list (list): List of numbers of panels to test in the convergence analysis.
        aerodynamic_model_type_list (list): List of aerodynamic model types (e.g., ["VSM", "LLT"]).
        allowed_error_list (list): List of allowed normalized errors for convergence (e.g., [1e-6]).
        core_radius_fraction_list (list): List of core radius fractions (e.g., [1e-20]).
        gamma_initial_distribution_type_list (list): List of initial gamma distribution types (e.g., ["elliptical", "zero", "previous"]).
        gamma_loop_type_list (list): List of gamma loop types (e.g., ["base", "non_linear"]).
        max_iterations_list (list): List of maximum iteration counts (e.g., [5000]).
        relaxation_factor_list (list): List of solver relaxation factors (e.g., [0.01]).
        is_with_corrected_polar_list (list): List of booleans for the corrected polar flag (e.g., [True, False]).
        polar_data_dir_list (list): List of directories for alternate polar data files.
        spanwise_panel_distribution_list (list): List of panel distribution types (e.g., ["uniform", "cosine", "split_provided","unchanged"]).

    Returns:
        Path: The directory where the convergence analysis CSV files have been saved.
    """

    parameter_list = [
        "aerodynamic_model_type",
        "allowed_error",
        "core_radius_fraction",
        "gamma_initial_distribution_type",
        "gamma_loop_type",
        "max_iterations",
        "relaxation_factor",
        "is_with_corrected_polar",
        "polar_data_dir",
        "spanwise_panel_distribution",
    ]
    value_list_list = [
        aerodynamic_model_type_list,
        allowed_error_list,
        core_radius_fraction_list,
        gamma_initial_distribution_type_list,
        gamma_loop_type_list,
        max_iterations_list,
        relaxation_factor_list,
        is_with_corrected_polar_list,
        polar_data_dir_list,
        spanwise_panel_distribution_list,
    ]
    # Write a custom dir for the parameter setup
    dir_name = ""
    for parameter, value_list in zip(parameter_list, value_list_list):
        if value_list is None:
            continue
        if len(dir_name) > 1:
            dir_name += "_"
        dir_name += f"{parameter}_(" + "_".join([str(v) for v in value_list]) + ")"
    convergence_results_dir = Path(convergence_analysis_dir, dir_name)
    # create the directory if it does not exist
    if not convergence_results_dir.exists():
        convergence_results_dir.mkdir(parents=True, exist_ok=True)

    for parameter, value_list in zip(parameter_list, value_list_list):
        if value_list is None:
            continue
        for value in value_list:
            print(f"\nConvergence analysis over n_panels with {parameter} = {value}")
            polar_data_dir_i = polar_data_dir
            spanwise_panel_distribution_i = spanwise_panel_distribution
            is_with_corrected_polar_i = is_with_corrected_polar
            solver_instance = None
            if parameter == "is_with_corrected_polar":
                is_with_corrected_polar_i = value
            elif parameter == "spanwise_panel_distribution":
                spanwise_panel_distribution_i = value
            elif parameter == "polar_data_dir":
                polar_data_dir_i = value
            else:
                solver_instance = Solver(**{parameter: value})

            df = generate_n_panel_sensivitity_df(
                n_panels_list,
                Umag,
                angle_of_attack,
                side_slip,
                yaw_rate,
                geometry_path,
                is_with_corrected_polar_i,
                polar_data_dir_i,
                spanwise_panel_distribution_i,
                solver_instance,
            )
            # concatenate the results
            df.to_csv(
                Path(convergence_results_dir, f"{parameter}_{value}.csv"), index=False
            )
    return convergence_results_dir


def plot_convergence(convergence_results_dir, name, plot_type="pdf"):
    """
    Create convergence plots for aerodynamic coefficients and runtime.

    Parameters:
    -----------
    convergence_results_dir : Path or str
        Directory containing CSV files with convergence data
    name : str
        Base name for the output plot file
    plot_type : str, optional
        Output plot file format (default: 'pdf')

    Returns:
    --------
    tuple
        Figure and axes objects from matplotlib
    """
    # Ensure the input is a Path object
    convergence_results_dir = Path(convergence_results_dir)

    # Set plot style (assuming this function exists in your plotting utilities)
    set_plot_style()

    # Read CSV files into a list of DataFrames
    dfs = []
    labels = []
    for file in convergence_results_dir.iterdir():
        if file.suffix == ".csv":
            dfs.append(pd.read_csv(file))
            labels.append(file.stem)

    # Define the metrics to plot
    metrics_rows = [
        [("cl", "$C_L$"), ("cd", "$C_D$"), ("cs", "$C_S$")],
        [("cmx", "$C_{M,x}$"), ("cmy", "$C_{M,y}$"), ("cmz", "$C_{M,z}$")],
        [("runtime", "t [s]")],
    ]

    # Create subplots
    fig, axes = plt.subplots(
        3, 3, figsize=(18, 12), gridspec_kw={"width_ratios": [1, 1, 1]}
    )

    # Prepare legend handles and labels
    legend_handles = []
    legend_labels = []

    # For each dataset
    for i, df in enumerate(dfs):
        # Get metadata values (assuming constant for this dataset)
        va_mag = df["va_mag"].iloc[0]
        alpha_val = df["alpha"].iloc[0]
        beta_val = df["beta"].iloc[0]
        yaw_rate = df["yaw_rate"].iloc[0]

        # Plot metrics for each row
        for row_idx, row_metrics in enumerate(metrics_rows):
            for col_idx, (metric, label) in enumerate(row_metrics):
                # Plot the data
                (h,) = axes[row_idx, col_idx].plot(
                    df["n_panels"],
                    df[metric],
                    marker="o",
                    linestyle="-",
                    label=labels[i],
                )

                # Set labels
                axes[row_idx, col_idx].set_xlabel("$n_{panels}$")
                axes[row_idx, col_idx].set_ylabel(label)

                # Store handles for legend (only for the first dataset)
                if row_idx == 0 and col_idx == 0:
                    legend_handles.append(h)
                    legend_labels.append(labels[i])

    # Add overall title with metadata info
    fig.suptitle(
        f"va_mag = {va_mag} m/s, $\\alpha = {alpha_val}^\\circ$, $\\beta = {beta_val}^\\circ$, yaw_rate = {yaw_rate} rad/s",
        fontsize=15,
        y=0.98,
    )

    # Remove the empty subplot in the runtime row
    axes[2, 1].axis("off")
    axes[2, 2].axis("off")

    # Add legend to the runtime row
    fig.legend(
        legend_handles,
        legend_labels,
        # loc="lower right",
        bbox_to_anchor=(0.9, 0.32),
        ncols=2,
        # fontsize=13,
    )

    plt.tight_layout()

    # Adjust layout to make room for legend and suptitle
    plt.subplots_adjust(top=0.9, bottom=0.1)

    # Save the figure
    save_path = Path(convergence_results_dir, f"{name}.{plot_type}")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Figure saved to {save_path}")
