import numpy as np
import time as time
from pathlib import Path
from VSM.core.WingGeometry import Wing
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.plotting import plot_polars, plot_distribution
from VSM.plot_styling import set_plot_style
import pandas as pd
import matplotlib.pyplot as plt


def generate_n_panel_sensitivity_df(
    n_panels_list,
    Umag,
    angle_of_attack,
    side_slip,
    yaw_rate,
    pitch_rate,
    roll_rate,
    config_path,
    spanwise_panel_distribution,
    solver_instance,
):

    if solver_instance is None:
        solver_instance = Solver()

    results_list = []
    for n_panels in n_panels_list:
        # Use the new instantiate method
        body_aero = BodyAerodynamics.instantiate(
            n_panels=n_panels,
            file_path=config_path,
            spanwise_panel_distribution=spanwise_panel_distribution,
            is_with_bridles=False,
        )
        # Initialize the aerodynamic model with the test conditions.
        body_aero.va_initialize(
            Umag,
            angle_of_attack,
            side_slip,
            yaw_rate,
            pitch_rate,
            roll_rate,
        )

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
                "pitch_rate": pitch_rate,
                "roll_rate": roll_rate,
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
    config_path,
    convergence_analysis_dir,
    spanwise_panel_distribution,
    Umag,
    angle_of_attack,
    side_slip,
    yaw_rate,
    pitch_rate,
    roll_rate,
    n_panels_list,
    aerodynamic_model_type_list=["VSM"],
    allowed_error_list=None,
    core_radius_fraction_list=None,
    gamma_initial_distribution_type_list=None,
    gamma_loop_type_list=None,
    max_iterations_list=None,
    relaxation_factor_list=None,
    spanwise_panel_distribution_list=None,
):
    """
    Generate CSV files for convergence analysis of the aerodynamic solver.

    This function performs a sensitivity analysis by varying both the number of panels and several solver parameters.
    It runs the aerodynamic solver for each combination of a particular parameter value and number of panels, collects
    the resulting aerodynamic coefficients and runtime data, and saves the results as CSV files. This allows the user
    to investigate how convergence is affected by changes in model settings and discretization resolution.

    Args:
        convergence_analysis_dir (str): Directory to save the CSV files for convergence analysis.
        config_path (str): Path to the geometry file defining the body.
        spanwise_panel_distribution (str): Type of panel distribution along the span.
        Umag (float): Magnitude of the freestream velocity (m/s).
        angle_of_attack (float): Angle of attack (in degrees).
        side_slip (float): Side slip angle (in degrees).
        yaw_rate (float): Yaw rate for the body (in appropriate units, e.g., rad/s).
        pitch_rate (float): Pitch rate for the body (in appropriate units, e.g., rad/s).
        roll_rate (float): Roll rate for the body (in appropriate units, e.g., rad/s).
        n_panels_list (list): List of numbers of panels to test in the convergence analysis.
        aerodynamic_model_type_list (list): List of aerodynamic model types (e.g., ["VSM", "LLT"]).
        allowed_error_list (list): List of allowed normalized errors for convergence (e.g., [1e-6]).
        core_radius_fraction_list (list): List of core radius fractions (e.g., [1e-20]).
        gamma_initial_distribution_type_list (list): List of initial gamma distribution types (e.g., ["elliptical", "zero", "previous"]).
        gamma_loop_type_list (list): List of gamma loop types (e.g., ["base", "non_linear"]).
        max_iterations_list (list): List of maximum iteration counts (e.g., [5000]).
        relaxation_factor_list (list): List of solver relaxation factors (e.g., [0.01]).
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
            spanwise_panel_distribution_i = spanwise_panel_distribution
            solver_instance = None
            if parameter == "spanwise_panel_distribution":
                spanwise_panel_distribution_i = value
            else:
                solver_instance = Solver(**{parameter: value})

            df = generate_n_panel_sensitivity_df(
                n_panels_list,
                Umag,
                angle_of_attack,
                side_slip,
                yaw_rate,
                pitch_rate,
                roll_rate,
                config_path,
                spanwise_panel_distribution_i,
                solver_instance,
            )
            # Save the DataFrame to a CSV file
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
        yaw_rate = df["yaw_rate"].iloc[0] if "yaw_rate" in df.columns else 0.0
        pitch_rate = df["pitch_rate"].iloc[0] if "pitch_rate" in df.columns else 0.0
        roll_rate = df["roll_rate"].iloc[0] if "roll_rate" in df.columns else 0.0

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
        (
            f"va_mag = {va_mag} m/s, $\\alpha = {alpha_val}^\\circ$, "
            f"$\\beta = {beta_val}^\\circ$, yaw = {yaw_rate} rad/s, "
            f"pitch = {pitch_rate} rad/s, roll = {roll_rate} rad/s"
        ),
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
    print(f"\n--> Figure saved to {save_path}")
