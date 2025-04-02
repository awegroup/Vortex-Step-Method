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
        # Create a Wing instance for the current n_panels value.
        wing = Wing(
            n_panels=n_panels, spanwise_panel_distribution=spanwise_panel_distribution
        )
        body_aero = BodyAerodynamics.from_file(
            wing,
            file_path=geometry_path,
            is_with_corrected_polar=is_with_corrected_polar,
            polar_data_dir=polar_data_dir,
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
    convergence_results_dir,
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
    if not convergence_results_dir.exists():
        convergence_results_dir.mkdir(parents=True, exist_ok=True)

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


# def testing_spanwise_distribution_effect(
#     sensitivity_results_dir,
#     file_path,
#     polar_data_dir,
#     spanwise_panel_distribution_list,
#     n_panels,
#     solver=None,
#     alpha_range=np.linspace(0, 25, 20),
#     alpha_range_distribution=[19, 20, 21, 22, 23],
#     beta_range=[0, 3, 6, 9, 12],
#     beta_range_distribution=[0, 3, 6],
#     Umag=3.15,
#     angle_of_attack=6.5,
#     side_slip=0,
#     yaw_rate=0,
# ):
#     """
#     Test the effects of different spanwise panel distributions on aerodynamic performance.

#     Parameters:
#     -----------
#     save_folder : Path or str
#         Directory to save the results
#     file_path : Path or str
#         Path to the wing geometry file
#     polar_data_dir : Path or str
#         Directory containing polar data files
#     spanwise_panel_distribution_list : list
#         List of spanwise panel distribution types to test
#     n_panels : int, default=50
#         Number of panels to use for all distribution tests
#     solver : Solver, optional
#         Solver instance to use. If None, a default solver will be created
#     alpha_range : array-like, default=np.linspace(0, 25, 20)
#         Range of angles of attack to test
#     alpha_range_distribution : list, default=[19, 20, 21, 22, 23]
#         Specific angles of attack to use for distribution plots
#     beta_range : list, default=[0, 3, 6, 9, 12]
#         Range of side slip angles to test
#     beta_range_distribution : list, default=[0, 3, 6]
#         Specific side slip angles to use for distribution plots
#     Umag : float, default=3.15
#         Magnitude of the freestream velocity
#     angle_of_attack : float, default=6.5
#         Default angle of attack
#     side_slip : float, default=0
#         Default side slip angle
#     yaw_rate : float, default=0
#         Default yaw rate
#     """
#     save_folder = Path(Path(sensitivity_results_dir) / "spanwise_distribution")
#     save_folder.mkdir(parents=True, exist_ok=True)

#     # Create default solver if not provided
#     if solver is None:
#         solver = Solver()

#     # Process spanwise distribution parameter
#     body_aero_list = []
#     label_list = []
#     y_coords_list = []

#     for distribution in spanwise_panel_distribution_list:
#         wing = Wing(n_panels=n_panels, spanwise_panel_distribution=distribution)
#         body_aero = BodyAerodynamics.from_file(
#             wing,
#             file_path=file_path,
#             is_with_corrected_polar=True,
#             path_polar_data_dir=polar_data_dir,
#         )

#         body_aero_list.append(body_aero)
#         label_list.append(f"distribution = {distribution}")
#         y_coords_list.append([panel.control_point[1] for panel in body_aero.panels])

#     label_list_copy = label_list.copy()

#     # Create a list of solvers (same solver for each wing)
#     solver_list = [solver] * len(spanwise_panel_distribution_list)

#     # Plotting alpha-polar
#     plot_polars(
#         solver_list=solver_list,
#         body_aero_list=body_aero_list,
#         label_list=label_list_copy,
#         literature_path_list=[],
#         angle_range=alpha_range,
#         angle_type="angle_of_attack",
#         angle_of_attack=angle_of_attack,
#         side_slip=side_slip,
#         yaw_rate=yaw_rate,
#         Umag=Umag,
#         title="alphasweep_spanwise_distribution",
#         data_type=".pdf",
#         save_path=save_folder,
#         is_save=True,
#         is_show=False,
#     )

#     # Plotting beta-polar
#     plot_polars(
#         solver_list=solver_list,
#         body_aero_list=body_aero_list,
#         label_list=label_list_copy,
#         literature_path_list=[],
#         angle_range=beta_range,
#         angle_type="side_slip",
#         angle_of_attack=angle_of_attack,
#         side_slip=side_slip,
#         yaw_rate=yaw_rate,
#         Umag=Umag,
#         title="betasweep_spanwise_distribution",
#         data_type=".pdf",
#         save_path=save_folder,
#         is_save=True,
#         is_show=False,
#     )

#     # Plotting distributions
#     for side_slip in beta_range_distribution:
#         for alpha in alpha_range_distribution:
#             print(f"\nalpha: {alpha}")
#             results_list = []
#             run_time_list = []

#             for i, body_aero in enumerate(body_aero_list):
#                 print(f"\nspanwise_distribution={spanwise_panel_distribution_list[i]}")

#                 body_aero.va_initialize(Umag, alpha, side_slip, yaw_rate)
#                 begin_time = time.time()
#                 results_list.append(solver.solve(body_aero, gamma_distribution=None))
#                 run_time_list.append(time.time() - begin_time)

#             plot_distribution(
#                 y_coordinates_list=y_coords_list,
#                 results_list=results_list,
#                 label_list=label_list_copy,
#                 title=f"spanwise_distribution_panel_distribution_beta_{side_slip}_alpha_{alpha}",
#                 data_type=".pdf",
#                 save_path=save_folder,
#                 is_save=True,
#                 is_show=False,
#                 run_time_list=run_time_list,
#             )

#     print("\nSpanwise distribution effect testing completed.")
