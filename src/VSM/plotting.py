import os
import sys
import logging
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time as time
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_pdf import PdfPages
from screeninfo import get_monitors
from matplotlib.lines import Line2D
from VSM.plot_styling import set_plot_style, plot_on_ax


def save_plot(fig, save_path, title, data_type=".pdf"):
    """
    Save a matplotlib figure to a file.

    Args:
        fig: matplotlib figure object
        save_path: path to save the plot
        title: title of the plot
        data_type: type of the data to be saved | default: ".pdf"

    Returns:
        None
    """

    if save_path is None:
        raise ValueError("save_path should be provided")
    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        full_path = Path(save_path) / (title + data_type)
        logging.debug(f"Attempting to save figure to: {full_path}")
        logging.debug(f"Current working directory: {os.getcwd()}")
        logging.debug(f"File exists before saving: {os.path.exists(full_path)}")

        try:
            if data_type.lower() == ".pdf":
                with PdfPages(full_path) as pdf:
                    pdf.savefig(fig)
                logging.debug("Figure saved using PdfPages")
            else:
                fig.savefig(full_path)
                logging.debug(f"Figure saved as {data_type}")

            if os.path.exists(full_path):
                logging.debug(f"File successfully saved to {full_path}")
                logging.debug(f"File size: {os.path.getsize(full_path)} bytes")
            else:
                logging.info(f"File does not exist after save attempt: {full_path}")

        except Exception as e:
            logging.info(f"Error saving figure: {str(e)}")
            logging.info(f"Error type: {type(e).__name__}")
            logging.info(f"Error details: {sys.exc_info()}")

        finally:
            logging.debug(
                f"File exists after save attempts: {os.path.exists(full_path)}"
            )
            if os.path.exists(full_path):
                logging.debug(f"Final file size: {os.path.getsize(full_path)} bytes")


def plot_line_segment(ax, segment, color, label, width: float = 3):
    """Plot a line segment in 3D.

    Args:
        ax: matplotlib axis object
        segment: list of two points defining the segment
        color: color of the segment
        label: label of the segment
        width: width of the segment | default: 3

    Returns:
        None
    """

    ax.plot(
        [segment[0][0], segment[1][0]],
        [segment[0][1], segment[1][1]],
        [segment[0][2], segment[1][2]],
        color=color,
        label=label,
        linewidth=width,
    )
    dir = segment[1] - segment[0]
    ax.quiver(
        segment[0][0],
        segment[0][1],
        segment[0][2],
        dir[0],
        dir[1],
        dir[2],
        color=color,
    )


def set_axes_equal(ax):
    """
    Set the axes of a 3D plot to be equal in scale.

    Args:
        ax: matplotlib axis object

    Returns:
        None
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range])

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
    ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
    ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])


def creating_geometry_plot(
    body_aero,
    title,
    view_elevation,
    view_azimuth,
):
    """
    Plots the wing panels and filaments in 3D.

    Args:
        body_aero: WingAerodynamics object
        title: title of the plot
        data_type: type of the data to be saved | default: ".pdf"
        save_path: path to save the plot | default: None
        is_save: boolean to save the plot | default: False
        is_show: boolean to show the plot | default: True
        view_elevation: elevation of the view | default: 15
        view_azimuth: azimuth of the view | default: -120

    Returns:
        None
    """

    # Set the plot style
    set_plot_style()

    # defining variables
    panels = body_aero.panels
    va = body_aero.va

    # Extract corner points, control points, and aerodynamic centers from panels
    corner_points = np.array([panel.corner_points for panel in panels])
    control_points = np.array([panel.control_point for panel in panels])
    aerodynamic_centers = np.array([panel.aerodynamic_center for panel in panels])

    # Create a 3D plot
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    # Plot each panel
    for i, panel in enumerate(panels):
        # Get the corner points of the current panel and close the loop by adding the first point again
        x_corners = np.append(corner_points[i, :, 0], corner_points[i, 0, 0])
        y_corners = np.append(corner_points[i, :, 1], corner_points[i, 0, 1])
        z_corners = np.append(corner_points[i, :, 2], corner_points[i, 0, 2])

        # Plot the panel edges
        ax.plot(
            x_corners,
            y_corners,
            z_corners,
            color="grey",
            label="Panel Edges" if i == 0 else "",
            linewidth=1,
        )

        # Create a list of tuples representing the vertices of the polygon
        verts = [list(zip(x_corners, y_corners, z_corners))]
        poly = Poly3DCollection(verts, color="grey", alpha=0.1)
        ax.add_collection3d(poly)

        # Plot the control point
        ax.scatter(
            control_points[i, 0],
            control_points[i, 1],
            control_points[i, 2],
            color="green",
            label="Control Points" if i == 0 else "",
        )

        # Plot the aerodynamic center
        ax.scatter(
            aerodynamic_centers[i, 0],
            aerodynamic_centers[i, 1],
            aerodynamic_centers[i, 2],
            color="b",
            label="Aerodynamic Centers" if i == 0 else "",
        )

        # Plot the filaments
        filaments = panel.calculate_filaments_for_plotting()
        legends = ["Bound Vortex", "side1", "side2", "wake_1", "wake_2"]

        for filament, legend in zip(filaments, legends):
            x1, x2, color = filament
            logging.debug("Legend: %s", legend)
            plot_line_segment(ax, [x1, x2], color, legend)

    # Plot the va_vector using the plot_segment
    max_chord = np.max([panel.chord for panel in panels])
    va_vector_begin = -2 * max_chord * va / np.linalg.norm(va)
    va_vector_end = va_vector_begin + 1.5 * va / np.linalg.norm(va)
    plot_line_segment(ax, [va_vector_begin, va_vector_end], "lightblue", "va")

    # Add legends for the first occurrence of each label
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Add axis labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Set equal axis limits
    set_axes_equal(ax)

    # Flip the z-axis (to stick to body reference frame)
    # ax.invert_zaxis()

    # Set the initial view
    ax.view_init(elev=view_elevation, azim=view_azimuth)

    # Ensure the figure is fully rendered
    fig.canvas.draw()

    return fig,ax


def plot_geometry(
    body_aero,
    title,
    data_type,
    save_path,
    is_save=False,
    is_show=False,
    view_elevation=15,
    view_azimuth=-120,
):
    """plot_geometry function

    Plots the wing panels and filaments in 3D.

    Args:
        body_aero: WingAerodynamics object
        title: title of the plot
        data_type: type of the data to be saved | default: ".pdf"
        save_path: path to save the plot | default: None
        is_save: boolean to save the plot | default: False
        is_show: boolean to show the plot | default: True
        view_elevation: elevation of the view | default: 15
        view_azimuth: azimuth of the view | default: -120

    Returns:
        None
    """
    # saving plot
    if is_save:
        # plot angled view
        fig = creating_geometry_plot(
            body_aero,
            title=title + "_angled_view",
            view_elevation=15,
            view_azimuth=-120,
        )
        save_plot(fig, save_path, title + "_angled_view", data_type)
        plt.close()
        # plot top view
        fig = creating_geometry_plot(
            body_aero,
            title=title + "_top_view",
            view_elevation=90,
            view_azimuth=0,
        )

        save_plot(fig, save_path, title + "_top_view", data_type)
        plt.close()
        # plot front view
        fig = creating_geometry_plot(
            body_aero,
            title=title + "_front_view",
            view_elevation=0,
            view_azimuth=0,
        )
        save_plot(fig, save_path, title + "_front_view", data_type)
        plt.close()
        # save side view
        fig = creating_geometry_plot(
            body_aero,
            title=title + "_side_view",
            view_elevation=0,
            view_azimuth=-90,
        )
        save_plot(fig, save_path, title + "_side_view", data_type)
        plt.close()

    # showing plot
    if is_show and not is_save:
        fig = creating_geometry_plot(
            body_aero,
            title=title,
            view_elevation=15,
            view_azimuth=-120,
        )
        plt.show()
    elif is_show and is_save:
        raise ValueError(
            "is_show and is_save are both True. Please set one of them to False."
        )


def plot_distribution(
    y_coordinates_list,
    results_list,
    label_list,
    title="spanwise_distribution",
    data_type=".pdf",
    save_path=None,
    is_save=True,
    is_show=True,
    run_time_list=None,
):
    """
    Plots a 3x3 spanwise distribution:
        Row 1: CD, CL, CS
        Row 2: CMx, CMy, CMz (non-dimensional moments)
        Row 3: alpha (geometric), alpha (corrected), alpha (uncorrected)

    Args:
        y_coordinates_list (list): list of y-coordinates (arrays or lists)
        results_list (list): list of dictionaries with keys like:
            "cd_distribution", "cl_distribution", "cs_distribution",
            "cmx_distribution", "cmy_distribution", "cmz_distribution",
            "alpha_geometric", "alpha_at_ac", "alpha_uncorrected",
            and total values "cd", "cl", "cs", "cmx", "cmy", "cmz"
        label_list (list): corresponding labels for each set of results
        title (str): plot title
        data_type (str): file extension for the saved plot, e.g. ".pdf"
        save_path (str): path to folder where plot will be saved
        is_save (bool): whether to save the plot
        is_show (bool): whether to show the plot

    Returns:
        None
    """
    set_plot_style()

    if len(y_coordinates_list) != len(results_list) != len(label_list):
        raise ValueError(
            f"The number of y_coordinates = number of results = number of labels. "
            f"Got {len(y_coordinates_list)} y_coordinates, {len(results_list)} results, {len(label_list)} labels."
        )
    # Create figure and axes: 3 rows x 3 columns
    fig, axs = plt.subplots(3, 3, figsize=(27, 15))
    # fig.suptitle(title)

    # --- Row 1: CD, CL, CS ---------------------------------------------------
    # Column 1: CD
    for y_coords, result, label in zip(y_coordinates_list, results_list, label_list):
        axs[0, 0].plot(
            y_coords,
            result["cd_distribution"],
            label=rf"$C_D$: {result['cd']:.2f}",
        )
    axs[0, 0].set_ylabel(r"$C_D$ Distribution")
    axs[0, 0].tick_params(labelbottom=False)
    axs[0, 0].legend()

    # Column 2: CL
    for y_coords, result, label in zip(y_coordinates_list, results_list, label_list):
        axs[0, 1].plot(
            y_coords,
            result["cl_distribution"],
            label=rf"$C_L$: {result['cl']:.2f}",
        )
    axs[0, 1].set_ylabel(r"$C_L$ Distribution")
    axs[0, 1].tick_params(labelbottom=False)
    axs[0, 1].legend()

    # Column 3: CS (side force coefficient)
    for y_coords, result, label in zip(y_coordinates_list, results_list, label_list):
        axs[0, 2].plot(
            y_coords,
            result["cs_distribution"],
            label=rf"$C_S$: {result['cs']:.2f}",
        )
    axs[0, 2].set_ylabel(r"$C_S$ Distribution")
    axs[0, 2].tick_params(labelbottom=False)
    axs[0, 2].legend()

    # --- Row 2: CMx, CMy, CMz ------------------------------------------------
    # Column 1: CMx
    for y_coords, result, label in zip(y_coordinates_list, results_list, label_list):
        axs[1, 0].plot(
            y_coords,
            result["cmx_distribution"],
            label=rf"$C_{{mx}}$: {result['cmx']:.2f}",
        )
    axs[1, 0].set_ylabel(r"$C_{mx}$ Distribution")
    axs[1, 0].tick_params(labelbottom=False)
    axs[1, 0].legend()

    # Column 2: CMy
    for y_coords, result, label in zip(y_coordinates_list, results_list, label_list):
        axs[1, 1].plot(
            y_coords,
            result["cmy_distribution"],
            label=rf"$C_{{my}}$: {result['cmy']:.2f}",
        )
    axs[1, 1].set_ylabel(r"$C_{my}$ Distribution")
    axs[1, 1].tick_params(labelbottom=False)
    axs[1, 1].legend()

    # Column 3: CMz
    for y_coords, result, label in zip(y_coordinates_list, results_list, label_list):
        axs[1, 2].plot(
            y_coords,
            result["cmz_distribution"],
            label=rf"$C_{{mz}}$: {result['cmz']:.2f}",
        )
    axs[1, 2].set_ylabel(r"$C_{mz}$ Distribution")
    axs[1, 2].tick_params(labelbottom=False)
    axs[1, 2].legend()

    # --- Row 3:  -----------------------------------------
    # Column 1: Gamma
    for y_coords, result, label in zip(y_coordinates_list, results_list, label_list):
        axs[2, 0].plot(
            y_coords,
            np.rad2deg(result["gamma_distribution"]),
            label=label,
        )
    axs[2, 0].set_xlabel(r"Spanwise Position $y/b$")
    axs[2, 0].set_ylabel(r"Gamma")
    # axs[2, 0].legend()

    # Column 2: Gamma distribution
    for y_coords, result, label in zip(y_coordinates_list, results_list, label_list):
        axs[2, 1].plot(
            y_coords,
            np.rad2deg(result["alpha_geometric"]),
            label=label,
        )
    axs[2, 1].set_xlabel(r"Spanwise Position $y/b$")
    axs[2, 1].set_ylabel(r"Geometric $\alpha$ (deg)")
    # axs[2, 1].legend()

    # Column 3: alpha (corrected to aerodynamic center)
    for y_coords, result, label in zip(y_coordinates_list, results_list, label_list):
        axs[2, 2].plot(
            y_coords,
            np.rad2deg(result["alpha_at_ac"]),
            # label=label,
        )
    axs[2, 2].set_xlabel(r"Spanwise Position $y/b$")
    axs[2, 2].set_ylabel(r"Corrected $\alpha$ (deg)")
    # axs[2, 2].legend()

    # --- Adjust y-limits intelligently --------------------------------------
    # We'll define allowed ranges per subplot index:
    #  0 -> CD
    #  1 -> CL
    #  2 -> CS
    #  3 -> CMx
    #  4 -> CMy
    #  5 -> CMz
    #  6 -> Gamma
    #  7 -> alpha corrected
    #  8 -> alpha uncorrected
    allowed_ranges = {
        0: (-1.0, 1.0),  # CD
        1: (-1.5, 2.0),  # CL
        2: (-1.5, 1.5),  # CS
        3: (-1.0, 1.0),  # CMx
        4: (-1.0, 1.0),  # CMy
        5: (-1.0, 1.0),  # CMz
        6: (-200, 500),  # Gamma
        7: (-10, 150),  # alpha geometric
        8: (-10, 90),  # alpha corrected
    }

    for i, ax in enumerate(axs.flat):
        y_min_allowed, y_max_allowed = allowed_ranges.get(i, ax.get_ylim())

        # Collect all y-data from lines in the current axis
        y_data = np.concatenate([line.get_ydata() for line in ax.get_lines()])
        # Identify data within the allowed range
        in_range = y_data[(y_data >= y_min_allowed) & (y_data <= y_max_allowed)]

        if in_range.size > 0:
            padding = 0.05 * (in_range.max() - in_range.min())
            ax.set_ylim(in_range.min() - padding, in_range.max() + padding)
        else:
            # If no data is within the range, fall back to a default
            ax.set_ylim(y_min_allowed, y_max_allowed)

    # Place the legend below the axes
    labels = []
    for i, label in enumerate(results_list):
        label = label_list[i]
        if run_time_list is not None:
            labels.append(label + f" t: {run_time_list[i]:.3f}s")
        else:
            labels.append(label)
    # labels = [label for label in label_list]
    handles = [axs[0, 0].get_lines()[i] for i in range(len(y_coordinates_list))]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=3)

    # Manually increase the bottom margin to make room for the legend
    fig.subplots_adjust(bottom=0.2)

    # Save or show plot
    if is_save:
        save_plot(fig, save_path, title, data_type)

    if is_show and not is_save:
        plt.show()
    elif is_show and is_save:
        raise ValueError(
            "is_show and is_save are both True. Please set one of them to False."
        )


def generate_3D_polar_data(
    solver,
    body_aero,
    angle_range,
    angle_type="angle_of_attack",
    angle_of_attack=0,
    side_slip=0,
    yaw_rate=0,
    Umag=10,
    steering="sideforce",  # sideforce or roll
):
    """
    Generates the polar data for the given solver and body_aero.

    Args:
        solver: solver object
        body_aero: body_aero object
        angle_range: range of angles to be considered
        angle_type: type of the angle | default: "angle_of_attack"
        angle_of_attack: angle of attack | default: 0
        side_slip: side slip angle | default: 0
        yaw_rate: yaw rate | default: 0
        Umag: magnitude of the velocity | default: 10

    Returns:
        list: list of polar data, in the following order:
            [angle_range, cl, cd, cs, gamma_distribution, cl_distribution, cd_distribution, cs_distribution]
    """

    cl = np.zeros(len(angle_range))
    aero_roll = np.zeros(len(angle_range))
    cl_total = np.zeros(len(angle_range))
    cd = np.zeros(len(angle_range))
    cs = np.zeros(len(angle_range))
    cmx = np.zeros(len(angle_range))
    cmy = np.zeros(len(angle_range))
    cmz = np.zeros(len(angle_range))
    gamma_distribution = np.zeros((len(angle_range), len(body_aero.panels)))
    cl_distribution = np.zeros((len(angle_range), len(body_aero.panels)))
    cd_distribution = np.zeros((len(angle_range), len(body_aero.panels)))
    cs_distribution = np.zeros((len(angle_range), len(body_aero.panels)))
    reynolds_number = np.zeros(len(angle_range))
    run_time = np.zeros(len(angle_range))
    # initialize the gamma with None
    gamma = None
    for i, angle_i in enumerate(angle_range):
        if angle_type == "angle_of_attack":
            body_aero.va_initialize(Umag, angle_i, side_slip, yaw_rate)
        elif angle_type == "side_slip":
            body_aero.va_initialize(Umag, angle_of_attack, angle_i, yaw_rate)
        else:
            raise ValueError(
                "angle_type should be either 'angle_of_attack' or 'side_slip'"
            )

        begin_time = time.time()
        results = solver.solve(body_aero, gamma_distribution=gamma)
        run_time[i] = time.time() - begin_time
        print(f"Angle: {angle_i:.1f}deg. Time: {run_time[i]:.4f}s")
        cl[i] = results["cl"]
        cd[i] = results["cd"]
        cs[i] = results["cs"]
        cl_total[i] = np.sqrt(results["cl"]**2+results["cs"]**2)
        aero_roll[i] = np.degrees(np.arctan2(results["cs"], results["cl"]))
        cmx[i] = results["cmx"]
        cmy[i] = results["cmy"]
        cmz[i] = results["cmz"]
        gamma_distribution[i] = results["gamma_distribution"]
        cl_distribution[i] = results["cl_distribution"]
        cd_distribution[i] = results["cd_distribution"]
        cs_distribution[i] = results["cs_distribution"]
        reynolds_number[i] = results["Rey"]
        gamma = gamma_distribution[i]

    polar_data = [
        angle_range,
        cl,
        cd,
        cs,
        cmx,
        cmy,
        cmz,
        gamma_distribution,
        cl_distribution,
        cd_distribution,
        cs_distribution,
        reynolds_number,
        run_time,
    ]
    if steering == "roll":
        polar_data[1] = cl_total
        polar_data[3] = aero_roll
    reynolds_number = results["Rey"]

    return polar_data, reynolds_number

def plot_polars(
    solver_list,
    body_aero_list,
    label_list,
    literature_path_list=None,
    angle_range=np.linspace(0, 20, 2),
    angle_type="angle_of_attack",
    angle_of_attack=0,
    side_slip=0,
    yaw_rate=0,
    Umag=10,
    title="polar",
    data_type=".pdf",
    steering = "sideforce",     # sideforce or roll
    save_path=None,
    is_save=True,
    is_show=True,
):
    """
    Plot polar data CL, CD, CS, and CL-CD over a specified angle for the given solvers and body_aeros.

    Args:
        solver_list: list of solver objects
        body_aero_list: list of body_aero objects
        label_list: list of labels for the results
        literature_path_list: list of paths to literature data | default: None
        angle_range: range of angles to be considered | default: np.linspace(0, 20, 2)
        angle_type: type of the angle | default: "angle_of_attack", options:
              - `"angle_of_attack`: will loop over an angle_of_attack range
              - `"side_slip"`: will loop over an side_slip range
        angle_of_attack: angle of attack [deg] | default: 0
        side_slip: side slip angle [deg] | default: 0
        yaw_rate: yaw rate | default: 0
        Umag: magnitude of the velocity | default: 10
        title: title of the plot | default: "polar"
        data_type: type of the data to be saved | default: ".pdf"
        save_path: path to save the plot | default: None
        is_save: boolean to save the plot | default: True
        is_show: boolean to show the plot | default: True

    Returns:
        fig, axs: figure and axes objects for further customization
    """
    # Set the plot style
    set_plot_style()

    # Initialize literature paths if None
    if literature_path_list is None:
        literature_path_list = []

    # Checking type and configuring the x_label
    if angle_type == "angle_of_attack":
        x_label = r"$\alpha$ [°]"
    elif angle_type == "side_slip":
        x_label = r"$\beta$ [°]"
    else:
        raise ValueError("angle_type should be either 'angle_of_attack' or 'side_slip'")

    # Validate input sizes
    total_solver_count = len(solver_list)
    total_data_count = total_solver_count + len(literature_path_list)
    
    if total_data_count != len(label_list) or len(solver_list) != len(body_aero_list):
        raise ValueError(
            f"Input count mismatch: {len(solver_list)} solvers, "
            f"{len(body_aero_list)} body_aero objects, "
            f"{len(literature_path_list)} literature files, "
            f"but {len(label_list)} labels provided. These counts must align."
        )

    # Generating polar data
    polar_data_list = []
    formatted_label_list = label_list.copy()  # Create a copy to preserve original labels
    
    print("\n=== Generating solver data ===")
    for i, (solver, body_aero, label) in enumerate(zip(solver_list, body_aero_list, label_list[:len(solver_list)])):
        print(f"\n=== label: {label} ===")
        polar_data, reynolds_number = generate_3D_polar_data(
            solver=solver,
            body_aero=body_aero,
            angle_range=angle_range,
            angle_type=angle_type,
            angle_of_attack=angle_of_attack,
            side_slip=side_slip,
            yaw_rate=yaw_rate,
            Umag=Umag,
            steering=steering,
        )
        polar_data_list.append(polar_data)
        # Append Reynolds number to the label
        formatted_label_list[i] = f"{label} (Re = {1e-5*reynolds_number:.1f}e5)"

    # Adding literature data to the polar_data_list
    for i, literature_path in enumerate(literature_path_list):
        df = pd.read_csv(literature_path)
        polar_data = [None, None, None, None, None, None, None]
        if angle_type == "angle_of_attack":
            polar_data[0] = df["alpha"].values
        elif angle_type == "side_slip":
            polar_data[0] = df["beta"].values
        if "CL" in df.columns:
            if steering == "sideforce":
                polar_data[1] = df["CL"].values
            elif steering == "roll":
                polar_data[1] = np.sqrt(df["CL"]**2 + df["CS"]**2)
        if "CD" in df.columns:
            polar_data[2] = df["CD"].values
        if "CS" in df.columns:
            if steering == "sideforce":
                polar_data[3] = df["CS"].values
            elif steering == "roll":
                polar_data[3] = np.degrees(np.arctan2(df["CS"], df["CL"]))
        if "CMx" in df.columns:
            polar_data[4] = df["CMx"].values
        if "CMy" in df.columns:
            polar_data[5] = df["CMy"].values
        if "CMz" in df.columns:
            polar_data[6] = df["CMz"].values
        
        polar_data_list.append(polar_data)

    # Initializing plot
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs_flat = axs.flatten()
    y_label_list = [
        r"$C_{\mathrm{L}}$",
        r"$C_{\mathrm{D}}$",
        r"$C_{\mathrm{L}}/C_{\mathrm{D}}$",
        r"$C_{mx}$ (Roll)",
        r"$C_{my}$ (Pitch)",
        r"$C_{mz}$ (Yaw)",
    ]
    if angle_type == "side_slip":
        if steering == "sideforce":
            y_label_list[2] = r"$C_{\mathrm{S}}$"
        elif steering == "roll":
            y_label_list[2] = r"$\phi_a$"

    # plotting the actyual data
    handle_list = []
    for data_idx, label in enumerate(label_list):
        for ax_idx,(ax,y_label) in enumerate(zip(axs_flat,y_label_list)):
            if polar_data_list[data_idx][ax_idx+1] is None:
                continue
            y_data = polar_data_list[data_idx][ax_idx+1]
            if angle_type == "angle_of_attack" and ax_idx == 2:
                y_data = polar_data_list[data_idx][1] / polar_data_list[data_idx][2]
            ax.plot(
                polar_data_list[data_idx][0],
                y_data,
                label=label,
                linestyle="-",
                marker="*",
                markersize=7,
            )
    # Adding one legend
    handles, labels = axs_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=min(3, len(handles)))
    fig.subplots_adjust(bottom=0.1)
    plt.tight_layout(rect=[0, 0.2, 1, 0.95])
    fig.canvas.draw()

    # Making it pretty
    for ax_idx,(ax,y_label) in enumerate(zip(axs_flat,y_label_list)):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid()

    # saving plot
    if is_save:
        if save_path is None:
            raise ValueError("save_path must be provided if is_save is True.")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Save the plot with the specified title and data type
        plt.savefig(Path(save_path) / f"{title}{data_type}", bbox_inches='tight')
        print(f"Plot saved as: {save_path}")

    # showing plot
    if is_show:
        plt.show()
    elif is_show and is_save:
        raise ValueError(
            "is_show and is_save are both True. Please set one of them to False."
        )
    
    
        
    return fig, axs


def plot_panel_coefficients(
    body_aero,
    panel_index,
    alpha_range=[-20, 30],
    title=None,
    data_type=".pdf",
    save_path=None,
    is_save=False,
    is_show=True,
):
    """
    Plot Cl, Cd, and Cm coefficients for a specific panel across a range of angles of attack.

    Args:
        body_aero (object): Wing aerodynamic object containing panels
        panel_index (int): Index of the panel to plot
        alpha_range (tuple, optional): Range of angles of attack in radians.
                                       Defaults to (-0.5, 0.5) radians.
    """
    set_plot_style()
    if title is None:
        title = f"2D_polar_of_panel_{panel_index}"

    # Select the specified panel
    panel = body_aero.panels[panel_index]

    # Create an array of angles of attack
    alpha_array = np.deg2rad(np.linspace(alpha_range[0], alpha_range[1], 50))

    # Calculate coefficients
    cl_array = np.array([panel.calculate_cl(alpha) for alpha in alpha_array])

    # For Cd and Cm, the method returns a tuple
    cd_array = np.array([panel.calculate_cd_cm(alpha)[0] for alpha in alpha_array])
    cm_array = np.array([panel.calculate_cd_cm(alpha)[1] for alpha in alpha_array])

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    ax1, ax2, ax3 = axs

    # Cl vs Alpha plot
    ax1.plot(
        np.rad2deg(alpha_array), cl_array, label=r"$C_{\mathrm{l}}$", color="black"
    )
    ax1.set_xlabel(r"$\alpha$ [deg] (angle of attack)")
    ax1.set_ylabel(r"$C_{\mathrm{l}}$")
    ax1.grid(True)
    ax1.legend()

    # Cd vs Alpha plot
    ax2.plot(
        np.rad2deg(alpha_array), cd_array, label=r"$C_{\mathrm{d}}$", color="black"
    )
    ax2.set_xlabel(r"$\alpha$ [deg] (angle of attack)")
    ax2.set_ylabel(r"$C_{\mathrm{d}}$")
    ax2.grid(True)
    ax2.legend()
    ax2.set_title(title)

    # Cm vs Alpha plot
    ax3.plot(
        np.rad2deg(alpha_array), cm_array, label=r"$C_{\mathrm{m}}$", color="black"
    )
    ax3.set_xlabel(r"$\alpha$ [deg] (angle of attack)")
    ax3.set_ylabel(r"$C_{\mathrm{m}}$")
    ax3.grid(True)
    ax3.legend()

    # Adjust layout and display
    plt.tight_layout()

    # Ensure the figure is fully rendered
    fig.canvas.draw()

    # saving plot
    if is_save and save_path is not None:
        save_plot(fig, save_path, title, data_type)

    # showing plot
    if is_show and not is_save:
        plt.show()
    elif is_show and is_save:
        raise ValueError(
            "is_show and is_save are both True. Please set one of them to False."
        )


def process_panel_coefficients_panel_i(body_aero, panel_index, PROJECT_DIR, n_panels):
    """
    Plot Cl, Cd, and Cm coefficients for a specific panel across a range of angles of attack.

    Args:
        body_aero (object): Wing aerodynamic object containing panels
        panel_index (int): Index of the panel to plot
        alpha_range (tuple, optional): Range of angles of attack in radians.
                                       Defaults to (-0.5, 0.5) radians.
    """

    def run_neuralfoil(alpha_array_deg, panel_index):
        """
        Run neuralfoil with a specific panel index.

        Parameters:
            PROJECT_DIR (str or Path): The base project directory.
            panel_index (int): An integer between 0 and 34.

        Returns:
            DataFrame: A pandas DataFrame containing alpha, CL, CD, and CM.
        """
        import neuralfoil as nf

        # Validate the panel_index
        if not (0 <= panel_index <= 34):
            raise ValueError("panel_index must be between 0 and 34.")

        # Compute the distance from the symmetry panel (panel 17)
        d = abs(panel_index - 17)

        # Map the distance d to the profile number using a dictionary.
        profile_mapping = {
            0: 1,
            1: 2,
            2: 2,
            3: 3,
            4: 3,
            5: 4,
            6: 4,
            7: 5,
            8: 6,
            9: 7,
            10: 8,
            11: 9,
            12: 9,
            13: 10,
            14: 10,
            15: 11,
            16: 11,
            17: 12,
        }
        profile_num = profile_mapping.get(d)
        if profile_num is None:
            raise ValueError(
                f"Unexpected value for d = {d} from panel_index = {panel_index}."
            )

        # Build the file path based on the profile number.
        dat_file_path = Path(
            PROJECT_DIR,
            "examples",
            "TUDELFT_V3_LEI_KITE",
            "polar_engineering",
            "profiles",
            "surfplan",
            f"prof_{profile_num}.dat",
        )

        # Define the operating parameters.
        Re = 5.6e5
        model_size = "xxlarge"

        # Compute the aerodynamic coefficients.
        aero = nf.get_aero_from_dat_file(
            filename=dat_file_path,
            alpha=alpha_array_deg,
            Re=Re,
            model_size=model_size,
        )

        # Package the results into a DataFrame.
        df_neuralfoil = pd.DataFrame(
            {
                "alpha": alpha_array_deg,
                "cl": aero["CL"],
                "cd": aero["CD"],
                "cm": aero["CM"],
            }
        )

        return df_neuralfoil

    # Select the specified panel
    panel = body_aero.panels[panel_index]

    # Create an array of angles of attack
    alpha_array_deg = np.linspace(-60, 60, 121)
    # Breukels Coefficients
    cl_br = np.array(
        [panel.calculate_cl(alpha) for alpha in np.deg2rad(alpha_array_deg)]
    )
    cd_br = np.array(
        [panel.calculate_cd_cm(alpha)[0] for alpha in np.deg2rad(alpha_array_deg)]
    )
    cm_br = np.array(
        [panel.calculate_cd_cm(alpha)[1] for alpha in np.deg2rad(alpha_array_deg)]
    )

    # Neuralfoil Coefficients
    df_neuralfoil = run_neuralfoil(alpha_array_deg, panel_index)
    alpha = df_neuralfoil["alpha"].values
    cl_nf = df_neuralfoil["cl"].values
    cd_nf = df_neuralfoil["cd"].values
    cm_nf = df_neuralfoil["cm"].values

    def get_value_at_alpha(alpha_array, data_array, alpha_value):
        """
        Given alpha_array (sorted) and data_array of the same length,
        return the data value at alpha_value by linear interpolation if
        alpha_value is between two existing alpha_array entries, or exact
        if alpha_value matches an entry.

        If alpha_value is below alpha_array[0], return data_array[0].
        If alpha_value is above alpha_array[-1], return data_array[-1].
        """
        if alpha_value <= alpha_array[0]:
            return data_array[0]
        if alpha_value >= alpha_array[-1]:
            return data_array[-1]

        # Find the insertion index
        i = np.searchsorted(alpha_array, alpha_value)

        # If it's an exact match, just return it.
        if i < len(alpha_array) and alpha_array[i] == alpha_value:
            return data_array[i]

        # Otherwise, linear interpolation between i-1 and i
        a1 = alpha_array[i - 1]
        a2 = alpha_array[i]
        d1 = data_array[i - 1]
        d2 = data_array[i]
        frac = (alpha_value - a1) / (a2 - a1)
        return d1 * (1 - frac) + d2 * frac

    # -------------------------------------------------
    # Suppose we have two sets of alpha boundaries and transitions:
    #   For CL
    cl_low_alpha = 0
    cl_high_alpha = 18
    cl_delta_trans_low = 15
    cl_delta_trans_high = 10

    #   For CD
    cd_low_alpha = -10
    cd_high_alpha = 18
    cd_delta_trans_low = 10
    cd_delta_trans_high = 15

    # alpha, cl_nf, cl_br, cd_nf, cd_br, etc. must be arrays of equal length
    # and alpha must be sorted ascending for the interpolation to work properly.
    # We'll assume you already have them in that form.
    # -------------------------------------------------

    # 1) Compute the "edge" values for CL transitions
    cl_nf_lower_edge = get_value_at_alpha(
        alpha_array_deg, cl_nf, cl_low_alpha - cl_delta_trans_low
    )
    cl_br_lower_edge = get_value_at_alpha(alpha_array_deg, cl_br, cl_low_alpha)

    cl_br_upper_edge = get_value_at_alpha(alpha_array_deg, cl_br, cl_high_alpha)
    cl_nf_upper_edge = get_value_at_alpha(
        alpha_array_deg, cl_nf, cl_high_alpha + cl_delta_trans_high
    )

    # 2) Compute the "edge" values for CD transitions
    cd_nf_lower_edge = get_value_at_alpha(
        alpha_array_deg, cd_nf, cd_low_alpha - cd_delta_trans_low
    )
    cd_br_lower_edge = get_value_at_alpha(alpha_array_deg, cd_br, cd_low_alpha)

    cd_br_upper_edge = get_value_at_alpha(alpha_array_deg, cd_br, cd_high_alpha)
    cd_nf_upper_edge = get_value_at_alpha(
        alpha_array_deg, cd_nf, cd_high_alpha + cd_delta_trans_high
    )

    cl_new, cd_new, cm_new = [], [], []

    for alpha_i, cl_nf_i, cd_nf_i, cm_nf_i, cl_br_i, cd_br_i in zip(
        alpha, cl_nf, cd_nf, cm_nf, cl_br, cd_br
    ):
        #
        # --------------------- CL LOGIC ---------------------
        #
        # Lower edge region => NF
        if alpha_i <= (cl_low_alpha - cl_delta_trans_low):
            cl = cl_nf_i

        # Lower transition zone => Interpolate from NF-edge to BR-edge
        elif (cl_low_alpha - cl_delta_trans_low) <= alpha_i < cl_low_alpha:
            # fraction from 0 at left edge to 1 at right edge
            denom = float(cl_delta_trans_low)  # in case it's an int
            frac = (alpha_i - (cl_low_alpha - cl_delta_trans_low)) / denom

            # Edge-based interpolation
            #   frac=0 => cl_nf_lower_edge
            #   frac=1 => cl_br_lower_edge
            cl = (1 - frac) * cl_nf_lower_edge + frac * cl_br_lower_edge

        # Between low_alpha and high_alpha => use BR
        elif cl_low_alpha <= alpha_i < cl_high_alpha:
            cl = cl_br_i

        # Upper transition zone => Interpolate from BR-edge back to NF-edge
        elif cl_high_alpha <= alpha_i < (cl_high_alpha + cl_delta_trans_high):
            denom = float(cl_delta_trans_high)
            frac = (alpha_i - cl_high_alpha) / denom

            #   frac=0 => cl_br_upper_edge
            #   frac=1 => cl_nf_upper_edge
            cl = (1 - frac) * cl_br_upper_edge + frac * cl_nf_upper_edge

        # Above high_alpha + delta_trans => NF
        elif (cl_high_alpha + cl_delta_trans_high) <= alpha_i:
            cl = cl_nf_i
        else:
            raise ValueError(
                "No condition met for CL; something is off in the algorithm."
            )

        #
        # --------------------- CD LOGIC ---------------------
        #
        # Lower edge region => NF
        if alpha_i <= (cd_low_alpha - cd_delta_trans_low):
            cd = cd_nf_i

        # Lower transition zone => Interpolate from NF-edge to BR-edge
        elif (cd_low_alpha - cd_delta_trans_low) <= alpha_i < cd_low_alpha:
            denom = float(cd_delta_trans_low)
            frac = (alpha_i - (cd_low_alpha - cd_delta_trans_low)) / denom
            #   frac=0 => cd_nf_lower_edge
            #   frac=1 => cd_br_lower_edge
            cd = (1 - frac) * cd_nf_lower_edge + frac * cd_br_lower_edge

        # Between cd_low_alpha and cd_high_alpha => BR
        elif cd_low_alpha <= alpha_i < cd_high_alpha:
            cd = cd_br_i

        # Upper transition zone => BR-edge back to NF-edge
        elif cd_high_alpha <= alpha_i < (cd_high_alpha + cd_delta_trans_high):
            denom = float(cd_delta_trans_high)
            frac = (alpha_i - cd_high_alpha) / denom
            #   frac=0 => cd_br_upper_edge
            #   frac=1 => cd_nf_upper_edge
            cd = (1 - frac) * cd_br_upper_edge + frac * cd_nf_upper_edge

        # Above cd_high_alpha + cd_delta_trans => NF
        elif (cd_high_alpha + cd_delta_trans_high) <= alpha_i:
            cd = cd_nf_i
        else:
            raise ValueError(
                "No condition met for CD; something is off in the algorithm."
            )

        #
        # --------------------- CM LOGIC ---------------------
        #
        # If you always take NF for cm, do so:
        cm = cm_nf_i
        # Or create your own boundary-based transitions similarly if needed.

        cl_new.append(cl)
        cd_new.append(cd)
        cm_new.append(cm)

    def smooth_values(alpha, y, window_size=7):
        """
        Smooths y-values by applying a moving average filter over a sorted alpha.

        Parameters
        ----------
        alpha : array-like
            The x-axis array (e.g., angle of attack).
        y : array-like
            The corresponding y-values (e.g., coefficients).
        window_size : int, optional
            The size of the smoothing window. Defaults to 5.

        Returns
        -------
        alpha_sorted : np.ndarray
            The sorted array of alpha values.
        y_smoothed : np.ndarray
            The y-values after applying the moving average smoothing.
        """

        # Ensure alpha and y are numpy arrays
        alpha = np.array(alpha)
        y = np.array(y)

        # Basic check for length mismatch
        if len(alpha) != len(y):
            raise ValueError(
                f"alpha and y must be the same length, got {len(alpha)} vs {len(y)}"
            )

        # Sort by alpha (in case it's not already monotonic)
        sort_idx = np.argsort(alpha)
        alpha_sorted = alpha[sort_idx]
        y_sorted = y[sort_idx]

        # Simple moving-average smoothing
        # 'same' mode ensures the output has the same length as the input
        kernel = np.ones(window_size) / window_size
        y_smoothed = np.convolve(y_sorted, kernel, mode="same")

        return y_smoothed

    cl_smooth = smooth_values(alpha_array_deg, cl_new)
    cd_smooth = smooth_values(alpha_array_deg, cd_new)
    cm_smooth = np.copy(cm_new)

    # taken only the smoothened values outside the defined ranges
    cl_smooth = np.where(
        np.logical_or(
            alpha_array_deg < (cl_low_alpha + 1),
            alpha_array_deg > (cl_high_alpha - 1),
        ),
        cl_smooth,
        cl_new,
    )
    cd_smooth = np.where(
        np.logical_or(
            alpha_array_deg < (cd_low_alpha + 1),
            alpha_array_deg > (cd_high_alpha - 1),
        ),
        cd_smooth,
        cd_new,
    )

    ## a second smoothening loop
    cl_smooth = smooth_values(alpha_array_deg, cl_new)
    cd_smooth = smooth_values(alpha_array_deg, cd_new)
    cm_smooth = np.copy(cm_new)

    # taken only the smoothened values outside the defined ranges
    cl_smooth = np.where(
        np.logical_or(
            alpha_array_deg < (cl_low_alpha + 1),
            alpha_array_deg > (cl_high_alpha - 1),
        ),
        cl_smooth,
        cl_new,
    )
    cd_smooth = np.where(
        np.logical_or(
            alpha_array_deg < (cd_low_alpha + 1),
            alpha_array_deg > (cd_high_alpha - 1),
        ),
        cd_smooth,
        cd_new,
    )

    # create a mask such that only alpha values between -40 and 40 remain
    mask = (alpha_array_deg >= -40) & (alpha <= 40)
    alpha_array_deg = alpha_array_deg[mask]
    cl_smooth = cl_smooth[mask]
    cd_smooth = cd_smooth[mask]
    cm_smooth = cm_smooth[mask]
    cl_new = np.array(cl_new)[mask]
    cd_new = np.array(cd_new)[mask]
    cm_new = np.array(cm_new)[mask]
    cl_br = cl_br[mask]
    cd_br = cd_br[mask]
    cm_br = cm_br[mask]
    cl_nf = cl_nf[mask]
    cd_nf = cd_nf[mask]
    cm_nf = cm_nf[mask]

    # Create a 1x3 subplot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

    ax1.plot(alpha_array_deg, cl_smooth, label="$C_l$ Smooth", color="blue")
    ax1.plot(alpha_array_deg, cl_new, label="$C_l$ Corrected", color="pink")
    ax1.plot(alpha_array_deg, cl_br, label="$C_l$ Breukels", color="black")
    ax1.plot(alpha_array_deg, cl_nf, label="$C_l$ NeuralFoil", color="red")
    ax1.set_xlabel(r"$\alpha$ [°]")
    ax1.set_ylabel(r"$C_{\mathrm{l}}$")
    ax1.grid(True)
    ax1.legend()

    # Cd vs Alpha plot
    ax2.plot(alpha_array_deg, cd_smooth, label="$C_d$ Smooth", color="blue")
    ax2.plot(alpha_array_deg, cd_new, label="$C_d$ Corrected", color="pink")
    ax2.plot(alpha_array_deg, cd_br, label="$C_d$ Breukels", color="black")
    ax2.plot(alpha_array_deg, cd_nf, label="$C_d$ NeuralFoil", color="red")
    ax2.set_xlabel(r"$\alpha$ [°]")
    ax2.set_ylabel(r"$C_{\mathrm{d}}$")
    ax2.grid(True)

    # Cm vs Alpha plot
    ax3.plot(alpha_array_deg, cm_smooth, label="$C_m$ Smooth", color="blue")
    ax3.plot(alpha_array_deg, cm_new, label="$C_m$ Corrected", color="pink")
    ax3.plot(alpha_array_deg, cm_br, label="$C_m$ Breukels", color="black")
    ax3.plot(alpha_array_deg, cm_nf, label="$C_m$ NeuralFoil", color="red")
    ax3.set_xlabel(r"$\alpha$ [°]")
    ax3.set_ylabel(r"$C_{\mathrm{m}}$")
    ax3.grid(True)

    # Adjust layout and display
    ax1.set_xlim(-40, 40)
    ax2.set_xlim(-40, 40)
    ax3.set_xlim(-40, 40)
    plt.tight_layout()
    # plt.show()
    polar_folder_path = Path(
        PROJECT_DIR, "examples", "TUDELFT_V3_LEI_KITE", "polar_engineering"
    )

    figure_path = Path(
        polar_folder_path,
        "figures",
        f"2D_polars_breukels_and_engineering_{panel_index}.pdf",
    )
    plt.savefig(figure_path)
    plt.close()

    df = pd.DataFrame(
        {
            "alpha": np.deg2rad(alpha_array_deg),
            "cl": cl_smooth,
            "cd": cd_smooth,
            "cm": cm_smooth,
            "cl_new": cl_new,
            "cd_new": cd_new,
            "cm_new": cm_new,
            "cl_breukels": cl_br,
            "cd_breukels": cd_br,
            "cm_breukels": cm_br,
            "cl_neuralfoil": cl_nf,
            "cd_neuralfoil": cd_nf,
            "cm_neuralfoil": cm_nf,
        }
    )
    df.to_csv(
        Path(polar_folder_path, "csv_files", f"polar_engineering_{panel_index}.csv"),
        index=False,
    )
    ### make sure first and last are added
    if panel_index == 0:
        df.to_csv(
            Path(polar_folder_path, "csv_files", f"corrected_polar_0.csv"), index=False
        )
    elif panel_index == (n_panels - 1):
        df.to_csv(
            Path(polar_folder_path, "csv_files", f"polar_engineering_{n_panels-1}.csv"),
            index=False,
        )
        df.to_csv(
            Path(polar_folder_path, "csv_files", f"corrected_polar_{n_panels}.csv"),
            index=False,
        )


def process_panel_coefficients(
    body_aero,
    PROJECT_DIR,
    n_panels,
    polar_folder_path,
):

    for i in range(n_panels):
        process_panel_coefficients_panel_i(
            body_aero=body_aero,
            panel_index=i,
            PROJECT_DIR=PROJECT_DIR,
            n_panels=n_panels,
        )

    # take the average for each panel of the side panels
    for i in np.arange(1, n_panels, 1):
        path_to_csv_i = Path(
            polar_folder_path,
            "csv_files",
            f"polar_engineering_{i-1}.csv",
        )
        df_polar_data_i = pd.read_csv(path_to_csv_i)

        path_to_csv_i_p1 = Path(
            polar_folder_path,
            "csv_files",
            f"polar_engineering_{i}.csv",
        )
        df_polar_data_i_p1 = pd.read_csv(path_to_csv_i_p1)

        # Compute the average for all columns with matching names
        df_polar_average = (df_polar_data_i + df_polar_data_i_p1) / 2

        # Save the averaged DataFrame
        df_polar_average.to_csv(
            Path(polar_folder_path, "csv_files", f"corrected_polar_{i}.csv"),
            index=False,
        )
