import os
import sys
import logging
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_pdf import PdfPages
from screeninfo import get_monitors
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


def show_plot(fig, dpi=130):
    """
    Display a matplotlib figure in full screen using the default system PDF viewer.

    :param fig: matplotlib figure object
    :param dpi: Dots per inch for the figure (default: 100)
    """
    # Get the screen resolution
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height

    # Calculate figure size in inches
    fig_width = screen_width / dpi
    fig_height = screen_height / dpi

    # Set figure size to match screen resolution
    fig.set_size_inches(fig_width, fig_height)

    # Save the figure to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        fig.savefig(temp_file.name, format="png", dpi=dpi, bbox_inches="tight")
        temp_file_name = temp_file.name

    # Open the temporary file with the default PDF viewer
    if sys.platform.startswith("darwin"):  # macOS
        subprocess.call(("open", temp_file_name))
    elif os.name == "nt":  # Windows
        os.startfile(temp_file_name)
    elif os.name == "posix":  # Linux
        subprocess.call(("xdg-open", temp_file_name))

    logging.debug(
        f"Plot opened in full screen with dimensions: {screen_width}x{screen_height} pixels at {dpi} DPI"
    )


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
    wing_aero,
    title,
    view_elevation,
    view_azimuth,
):
    """
    Plots the wing panels and filaments in 3D.

    Args:
        wing_aero: WingAerodynamics object
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
    panels = wing_aero.panels
    va = wing_aero.va

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

    return fig


def plot_geometry(
    wing_aero,
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
        wing_aero: WingAerodynamics object
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
            wing_aero,
            title=title + "_angled_view",
            view_elevation=15,
            view_azimuth=-120,
        )
        save_plot(fig, save_path, title + "_angled_view", data_type)
        plt.close()
        # plot top view
        fig = creating_geometry_plot(
            wing_aero,
            title=title + "_top_view",
            view_elevation=90,
            view_azimuth=0,
        )

        save_plot(fig, save_path, title + "_top_view", data_type)
        plt.close()
        # plot front view
        fig = creating_geometry_plot(
            wing_aero,
            title=title + "_front_view",
            view_elevation=0,
            view_azimuth=0,
        )
        save_plot(fig, save_path, title + "_front_view", data_type)
        plt.close()
        # save side view
        fig = creating_geometry_plot(
            wing_aero,
            title=title + "_side_view",
            view_elevation=0,
            view_azimuth=-90,
        )
        save_plot(fig, save_path, title + "_side_view", data_type)
        plt.close()

    # showing plot
    if is_show:
        fig = creating_geometry_plot(
            wing_aero,
            title=title,
            view_elevation=15,
            view_azimuth=-120,
        )
        plt.show()


# def plot_distribution(
#     y_coordinates_list,
#     results_list,
#     label_list,
#     title="spanwise_distribution",
#     data_type=".pdf",
#     save_path=None,
#     is_save=True,
#     is_show=True,
# ):
#     """
#     Plots the spanwise distribution of the results.

#     Args:
#         y_coordinates_list: list of y coordinates
#         results_list: list of results dictionaries
#         label_list: list of labels for the results
#         title: title of the plot
#         data_type: type of the data to be saved | default: ".pdf"
#         save_path: path to save the plot | default: None
#         is_save: boolean to save the plot | default: True
#         is_show: boolean to show the plot | default: True

#     Returns:
#         None
#     """
#     set_plot_style()

#     if len(results_list) != len(label_list):
#         raise ValueError(
#             "The number of results and labels should be the same. Got {} results and {} labels".format(
#                 len(results_list), len(label_list)
#             )
#         )

#     # Set the plot style
#     set_plot_style()

#     # Initializing plot
#     fig, axs = plt.subplots(4, 3, figsize=(20, 15))
#     fig.suptitle(title)  # , fontsize=16)

#     # CL plot
#     for y_coordinates_i, result_i, label_i in zip(
#         y_coordinates_list, results_list, label_list
#     ):
#         axs[0, 0].plot(
#             y_coordinates_i,
#             result_i["cl_distribution"],
#             label=label_i + rf" $C_L$: {result_i['cl']:.2f}",
#         )
#     # axs[0, 0].set_title(rf"$C_L$ Distribution")
#     # axs[0, 0].set_xlabel(r"Spanwise Position $y/b$")
#     axs[0, 0].tick_params(labelbottom=False)
#     axs[0, 0].set_ylabel(r"$C_L$ distribution")
#     axs[0, 0].legend()

#     # CD plot
#     for y_coordinates_i, result_i, label_i in zip(
#         y_coordinates_list, results_list, label_list
#     ):
#         axs[0, 1].plot(
#             y_coordinates_i,
#             result_i["cd_distribution"],
#             label=label_i + rf" $C_D$: {result_i['cd']:.2f}",
#         )
#     # axs[0, 1].set_title(r"$C_D$ Distribution")
#     # axs[0, 1].set_xlabel(r"Spanwise Position $y/b$")
#     axs[0, 1].set_ylabel(r"$C_D$ Distribution")
#     axs[0, 1].tick_params(labelbottom=False)
#     axs[0, 1].legend()

#     # Gamma plot
#     for y_coordinates_i, result_i, label_i in zip(
#         y_coordinates_list, results_list, label_list
#     ):
#         axs[0, 2].plot(y_coordinates_i, result_i["gamma_distribution"], label=label_i)
#     # axs[0, 2].set_title(r"$\Gamma$ Distribution")
#     # axs[0, 2].set_xlabel(r"Spanwise Position $y/b$")
#     axs[0, 2].set_ylabel(r"Circulation $\Gamma$  distribution")
#     axs[0, 2].tick_params(labelbottom=False)
#     axs[0, 2].legend()

#     # Geometric Alpha plot
#     for y_coordinates_i, result_i, label_i in zip(
#         y_coordinates_list, results_list, label_list
#     ):
#         axs[1, 0].plot(
#             y_coordinates_i, np.rad2deg(result_i["alpha_geometric"]), label=label_i
#         )
#     # axs[1, 0].set_title(r"$\alpha$ Geometric")
#     # axs[1, 0].set_xlabel(r"Spanwise Position $y/b$")
#     axs[1, 0].set_ylabel(r"Geometric $\alpha$ (deg)")
#     axs[1, 0].tick_params(labelbottom=False)
#     axs[1, 0].legend()

#     # Calculated/Corrected Alpha plot
#     for y_coordinates_i, result_i, label_i in zip(
#         y_coordinates_list, results_list, label_list
#     ):
#         axs[1, 1].plot(
#             y_coordinates_i, np.rad2deg(result_i["alpha_at_ac"]), label=label_i
#         )
#     # axs[1, 1].set_title(r"$\alpha$ result (corrected to aerodynamic center)")
#     # axs[1, 1].set_xlabel(r"Spanwise Position $y/b$")
#     axs[1, 1].set_ylabel(r"Corrected $\alpha$ (to geometric center) [deg]")
#     axs[1, 1].tick_params(labelbottom=False)
#     axs[1, 1].legend()

#     # Uncorrected Alpha plot
#     for y_coordinates_i, result_i, label_i in zip(
#         y_coordinates_list, results_list, label_list
#     ):
#         axs[1, 2].plot(
#             y_coordinates_i,
#             np.rad2deg(result_i["alpha_uncorrected"]),
#             label=label_i,
#         )
#     # axs[1, 2].set_title(r"$\alpha$ Uncorrected (if VSM, at the control point)")
#     # axs[1, 2].set_xlabel(r"Spanwise Position $y/b$")
#     axs[1, 2].set_ylabel(r"Uncorrected $\alpha$ (if VSM, at 3/4c control point) [deg]")
#     axs[1, 2].tick_params(labelbottom=False)
#     axs[1, 2].legend()

#     for y_coordinates_i, result_i, label_i in zip(
#         y_coordinates_list, results_list, label_list
#     ):
#         axs[2, 0].plot(
#             y_coordinates_i,
#             [force[0] for force in result_i["F_distribution"]],
#             label=label_i + rf" $\sum{{F_x}}$: {result_i['Fx']:.2f}N",
#         )
#     # axs[2, 0].set_title(f"Force in x direction")
#     # axs[2, 0].set_xlabel(r"Spanwise Position $y/b$")
#     axs[2, 0].set_ylabel(r"$F_x$ distribution")
#     axs[2, 0].tick_params(labelbottom=False)
#     axs[2, 0].legend()

#     # Force in y
#     for y_coordinates_i, result_i, label_i in zip(
#         y_coordinates_list, results_list, label_list
#     ):
#         axs[2, 1].plot(
#             y_coordinates_i,
#             [force[1] for force in result_i["F_distribution"]],
#             label=label_i + rf" $\sum{{F_y}}$: {result_i['Fy']:.2f}N",
#         )
#     # axs[2, 1].set_title(f"Force in y direction")
#     # axs[2, 1].set_xlabel(r"Spanwise Position $y/b$")
#     axs[2, 1].set_ylabel(r"$F_y$ distribution")
#     axs[2, 1].tick_params(labelbottom=False)
#     axs[2, 1].legend()

#     # Force in z
#     for y_coordinates_i, result_i, label_i in zip(
#         y_coordinates_list, results_list, label_list
#     ):
#         axs[2, 2].plot(
#             y_coordinates_i,
#             [force[2] for force in result_i["F_distribution"]],
#             label=label_i + rf" $\sum{{F_z}}$: {result_i['Fz']:.2f}N",
#         )
#     # axs[2, 2].set_title(f"Force in z direction")
#     # axs[2, 2].set_xlabel(r"Spanwise Position $y/b$")
#     axs[2, 2].set_ylabel(r"$F_z$ distribution")
#     axs[2, 2].tick_params(labelbottom=False)
#     axs[2, 2].legend()

#     # Moment in x
#     for y_coordinates_i, result_i, label_i in zip(
#         y_coordinates_list, results_list, label_list
#     ):
#         axs[3, 0].plot(
#             y_coordinates_i,
#             [moment[0] for moment in result_i["M_distribution"]],
#             label=label_i + rf" $\sum{{M_x}}$: {result_i['Mx']:.2f}Nm",
#         )
#     # axs[3, 0].set_title(f"Moment in x direction")
#     axs[3, 0].set_xlabel(r"Spanwise Position $y/b$")
#     axs[3, 0].set_ylabel(r"$M_x$ distribution")
#     axs[3, 0].legend()

#     # Moment in y
#     for y_coordinates_i, result_i, label_i in zip(
#         y_coordinates_list, results_list, label_list
#     ):
#         axs[3, 1].plot(
#             y_coordinates_i,
#             [moment[1] for moment in result_i["M_distribution"]],
#             label=label_i + rf" $\sum{{M_y}}$: {result_i['My']:.2f}Nm",
#         )
#     # axs[3, 1].set_title(f"Moment in y direction")
#     axs[3, 1].set_xlabel(r"Spanwise Position $y/b$")
#     axs[3, 1].set_ylabel(r"$M_y$ distribution")
#     axs[3, 1].legend()

#     # Moment in z
#     for y_coordinates_i, result_i, label_i in zip(
#         y_coordinates_list, results_list, label_list
#     ):
#         axs[3, 2].plot(
#             y_coordinates_i,
#             [moment[2] for moment in result_i["M_distribution"]],
#             label=label_i + rf" $\sum{{M_z}}$: {result_i['Mz']:.2f}Nm",
#         )
#     # axs[3, 2].set_title(f"Moment in z direction")
#     axs[3, 2].set_xlabel(r"Spanwise Position $y/b$")
#     axs[3, 2].set_ylabel(r"$M_z$ distribution")
#     axs[3, 2].legend()

#     ### Ensuring that a value does not ruin the naturally zooomed in ylim
#     for i, ax in enumerate(axs.flat):
#         if i == 0:  # CL
#             y_min_allowed, y_max_allowed = -1.5, 1.5
#         elif i == 1:  # CD
#             y_min_allowed, y_max_allowed = -1.5, 1.5
#         elif i == 2:  # Gamma
#             y_min_allowed, y_max_allowed = -1, 25
#         elif i == 3:  # alpha geometric
#             y_min_allowed, y_max_allowed = -10, 150
#         elif i == 4:  # alpha corrected
#             y_min_allowed, y_max_allowed = -10, 90
#         elif i == 5:  # alpha uncorrected
#             y_min_allowed, y_max_allowed = -10, 90
#         else:
#             y_min_allowed, y_max_allowed = ax.get_ylim()

#         # Collect all y-data from the lines in the current axis
#         y_data = np.concatenate([line.get_ydata() for line in ax.get_lines()])

#         # Identify data within the allowed range
#         in_range = y_data[(y_data >= y_min_allowed) & (y_data <= y_max_allowed)]

#         if in_range.size > 0:
#             # Optionally add some padding to the y-limits
#             padding = 0.05 * (in_range.max() - in_range.min())
#             ax.set_ylim(in_range.min() - padding, in_range.max() + padding)
#         else:
#             # If no data is within the range, you might choose to set default limits or skip
#             pass  # Or set default limits, e.g., ax.set_ylim(y_min_allowed, y_max_allowed)

#     plt.tight_layout()

#     # Ensure the figure is fully rendered
#     fig.canvas.draw()

#     # saving plot
#     if is_save:
#         save_plot(fig, save_path, title, data_type)

#     # showing plot
#     if is_show:
#         # plt.show()
#         show_plot(fig)


def plot_distribution(
    y_coordinates_list,
    results_list,
    label_list,
    title="spanwise_distribution",
    data_type=".pdf",
    save_path=None,
    is_save=True,
    is_show=True,
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

    if len(results_list) != len(label_list):
        raise ValueError(
            f"The number of results and labels should match. "
            f"Got {len(results_list)} results and {len(label_list)} labels."
        )

    # Create figure and axes: 3 rows x 3 columns
    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    # fig.suptitle(title)

    # --- Row 1: CD, CL, CS ---------------------------------------------------
    # Column 1: CD
    for y_coords, result, label in zip(y_coordinates_list, results_list, label_list):
        axs[0, 0].plot(
            y_coords,
            result["cd_distribution"],
            label=label + rf" $C_D$: {result['cd']:.2f}",
        )
    axs[0, 0].set_ylabel(r"$C_D$ Distribution")
    axs[0, 0].tick_params(labelbottom=False)
    axs[0, 0].legend()

    # Column 2: CL
    for y_coords, result, label in zip(y_coordinates_list, results_list, label_list):
        axs[0, 1].plot(
            y_coords,
            result["cl_distribution"],
            label=label + rf" $C_L$: {result['cl']:.2f}",
        )
    axs[0, 1].set_ylabel(r"$C_L$ Distribution")
    axs[0, 1].tick_params(labelbottom=False)
    axs[0, 1].legend()

    # Column 3: CS (side force coefficient)
    for y_coords, result, label in zip(y_coordinates_list, results_list, label_list):
        axs[0, 2].plot(
            y_coords,
            result["cs_distribution"],
            label=label + rf" $C_S$: {result['cs']:.2f}",
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
            label=label + rf" $C_{{mx}}$: {result['cmx']:.2f}",
        )
    axs[1, 0].set_ylabel(r"$C_{mx}$ Distribution")
    axs[1, 0].tick_params(labelbottom=False)
    axs[1, 0].legend()

    # Column 2: CMy
    for y_coords, result, label in zip(y_coordinates_list, results_list, label_list):
        axs[1, 1].plot(
            y_coords,
            result["cmy_distribution"],
            label=label + rf" $C_{{my}}$: {result['cmy']:.2f}",
        )
    axs[1, 1].set_ylabel(r"$C_{my}$ Distribution")
    axs[1, 1].tick_params(labelbottom=False)
    axs[1, 1].legend()

    # Column 3: CMz
    for y_coords, result, label in zip(y_coordinates_list, results_list, label_list):
        axs[1, 2].plot(
            y_coords,
            result["cmz_distribution"],
            label=label + rf" $C_{{mz}}$: {result['cmz']:.2f}",
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
    axs[2, 0].legend()

    # Column 2: Gamma distribution
    for y_coords, result, label in zip(y_coordinates_list, results_list, label_list):
        axs[2, 1].plot(
            y_coords,
            np.rad2deg(result["alpha_geometric"]),
            label=label,
        )
    axs[2, 1].set_xlabel(r"Spanwise Position $y/b$")
    axs[2, 1].set_ylabel(r"Geometric $\alpha$ (deg)")
    axs[2, 1].legend()

    # Column 3: alpha (corrected to aerodynamic center)
    for y_coords, result, label in zip(y_coordinates_list, results_list, label_list):
        axs[2, 2].plot(
            y_coords,
            np.rad2deg(result["alpha_at_ac"]),
            label=label,
        )
    axs[2, 2].set_xlabel(r"Spanwise Position $y/b$")
    axs[2, 2].set_ylabel(r"Corrected $\alpha$ (deg)")
    axs[2, 2].legend()

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

    plt.tight_layout()

    # Save or show plot
    if is_save:
        save_plot(fig, save_path, title, data_type)

    if is_show:
        show_plot(fig)


def generate_polar_data(
    solver,
    wing_aero,
    angle_range,
    angle_type="angle_of_attack",
    angle_of_attack=0,
    side_slip=0,
    yaw_rate=0,
    Umag=10,
):
    """
    Generates the polar data for the given solver and wing_aero.

    Args:
        solver: solver object
        wing_aero: wing_aero object
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
    cd = np.zeros(len(angle_range))
    cs = np.zeros(len(angle_range))
    cmx = np.zeros(len(angle_range))
    cmy = np.zeros(len(angle_range))
    cmz = np.zeros(len(angle_range))
    gamma_distribution = np.zeros((len(angle_range), len(wing_aero.panels)))
    cl_distribution = np.zeros((len(angle_range), len(wing_aero.panels)))
    cd_distribution = np.zeros((len(angle_range), len(wing_aero.panels)))
    cs_distribution = np.zeros((len(angle_range), len(wing_aero.panels)))
    reynolds_number = np.zeros(len(angle_range))
    # initialize the gamma with None
    gamma = None
    for i, angle_i in enumerate(angle_range):
        if angle_type == "angle_of_attack":
            wing_aero.va_initialize(Umag, angle_i, side_slip, yaw_rate)
        elif angle_type == "side_slip":
            wing_aero.va_initialize(Umag, angle_of_attack, angle_i, yaw_rate)
        else:
            raise ValueError(
                "angle_type should be either 'angle_of_attack' or 'side_slip'"
            )

        # Set the inflow conditions

        results = solver.solve(wing_aero)
        cl[i] = results["cl"]
        cd[i] = results["cd"]
        cs[i] = results["cs"]
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
    ]
    reynolds_number = results["Rey"]

    return polar_data, reynolds_number


def plot_polars(
    solver_list,
    wing_aero_list,
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
    save_path=None,
    is_save=True,
    is_show=True,
):
    """
    Plot polar data CL, CD, CS, and CL-CD over a specified angle for the given solvers and wing_aeros.

    Args:
        solver_list: list of solver objects
        wing_aero_list: list of wing_aero objects
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
        None
    """
    # Set the plot style
    set_plot_style()

    # Checking type and configuring the x_label
    if angle_type == "angle_of_attack":
        x_label = r"$\alpha$ [°]"
    elif angle_type == "side_slip":
        x_label = r"$\beta$ [°]"
    else:
        raise ValueError("angle_type should be either 'angle_of_attack' or 'side_slip'")

    if (len(wing_aero_list) + len(literature_path_list)) != len(label_list) or len(
        solver_list
    ) != len(wing_aero_list):
        raise ValueError(
            "The number of solvers, results and labels should be the same. Got {} solvers and {} results and {} labels".format(
                (len(solver_list) + len(literature_path_list)),
                (len(wing_aero_list) + len(literature_path_list)),
                len(label_list),
            )
        )

    # generating polar data
    polar_data_list = []
    for i, (solver, wing_aero) in enumerate(zip(solver_list, wing_aero_list)):
        polar_data, reynolds_number = generate_polar_data(
            solver=solver,
            wing_aero=wing_aero,
            angle_range=angle_range,
            angle_type=angle_type,
            angle_of_attack=angle_of_attack,
            side_slip=side_slip,
            yaw_rate=yaw_rate,
            Umag=Umag,
        )
        polar_data_list.append(polar_data)
        # Appending Reynolds numbers to the labels of the solvers
        label_list[i] += f" Re = {1e-5*reynolds_number:.1f}e5"

    # Grabbing additional data from literature
    if literature_path_list is not None:
        for literature_path in literature_path_list:
            CL, CD, angle = np.loadtxt(
                literature_path, delimiter=",", skiprows=1, unpack=True
            )
            polar_data_list.append([angle, CL, CD])

    # Initializing plot
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    n_solvers = len(solver_list)
    # CL plot
    for i, (polar_data, label) in enumerate(zip(polar_data_list, label_list)):
        if i < n_solvers:
            linestyle = "-"
            marker = "*"
            markersize = 7
        else:
            linestyle = "-"
            marker = "."
            markersize = 5
        axs[0, 0].plot(
            polar_data[0],
            polar_data[1],
            label=label,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
        )
        # if CL is greater than 10, limit the yrange
        if max(polar_data[1]) > 10:
            axs[0, 0].set_ylim([-0.5, 2])
    axs[0, 0].set_ylabel(r"$C_{\mathrm{L}}$")

    # CD plot
    for i, (polar_data, label) in enumerate(zip(polar_data_list, label_list)):
        if i < n_solvers:
            linestyle = "-"
            marker = "*"
            markersize = 7
        else:
            linestyle = "-"
            marker = "."
            markersize = 5
        axs[0, 1].plot(
            polar_data[0],
            polar_data[2],
            label=label,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
        )
        # if CD is greater than 10, limit the range
        if max(polar_data[2]) > 10:
            axs[0, 1].set_ylim([-0.2, 0.5])
    axs[0, 1].set_ylabel(r"$C_{\mathrm{D}}$")
    axs[0, 1].legend(loc="best")

    # CL-CD plot
    if angle_type == "angle_of_attack":
        for i, (polar_data, label) in enumerate(zip(polar_data_list, label_list)):
            if i < n_solvers:
                linestyle = "-"
                marker = "*"
                markersize = 7
            else:
                linestyle = "-"
                marker = "."
                markersize = 5
            axs[0, 2].plot(
                polar_data[0],
                polar_data[1] / polar_data[2],
                label=label,
                linestyle=linestyle,
                marker=marker,
                markersize=markersize,
            )

        axs[0, 2].set_ylabel(r"$C_{\mathrm{L}}$ / $C_{\mathrm{D}}$")

    elif angle_type == "side_slip":
        # CS plot
        for i, (polar_data, label) in enumerate(zip(polar_data_list, label_list)):
            # Build-in a check, since the literature polars might not have the CS coefficient
            if len(polar_data) > 3:
                if i < n_solvers:
                    linestyle = "-"
                    marker = "*"
                    markersize = 7
                else:
                    linestyle = "-"
                    marker = "."
                    markersize = 5
                axs[0, 2].plot(
                    polar_data[0],
                    polar_data[3],
                    label=label,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=markersize,
                )
        axs[0, 2].set_ylabel(r"$C_{\mathrm{S}}$")

    # cmx, cmy,cmz plots
    for i, (polar_data, label) in enumerate(zip(polar_data_list, label_list)):
        if i >= len(wing_aero_list):
            continue
        else:
            linestyle = "-"
            marker = "."
            markersize = 5
            axs[1, 0].plot(
                polar_data[0],
                polar_data[4],
                label=label + r" $C_{\mathrm{m,x}}$",
                linestyle=linestyle,
                marker=marker,
                markersize=markersize,
            )
            axs[1, 0].set_ylabel(r"$C_{\mathrm{m,x}}$")
            axs[1, 1].plot(
                polar_data[0],
                polar_data[5],
                label=label + r" $C_{\mathrm{m,y}}$",
                linestyle=linestyle,
                marker=marker,
                markersize=markersize,
            )
            axs[1, 1].set_ylabel(r"$C_{\mathrm{m,y}}$")
            axs[1, 2].plot(
                polar_data[0],
                polar_data[6],
                label=label + r" $C_{\mathrm{m,z}}$",
                linestyle=linestyle,
                marker=marker,
                markersize=markersize,
            )
            axs[1, 2].set_ylabel(r"$C_{\mathrm{m,z}}$")

    # Add axis labels
    for ax in axs.flat:
        ax.set_xlabel(x_label)

    ### Ensuring that a value does not ruin the naturally zooomed in ylim
    for i, ax in enumerate(axs.flat):
        if angle_type == "angle_of_attack":
            if i == 2:  # Cl/Cd
                y_min_allowed, y_max_allowed = -15, 15
            elif i == 4:  # CMy
                y_min_allowed, y_max_allowed = -2, 2
            else:
                y_min_allowed, y_max_allowed = -1.5, 1.5
        elif angle_type == "side_slip":
            y_min_allowed, y_max_allowed = -1.5, 1.5

        # Collect all y-data from the lines in the current axis
        y_data = np.concatenate([line.get_ydata() for line in ax.get_lines()])

        # Identify data within the allowed range
        in_range = y_data[(y_data >= y_min_allowed) & (y_data <= y_max_allowed)]

        if in_range.size > 0:
            # Optionally add some padding to the y-limits
            padding = 0.05 * (in_range.max() - in_range.min())
            ax.set_ylim(in_range.min() - padding, in_range.max() + padding)
        else:
            # If no data is within the range, you might choose to set default limits or skip
            pass  # Or set default limits, e.g., ax.set_ylim(y_min_allowed, y_max_allowed)

    # Ensure the figure is fully rendered
    fig.canvas.draw()

    # saving plot
    if is_save:
        save_plot(fig, save_path, title, data_type)

    # showing plot
    if is_show:
        show_plot(fig)


def plot_panel_coefficients(
    wing_aero,
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
        wing_aero (object): Wing aerodynamic object containing panels
        panel_index (int): Index of the panel to plot
        alpha_range (tuple, optional): Range of angles of attack in radians.
                                       Defaults to (-0.5, 0.5) radians.
    """
    set_plot_style()
    if title is None:
        title = f"2D_polar_of_panel_{panel_index}"

    # Select the specified panel
    panel = wing_aero.panels[panel_index]

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
    if is_show:
        show_plot(fig)


# def process_panel_coefficients_panel_i(
#     wing_aero, panel_index, PROJECT_DIR, n_panels, alpha_range=[-40, 40]
# ):
#     """
#     Plot Cl, Cd, and Cm coefficients for a specific panel across a range of angles of attack.

#     Args:
#         wing_aero (object): Wing aerodynamic object containing panels
#         panel_index (int): Index of the panel to plot
#         alpha_range (tuple, optional): Range of angles of attack in radians.
#                                        Defaults to (-0.5, 0.5) radians.
#     """
#     # Select the specified panel
#     panel = wing_aero.panels[panel_index]

#     # Create an array of angles of attack
#     alpha_array_deg = np.linspace(-40, 40, 81)
#     alpha_array = np.deg2rad(alpha_array_deg)

#     # Calculate coefficients
#     cl_array = np.array(
#         [panel.calculate_cl(alpha) for alpha in np.deg2rad(alpha_array_deg)]
#     )
#     cd_array = np.array(
#         [panel.calculate_cd_cm(alpha)[0] for alpha in np.deg2rad(alpha_array_deg)]
#     )
#     cm_array = np.array(
#         [panel.calculate_cd_cm(alpha)[1] for alpha in np.deg2rad(alpha_array_deg)]
#     )

#     def run_neuralfoil(PROJECT_DIR=PROJECT_DIR):
#         import neuralfoil as nf

#         Re = 5.6e5
#         model_size = "xxlarge"
#         dat_file_path = Path(
#             PROJECT_DIR,
#             "examples",
#             "TUDELFT_V3_LEI_KITE",
#             "polar_engineering",
#             "profiles",
#             "y1_corrected.dat",
#         )
#         alpha_values_deg = np.linspace(-40, 40, 81)
#         neuralfoil_alphas = alpha_values_deg

#         aero = nf.get_aero_from_dat_file(
#             filename=dat_file_path,
#             alpha=neuralfoil_alphas,
#             Re=Re,
#             model_size=model_size,
#         )
#         df_neuralfoil = pd.DataFrame(
#             {
#                 "alpha": neuralfoil_alphas,
#                 "cl": aero["CL"],
#                 "cd": aero["CD"],
#                 "cm": aero["CM"],
#             }
#         )
#         return df_neuralfoil

#     df_neuralfoil = run_neuralfoil(PROJECT_DIR)
#     cm_array_new = df_neuralfoil["cm"].values
#     print(f"len(cm_array): {len(cm_array)}, len(cm_array_new): {len(cm_array_new)}")

#     ### Adding NeuralFoil to Breukels Polars outside the -3 to 20 range

#     cl_array_new = []
#     cd_array_new = []
#     cm_array_new = []
#     for alpha in alpha_array:

#         if np.rad2deg(alpha) < -1 and np.rad2deg(alpha) > -18:
#             cl = -0.3 - np.abs(0.7 * (np.rad2deg(alpha) / 40))
#             cd = panel.calculate_cd_cm(alpha)[0]
#         elif np.rad2deg(alpha) < -18:
#             cl = -0.3 - 0.7 * np.abs(np.rad2deg(alpha) / 40)
#             blend_factor = (np.rad2deg(alpha) + 18) / 2  # Smooth transition factor
#             cd = (1 - blend_factor) * panel.calculate_cd_cm(np.deg2rad(-18))[
#                 0
#             ] + blend_factor * (-0.01 * (np.rad2deg(alpha) + 18))
#         elif np.rad2deg(alpha) > 19:
#             cd = (
#                 0.01 * (np.rad2deg(alpha) - 19)
#                 + panel.calculate_cd_cm(np.deg2rad(19))[0]
#             )
#             cl = 1.05 + 0.2 * np.abs(np.rad2deg(alpha) / 40)
#         else:
#             cl = panel.calculate_cl(alpha)
#             cd = panel.calculate_cd_cm(alpha)[0]

#         cl_array_new.append(cl)
#         cd_array_new.append(cd)

#     def smooth_discontinuous_values(
#         alpha, y, method="cubic", smoothing_window=1, use_gaussian=True
#     ):
#         """
#         Smooths y-values corresponding to discontinuous alpha values.

#         Args:
#             alpha (np.ndarray): Array of alpha values (x-axis).
#             y (np.ndarray): Array of y values (y-axis).
#             method (str): Interpolation method ('linear', 'cubic', etc.). Default is 'linear'.
#             smoothing_window (int): Size of the smoothing window. Default is 5.
#             use_gaussian (bool): Whether to use Gaussian filter for smoothing. Default is True.

#         Returns:
#             np.ndarray, np.ndarray: Smoothed alpha and y values.
#         """
#         from scipy.interpolate import interp1d
#         from scipy.ndimage import gaussian_filter1d

#         # Interpolate y values on the regular alpha grid
#         print(f"len(alpha): {len(alpha)}, len(y): {len(y)}")
#         interpolator = interp1d(alpha, y, kind=method, fill_value="extrapolate")
#         y_interpolated = interpolator(alpha)

#         # Apply smoothing
#         if use_gaussian:
#             y_smoothed = gaussian_filter1d(y_interpolated, smoothing_window)
#         else:
#             # Simple moving average
#             y_smoothed = np.convolve(
#                 y_interpolated,
#                 np.ones(smoothing_window) / smoothing_window,
#                 mode="same",
#             )

#         return y_smoothed

#     def smooth_values(alpha, y, window_size=5):
#         """
#         Smooths y-values by applying a moving average filter over a sorted alpha.

#         Parameters
#         ----------
#         alpha : array-like
#             The x-axis array (e.g., angle of attack).
#         y : array-like
#             The corresponding y-values (e.g., coefficients).
#         window_size : int, optional
#             The size of the smoothing window. Defaults to 5.

#         Returns
#         -------
#         alpha_sorted : np.ndarray
#             The sorted array of alpha values.
#         y_smoothed : np.ndarray
#             The y-values after applying the moving average smoothing.
#         """

#         # Ensure alpha and y are numpy arrays
#         alpha = np.array(alpha)
#         y = np.array(y)

#         # Basic check for length mismatch
#         if len(alpha) != len(y):
#             raise ValueError(
#                 f"alpha and y must be the same length, got {len(alpha)} vs {len(y)}"
#             )

#         # Sort by alpha (in case it's not already monotonic)
#         sort_idx = np.argsort(alpha)
#         alpha_sorted = alpha[sort_idx]
#         y_sorted = y[sort_idx]

#         # Simple moving-average smoothing
#         # 'same' mode ensures the output has the same length as the input
#         kernel = np.ones(window_size) / window_size
#         y_smoothed = np.convolve(y_sorted, kernel, mode="same")

#         return alpha_sorted, y_smoothed

#     cl_smoothed = smooth_discontinuous_values(
#         np.copy(alpha_array_deg), np.copy(cl_array_new)
#     )
#     cd_smoothed = smooth_discontinuous_values(
#         np.copy(alpha_array_deg), np.copy(cd_array_new)
#     )
#     # cm_smoothed = smooth_discontinuous_values(
#     #     np.copy(alpha_array_deg), np.copy(cm_array_new)
#     # )

#     # Create a new smooth array that uses the old values for alpha < -3 and alpha > 19
#     # cl_smoothed = np.where(
#     #     np.logical_or(alpha_array < np.deg2rad(-1), alpha_array > np.deg2rad(50)),
#     #     cl_smoothed,
#     #     cl_array,
#     # )
#     # cd_smoothed = np.where(
#     #     np.logical_or(alpha_array < np.deg2rad(-18), alpha_array > np.deg2rad(19)),
#     #     cd_smoothed,
#     #     cd_array,
#     # )
#     # cm_smoothed = np.where(
#     #     np.logical_or(alpha_array < np.deg2rad(-1), alpha_array > np.deg2rad(19)),
#     #     cm_smoothed,
#     #     cm_array,
#     # )

#     # cl_smoothed = smooth_values(alpha_array_deg, cl_array_new)
#     # cd_smoothed = smooth_values(alpha_array_deg, cd_array_new)
#     # cm_smoothed = smooth_values(alpha_array_deg, cm_array_new)
#     # cl_smoothed = np.copy(cl_array_new)
#     # cd_smoothed = np.copy(cd_array_new)
#     cm_smoothed = np.copy(df_neuralfoil["cm"].values)

#     # Create a 1x3 subplot
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

#     # Cl vs Alpha plot
#     alpha_array = np.deg2rad(alpha_array_deg)
#     print(f"len(alpha_array): {len(alpha_array)}, len(cl_smoothed): {len(cl_smoothed)}")
#     print(f"len(cl_array): {len(cl_array)}")
#     print(f"len(cl_array_new): {len(cl_array_new)}")
#     print(f"len cl smooth: {len(cl_smoothed)}")
#     print(f"len cd smooth: {len(cd_smoothed)}")
#     print(f"len cd array: {len(cd_array)}")
#     print(f"len cm smooth: {len(cm_smoothed)}")
#     print(f"len cm array: {len(cm_array)}")

#     ax1.plot(
#         np.rad2deg(alpha_array), cl_smoothed, label="$C_l$ Corrected", color="blue"
#     )
#     ax1.plot(np.rad2deg(alpha_array), cl_array, label="$C_l$ Breukels", color="black")
#     ax1.plot(
#         np.rad2deg(alpha_array),
#         df_neuralfoil["cl"].values,
#         label="$C_l$ NeuralFoil",
#         color="red",
#     )
#     ax1.set_xlabel(r"$\alpha$ [°]")
#     ax1.set_ylabel(r"$C_{\mathrm{l}}$")
#     ax1.grid(True)
#     ax1.legend()

#     # Cd vs Alpha plot
#     ax2.plot(
#         np.rad2deg(alpha_array), cd_smoothed, label="$C_d$ Corrected", color="blue"
#     )
#     ax2.plot(np.rad2deg(alpha_array), cd_array, label="$C_d$ Breukels", color="black")
#     ax2.plot(
#         np.rad2deg(alpha_array),
#         df_neuralfoil["cd"].values,
#         label="$C_d$ NeuralFoil",
#         color="red",
#     )
#     ax2.set_xlabel(r"$\alpha$ [°]")
#     ax2.set_ylabel(r"$C_{\mathrm{d}}$")
#     ax2.grid(True)

#     # Cm vs Alpha plot
#     ax3.plot(
#         np.rad2deg(alpha_array), cm_smoothed, label="$C_m$ Corrected", color="blue"
#     )
#     ax3.plot(np.rad2deg(alpha_array), cm_array, label="$C_m$ Breukels", color="black")
#     ax3.plot(
#         np.rad2deg(alpha_array),
#         df_neuralfoil["cm"].values,
#         label="$C_m$ NeuralFoil",
#         color="red",
#     )
#     ax3.set_xlabel(r"$\alpha$ [°]")
#     ax3.set_ylabel(r"$C_{\mathrm{m}}$")
#     ax3.grid(True)

#     # Adjust layout and display
#     ax1.set_xlim(-40, 40)
#     ax2.set_xlim(-40, 40)
#     ax3.set_xlim(-40, 40)
#     plt.tight_layout()
#     # plt.show()
#     polar_folder_path = Path(
#         PROJECT_DIR, "examples", "TUDELFT_V3_LEI_KITE", "polar_engineering"
#     )

#     figure_path = Path(
#         polar_folder_path,
#         "figures",
#         f"2D_polars_breukels_and_engineering_{panel_index}.pdf",
#     )
#     plt.savefig(figure_path)
#     plt.close()

#     df = pd.DataFrame(
#         {
#             "alpha": alpha_array,
#             "cl": cl_smoothed,
#             "cd": cd_smoothed,
#             "cm": cm_smoothed,
#             "cl_breukels": cl_array,
#             "cd_breukels": cd_array,
#             "cm_breukels": cm_array,
#         }
#     )
#     df.to_csv(
#         Path(polar_folder_path, "csv_files", f"polar_engineering_{panel_index}.csv"),
#         index=False,
#     )
#     ### make sure first and last are added
#     if panel_index == 0:
#         df.to_csv(
#             Path(polar_folder_path, "csv_files", f"corrected_polar_0.csv"), index=False
#         )
#     elif panel_index == (n_panels - 1):
#         df.to_csv(
#             Path(polar_folder_path, "csv_files", f"polar_engineering_{n_panels-1}.csv"),
#             index=False,
#         )
#         df.to_csv(
#             Path(polar_folder_path, "csv_files", f"corrected_polar_{n_panels}.csv"),
#             index=False,
#         )


def process_panel_coefficients_panel_i(wing_aero, panel_index, PROJECT_DIR, n_panels):
    """
    Plot Cl, Cd, and Cm coefficients for a specific panel across a range of angles of attack.

    Args:
        wing_aero (object): Wing aerodynamic object containing panels
        panel_index (int): Index of the panel to plot
        alpha_range (tuple, optional): Range of angles of attack in radians.
                                       Defaults to (-0.5, 0.5) radians.
    """

    # def run_neuralfoil(alpha_array_deg, PROJECT_DIR=PROJECT_DIR):
    #     import neuralfoil as nf

    #     Re = 5.6e5
    #     model_size = "xxlarge"
    #     dat_file_path = Path(
    #         PROJECT_DIR,
    #         "examples",
    #         "TUDELFT_V3_LEI_KITE",
    #         "polar_engineering",
    #         "profiles",
    #         "y1_corrected.dat",
    #     )

    #     aero = nf.get_aero_from_dat_file(
    #         filename=dat_file_path,
    #         alpha=alpha_array_deg,
    #         Re=Re,
    #         model_size=model_size,
    #     )
    #     df_neuralfoil = pd.DataFrame(
    #         {
    #             "alpha": alpha_array_deg,
    #             "cl": aero["CL"],
    #             "cd": aero["CD"],
    #             "cm": aero["CM"],
    #         }
    #     )
    #     return df_neuralfoil

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
    panel = wing_aero.panels[panel_index]

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
    wing_aero,
    PROJECT_DIR,
    n_panels,
    polar_folder_path,
):

    for i in range(n_panels):
        process_panel_coefficients_panel_i(
            wing_aero=wing_aero,
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
