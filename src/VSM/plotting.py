import os
import sys
import logging
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_pdf import PdfPages
from screeninfo import get_monitors
from VSM.color_palette import set_plot_style, get_color


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
    Plots the spanwise distribution of the results.

    Args:
        y_coordinates_list: list of y coordinates
        results_list: list of results dictionaries
        label_list: list of labels for the results
        title: title of the plot
        data_type: type of the data to be saved | default: ".pdf"
        save_path: path to save the plot | default: None
        is_save: boolean to save the plot | default: True
        is_show: boolean to show the plot | default: True

    Returns:
        None
    """

    if len(results_list) != len(label_list):
        raise ValueError(
            "The number of results and labels should be the same. Got {} results and {} labels".format(
                len(results_list), len(label_list)
            )
        )

    # Set the plot style
    set_plot_style()

    # Initializing plot
    fig, axs = plt.subplots(3, 3, figsize=(16, 10))
    fig.suptitle(title, fontsize=16)

    # CL plot
    for y_coordinates_i, result_i, label_i in zip(
        y_coordinates_list, results_list, label_list
    ):
        axs[0, 0].plot(
            y_coordinates_i,
            result_i["cl_distribution"],
            label=label_i + rf" $C_L$: {result_i['cl']:.2f}",
        )
    axs[0, 0].set_title(rf"$C_L$ Distribution")
    axs[0, 0].set_xlabel(r"Spanwise Position $y/b$")
    axs[0, 0].set_ylabel(r"Lift Coefficient $C_L$")
    axs[0, 0].legend()

    # CD plot
    for y_coordinates_i, result_i, label_i in zip(
        y_coordinates_list, results_list, label_list
    ):
        axs[0, 1].plot(
            y_coordinates_i,
            result_i["cd_distribution"],
            label=label_i + rf" $C_D$: {result_i['cd']:.2f}",
        )
    axs[0, 1].set_title(r"$C_D$ Distribution")
    axs[0, 1].set_xlabel(r"Spanwise Position $y/b$")
    axs[0, 1].set_ylabel(r"Drag Coefficient $C_D$")
    axs[0, 1].legend()

    # Gamma plot
    for y_coordinates_i, result_i, label_i in zip(
        y_coordinates_list, results_list, label_list
    ):
        axs[0, 2].plot(y_coordinates_i, result_i["gamma_distribution"], label=label_i)
    axs[0, 2].set_title(r"$\Gamma$ Distribution")
    axs[0, 2].set_xlabel(r"Spanwise Position $y/b$")
    axs[0, 2].set_ylabel(r"Circulation $\Gamma$")
    axs[0, 2].legend()

    # Geometric Alpha plot
    for y_coordinates_i, result_i, label_i in zip(
        y_coordinates_list, results_list, label_list
    ):
        axs[1, 0].plot(
            y_coordinates_i, np.rad2deg(result_i["alpha_geometric"]), label=label_i
        )
    axs[1, 0].set_title(r"$\alpha$ Geometric")
    axs[1, 0].set_xlabel(r"Spanwise Position $y/b$")
    axs[1, 0].set_ylabel(r"Angle of Attack $\alpha$ (deg)")
    axs[1, 0].legend()

    # Calculated/Corrected Alpha plot
    for y_coordinates_i, result_i, label_i in zip(
        y_coordinates_list, results_list, label_list
    ):
        axs[1, 1].plot(
            y_coordinates_i, np.rad2deg(result_i["alpha_at_ac"]), label=label_i
        )
    axs[1, 1].set_title(r"$\alpha$ result (corrected to aerodynamic center)")
    axs[1, 1].set_xlabel(r"Spanwise Position $y/b$")
    axs[1, 1].set_ylabel(r"Angle of Attack $\alpha$ (deg)")
    axs[1, 1].legend()

    # Uncorrected Alpha plot
    for y_coordinates_i, result_i, label_i in zip(
        y_coordinates_list, results_list, label_list
    ):
        axs[1, 2].plot(
            y_coordinates_i,
            np.rad2deg(result_i["alpha_uncorrected"]),
            label=label_i,
        )
    axs[1, 2].set_title(r"$\alpha$ Uncorrected (if VSM, at the control point)")
    axs[1, 2].set_xlabel(r"Spanwise Position $y/b$")
    axs[1, 2].set_ylabel(r"Angle of Attack $\alpha$ (deg)")
    axs[1, 2].legend()

    for y_coordinates_i, result_i, label_i in zip(
        y_coordinates_list, results_list, label_list
    ):
        axs[2, 0].plot(
            y_coordinates_i,
            [force[0] for force in result_i["F_distribution"]],
            label=label_i + rf" $\sum{{F_x}}$: {result_i['Fx']:.2f}N",
        )
    axs[2, 0].set_title(f"Force in x direction")
    axs[2, 0].set_xlabel(r"Spanwise Position $y/b$")
    axs[2, 0].set_ylabel(r"$F_x$")
    axs[2, 0].legend()

    # Force in y
    for y_coordinates_i, result_i, label_i in zip(
        y_coordinates_list, results_list, label_list
    ):
        axs[2, 1].plot(
            y_coordinates_i,
            [force[1] for force in result_i["F_distribution"]],
            label=label_i + rf" $\sum{{F_y}}$: {result_i['Fy']:.2f}N",
        )
    axs[2, 1].set_title(f"Force in y direction")
    axs[2, 1].set_xlabel(r"Spanwise Position $y/b$")
    axs[2, 1].set_ylabel(r"$F_y$")
    axs[2, 1].legend()

    # Force in z
    for y_coordinates_i, result_i, label_i in zip(
        y_coordinates_list, results_list, label_list
    ):
        axs[2, 2].plot(
            y_coordinates_i,
            [force[2] for force in result_i["F_distribution"]],
            label=label_i + rf" $\sum{{F_z}}$: {result_i['Fz']:.2f}N",
        )
    axs[2, 2].set_title(f"Force in z direction")
    axs[2, 2].set_xlabel(r"Spanwise Position $y/b$")
    axs[2, 2].set_ylabel(r"$F_z$")
    axs[2, 2].legend()

    plt.tight_layout()

    # Ensure the figure is fully rendered
    fig.canvas.draw()

    # saving plot
    if is_save:
        save_plot(fig, save_path, title, data_type)

    # showing plot
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
    gamma_distribution = np.zeros((len(angle_range), len(wing_aero.panels)))
    cl_distribution = np.zeros((len(angle_range), len(wing_aero.panels)))
    cd_distribution = np.zeros((len(angle_range), len(wing_aero.panels)))
    cs_distribution = np.zeros((len(angle_range), len(wing_aero.panels)))
    reynolds_number = np.zeros(len(angle_range))
    # initialize the gamma with None
    gamma = None
    for i, angle_i in enumerate(angle_range):
        if angle_type == "angle_of_attack":
            angle_of_attack = np.deg2rad(angle_i)
        elif angle_type == "side_slip":
            side_slip = np.deg2rad(angle_i)
        else:
            raise ValueError(
                "angle_type should be either 'angle_of_attack' or 'side_slip'"
            )

        # Set the inflow conditions
        wing_aero.va = (
            np.array(
                [
                    np.cos(angle_of_attack) * np.cos(side_slip),
                    np.sin(side_slip),
                    np.sin(angle_of_attack),
                ]
            )
            * Umag,
            yaw_rate,
        )

        results = solver.solve(wing_aero, gamma_distribution=gamma)
        cl[i] = results["cl"]
        cd[i] = results["cd"]
        cs[i] = results["cs"]
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
        angle_of_attack: angle of attack | default: 0
        side_slip: side slip angle | default: 0
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

    # Set the plot style
    set_plot_style()

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
        label_list[i] += f" Re = {1e-5*reynolds_number:.0f}e5"

    # Grabbing additional data from literature
    if literature_path_list is not None:
        for literature_path in literature_path_list:
            CL, CD, angle = np.loadtxt(
                literature_path, delimiter=",", skiprows=1, unpack=True
            )
            polar_data_list.append([angle, CL, CD])

    # Initializing plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 14))

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
    axs[0, 0].set_title(r"$C_L$ vs {}".format(angle_type))
    axs[0, 0].set_xlabel(r"{}[deg]".format(angle_type))
    axs[0, 0].set_ylabel(r"$C_L$")
    axs[0, 0].legend()

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
    axs[0, 1].set_title(r"$C_D$ vs {}".format(angle_type))
    axs[0, 1].set_xlabel(r"{}[deg]".format(angle_type))
    axs[0, 1].set_ylabel(r"$C_D$")
    axs[0, 1].legend()

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
            axs[1, 0].plot(
                polar_data[0],
                polar_data[3],
                label=label,
                linestyle=linestyle,
                marker=marker,
                markersize=markersize,
            )
    axs[1, 0].set_title(r"$C_S$ vs {}".format(angle_type))
    axs[1, 0].set_xlabel(r"{}[deg]".format(angle_type))
    axs[1, 0].set_ylabel(r"$C_S$")
    axs[1, 0].legend()

    # CL-CD plot
    for i, (polar_data, label) in enumerate(zip(polar_data_list, label_list)):
        if i < n_solvers:
            linestyle = "-"
            marker = "*"
            markersize = 7
        else:
            linestyle = "-"
            marker = "."
            markersize = 5
        axs[1, 1].plot(
            polar_data[2],
            polar_data[1],
            label=label,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
        )
        if max(polar_data[1]) > 10 or max(polar_data[2]) > 10:
            axs[1, 1].set_ylim([-0.5, 2])
            axs[1, 1].set_xlim([-0.2, 0.5])
    axs[1, 1].set_title(r"$C_L$ vs $C_D$ (over {} range)".format(angle_type))
    axs[1, 1].set_xlabel(r"$C_D$")
    axs[1, 1].set_ylabel(r"$C_L$")
    axs[1, 1].legend()

    # Ensure the figure is fully rendered
    fig.canvas.draw()

    # saving plot
    if is_save:
        save_plot(fig, save_path, title, data_type)

    # showing plot
    if is_show:
        show_plot(fig, dpi=110)
