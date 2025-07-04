import logging
import numpy as np
import matplotlib.pyplot as plt
import time as time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from VSM.plot_styling import set_plot_style
from VSM.plotting import save_plot


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
        filaments = panel.compute_filaments_for_plotting()
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

    return fig, ax


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
