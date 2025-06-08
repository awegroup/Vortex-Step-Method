from typing import List, Tuple, Dict, Any
import numpy as np
import plotly.graph_objects as go
from VSM.Solver import Solver


def add_panel_edges(fig: go.Figure, panel: Any, is_first: bool, is_last: bool) -> None:
    """Add panel edges to the figure with different thicknesses."""
    leadinge_edge_line_color = "black"
    leadinge_edge_line_width = 15
    # Leading edge
    if is_first:
        # Include 1-side edge
        fig.add_trace(
            go.Scatter3d(
                x=[panel.LE_point_1[0], panel.TE_point_1[0]],
                y=[panel.LE_point_1[1], panel.TE_point_1[1]],
                z=[panel.LE_point_1[2], panel.TE_point_1[2]],
                mode="lines",
                line=dict(
                    color=leadinge_edge_line_color, width=leadinge_edge_line_width
                ),
                name="Leading Edge",
                showlegend=False,
            )
        )
    elif is_last:
        # Include 1-side edge
        fig.add_trace(
            go.Scatter3d(
                x=[panel.LE_point_2[0], panel.TE_point_2[0]],
                y=[panel.LE_point_2[1], panel.TE_point_2[1]],
                z=[panel.LE_point_2[2], panel.TE_point_2[2]],
                mode="lines",
                line=dict(
                    color=leadinge_edge_line_color, width=leadinge_edge_line_width
                ),
                name="Leading Edge",
                showlegend=False,
            )
        )
    # Standard leading edge
    fig.add_trace(
        go.Scatter3d(
            x=[panel.LE_point_1[0], panel.LE_point_2[0]],
            y=[panel.LE_point_1[1], panel.LE_point_2[1]],
            z=[panel.LE_point_1[2], panel.LE_point_2[2]],
            mode="lines",
            line=dict(color=leadinge_edge_line_color, width=leadinge_edge_line_width),
            name="Leading Edge",
            showlegend=is_first,
        )
    )

    # Trailing edge
    fig.add_trace(
        go.Scatter3d(
            x=[panel.TE_point_1[0], panel.TE_point_2[0]],
            y=[panel.TE_point_1[1], panel.TE_point_2[1]],
            z=[panel.TE_point_1[2], panel.TE_point_2[2]],
            mode="lines",
            line=dict(color="black", width=2),
            name="Trailing Edge",
            showlegend=is_first,
        )
    )

    # Side edges
    for i, points in enumerate(
        [(panel.LE_point_1, panel.TE_point_1), (panel.LE_point_2, panel.TE_point_2)]
    ):
        fig.add_trace(
            go.Scatter3d(
                x=[points[0][0], points[1][0]],
                y=[points[0][1], points[1][1]],
                z=[points[0][2], points[1][2]],
                mode="lines",
                line=dict(color="black", width=0.8),
                name=(
                    "Side Edge" if is_first and i == 0 else None
                ),  # Legend only for the first side edge
                showlegend=is_first and i == 0,
            )
        )


def add_panel_surface(fig: go.Figure, panel: Any, is_first: bool) -> None:

    fig.add_trace(
        go.Mesh3d(
            x=[
                panel.LE_point_1[0],
                panel.LE_point_2[0],
                panel.TE_point_2[0],
                panel.TE_point_1[0],
            ],
            y=[
                panel.LE_point_1[1],
                panel.LE_point_2[1],
                panel.TE_point_2[1],
                panel.TE_point_1[1],
            ],
            z=[
                panel.LE_point_1[2],
                panel.LE_point_2[2],
                panel.TE_point_2[2],
                panel.TE_point_1[2],
            ],
            i=[0, 1, 2, 3],
            j=[1, 2, 3, 0],
            k=[2, 3, 0, 1],
            color="lightgrey",
            opacity=0.6,
            name="Panel Surface",
            showlegend=is_first,
        )
    )


def add_filaments(fig: go.Figure, panel: Any, is_first: bool = False) -> None:
    """Add aerodynamic visualization details to the figure."""
    filaments = panel.calculate_filaments_for_plotting()
    colors = ["blue", "blue", "blue", "blue", "blue"]
    names = ["Bound Vortex", "Side 1", "Side 2", "Wake 1", "Wake 2"]

    for filament, color, name in zip(filaments, colors, names):
        x1, x2, _ = filament
        fig.add_trace(
            go.Scatter3d(
                x=[x1[0], x2[0]],
                y=[x1[1], x2[1]],
                z=[x1[2], x2[2]],
                mode="lines",
                line=dict(color=color, width=2),
                name=name,
                showlegend=is_first,
            )
        )


def add_control_and_aero_centers(fig: go.Figure, panels: List[Any]) -> None:
    """Add control points and aerodynamic centers to the figure."""
    control_points = np.array([panel.control_point for panel in panels])
    aerodynamic_centers = np.array([panel.aerodynamic_center for panel in panels])

    fig.add_trace(
        go.Scatter3d(
            x=control_points[:, 0],
            y=control_points[:, 1],
            z=control_points[:, 2],
            mode="markers",
            marker=dict(color="blue", size=4),
            name="Control Points (3/4 chord)",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=aerodynamic_centers[:, 0],
            y=aerodynamic_centers[:, 1],
            z=aerodynamic_centers[:, 2],
            mode="markers",
            marker=dict(color="red", size=4),
            name="Aerodynamic Centers (1/4 chord)",
        )
    )


def add_aerodynamic_vectors(
    fig: go.Figure,
    panel: List[Any],
    is_first: bool,
    force_vector: np.ndarray,
    scale: float = 0.5,
    max_force: float = 1.0,
):
    """
    Add aerodynamic force vectors to a given panel on the plot.

    Args:
        fig: Plotly figure
        panel: Panel object
        force_of_panel: Aerodynamic force vector
        scale: Scaling factor for the force vector
    """
    # Compute vector endpoint
    aerodynamic_center = panel.aerodynamic_center
    vector = (force_vector / max_force) * scale * 0.5
    vector_endpoint = aerodynamic_center + vector

    # Add the vector as a line
    fig.add_trace(
        go.Scatter3d(
            x=[aerodynamic_center[0], vector_endpoint[0]],
            y=[aerodynamic_center[1], vector_endpoint[1]],
            z=[aerodynamic_center[2], vector_endpoint[2]],
            mode="lines",
            line=dict(color="red", width=4),
            name="Aerodynamic Vector",
            showlegend=is_first,
        )
    )
    # Add the arrowhead as a cone
    sizeref = np.linalg.norm(vector) / 10  # Adjust for the size of the arrowhead
    if np.isnan(sizeref) or sizeref <= 0:
        sizeref = 1  # or some reasonable default

    fig.add_trace(
        go.Cone(
            x=[vector_endpoint[0]],
            y=[vector_endpoint[1]],
            z=[vector_endpoint[2]],
            u=[vector[0]],
            v=[vector[1]],
            w=[vector[2]],
            sizemode="absolute",
            sizeref=sizeref,  # Adjust for the size of the arrowhead
            anchor="tip",
            colorscale=[[0, "red"], [1, "red"]],
            showscale=False,
            name="Arrowhead",
            showlegend=False,
        )
    )

    # add aerodynamic centers
    fig.add_trace(
        go.Scatter3d(
            x=[aerodynamic_center[0]],
            y=[aerodynamic_center[1]],
            z=[aerodynamic_center[2]],
            mode="markers",
            marker=dict(color="red", size=2),
            name="Aerodynamic Centers",
            showlegend=is_first,
        )
    )


def create_3D_plot(
    fig: go.Figure,
    wing_aero: object,
    forces_of_panels: List[np.ndarray],
    is_with_aerodynamic_details: bool,
) -> go.Figure:
    """
    Creates an interactive 3D plot of wing geometry using Plotly.

    Args:
        wing_aero: WingAerodynamics object containing panels
        title: Title of the plot
        is_with_aerodynamic_details: Boolean to show/hide aerodynamic visualization details

    Returns:
        plotly.graph_objects.Figure
    """
    panels = wing_aero.panels

    chord_average = np.max([panel.chord for panel in panels])
    max_force = np.max([np.linalg.norm(force) for force in forces_of_panels])

    if is_with_aerodynamic_details:
        add_control_and_aero_centers(fig, panels)

    # Add geometric elements
    for i, panel in enumerate(panels):
        is_first = False
        is_last = False
        if i == 0:
            is_first = True
        elif i == len(panels) - 1:
            is_last = True

        if is_with_aerodynamic_details:
            add_filaments(fig, panel, is_first)

        add_panel_edges(fig, panel, is_first, is_last)
        add_panel_surface(fig, panel, is_first)
        add_aerodynamic_vectors(
            fig,
            panel,
            is_first,
            np.array(forces_of_panels[i]),
            scale=chord_average,
            max_force=max_force,
        )

    return fig


def calculate_kite_geometry_ranges(panels: List[Any]) -> Tuple[Dict[str, float], float]:
    """Calculate axis ranges and tick spacing."""
    all_points = []
    for panel in panels:
        all_points.extend(
            [panel.LE_point_1, panel.LE_point_2, panel.TE_point_1, panel.TE_point_2]
        )
    all_points = np.array(all_points)

    kite_geometry_ranges = {
        "x": [np.min(all_points[:, 0]), np.max(all_points[:, 0])],
        "y": [np.min(all_points[:, 1]), np.max(all_points[:, 1])],
        "z": [np.min(all_points[:, 2]), np.max(all_points[:, 2])],
    }
    return kite_geometry_ranges


def calculate_axis_parameters_from_fig(
    fig: go.Figure,
) -> Tuple[Dict[str, float], float]:
    """Calculate axis ranges and tick spacing based on the figure's data."""
    # Extract all data points from the figure traces
    all_points = []
    for trace in fig.data:
        if isinstance(trace, go.Scatter3d):
            all_points.extend(zip(trace.x, trace.y, trace.z))

    # Convert to numpy array for calculations
    all_points = np.array(all_points)

    # Calculate ranges for each axis
    ranges = {
        "x": [np.min(all_points[:, 0]), np.max(all_points[:, 0])],
        "y": [np.min(all_points[:, 1]), np.max(all_points[:, 1])],
        "z": [np.min(all_points[:, 2]), np.max(all_points[:, 2])],
    }

    # Calculate tick spacing based on the smallest range
    min_range = min(
        ranges["x"][1] - ranges["x"][0],
        ranges["y"][1] - ranges["y"][0],
        ranges["z"][1] - ranges["z"][0],
    )
    tick_spacing = min_range / 10

    return ranges, tick_spacing


def update_fig_layout(fig: go.Figure, panels: List[Any], title: str) -> go.Figure:

    # Calculate axis parameters
    kite_geometry_ranges = calculate_kite_geometry_ranges(panels)
    ranges, tick_spacing = calculate_axis_parameters_from_fig(fig)
    padding = tick_spacing * 0.5  # padding of half a tick spacing

    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(
                range=[ranges["x"][0] - padding, ranges["x"][1] + padding],
                showgrid=False,
                showbackground=False,
                zeroline=False,
                showline=True,  # Add the axis line
                linecolor="black",  # Color of the axis line
                linewidth=2,  # Thickness of the axis line
                tickvals=[
                    kite_geometry_ranges["x"][0],
                    kite_geometry_ranges["x"][1],
                ],  # Only show ticks at the ends
                ticktext=[
                    f"{kite_geometry_ranges['x'][0]:.2f}",
                    f"{kite_geometry_ranges['x'][1]:.2f}",
                ],  # Label for the ticks
            ),
            yaxis=dict(
                range=[ranges["y"][0] - padding, ranges["y"][1] + padding],
                showgrid=False,
                showbackground=False,
                zeroline=False,
                showline=True,  # Add the axis line
                linecolor="black",  # Color of the axis line
                linewidth=2,  # Thickness of the axis line
                tickvals=[
                    kite_geometry_ranges["y"][0],
                    kite_geometry_ranges["y"][1],
                ],  # Only show ticks at the ends
                ticktext=[
                    f"{kite_geometry_ranges['y'][0]:.2f}",
                    f"{kite_geometry_ranges['y'][1]:.2f}",
                ],  # Label for the ticks
            ),
            zaxis=dict(
                range=[ranges["z"][0] - padding, ranges["z"][1] + padding],
                showgrid=False,
                showbackground=False,
                zeroline=False,
                showline=True,  # Add the axis line
                linecolor="black",  # Color of the axis line
                linewidth=2,  # Thickness of the axis line
                tickvals=[
                    kite_geometry_ranges["z"][0],
                    kite_geometry_ranges["z"][1],
                ],  # Only show ticks at the ends
                ticktext=[
                    f"{kite_geometry_ranges['z'][0]:.2f}",
                    f"{kite_geometry_ranges['z'][1]:.2f}",
                ],  # Label for the ticks
            ),
            aspectmode="data",  # Allow different axis lengths
            aspectratio=dict(x=1, y=1, z=1),  # Keep aspect ratio 1:1:1
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5),
            ),
            bgcolor="white",
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=0,
            title="Legend (click items to hide)",
            font=dict(size=12),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


# Example running_VSM with AoA parameter
def running_VSM(
    wing_aero: object,
    vel: float,
    angle_of_attack: float,
    side_slip: float,
    yaw_rate: float,
) -> Dict[str, Any]:
    """Run the Vortex Source Method on the given wing_aero object, based on AoA."""
    # setting va

    wing_aero.va_initialize(vel, angle_of_attack, side_slip, yaw_rate)

    # configuring the solver
    VSM_solver = Solver()
    # solving
    results = VSM_solver.solve(wing_aero)
    return results


# Function to add text annotations
def add_text_annotations(fig, x, y, title: str = "Your Text Here"):
    """
    Add custom text annotations to the plot (e.g., in the top-right corner).
    """
    fig.add_annotation(
        x=x,  # x-position (1 means far right)
        y=y,  # y-position (1 means top)
        xref="paper",  # Use relative positioning on the paper
        yref="paper",  # Use relative positioning on the paper
        text=title,
        showarrow=False,  # No arrow pointing to the text
        font=dict(size=16, color="black"),
        align="right",  # Align text to the right
        # borderpad=4,  # Padding around the text
        bgcolor="rgba(255, 255, 255, 0.7)",  # Background color with some transparency
        # bordercolor="black",  # Border color around the text
        # borderwidth=1,  # Border width around the text
        opacity=0.7,  # Text opacity
    )


def add_case_information(
    fig,
    panels,
    vel: float,
    angle_of_attack: float,
    side_slip: float,
    yaw_rate: float,
    results: Dict[str, Any],
) -> go.Figure:
    kite_geometry_ranges = calculate_kite_geometry_ranges(panels)
    chord = kite_geometry_ranges["x"][1] - kite_geometry_ranges["x"][0]
    span = kite_geometry_ranges["y"][1] - kite_geometry_ranges["y"][0]
    height = kite_geometry_ranges["z"][1] - kite_geometry_ranges["z"][0]

    add_text_annotations(fig, x=1, y=1.00, title=f"velocity = {vel:.2f} [m/s]")
    add_text_annotations(
        fig, x=1, y=0.97, title=f"angle of attack = {angle_of_attack:.2f} [deg]"
    )
    add_text_annotations(fig, x=1, y=0.94, title=f"side slip = {side_slip:.2f} [deg]")
    add_text_annotations(fig, x=1, y=0.91, title=f"yaw rate = {yaw_rate:.2f} [rad/s]")
    add_text_annotations(fig, x=1, y=0.88, title="-------------------")
    add_text_annotations(
        fig,
        x=1,
        y=0.85,
        title=f"CL = {results['cl']:.2f}",
    )
    add_text_annotations(
        fig,
        x=1,
        y=0.82,
        title=f"CD = {results['cd']:.2f}",
    )
    add_text_annotations(
        fig,
        x=1,
        y=0.79,
        title=f"CS = {results['cs']:.2f}",
    )

    add_text_annotations(fig, x=1, y=0.76, title="-------------------")
    add_text_annotations(
        fig,
        x=1,
        y=0.73,
        title=f"span = {span:.3f} [m]",
    )
    add_text_annotations(
        fig,
        x=1,
        y=0.70,
        title=f"chord = {chord:.3f} [m]",
    )
    add_text_annotations(
        fig,
        x=1,
        y=0.67,
        title=f"height = {height:.3f} [m]",
    )

    return fig


# Function to update the plot based on AoA change
def update_plot(
    fig,
    wing_aero: object,
    vel: float,
    angle_of_attack: float,
    side_slip: float,
    yaw_rate: float,
    is_with_aerodynamic_details: bool,
    title: str,
):
    # Update AoA and rerun VSM
    results = running_VSM(wing_aero, vel, angle_of_attack, side_slip, yaw_rate)

    # Populating the plot with updated aerodynamic details
    fig.data = []  # Clear the previous plot data
    fig = create_3D_plot(
        fig, wing_aero, results["F_distribution"], is_with_aerodynamic_details
    )
    fig = add_case_information(
        fig, wing_aero.panels, vel, angle_of_attack, side_slip, yaw_rate, results
    )
    fig = update_fig_layout(fig, wing_aero.panels, title)


# Define the interactive plot function with slider for AoA
def interactive_plot(
    wing_aero: object,
    vel: float = 10,
    angle_of_attack: float = 10,
    side_slip: float = 0,
    yaw_rate: float = 0,
    title: str = "Interactive plot",
    is_with_aerodynamic_details: bool = False,
    save_path: str = None,
    is_save: bool = False,
    filename="wing_geometry",
    is_show: bool = True,
):
    """
    Creates and optionally saves multiple views of the wing geometry with interactive AoA slider.
    """

    # Create the figure with a default orientation
    fig = go.Figure()
    fig.update_layout(
        scene_camera=dict(
            eye=dict(
                x=-4, y=0, z=0
            )  # Adjust the x, y, and z values for the desired view angles
        )
    )

    # Add initial plot based on initial AoA
    update_plot(
        fig,
        wing_aero,
        vel,
        angle_of_attack,
        side_slip,
        yaw_rate,
        is_with_aerodynamic_details,
        title,
    )

    # Save or show the plot if requested
    if is_save:
        if save_path is None:
            save_path = "."

        # Save as HTML for interactivity
        fig.write_html(f"{save_path}/{filename}.html")

        # Save as static image
        fig.write_image(f"{save_path}/{filename}.png")

    if is_show:
        fig.show()
