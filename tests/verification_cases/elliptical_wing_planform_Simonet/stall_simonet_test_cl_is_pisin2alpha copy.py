"""
This is the same test as described in p.10 and 11
of Simonet et al. (2024) in Ch.4.1
URL: https://doi.org/10.21203/rs.3.rs-3955527/v1
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
from copy import deepcopy

import pandas as pd
from pathlib import Path
from scipy.optimize import root

from VSM.WingGeometry import Wing
from VSM.BodyAerodynamics import BodyAerodynamics
from VSM.Solver import Solver
from VSM.plotting import plot_polars, plot_distribution
from VSM.interactive import interactive_plot
from VSM.plot_styling import set_plot_style, plot_on_ax


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent

save_folder = (
    Path(PROJECT_DIR)
    / "tests"
    / "verification_cases"
    / "elliptical_wing_planform"
    / "results"
)


def cosine_spacing(min, max, n_points):
    """
    Create an array with cosine spacing, from min to max values, with n points

    """
    mean = (max + min) / 2
    amp = (max - min) / 2

    return mean + amp * np.cos(np.linspace(np.pi, 0, n_points))


def offcenter_cosine_space(a, b, n):
    """
    Generate n points in [a, b] using an off-center 'cosine-like' distribution,
    often used so that the control points (or boundaries) are shifted from the endpoints.

    Formula (for k = 0..n-1):
        y_k = 0.5 * [ (b + a) - (b - a) * cos( (2k + 1)*pi / (2n) ) ]

    This ensures that the points are not exactly at a or b, but 'off-center.'
    """
    k = np.arange(n)
    return 0.5 * ((b + a) - (b - a) * np.cos((2 * k + 1) * np.pi / (2 * n)))


def generate_coordinates_el_wing(max_chord, span, N, dist):
    """
    Generate 2D coordinates (leading edge and trailing edge) for N spanwise sections
    of an 'elliptical' wing. The distribution of spanwise stations can be:
        - 'lin' : uniform spacing
        - 'cos' : standard cosine spacing
        - 'off_center' : off-center distribution (Phillips & Snyder style)
             panel midpoints are cosine distributed, rather then the edges.

    Args:
        max_chord (float): maximum chord (root chord).
        span (float): total wing span.
        N (int): number of sections (the code returns 2*N points).
        dist (str): distribution type: 'lin', 'cos', or 'off_center'.

    Returns:
        coord (np.ndarray of shape (2*N, 3)):
            For each i in [0..N-1], the leading edge is coord[2*i]
            and the trailing edge is coord[2*i+1].
    """
    coord = np.empty((2 * N, 3))
    start = span * 1e-5  # a tiny offset to avoid exactly -span/2 or +span/2

    # 1) Choose the y-array distribution
    if dist == "uniform":
        # Uniform spacing
        y_arr = np.linspace(-span / 2 + start, span / 2 - start, N)
    elif dist == "cosine":
        # Cosine spacing
        y_arr = cosine_spacing(-span / 2 + start, span / 2 - start, N)
    elif dist == "off_center_cosine":
        # Off-center (Phillips & Snyder style)
        y_arr = offcenter_cosine_space(-span / 2 + start, span / 2 - start, N)
    else:
        raise ValueError(
            f"Invalid dist='{dist}'. Must be 'lin', 'cos', or 'off_center'."
        )

    # 2) Elliptical chord distribution for each station
    #    c(y) = 2 * sqrt(1 - (y/(span/2))^2 ) * max_chord / 2
    c_arr = 2.0 * np.sqrt(1.0 - (y_arr / (span / 2.0)) ** 2) * (max_chord / 2.0)

    # 3) For each spanwise station, define LE & TE in x-direction:
    #    Leading edge at x = -0.25*c, trailing edge at x= +0.75*c
    #    y stays at y_arr[i], z=0 (flat wing)
    for i in range(N):
        coord[2 * i, :] = [-0.25 * c_arr[i], y_arr[i], 0.0]  # LE
        coord[2 * i + 1, :] = [0.75 * c_arr[i], y_arr[i], 0.0]  # TE

    return coord


def flip_created_coord_in_pairs(coord):
    # Reshape the array into pairs
    reshaped = coord.reshape(-1, 2, 3)

    # Reverse the order of the pairs
    flipped = np.flip(reshaped, axis=0)

    # Flatten back to the original shape
    return flipped.reshape(-1, 3)


def build_elliptical_wing_instance(
    max_chord: float, span: float, N: int, dist: str, airfoil_data=None
) -> Wing:
    """
    Build a Wing instance with an elliptical planform using the function
    'generate_coordinates_el_wing' from 'thesis_functions'. Each spanwise
    section is added to the Wing with the specified airfoil data.

    Args:
        max_chord (float): Maximum chord (e.g. at wing root).
        span (float): Total wing span.
        N (int): Number of spanwise sections (the Wing will have N panels).
        dist (str): Spanwise distribution, e.g. "cosine" or "uniform".
        airfoil_data (list or None): The airfoil specification for each section.
            For instance, ["inviscid"] or ["polar_data", data_airf].

    Returns:
        Wing: A Wing object representing the elliptical planform.
    """
    if airfoil_data is None:
        # Default to a simple inviscid airfoil if not provided
        airfoil_data = ["cl_is_pisin2alpha"]

    # 1) Generate the leading/trailing edge coordinates
    coord = generate_coordinates_el_wing(max_chord, span, N, dist)

    # 2) Optionally 'flip' or reorder these coordinate pairs
    coord_left_to_right = flip_created_coord_in_pairs(deepcopy(coord))

    # 3) Create the Wing object
    wing_instance = Wing(N, "unchanged")

    # 4) Add each section to the Wing
    #    The array 'coord_left_to_right' has shape (2*N, 3),
    #    so each panel i uses rows (2*i) and (2*i+1).
    for idx in range(N):
        LE_point = coord_left_to_right[2 * idx]
        TE_point = coord_left_to_right[2 * idx + 1]
        # airfoil_data can be the same for each panel, or you can adapt logic here
        wing_instance.add_section(LE_point, TE_point, airfoil_data)

    logging.info(
        f"Built elliptical wing with max_chord={max_chord}, span={span}, N={N}, dist={dist}"
    )
    return wing_instance


def elliptical_aspect_ratio(span, max_chord):
    return 4 * span / (np.pi * max_chord)


# ==========================
#   Prandlts semi-analytical solution
# ==========================


def prandtl_semianalytical_simonet(alpha, AR):
    """
    Solve eq. (31) for A1, then compute CL(α) = π * AR * A1.

    Equations from reference [22]:
        (30)  CL(α) = π * AR * A1
        (31)  A1    = (1 / AR) * sin( 2α - 2 arctan(A1) )

    Args:
        alpha (float): Angle of attack in deg.
        AR (float): Aspect ratio of the wing.

    Returns:
        float: The lift coefficient CL(α) from the semi-analytical approach.
    """

    alpha = np.deg2rad(alpha)

    # Define the residual of eq. (31): f(A1) = A1 - (1/AR)*sin(2α - 2arctan(A1)) = 0
    def eq_31_res(A1):
        return A1 - ((1.0 / AR) * np.sin(2.0 * alpha - 2.0 * np.arctan(A1)))

    # Solve for A1 using a root-finding method. We'll pick a small positive initial guess.
    sol = root(eq_31_res, 0.1)
    if not sol.success:
        raise RuntimeError(
            f"Failed to solve eq. (31) for alpha={np.rad2deg(alpha)} rad, AR={AR}"
        )

    A1 = sol.x[0]

    # Once A1 is found, eq. (30) => CL = π * AR * A1
    CL = np.pi * AR * A1
    return CL


# ==========================
# Settings
# AR_chattot = 12.7
# n_chattot = 51
# AR_simonet = 5
# n_simonet = 50
# ==========================

span_chattot = 10  # gives AR=12.7
span_simonet = 3.927  # gives AR=5
span = span_simonet
chord = 1
AR = elliptical_aspect_ratio(span, chord)
print(f"Chattot AR=12.7 and our AR={AR:.2f} ")
density = 1
mu = 2e-5
Umag = 20
print(f"Re should be 1e6 and ={density * Umag * chord / mu/1e6:.2f}e6")
n_panels = 50
alpha_range = np.linspace(0, 90, 20)
print(f"alpha_range: {alpha_range}")
# ==========================
# Building the wing instance
# ==========================
body_aero_uniform = BodyAerodynamics(
    [
        build_elliptical_wing_instance(
            max_chord=chord,
            span=span,
            N=n_panels,
            dist="uniform",
            airfoil_data=["cl_is_pisin2alpha"],
        )
    ]
)
body_aero_cosine = BodyAerodynamics(
    [
        build_elliptical_wing_instance(
            max_chord=chord,
            span=span,
            N=n_panels,
            dist="cosine",
            airfoil_data=["cl_is_pisin2alpha"],
        )
    ]
)
body_aero_off_center_cosine = BodyAerodynamics(
    [
        build_elliptical_wing_instance(
            max_chord=chord,
            span=span,
            N=n_panels,
            dist="off_center_cosine",
            airfoil_data=["cl_is_pisin2alpha"],
        )
    ]
)
# ==========================
#   INTERACTIVE PLOT
# ==========================
# angle_of_attack = 10
# side_slip = 0
# yaw_rate = 0
# interactive_plot(
#     body_aero_uniform,
#     vel=Umag,
#     angle_of_attack=angle_of_attack,
#     side_slip=side_slip,
#     yaw_rate=yaw_rate,
#     is_with_aerodynamic_details=True,
# )
# interactive_plot(
#     body_aero_cosine,
#     vel=Umag,
#     angle_of_attack=angle_of_attack,
#     side_slip=side_slip,
#     yaw_rate=yaw_rate,
#     is_with_aerodynamic_details=True,
# )
# interactive_plot(
#     body_aero_off_center_cosine,
#     vel=Umag,
#     angle_of_attack=angle_of_attack,
#     side_slip=side_slip,
#     yaw_rate=yaw_rate,
#     is_with_aerodynamic_details=True,
# )


# ==========================
# Plotting CL-alpha (Fig. 7)
# ==========================
solver_base = Solver(
    # max_iterations=5e3,
    allowed_error=1e-6,
    # relaxation_factor=0.05,
    gamma_loop_type="non_linear",
)

df = pd.DataFrame(
    columns=[
        "alpha",
        "CL_VSM_uniform",
        "CL_VSM_cosine",
        "CL_VSM_off_center_cosine",
        "CL_analytical_simonet",
    ]
)

for alpha in alpha_range:
    body_aero_uniform.va_initialize(Umag, alpha, side_slip=0, yaw_rate=0)
    results_VSM_uniform = solver_base.solve(body_aero_uniform, gamma_distribution=None)
    CL_VSM_uniform = results_VSM_uniform["cl"]

    body_aero_cosine.va_initialize(Umag, alpha, side_slip=0, yaw_rate=0)
    results_VSM_cosine = solver_base.solve(body_aero_cosine, gamma_distribution=None)
    CL_VSM_cosine = results_VSM_cosine["cl"]

    body_aero_off_center_cosine.va_initialize(Umag, alpha, side_slip=0, yaw_rate=0)
    results_VSM_off_center_cosine = solver_base.solve(
        body_aero_off_center_cosine, gamma_distribution=None
    )
    CL_VSM_off_center_cosine = results_VSM_off_center_cosine["cl"]

    CL_analytical_simonet = prandtl_semianalytical_simonet(alpha, AR)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "alpha": [alpha],
                    "CL_VSM_uniform": [CL_VSM_uniform],
                    "CL_VSM_cosine": [CL_VSM_cosine],
                    "CL_VSM_off_center_cosine": [CL_VSM_off_center_cosine],
                    "CL_analytical_simonet": [CL_analytical_simonet],
                }
            ),
        ],
        ignore_index=True,
    )


set_plot_style()
fig, ax = plt.subplots(figsize=(8, 6))

plot_on_ax(
    ax,
    df["alpha"],
    np.pi * np.sin(2 * np.deg2rad(df["alpha"])),
    label="2D Analytical",
    color="black",
)
plot_on_ax(
    ax,
    df["alpha"],
    df["CL_analytical_simonet"],
    label="Analytical Simonet",
    color="black",
    marker="o",
    linestyle="None",
)
plot_on_ax(ax, df["alpha"], df["CL_VSM_uniform"], label="VSM uniform")
plot_on_ax(ax, df["alpha"], df["CL_VSM_cosine"], label="VSM cosine")
plot_on_ax(
    ax,
    df["alpha"],
    df["CL_VSM_off_center_cosine"],
    label="VSM off-center_cosine",
    is_with_grid=True,
)

ax.grid(True)
ax.legend()
ax.set_xlabel(r"$\alpha$ [$^{\circ}$]")
ax.set_ylabel(r"$C_L$")
ax.set_title(r"$C_L$ vs $\alpha$")
plt.tight_layout()
plt.savefig(Path(save_folder) / f"CL_vs_alpha_AR_{AR:.1f}.pdf")
# plt.show()


# ==========================
# Plotting Circulation (Fig. 8)
# ==========================
solver_base = Solver(
    # max_iterations=5e3,
    allowed_error=1e-6,
    # relaxation_factor=0.05,
    gamma_loop_type="non_linear",
)

df = pd.DataFrame(
    columns=[
        "alpha",
        "gama_VSM_uniform",
        "gama_VSM_cosine",
        "gama_VSM_off_center_cosine",
    ]
)


set_plot_style()
y_coordinates = [panel.control_point[1] for panel in body_aero_uniform.panels]
fig, ax = plt.subplots(figsize=(8, 6))
fva = 0
for alpha in [40, 50, 60, 70, 80]:
    body_aero_uniform.va_initialize(Umag, alpha, side_slip=0, yaw_rate=0)
    results_VSM_uniform = solver_base.solve(body_aero_uniform, gamma_distribution=None)
    gamma_VSM_uniform = results_VSM_uniform["gamma_distribution"]
    plot_on_ax(ax, y_coordinates, gamma_VSM_uniform, label=r"$\alpha$" + f"={alpha}")

ax.grid(True)
ax.legend()
ax.set_xlabel(r"$y$ [m]")
ax.set_ylabel(r"$\Gamma$")
ax.set_title(r"$\Gamma$ vs $y$")
plt.tight_layout()
plt.savefig(Path(save_folder) / f"Gamma_vs_alpha_AR_{AR:.1f}_fva_{fva:.1f}.pdf")
