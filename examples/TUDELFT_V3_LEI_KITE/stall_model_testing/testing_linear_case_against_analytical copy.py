"""
This is the same test as described in p.10 and 11
of Simonet et al. (2024) 
URL: https://doi.org/10.21203/rs.3.rs-3955527/v1
"""

import numpy as np
import logging
import matplotlib.pyplot as plt

import pandas as pd
from pathlib import Path
from VSM.WingGeometry import Wing
from VSM.BodyAerodynamics import BodyAerodynamics
from VSM.Solver import Solver
from VSM.plotting import plot_polars, plot_distribution
from VSM.interactive import interactive_plot


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
save_folder = (
    Path(PROJECT_DIR)
    / "examples"
    / "TUDELFT_V3_LEI_KITE"
    / "stall_model_testing"
    / "results"
    / "testing_simonet_artificial_viscosity_vs_no_stall"
)

# Create a wing object with specific properties, accepting the standard default spanwise direction
wing = Wing(n_panels=20, spanwise_panel_distribution="linear")


# Add sections to the wing, here only the tip-sections are specified and and an "inviscid" airfoil model is chosen
AR = 5
span = 5
chord = 1
wing_area = span * chord
print(f"AR should be 5 and ={span ** 2 / wing_area:.2f} ")

wing.add_section([0, span / 2, 0], [1, span / 2, 0], ["inviscid"])
wing.add_section([0, -span / 2, 0], [1, -span / 2, 0], ["inviscid"])

# Initialize wing aerodynamics
wing_aero = BodyAerodynamics([wing])

#
# reynolds_number = density * va_mag * max_chord / mu
density = 1
mu = 2e-5
max_chord = 1
Umag = 20
reynolds_number = density * Umag * max_chord / mu
print(f"Re should be 1e6 and ={reynolds_number/1e6:.2f}e6")


# Operating Conditions
# Umag = .. # set such that Reynolds number is 1e6
AR = 5.0
side_slip = 0
yaw_rate = 0
alpha_range = np.linspace(-10, 20, 4)

# Define inflow conditions
Umag = 20  # Magnitude of the inflow velocity
aoa = 30  # Angle of attack in degrees
aoa = np.deg2rad(aoa)  # Convert angle of attack to radians
vel_app = np.array([np.cos(aoa), 0, np.sin(aoa)]) * Umag  # Define the inflow vector
yaw_rate = 0
wing_aero.va = vel_app, yaw_rate  # Set the inflow conditions

# one can also set the inflow condition without specifying the yaw rate
wing_aero.va = vel_app


def classical_prandtl_CL(alpha_rad, AR=5.0):
    """
    Classical Prandtl lifting-line formula for lift coefficient:
      CL = 2 * pi * AR / (2 + AR) * alpha (in radians),
    valid for small angles and an untwisted, constant airfoil distribution.

    For reference, eq. (28) or any standard text might differ slightly in constants.
    Adjust if your reference has a different formula.

    Args:
        alpha_rad (float): Angle of attack in radians
        AR (float): Aspect ratio

    Returns:
        float: The theoretical lift coefficient from Prandtl's LLT
    """
    return (2.0 * np.pi * AR / (2.0 + AR)) * alpha_rad


def build_linear_test_wing(
    n_panels=50, AR=5.0, spanwise_distribution="linear", twist_deg=0.0
):
    """
    Build a simple wing instance with aspect ratio AR, no taper or elliptical chord,
    for a linear "test" geometry. Optionally add a constant or linear twist.

    Args:
        n_panels (int): Number of spanwise panels
        AR (float): Aspect ratio = b^2 / S
        spanwise_distribution (str): e.g. "linear"
        twist_deg (float or tuple): If float, we apply a uniform twist across the span.
                                   If a (root_twist, tip_twist) tuple, we do a linear variation.

    Returns:
        Wing: a Wing object suitable for BodyAerodynamics
    """
    # Suppose we choose a semispan b/2 = 1. Then total span b=2, area S = b^2/AR => 4/AR,
    # so chord = S/b = (4/AR)/2 = 2/AR
    # This is purely arbitrary; you can choose different scaling.
    b = 2.0
    S = b * b / AR  # => 4/AR
    c = S / b  # => 2/AR

    # We'll define root at y=-b/2, tip at y=b/2, and the chord is c at every section.
    # If you want an elliptical chord distribution or linear twist, you can code that.
    # For simplicity, let's do a constant chord = c, optional linear twist from root to tip.

    if isinstance(twist_deg, (float, int)):
        root_twist_deg = twist_deg
        tip_twist_deg = twist_deg
    else:
        root_twist_deg, tip_twist_deg = twist_deg

    # Create the Wing object
    wing = Wing(n_panels, spanwise_distribution)
    # We'll manually define each "rib" from y_i to y_{i+1}.
    # A "rib" = [LE, TE, airfoil_data], where:
    #   LE, TE are the 3D coords of leading/trailing edges
    #   airfoil_data might be something like ["inviscid", [0.0]] or your custom approach.

    y_coords = np.linspace(-b / 2, b / 2, n_panels + 1)
    for i in range(n_panels):
        y1 = y_coords[i]
        y2 = y_coords[i + 1]
        # chord might be constant => c
        chord_i = c
        # twist might vary linearly
        frac = (y1 + b / 2) / b  # from 0 at root to 1 at tip
        twist_here_deg = root_twist_deg + frac * (tip_twist_deg - root_twist_deg)
        twist_here_rad = np.deg2rad(twist_here_deg)

        # We'll define leading edge at x=0, z=0 for simplicity, and the chord extends in x direction
        # If twist != 0, we rotate around the y axis? For a simple 2D approach, let's just do no dihedral => z=0
        LE_x = 0.0
        LE_y = y1
        LE_z = 0.0
        TE_x = chord_i * np.cos(twist_here_rad)
        TE_y = y1
        TE_z = -chord_i * np.sin(twist_here_rad)

        # We can store some simple airfoil data => "inviscid" or "lei_airfoil_breukels", etc.
        # For a linear test, "inviscid" might suffice => 2 * pi * alpha
        airfoil_data = ["inviscid", []]

        wing.add_section(
            np.array([LE_x, LE_y, LE_z]), np.array([TE_x, TE_y, TE_z]), airfoil_data
        )

    return wing


def run_linear_test_case():
    """
    Demonstrate a 'linear test case' with AR=5, no twist, and compare
    the computed lift coefficient vs. angle of attack with the classical
    Prandtl formula.
    """
    logging.basicConfig(level=logging.INFO)

    # Wing geometry
    AR = 5.0
    n_panels = 50
    twist_deg = 0.0  # or (0.0, 10.0) if you want a linear twist from 0 to 10 deg
    wing = build_linear_test_wing(n_panels, AR, "linear", twist_deg=twist_deg)

    # Create BodyAerodynamics
    body_aero = BodyAerodynamics([wing])  # minimal constructor => no bridle
    # We can set some default inflow, e.g. free-stream velocity magnitude=1, alpha=0 for now
    Umag = 1.0
    alpha0 = 0.0
    side_slip = 0.0
    yaw_rate = 0.0
    body_aero.va_initialize(Umag, alpha0, side_slip, yaw_rate)

    # Create a solver
    solver = Solver(
        gamma_loop_type="gamma_loop",  # or "non_linear" if you want your nonlinear approach
        max_iterations=1000,
        allowed_error=1e-5,
        relaxation_factor=0.01,
    )

    # Angles of attack to test
    alpha_deg_array = np.linspace(-5, 15, 6)  # e.g. -5,0,5,10,15 deg
    CL_numeric = []
    CL_prandtl = []

    for alpha_deg in alpha_deg_array:
        alpha_rad = np.deg2rad(alpha_deg)
        # re-initialize inflow
        body_aero.va_initialize(Umag, alpha_deg, side_slip, yaw_rate)

        # Solve
        results = solver.solve(body_aero)
        # Extract total lift coefficient from results
        # Suppose results has "CL" => results["CL"]
        CL_num = results["CL"]  # adjust if your code uses a different key
        CL_numeric.append(CL_num)

        # Compare with classical Prandtl
        CL_theory = classical_prandtl_CL(alpha_rad, AR=AR)
        CL_prandtl.append(CL_theory)

        logging.info(
            f"Alpha = {alpha_deg:.1f} deg => CL_numerical = {CL_num:.4f}, CL_theory = {CL_theory:.4f}"
        )

    # Plot
    plt.figure()
    plt.plot(alpha_deg_array, CL_numeric, "ro-", label="3DN3LM numeric")
    plt.plot(alpha_deg_array, CL_prandtl, "b--", label="Prandtl LLT theory")
    plt.xlabel("Angle of attack (deg)")
    plt.ylabel("Lift Coefficient (CL)")
    plt.legend()
    plt.title("Linear test case: AR=5 comparison with Prandtl's LLT")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_linear_test_case()
