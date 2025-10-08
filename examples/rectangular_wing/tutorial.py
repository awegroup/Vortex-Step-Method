# Step 1: Import necessary libraries
import numpy as np
from VSM.core.Solver import Solver
from VSM.core.WingGeometry import Wing
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.plot_geometry_matplotlib import plot_geometry
from VSM.plot_geometry_plotly import interactive_plot


def main():
    """
    Rectangular Wing Example: Manual Construction and Visualization with VSM

    This script demonstrates how to manually construct a rectangular wing, set up the aerodynamic model,
    define inflow conditions, and visualize the geometry using both Matplotlib and Plotly.

    Workflow:
    ---------
    1. **Wing Construction**:
        - Create a `Wing` instance.
        - Add tip sections with inviscid airfoil data.
        - The airfoil data can be inviscid or from other supported models (see AirfoilAerodynamics).
        - Sections are defined by leading and trailing edge points and a polar data array.

    2. **Aerodynamic Model Setup**:
        - Initialize a `BodyAerodynamics` object with the constructed wing.
        - This object handles aerodynamic calculations for the body and can support multiple wings.

    3. **Inflow Condition Definition**:
        - Set the apparent wind vector and yaw rate.
        - The inflow can be set as a vector or as magnitude and angle of attack.

    4. **Geometry Visualization**:
        - Plot the geometry using Matplotlib (`plot_geometry`).
        - Create an interactive 3D plot using Plotly (`interactive_plot`).

    Returns:
        None
    """

    # 1. Wing Construction
    span = 20
    n_panels = 20
    wing = Wing(n_panels=n_panels, spanwise_panel_distribution="uniform")

    # Use inviscid airfoil data for both sections
    """
    You can use various airfoil aerodynamic models. See AirfoilAerodynamics for details.
    Supported types include:

       - breukels_regression:
           t: Tube diameter (non-dimensionalized by chord)
           kappa: Maximum camber height (non-dimensionalized by chord)
       - neuralfoil:
           dat_file_path: Path to airfoil .dat file (x, y columns)
           model_size: NeuralFoil model size (e.g., "xxxlarge")
           xtr_lower: Lower transition location (0=forced, 1=free)
           xtr_upper: Upper transition location
           n_crit: Critical amplification factor (see guidelines below)
             n_crit guidelines:
               Sailplane:           12–14
               Motorglider:         11–13
               Clean wind tunnel:   10–12
               Average wind tunnel: 9   (standard "e^9 method")
               Dirty wind tunnel:   4–8
       - polars:
           csv_file_path: Path to polar CSV file (columns: alpha [rad], cl, cd, cm)
       - masure_regression:
           t, eta, kappa, delta, lambda, phi: Regression parameters
     ---------------------------------------------------------------
    """
    alpha_range = np.arange(-10, 31, 1)
    alpha_rad = np.deg2rad(alpha_range)
    cl = 2 * np.pi * alpha_rad
    cd = np.zeros_like(alpha_rad)
    cm = np.zeros_like(alpha_rad)
    polar_data = np.column_stack([alpha_rad, cl, cd, cm])

    # Add sections to the wing at the tips
    """   The sections are defined from the leading edge to the trailing edge.
    The leading edge is at [0, span/2, 0] and the trailing edge is at [1, span/2, 0]
    The solver sorts the sections automatically based on the spanwise direction.
    """
    wing.add_section([0, span / 2, 0], [1, span / 2, 0], polar_data)
    wing.add_section([0, -span / 2, 0], [1, -span / 2, 0], polar_data)

    # Step 2: Initialize BodyAerodynamics
    """ BodyAerodynamics is the main class that handles the aerodynamic calculations for the body.
    It takes a list of Wing objects and can handle multiple wings.
    """
    body_aero = BodyAerodynamics([wing])

    # 3. Inflow Condition Definition
    Umag = 20  # Inflow velocity magnitude
    aoa_deg = 30  # Angle of attack in degrees
    aoa_rad = np.deg2rad(aoa_deg)
    vel_app = np.array([np.cos(aoa_rad), 0, np.sin(aoa_rad)]) * Umag
    body_aero.va = vel_app

    # Optionally, set the inflow condition without specifying the yaw rate
    body_aero.va = vel_app

    # Step 4: Plotting the geometry using Matplotlib
    plot_geometry(
        body_aero,
        title="rectangular_wing_geometry",
        data_type=".pdf",
        save_path=".",
        is_save=False,
        is_show=True,
    )

    # Plot use Plotly
    interactive_plot(
        body_aero,
        vel=Umag,
        angle_of_attack=aoa_rad,
        is_with_aerodynamic_details=True,
    )

    # Step 5: Setting up the solver
    """
     You can configure the solver with various parameters. Here are some common options:
       aerodynamic_model_type: "VSM" (default) or "LLT"
       max_iterations: Maximum number of iterations (default: 5000)
       allowed_error: Convergence tolerance (default: 1e-6)
       relaxation_factor: Relaxation factor for iterative solver (default: 0.01)
       core_radius_fraction: Core radius as a fraction of chord (default: 1e-20)
       gamma_loop_type: Type of gamma update loop ("base" by default)
       gamma_initial_distribution_type: Initial gamma distribution ("elliptical" by default)
       is_only_f_and_gamma_output: If True, only force and gamma outputs are returned
       is_with_viscous_drag_correction: If True, applies viscous drag correction
       reference_point: Reference point for moment calculations (default: [0, 0, 0])
       mu: Dynamic viscosity (default: 1.81e-5)
       rho: Air density (default: 1.225)
    """
    solver_VSM = Solver(aerodynamic_model_type="VSM")
    solver_LLT = Solver(aerodynamic_model_type="LLT")

    # Step 6: Run the simulation
    """
    The `.solve` method takes the BodyAerodynamics object and returns a dictionary of results.

    The output dictionary contains the following keys:

    - Global wing aerodynamics:
        - "Fx", "Fy", "Fz": Total aerodynamic force components (global 3D sum)
        - "Mx", "My", "Mz": Total aerodynamic moment components (global 3D sum)
        - "lift", "drag", "side": Total lift, drag, and side force for the wing
        - "cl", "cd", "cs": Global lift, drag, and side force coefficients
        - "cmx", "cmy", "cmz": Global moment coefficients

    - Local panel aerodynamics:
        - "cl_distribution", "cd_distribution", "cs_distribution": Spanwise distributions of lift, drag, and side force coefficients
        - "F_distribution", "M_distribution": Spanwise distributions of force and moment vectors

    - Additional info:
        - "cfx_distribution", "cfy_distribution", "cfz_distribution": Spanwise force coefficient distributions
        - "cmx_distribution", "cmy_distribution", "cmz_distribution": Spanwise moment coefficient distributions
        - "alpha_at_ac": Effective angle of attack at aerodynamic center
        - "alpha_uncorrected": Uncorrected angle of attack
        - "alpha_geometric": Geometric angle of attack
        - "gamma_distribution": Circulation distribution
        - "area_all_panels": Area of each panel
        - "projected_area": Total projected wing area
        - "wing_span": Wing span
        - "aspect_ratio_projected": Projected aspect ratio
        - "Rey": Reynolds number
        - "center_of_pressure": Center of pressure location
        - "panel_cp_locations": Spanwise locations of panel centers of pressure

    These outputs allow you to analyze both global and local aerodynamic properties of the wing.
    """

    results_VSM = solver_VSM.solve(body_aero)
    results_LLT = solver_LLT.solve(body_aero)
    print(f"\n VSM Results CL: {results_VSM['cl']:.3f}, CD: {results_VSM['cd']:.3f}")
    print(f"\n LLT Results CL: {results_LLT['cl']:.3f}, CD: {results_LLT['cd']:.3f}")


if __name__ == "__main__":
    main()
