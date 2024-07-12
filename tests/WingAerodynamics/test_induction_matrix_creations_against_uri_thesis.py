import pytest
import numpy as np
import logging
import pprint
from copy import deepcopy
from VSM.Solver import Solver
from VSM.Panel import Panel
from VSM.WingAerodynamics import WingAerodynamics
from VSM.WingGeometry import Wing


import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_path)
from tests.WingAerodynamics.test_wing_aero_object_against_create_geometry_general import (
    # create_geometry_general,
    create_geometry_from_wing_object,
)
from tests.utils import (
    generate_coordinates_el_wing,
    generate_coordinates_rect_wing,
    generate_coordinates_curved_wing,
    asserting_all_elements_in_list_dict,
    asserting_all_elements_in_list_list_dict,
    create_ring_from_wing_object,
)
from tests.thesis_functions_oriol_cayon import (
    vec_norm,
    dot_product,
    vector_projection,
    update_Gamma_single_ring,
    velocity_induced_single_ring_semiinfinite,
    velocity_induced_bound_2D,
    velocity_induced_single_ring_semiinfinite_nocore,
    create_geometry_general,
    update_Gamma_single_ring,
)


def thesis_induction_matrix_creation(
    ringvec,
    controlpoints,
    rings,
    Uinf,
    Gamma0,
    data_airf,
    conv_crit,
    model,
):
    """
    Solve the VSM or LLM by finding the distribution of Gamma

    Parameters
    ----------
    ringvec : List of dictionaries containing the vectors that define each ring
    controlpoints :List of dictionaries with the variables needed to define each wing section
    rings : List of list with the definition of each vortex filament
    Uinf : Wind speed velocity vector
    data_airf : 2D airfoil data with alpha, Cl, Cd, Cm
    recalc_alpha : True if you want to recalculate the induced angle of attack at 1/4 of the chord (VSM)
    Gamma0 : Initial Guess of Gamma
    model : VSM: Vortex Step method/ LLT: Lifting Line Theory

    Returns
    -------
    F: Lift, Drag and Moment at each section
    Gamma: Gamma at each section
    aero_coeffs: alpha, cl, cd, cm at each wing section

    """

    nocore = False  # To shut down core corrections input True
    # Initialization of the parameters
    velocity_induced = []
    u = 0
    v = 0
    w = 0
    N = len(rings)
    Gammaini = Gamma0
    Gamma = np.zeros(N)
    GammaNew = Gammaini
    Lift = np.zeros(N)
    Drag = np.zeros(N)
    Ma = np.zeros(N)
    alpha = np.zeros(N)
    cl = np.zeros(N)
    cd = np.zeros(N)
    cm = np.zeros(N)
    MatrixU = np.empty((N, N))
    MatrixV = np.empty((N, N))
    MatrixW = np.empty((N, N))
    U_2D = np.zeros((N, 3))

    # Number of iterations and convergence criteria
    Niterations = conv_crit["Niterations"]
    errorlimit = conv_crit["error"]
    ConvWeight = conv_crit["Relax_factor"]
    converged = False
    rho = 1.225

    coord_cp = [controlpoints[icp]["coordinates"] for icp in range(N)]
    chord = [controlpoints[icp]["chord"] for icp in range(N)]
    airf_coord = [controlpoints[icp]["airf_coord"] for icp in range(N)]

    for icp in range(N):

        if model == "VSM":
            # Velocity induced by a infinte bound vortex with Gamma = 1
            U_2D[icp] = velocity_induced_bound_2D(ringvec[icp])

        for jring in range(N):
            rings[jring] = update_Gamma_single_ring(rings[jring], 1, 1)
            # Calculate velocity induced by a ring to a control point
            velocity_induced = velocity_induced_single_ring_semiinfinite(
                rings[jring], coord_cp[icp], model, vec_norm(Uinf)
            )
            # If CORE corrections are deactivated
            if nocore == True:
                # Calculate velocity induced by a ring to a control point
                velocity_induced = velocity_induced_single_ring_semiinfinite_nocore(
                    rings[jring], coord_cp[icp], model
                )

            # AIC Matrix
            MatrixU[icp, jring] = velocity_induced[0]
            MatrixV[icp, jring] = velocity_induced[1]
            MatrixW[icp, jring] = velocity_induced[2]

    return MatrixU, MatrixV, MatrixW


def test_induction_matrix_creation():

    n_panels = 2
    # N = number of SECTIONS
    N = n_panels + 1
    max_chord = 1
    span = 2.36
    AR = span**2 / (np.pi * span * max_chord / 4)
    dist = "cos"
    coord = generate_coordinates_el_wing(max_chord, span, N, dist)
    Atot = max_chord / 2 * span / 2 * np.pi
    logging.info(f"N: {N}")
    logging.info(f"len(coord): {len(coord)}")

    Umag = 20
    aoa = 5.7106 * np.pi / 180
    Uinf = np.array([np.cos(aoa), 0, np.sin(aoa)]) * Umag
    # Uinf = np.array([np.sqrt(0.99),0,0.1])

    conv_crit = {"Niterations": 1500, "error": 1e-5, "Relax_factor": 0.05}

    Gamma0 = np.zeros(N - 1)

    ring_geo = "5fil"
    model = "LLT"

    alpha_airf = np.arange(-10, 30)
    data_airf = np.zeros((len(alpha_airf), 4))
    data_airf[:, 0] = alpha_airf
    data_airf[:, 1] = alpha_airf / 180 * np.pi * 2 * np.pi
    data_airf[:, 2] = alpha_airf * 0
    data_airf[:, 3] = alpha_airf * 0

    #########################
    ### GEOMETRY CREATION ###
    #########################

    ### THESIS ###
    # Define system of vorticity
    controlpoints, rings, bladepanels, ringvec, coord_L = create_geometry_general(
        coord, Uinf, N, ring_geo, model
    )

    ### NEW ###
    ## Elliptical Wing
    core_radius_fraction = 1e-20  # only value I could find
    wing = Wing(n_panels, "unchanged")
    for idx in range(int(len(coord) / 2)):
        wing.add_section(coord[2 * idx], coord[2 * idx + 1], ["inviscid"])
    wing_aero = WingAerodynamics([wing])

    wing_aero.va = Uinf
    # Generate geometry from wing object
    new_controlpoints, new_rings, new_wingpanels, new_ringvec, new_coord_L = (
        create_geometry_from_wing_object(wing_aero, model)
    )

    #### Check if geometry input is the same
    # Check lengths
    assert np.allclose(len(controlpoints), n_panels)
    assert np.allclose(len(rings), n_panels)
    assert np.allclose(len(wing_aero.panels), n_panels)
    assert np.allclose(len(ringvec), n_panels)
    assert np.allclose(len(coord_L), n_panels)
    assert np.allclose(len(new_controlpoints), n_panels)
    assert np.allclose(len(new_rings), n_panels)
    assert np.allclose(len(new_wingpanels), n_panels)
    assert np.allclose(len(new_ringvec), n_panels)
    assert np.allclose(len(new_coord_L), n_panels)

    # check items in the dictionaries
    asserting_all_elements_in_list_dict(controlpoints, new_controlpoints)
    asserting_all_elements_in_list_list_dict(rings, new_rings)
    asserting_all_elements_in_list_dict(bladepanels, new_wingpanels)
    asserting_all_elements_in_list_dict(ringvec, new_ringvec)
    assert np.allclose(coord_L, new_coord_L, atol=1e-5)

    asserting_all_elements_in_list_dict(new_controlpoints, controlpoints)
    asserting_all_elements_in_list_list_dict(new_rings, rings)
    asserting_all_elements_in_list_dict(new_wingpanels, bladepanels)
    asserting_all_elements_in_list_dict(new_ringvec, ringvec)
    assert np.allclose(new_coord_L, coord_L, atol=1e-5)

    #############################
    ### CREATING AIC MATRICES ###
    #############################

    ### THESIS ###
    asserting_all_elements_in_list_list_dict(rings, new_rings)

    MatrixU, MatrixV, MatrixW = thesis_induction_matrix_creation(
        deepcopy(ringvec),
        deepcopy(controlpoints),
        deepcopy(rings),
        deepcopy(Uinf),
        deepcopy(Gamma0),
        deepcopy(data_airf),
        deepcopy(conv_crit),
        deepcopy(model),
    )
    asserting_all_elements_in_list_list_dict(rings, new_rings)

    ### NEW ###
    # TODO: this method should be properly tested against the old code and analytics
    def calculate_AIC_matrices(self, model, core_radius_fraction):
        """Calculates the AIC matrices for the given aerodynamic model

        Args:
            model (str): The aerodynamic model to be used, either VSM or LLT

        Returns:
            MatrixU (np.array): The x-component of the AIC matrix
            MatrixV (np.array): The y-component of the AIC matrix
            MatrixW (np.array): The z-component of the AIC matrix
            U_2D (np.array): The 2D velocity induced by a bound vortex
        """

        n_panels = self.n_panels
        AIC_x = np.empty((n_panels, n_panels))
        AIC_y = np.empty((n_panels, n_panels))
        AIC_z = np.empty((n_panels, n_panels))

        if model == "VSM":
            evaluation_point = "control_point"
        elif model == "LLT":
            evaluation_point = "aerodynamic_center"
        else:
            raise ValueError("Invalid aerodynamic model type, should be VSM or LLT")

        rings = create_ring_from_wing_object(self, 1)

        for icp, panel_icp in enumerate(self.panels):

            for jring, panel_jring in enumerate(self.panels):
                # TODO: get this to work
                ### OLD
                velocity_induced = panel_jring.calculate_velocity_induced_horseshoe(
                    getattr(panel_icp, evaluation_point),
                    gamma=1,
                    core_radius_fraction=core_radius_fraction,
                    model=model,
                )
                velocity_induced = velocity_induced_single_ring_semiinfinite(
                    rings[jring],
                    getattr(panel_icp, evaluation_point),
                    model,
                    np.linalg.norm(self.va),
                )
                ##################

                # AIC Matrix
                AIC_x[icp, jring] = velocity_induced[0]
                AIC_y[icp, jring] = velocity_induced[1]
                AIC_z[icp, jring] = velocity_induced[2]

                # Only apply correction term when dealing with same horshoe vortex (see p.27 Uri Thesis)
                if icp == jring:
                    if evaluation_point != "aerodynamic_center":  # if VSM and not LLT
                        # CORRECTION TERM (S.T.Piszkin and E.S.Levinsky,1976)
                        # Not present in classic LLT, added to allow for "arbitrary" (3/4c) control point location [37].
                        U_2D = panel_jring.calculate_velocity_induced_bound_2D(
                            getattr(panel_icp, evaluation_point),
                            gamma=1,
                            core_radius_fraction=core_radius_fraction,
                        )
                        AIC_x[icp, jring] -= U_2D[0]
                        AIC_y[icp, jring] -= U_2D[1]
                        AIC_z[icp, jring] -= U_2D[2]

        return AIC_x, AIC_y, AIC_z

    AIC_x, AIC_y, AIC_z = calculate_AIC_matrices(wing_aero, model, core_radius_fraction)

    # Check if the matrices are the same
    assert np.allclose(MatrixU, AIC_x, atol=1e-8)
    assert np.allclose(MatrixV, AIC_y, atol=1e-8)
    assert np.allclose(MatrixW, AIC_z, atol=1e-8)
    # assert np.allclose(MatrixV, AIC_x, atol=1e-8)
