"""Parametric design of a LEI kite profile"""

import numpy as np
from math import *
import matplotlib.pyplot as plt
from pathlib import Path


# Cubic polynomial function
def interpolation3(P1, P2, t1, t2, n=100):
    # start and end point coordinates
    x1 = P1[0]
    x2 = P2[0]
    y1 = P1[1]
    y2 = P2[1]

    # slope at end points
    s1, s2 = np.tan(t1), np.tan(t2)

    # AX = Y
    A = np.array(
        [
            [x1**3, x1**2, x1, 1],
            [x2**3, x2**2, x2, 1],
            [3 * x1**2, 2 * x1, 1, 0],
            [3 * x2**2, 2 * x2, 1, 0],
        ]
    )

    Y = np.array([y1, y2, s1, s2])
    X = np.linalg.solve(A, Y)
    x_list = np.linspace(x1, x2, n)
    y_list = X[0] * x_list**3 + X[1] * x_list**2 + X[2] * x_list + X[3]

    # 2D array of the x and y coordinate of each point
    points = np.transpose(np.vstack((x_list, y_list)))
    return points


# Cubic Bezier curve function
def cubic_bezier(P0, P1, P2, P3, t):
    # Bezier curve x and y coordinates
    x_bezier = (
        (1 - t) ** 3 * P0[0]
        + 3 * (1 - t) ** 2 * t * P1[0]
        + 3 * (1 - t) * t**2 * P2[0]
        + t**3 * P3[0]
    )
    y_bezier = (
        (1 - t) ** 3 * P0[1]
        + 3 * (1 - t) ** 2 * t * P1[1]
        + 3 * (1 - t) * t**2 * P2[1]
        + t**3 * P3[1]
    )

    # Bezier derivatives in x and y to retrieve the slope
    dx_bezier = (
        3 * (1 - t) ** 2 * (P1[0] - P0[0])
        + 6 * (1 - t) * t * (P2[0] - P1[0])
        + 3 * t**2 * (P3[0] - P2[0])
    )
    dy_bezier = (
        3 * (1 - t) ** 2 * (P1[1] - P0[1])
        + 6 * (1 - t) * t * (P2[1] - P1[1])
        + 3 * t**2 * (P3[1] - P2[1])
    )
    slope = np.divide(
        dy_bezier, dx_bezier, out=np.full_like(dy_bezier, 1000), where=dx_bezier != 0
    )

    # 2D array of the x and y coordinate of each point
    points = np.transpose(np.vstack((x_bezier, y_bezier)))
    return points, slope


# Dynamic seam angle position function
def LE_seam_angle(tube_size, c_x, c_y):
    for angle in range(0, 91):
        seam_a = np.radians(angle)
        radius = tube_size / 2
        s = (np.pi / 2) - seam_a
        P1 = [radius * (1 - np.cos(seam_a)), radius * np.sin(seam_a)]
        P2 = [c_x, c_y]
        poly = interpolation3(P1, P2, s, 0)
        maximum = round(
            max(poly[:-1, 1]), 4
        )  # For error prevention disregard last point

        # If highest point is below c_y
        if maximum <= c_y:
            print(angle - 10)
            return np.radians(angle - 10)


# LEI kite profile coordinates and control points
def LEI_airfoil(
    tube_size,
    c_x,
    c_y,
    TE_angle,
    TE_cam_tension,
    LE_tension,
    ##############################
    # ---- FIXED VALUES ----
    ##############################
    TE_tension=0.2,
    e=0.0005,
    LE_fillet=0.06,
    LE_config=3,
    manual_D_fillet1=0.04,
    manual_D_fillet2=0.04,
    manual_fillet_a=40,
    fillet_automatic=True,
):
    """
    Generate a complete Leading Edge Inflatable (LEI) kite airfoil profile geometry.

    Creates a parametric LEI airfoil using cubic Bézier curves for smooth transitions
    between the leading edge tube, airfoil surfaces, trailing edge, and fillet region.
    Designed for aerodynamic analysis and CFD preprocessing of inflatable kite profiles.

    Parameters
    ----------
    Required Design Parameters:
        tube_size : float
            t – Diameter of the inflatable leading edge tube (non-dimensionalized by chord)
        c_x : float
            η (eta) – Chordwise position of maximum camber (0 to 1); values too close to 0 may yield invalid geometry
        c_y : float
            κ (kappa) – Maximum camber height (can be negative); if ≤ tube radius, triggers flat mode
        TE_angle : float
            δ (delta) – Trailing edge reflex angle in degrees (negative = downward deflection)
        TE_cam_tension : float
            λ (lambda) – Camber tension parameter controlling rear spline curvature at max camber
        LE_tension : float
            φ (phi) – Curvature parameter for tube-to-upper-surface transition

    Optional Parameters:
        TE_tension : float, default=0.2
            Trailing edge curvature tension
        e : float, default=0.0005
            Minimum airfoil thickness (fixed to ensure physical realism)
        LE_fillet : float, default=0.06
            Size of the fillet blending the tube into the lower surface
        LE_config : int, default=3
            Leading edge Bézier control configuration:
            - 1: Move only P11 (P12 fixed)
            - 2: Move only P12 (P11 fixed)
            - 3: Move both P11 and P12 proportionally
        manual_D_fillet1 : float, default=0.04
            Tube-side Bézier control distance (only if fillet_automatic=False)
        manual_D_fillet2 : float, default=0.04
            Surface-side Bézier control distance (only if fillet_automatic=False)
        manual_fillet_a : float, default=40
            Tube intersection angle in degrees (only if fillet_automatic=False)
        fillet_automatic : bool, default=True
            Enable dynamic (automatic) calculation of fillet parameters

    Returns
    -------
    tuple
        29-element tuple containing coordinates, control points, reference data, and derivatives:
        - Geometry: LE_tube_points, P1, P11, P12, ..., fillet_points
        - Reference: origin_LE_tube, round_TE_mid, seam_a
        - Derivatives: LE_dyu_dx, LE_d2yu_dx2, circ_dyu_dx, circ_d2yu_dx2, both_array

    Flat Mode (if c_y ≤ tube_radius)
    --------------------------------
        - Leading edge is linearly interpolated to trailing edge
        - Max camber point is fixed
        - Camber tension, TE reflex, and tube diameter still influence shape
        - Ensures structural and geometric simplicity for low-camber profiles

    Notes
    -----
        - All dimensions normalized by chord length
        - Smooth transitions enforced with C1-continuous Bézier curves
        - Suitable for real-world inflatable kite design with physical constraints
    """

    radius = tube_size / 2  # Radius LE tube
    P2 = np.array([c_x, c_y])  # Max camber position
    P3 = np.array([1, 0])  # TE top side

    # Check if c_y is below tube radius
    if P2[1] < radius:
        flat = True
    else:
        flat = False

    # Normal configuration with camber
    if not flat:
        if P2[1] == radius:
            seam_a = np.radians(90)
            P1 = np.array([radius, radius])  # Seam location
            P11 = np.array([0.33 * (P2[0] - P1[0]) + P1[0], P1[1]])
            P12 = np.array([0.66 * (P2[0] - P1[0]) + P1[0], P1[1]])

        else:
            ### LE ###
            seam_a = LE_seam_angle(tube_size, c_x, c_y)  # LE_seam_angle
            P1_s = np.tan((np.pi / 2) - seam_a)  # Tangency at seam location
            P1 = np.array(
                [radius * (1 - np.cos(seam_a)), radius * np.sin(seam_a)]
            )  # Seam location
            P11_max = np.array(
                [P1[0] + (c_y - P1[1]) / P1_s, P2[1]]
            )  # P11 at c_y height

            # Compute control points P11 and P12
            # config 1: only P11 can move, P12 at P11_max (c_y)
            # config 2: only P12 can move, P11 at P11_max (c_y)
            # config 3: P11 and P12
            if LE_config == 1:
                P11 = np.array(
                    [
                        (1 - LE_tension) * P1[0] + LE_tension * P11_max[0],
                        (1 - LE_tension) * P1[1] + LE_tension * P11_max[1],
                    ]
                )
                P12 = P11_max

            if LE_config == 2:
                P11 = P11_max
                P12 = np.array(
                    [(P2[0] - P11_max[0]) * (1 - LE_tension) + P11_max[0], P2[1]]
                )

            if LE_config == 3:
                P11 = np.array(
                    [
                        (1 - LE_tension) * P1[0] + LE_tension * P11_max[0],
                        (1 - LE_tension) * P1[1] + LE_tension * P11_max[1],
                    ]
                )
                P12 = np.array(
                    [(P2[0] - P11_max[0]) * (1 - LE_tension) + P11_max[0], P2[1]]
                )

        ### TE ###
        P21 = np.array(
            [c_x + TE_cam_tension * (1 - c_x), c_y]
        )  # TE top first control point
        D_P21_TE = sqrt(
            (1 - P21[0]) ** 2 + P21[1] ** 2
        )  # distance from max camber to TE
        reflex_angle = np.radians(TE_angle) + atan(
            P21[1] / (1 - P21[0])
        )  # reflex angle
        P22 = np.array(
            [
                P3[0] - D_P21_TE * TE_tension * cos(reflex_angle),
                D_P21_TE * TE_tension * sin(reflex_angle),
            ]
        )  # TE top second control point, based on tension para related to the TE distance

    # Flat configuration if cy is below tube radius
    else:
        c_x = 4 * radius
        seam_a = np.radians(
            90 + np.degrees(np.arcsin(radius / (1 - radius)))
        )  # Seam angle
        P1 = np.array(
            [radius * (1 - np.cos(seam_a)), radius * np.sin(seam_a)]
        )  # Seam location

        # Compute slope and intercept of straight profile
        slope = (P3[1] - P1[1]) / (P3[0] - P1[0])
        intercept = P1[1] - slope * P1[0]

        P11 = np.array(
            [
                (c_x - P1[0]) * 0.33 + P1[0],
                slope * ((c_x - P1[0]) * 0.33 + P1[0]) + intercept,
            ]
        )
        P12 = np.array(
            [
                (c_x - P1[0]) * 0.66 + P1[0],
                slope * ((c_x - P1[0]) * 0.66 + P1[0]) + intercept,
            ]
        )
        P2 = np.array([c_x, slope * c_x + intercept])
        P21 = np.array(
            [
                c_x + TE_cam_tension * (1 - c_x),
                slope * (c_x + TE_cam_tension * (1 - c_x)) + intercept,
            ]
        )
        D_P21_TE = sqrt(
            (1 - P21[0]) ** 2 + P21[1] ** 2
        )  # distance from max camber to TE
        reflex_angle = np.radians(TE_angle) + atan(
            P21[1] / (1 - P21[0])
        )  # reflex angle
        P22 = np.array(
            [
                P3[0] - D_P21_TE * TE_tension * cos(reflex_angle),
                D_P21_TE * TE_tension * sin(reflex_angle),
            ]
        )  # TE top second control point, based on tension para

    # LE bezier curve
    t_LE = np.linspace(0, 1, 80)
    LE_points = cubic_bezier(P1, P11, P12, P2, t_LE)[
        0
    ]  # 2d array of points [[x1,y1],[xn, yn]]

    # TE bezier curve
    t_TE = np.linspace(0, 1, 100)
    TE_points = cubic_bezier(P2, P21, P22, P3, t_TE)[
        0
    ]  # 2d array of points [[x1,y1],[xn, yn]]

    ### TE lower side control points ###
    P5 = np.array([c_x, P2[1] - e])  # Max camber point on lower side
    P51 = np.array(
        [c_x + TE_cam_tension * (1 - c_x), P21[1] - e]
    )  # TE lower skin first control point, closest to max camber
    P4 = np.array(
        [1 - e * sin(reflex_angle), 0 - e * cos(reflex_angle)]
    )  # TE lower point
    P52 = np.array(
        [
            P4[0] - D_P21_TE * TE_tension * cos(reflex_angle),
            D_P21_TE * TE_tension * sin(reflex_angle) + P4[1],
        ]
    )  # TE lower skin 2nd control point

    ### Round TE ###
    round_TE = []  # List of round TE points
    round_TE_mid = np.array(
        [0.5 * (P4[0] + 1), 0.5 * P4[1]]
    )  # Middle point of round TE

    # Discretize the round TE
    for i in np.linspace(0, np.pi, 30):
        round_TE_point = [
            round_TE_mid[0] + e / 2 * sin(reflex_angle + i),
            round_TE_mid[1] + e / 2 * cos(reflex_angle + i),
        ]
        round_TE.append(round_TE_point)
    round_TE_points = np.array(round_TE)  # 2d array of points [[x1,y1],[xn, yn]]

    ### LE and TE lower surface and slope at P63 ###
    LE_lower_points = LE_points.copy()  # Make a copy of LE_points
    LE_lower_points[:, 1] -= e  # Subtract e from the column Y
    LE_lower_points = LE_lower_points[:-1]  # Remove the last row

    t_TE_lower = np.linspace(0, 1, 100)
    TE_lower_points_init = cubic_bezier(P5, P51, P52, P4, t_TE_lower)[
        0
    ]  # Initial TE lower points from P5 to P4

    # Find the index of the closest value
    lower_surface_init = np.vstack(
        (LE_lower_points, TE_lower_points_init)
    )  # Initial lower surface points from P4 till LE

    if fillet_automatic:
        fillet_a = np.radians(110 - np.degrees(seam_a))  # Fillet angle
        P6 = np.array(
            [radius * (1 + np.cos(fillet_a)), radius * np.sin(fillet_a)]
        )  # Fillet seam position
        index_lower = np.abs(
            lower_surface_init[:, 0] - P6[0] * 1.18
        ).argmin()  # Defines the intersection point P63
    else:
        fillet_a = np.radians(manual_fillet_a)  # Fillet angle
        P6 = np.array(
            [radius * (1 + np.cos(fillet_a)), radius * np.sin(fillet_a)]
        )  # Fillet seam position
        index_lower = np.abs(
            lower_surface_init[:, 0] - LE_fillet
        ).argmin()  # Defines the intersection point P63

    TE_lower_points = lower_surface_init[
        index_lower:
    ]  # Defining TE array based on fillet percentage

    LE_fillet_slopes = cubic_bezier(P1, P11, P12, P2, t_LE)[1]  # Array of fillet slopes
    TE_lower_slopes = cubic_bezier(P5, P51, P52, P4, t_TE_lower)[
        1
    ]  # Array of TE slopes
    P63_slope = np.hstack((LE_fillet_slopes, TE_lower_slopes))[
        index_lower
    ]  # returns the slope at P63 based on the fillet percentage

    ### LE fillet ###
    P6_s = -np.tan((np.pi / 2) - fillet_a)  # Slope at the fillet seam
    P63 = np.array(
        [TE_lower_points[0, 0], TE_lower_points[0, 1]]
    )  # Intersection between LE fillet and TE lower side

    if fillet_automatic:
        D_fillet1 = 0.55 * np.linalg.norm(
            P63 - P6
        )  # LE_fillet distance P61 closest to tube
        D_fillet2 = 0.65 * np.linalg.norm(P63 - P6)  # LE_fillet distance P62
    else:
        D_fillet1 = manual_D_fillet1
        D_fillet2 = manual_D_fillet2

    # LE fillet 1st control point (closest to LE)
    if fillet_a < 0:
        P61 = np.array(
            [
                P6[0] + D_fillet1 / sqrt(1 + P6_s**2),
                P6[1] - P6_s * (-D_fillet1 / sqrt(1 + P6_s**2)),
            ]
        )
    else:
        P61 = np.array(
            [
                P6[0] - D_fillet1 / sqrt(1 + P6_s**2),
                P6[1] + P6_s * (-D_fillet1 / sqrt(1 + P6_s**2)),
            ]
        )

    # LE fillet 2nd control point
    P62 = np.array([P63[0] - D_fillet2, P63_slope * -D_fillet2 + P63[1]])

    # LE fillet bezier curve
    t_LE_u = np.linspace(0, 1, 50)
    fillet_points = cubic_bezier(P6, P61, P62, P63, t_LE_u)[
        0
    ]  # 2d array of points [[x1,y1],[xn, yn]]

    ### LE tube ###
    circle_n_points = 80  # Number of LE tube points
    theta = np.linspace(
        np.pi - seam_a, fillet_a + np.pi * 2, circle_n_points
    )  # Array of LE tube angles
    Origin_LE_tube = [radius, 0]  # Origin of the LE tube

    x_cr = Origin_LE_tube[0] + radius * np.cos(
        theta
    )  # x location of the LE tube points
    y_cr = Origin_LE_tube[1] + radius * np.sin(
        theta
    )  # y location of the LE tube points
    LE_tube_points = np.column_stack(
        (x_cr, y_cr)
    )  # LE tube points from seam to fillet seam, 2d array of points [[x1,y1],[xn, yn]]
    both_array = np.vstack((LE_tube_points[::-1][:-1], LE_points))

    ### Curvature ###
    # front spline
    LE_dyu_dx = np.gradient(
        LE_points[:, 1], LE_points[:, 0]
    )  # First derivative (slope)
    LE_d2yu_dx2 = np.gradient(
        LE_dyu_dx[1:-1], LE_points[1:-1, 0]
    )  # Second derivative (curvature)

    # circle tube
    circ_dyu_dx = np.gradient(
        both_array[:, 1], both_array[:, 0]
    )  # First derivative (slope)
    circ_d2yu_dx2 = np.gradient(
        circ_dyu_dx, both_array[:, 0]
    )  # Second derivative (curvature)
    return (
        LE_tube_points,
        P1,
        P11,
        P12,
        LE_points,
        TE_points,
        P2,
        P21,
        P22,
        P3,
        round_TE_points,
        P4,
        P5,
        P51,
        P52,
        TE_lower_points,
        P6,
        P61,
        P62,
        P63,
        fillet_points,
        Origin_LE_tube,
        round_TE_mid,
        seam_a,
        LE_dyu_dx,
        LE_d2yu_dx2,
        circ_dyu_dx,
        circ_d2yu_dx2,
        both_array,
    )


# Boundary layer height function
def wall_height(Re):
    # Given constants
    y_plus = 1.0  # y+ value [-]
    rho = 1  # Air density [kg/m^3]
    U_inf = 1  # Free-stream velocity [m/s]

    mu = 1 / Re  # Dynamic viscosity [kg/m/s]
    c_f = 0.027 * Re ** (-1 / 7)  # Flat plate skin friction coefficient [-]
    tau_w = 0.5 * c_f * rho * U_inf**2  # Wall shear stress [kg/m/s^2]
    u_tau = sqrt(tau_w / rho)  # Friction velocity [m/s]
    yw = (y_plus * mu) / (u_tau * rho) * 1.5  # First cell layer height [m]
    return round(yw, 7)


def plot_airfoil(
    fig_file_path,
    profile_name,
    LE_tube_points,
    P1,
    P11,
    P12,
    LE_points,
    TE_points,
    P2,
    P21,
    P22,
    P3,
    round_TE_points,
    P4,
    P5,
    P51,
    P52,
    TE_lower_points,
    P6,
    P61,
    P62,
    P63,
    fillet_points,
    seam_a,
):
    fig = plt.figure(figsize=(16, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.yaxis.grid(color="gray")
    ax.xaxis.grid(color="gray")
    ax.set_axisbelow(True)

    line_t = 2
    control_s = 30
    line_type = "-"
    control = True
    cfd_fillet = False

    # LE tube
    if cfd_fillet:
        plt.plot(
            LE_tube_points[:, 0],
            LE_tube_points[:, 1],
            "-o",
            linewidth=line_t,
            markersize=0.1,
            label="circ",
        )

    # Plot LE full circle
    eta = np.linspace(0, 2 * np.pi, 100)  # array of LE tube angles
    radius = -min(LE_tube_points[:, 1])
    Origin_Circle = [radius, 0]  # origin of the LE tube
    x_cr = Origin_Circle[0] + radius * np.cos(eta)  # x location of the LE tube points
    y_cr = Origin_Circle[1] + radius * np.sin(eta)  # y location of the LE tube points
    circ_full = np.column_stack((x_cr, y_cr))  # 2d array of points [[x1,y1],[xn, yn]]
    plt.plot(
        circ_full[:, 0],
        circ_full[:, 1],
        "--",
        linewidth=line_t,
        markersize=0.1,
        color="#3776ab",
        label="Circular tube",
    )

    # Plot front spline
    plt.plot(
        LE_points[:, 0],
        LE_points[:, 1],
        line_type,
        color="#ff7f0e",
        linewidth=line_t,
        markersize=0.5,
        label="Front spline",
    )
    if control:
        control_points = np.array([P1, P11, P12, P2])
        plt.plot(
            control_points[:, 0],
            control_points[:, 1],
            "--",
            linewidth=line_t,
            color="gray",
        )
        plt.scatter(
            control_points[:, 0],
            control_points[:, 1],
            color="#ff7f0e",
            s=control_s,
            label="Control front",
        )

    # Plot Rear spline
    plt.plot(
        TE_points[:, 0],
        TE_points[:, 1],
        line_type,
        color="#2CA02C",
        linewidth=line_t,
        markersize=0.5,
        label="Rear spline",
    )
    if control:
        control_points = np.array([P2, P21, P22, P3])
        plt.plot(
            control_points[:, 0],
            control_points[:, 1],
            "--",
            linewidth=line_t,
            color="gray",
        )
        plt.scatter(
            control_points[:, 0],
            control_points[:, 1],
            color="#2CA02C",
            s=control_s,
            label="Control rear",
        )

    # Plot LE fillet
    if cfd_fillet:
        plt.plot(
            fillet_points[:, 0],
            fillet_points[:, 1],
            line_type,
            color="#D62728",
            linewidth=line_t,
            markersize=0.5,
            label="LE fillet",
        )
        if control:
            control_points = np.array([P6, P61, P62, P63])
            plt.plot(
                control_points[:, 0],
                control_points[:, 1],
                "--",
                linewidth=line_t,
                color="gray",
            )
            plt.scatter(
                control_points[:, 0],
                control_points[:, 1],
                color="#D62728",
                s=control_s,
                label="Control LE fillet",
            )

    # Plot rear lower spline
    if cfd_fillet:
        plt.plot(
            TE_lower_points[:, 0],
            TE_lower_points[:, 1],
            line_type,
            color="teal",
            linewidth=line_t,
            markersize=0.5,
            label="TE lower",
        )
        if control:
            control_points = np.array([P5, P51, P52, P4])
            plt.plot(
                control_points[:, 0],
                control_points[:, 1],
                "--",
                linewidth=line_t,
                color="gray",
            )
            plt.scatter(
                control_points[:, 0],
                control_points[:, 1],
                color="teal",
                s=control_s,
                label="Control TE lower",
            )

    # Plot round TE
    if cfd_fillet:
        plt.plot(
            round_TE_points[:, 0],
            round_TE_points[:, 1],
            "-",
            markersize=0.5,
            color="k",
            label="Round TE",
        )

    plt.scatter(
        Origin_Circle[0],
        Origin_Circle[1],
        marker="*",
        color="b",
        s=control_s,
        label="LE tube centre",
    )
    plt.scatter(
        TE_points[-1, 0],
        TE_points[-1, 1],
        marker="*",
        color="r",
        s=control_s,
        label="TE position",
    )
    plt.scatter(
        LE_points[0, 0],
        LE_points[0, 1],
        marker="*",
        color="g",
        s=control_s,
        label="Tube-canopy intersection",
    )
    plt.scatter(
        LE_points[-1, 0],
        LE_points[-1, 1],
        marker="*",
        color="k",
        s=control_s,
        label="Max. camber position",
    )

    # Reflex angle for legend
    plt.scatter(
        [0.5], [-2], marker="$r$", color="k", s=control_s - 10, label="Reflex angle"
    )

    max_y = max(LE_points[:, 1])
    min_y = min(LE_tube_points[:, 1])

    plt.xlabel("x/c [-]")
    plt.ylabel("y/c [-]")
    plt.xticks(np.arange(0, 1.05, 0.1))
    plt.yticks(
        np.arange(
            np.ceil(1.5 * min_y / 0.05) * 0.05, np.ceil(1.2 * max_y / 0.05) * 0.05, 0.05
        )
    )
    plt.axis("equal")
    plt.xlim([0, 1.05])
    plt.ylim([1.5 * min_y, 1.2 * max_y])

    plt.title(profile_name, fontsize=16, pad=5)
    plt.savefig(fig_file_path, format="png", dpi=200)  # Saving figure to results
    # plt.close()
    plt.show()


def generate_profile(
    t_val,
    eta_val,
    kappa_val,
    delta_val,
    lambda_val,
    phi_val,
    ################################
    # ---- OPTIONAL FIXED VALUES ----
    ################################
    TE_tension=0.2,
    e=0.006,
    LE_fillet=0.06,
    LE_config=3,
    manual_D_fillet1=0.04,
    manual_D_fillet2=0.04,
    manual_fillet_a=40,
    fillet_automatic=True,
):
    """
    Generate LEI airfoil profile geometry.

    This function generates a complete LEI airfoil using the parametric design
    function and returns the resulting coordinates and profile name.

    Parameters:
    ----------
    t_val : float
        Tube size parameter (tube diameter / chord)
    eta_val : float
        Maximum camber chordwise location (x/c)
    kappa_val : float
        Maximum camber height (y/c)
    delta_val : float
        Trailing edge reflex angle in degrees
    lambda_val : float
        Camber tension parameter
    phi_val : float
        Leading edge curvature parameter
    TE_tension : float, optional (default=0.2)
        Trailing edge tension parameter
    e : float, optional (default=0.006)
        Airfoil thickness parameter
    LE_fillet : float, optional (default=0.06)
        Leading edge fillet size
    LE_config : int, optional (default=3)
        Leading edge configuration (1, 2, or 3)
    manual_D_fillet1 : float, optional (default=0.04)
        Manual fillet control point distance (tube-side)
    manual_D_fillet2 : float, optional (default=0.04)
        Manual fillet control point distance (surface-side)
    manual_fillet_a : float, optional (default=40)
        Manual tube intersection angle in degrees
    fillet_automatic : bool, optional (default=True)
        Enable automatic fillet parameter calculation

    Returns:
    -------
    tuple
        (all_points, profile_name, seam_a) where:
        - all_points: numpy array of (x, y) coordinates forming the complete airfoil contour
        - profile_name: string identifier for the profile
        - seam_a: seam angle in radians

    """

    # Generate LEI airfoil geometry
    (
        LE_tube_points,
        P1,
        P11,
        P12,
        LE_points,
        TE_points,
        P2,
        P21,
        P22,
        P3,
        round_TE_points,
        P4,
        P5,
        P51,
        P52,
        TE_lower_points,
        P6,
        P61,
        P62,
        P63,
        fillet_points,
        Origin_LE_tube,
        round_TE_mid,
        seam_a,
        LE_dyu_dx,
        LE_d2yu_dx2,
        circ_dyu_dx,
        circ_d2yu_dx2,
        both_array,
    ) = LEI_airfoil(
        tube_size=t_val,
        c_x=eta_val,
        c_y=kappa_val,
        TE_angle=delta_val,
        TE_cam_tension=lambda_val,
        LE_tension=phi_val,
        TE_tension=TE_tension,
        e=e,
        LE_fillet=LE_fillet,
        LE_config=LE_config,
        manual_D_fillet1=manual_D_fillet1,
        manual_D_fillet2=manual_D_fillet2,
        manual_fillet_a=manual_fillet_a,
        fillet_automatic=fillet_automatic,
    )

    # Create profile name with parameters
    profile_name = (
        f"LEI_t{t_val:.3f}_eta{eta_val:.3f}_kappa{kappa_val:.3f}_delta{delta_val:.1f}_"
        f"lambda{lambda_val:.3f}_phi{phi_val:.3f}_e{e:.4f}"
    )

    # Combine all points into a single array following the correct order for airfoil contour
    # Order: LE_points -> TE_points (skip first to avoid duplicate) -> round_TE_points (skip first and last)
    #        -> TE_lower_points (reversed) -> fillet_points (reversed, skip last) -> LE_tube_points (reversed)
    all_points = np.vstack(
        (
            LE_points,  # Leading edge upper surface
            TE_points[1:],  # Trailing edge upper surface (skip duplicate point)
            round_TE_points[1:-1],  # Rounded trailing edge (skip first and last)
            TE_lower_points[::-1],  # Trailing edge lower surface (reversed)
            fillet_points[:-1][
                ::-1
            ],  # Fillet curve (reversed, skip last to avoid duplicate)
            LE_tube_points[::-1],  # Leading edge tube (reversed)
        )
    )

    return all_points, profile_name, seam_a


def save_profile_as_dat_file(all_points, profile_name, output_file_path, seam_a=None):
    """
    Save LEI airfoil profile geometry to a .dat file.

    This function saves the airfoil coordinates to a standard airfoil .dat file
    format suitable for CFD analysis or other aerodynamic applications.

    Parameters:
    ----------
    all_points : numpy.ndarray
        Array of (x, y) coordinates forming the complete airfoil contour
    profile_name : str
        Name identifier for the profile
    output_file_path : str or Path
        Path where the .dat file will be saved
    seam_a : float, optional
        Seam angle in radians for printing statistics

    Returns:
    -------
    str
        Path to the saved .dat file

    Notes:
    -----
    The .dat file contains the complete airfoil contour starting from the leading
    edge upper surface, proceeding through trailing edge, lower surface, and back
    to the leading edge, forming a closed profile suitable for mesh generation.
    """

    # Ensure output directory exists
    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write profile points to dat file
    with open(output_path, "w") as f:
        f.write(f"{profile_name}\n")
        for point in all_points:
            f.write(f"{point[0]:12.8f} {point[1]:12.8f}\n")

    # Calculate statistics
    total_points = len(all_points)

    print(f"Profile saved to: {output_path}")
    print(f"Profile name: {profile_name}")
    print(f"Total points: {total_points}")
    if seam_a is not None:
        print(f"Seam angle: {np.degrees(seam_a):.1f}°")

    return str(output_path)


def reading_profile_from_airfoil_dat_files(filepath):
    """
    Read airfoil profile data from .dat file and extract key parameters.

    Handles multiple file formats:
    - Space-separated values (standard .dat format)
    - Comma-separated values (CSV format)
    - Files with headers/comments starting with #
    - Files with 2 or 3 columns (x,y or x,y,z)

    Args:
        filepath (Path): Path to the .dat file

    Returns:
        dict: Dictionary containing profile information
    """
    with open(filepath, "r") as file:
        lines = file.readlines()

    # Extract profile name from first non-empty, non-comment line before coordinate data
    profile_name = "Unnamed Profile"
    coord_start_idx = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        elif line.startswith("#"):  # Skip comment lines
            if profile_name == "Unnamed Profile":
                # Use comment as profile name (remove # and extra spaces)
                profile_name = line.lstrip("#").strip()
            continue
        elif line[0].isdigit() or line[0] == "-":  # Found coordinate data
            coord_start_idx = i
            break
        else:  # This should be the profile name
            profile_name = line
    else:
        raise ValueError("No valid profile data found in the file.")

    # Extract all coordinate points starting from coordinate data
    points = []
    for line in lines[coord_start_idx:]:
        line = line.strip()
        if not line or line.startswith("#"):  # Skip empty lines and comments
            continue

        # Handle both comma-separated and space-separated formats
        if "," in line:
            # CSV format
            parts = line.split(",")
        else:
            # Space-separated format
            parts = line.split()

        if len(parts) >= 2:
            try:
                x, y = map(
                    float, parts[:2]
                )  # Take only first 2 columns (ignore z if present)
                points.append([x, y])
            except ValueError:
                continue  # Skip lines that can't be parsed as coordinates

    if not points:
        raise ValueError("No valid coordinate points found in the file.")

    return {
        "name": profile_name,
        "points": points,
    }
