import numpy as np
import logging
from VSM.Filament import BoundFilament
from . import jit_cross, jit_norm, jit_dot


class Panel:
    """Class for Panel object

    Args:
        - section_1 (Section Object): First section of the panel
        - section_2 (Section Object): Second section of the panel
        - aerodynamic_center (np.array): Aerodynamic center of the panel
        - control_point (np.array): Control point of the panel
        - bound_point_1 (np.array): First bound point of the panel
        - bound_point_2 (np.array): Second bound point of the panel
        - x_airf (np.array): Unit vector pointing upwards from the chord-line, perpendicular to the panel
        - y_airf (np.array): Unit vector pointing parallel to the chord-line, from LE-to-TE
        - z_airf (np.array): Unit vector pointing in the airfoil plane, so that is towards left-tip in spanwise direction

    Returns:
        Panel Object: Panel object with the given attributes

    Methods:
        - calculate_relative_alpha_and_relative_velocity(induced_velocity: np.array): Calculates the relative angle of attack and relative velocity of the panel
        - calculate_cl(alpha): Get the lift coefficient (Cl) for a given angle of attack
        - calculate_cd_cm(alpha): Get the lift, drag, and moment coefficients (Cl, Cd, Cm) for a given angle of attack
        - calculate_velocity_induced_bound_2D(evaluation_point): Calculates velocity induced by bound vortex filaments at the control point
        - calculate_velocity_induced_single_ring_semiinfinite(evaluation_point, evaluation_point_on_bound, va_norm, va_unit, gamma, core_radius_fraction): Calculates the velocity induced by a ring at a certain controlpoint
        - calculate_filaments_for_plotting(): Calculates the filaments for plotting

    Properties:
        - z_airf (np.array): Unit vector pointing in the airfoil plane, so that is towards left-tip in spanwise direction
        - x_airf (np.array): Unit vector pointing upwards from the chord-line, perpendicular to the panel
        - y_airf (np.array): Unit vector pointing parallel to the chord-line, from LE-to-TE
        - va (np.array): Relative velocity of the panel
        - aerodynamic_center (np.array): The aerodynamic center of the panel, also LLTpoint, at 1/4c
        - control_point (np.array): The control point of the panel, also VSMpoint, at 3/4c
        - corner_points (np.array): Corner points of the panel
        - bound_point_1 (np.array): First bound point of the panel
        - bound_point_2 (np.array): Second bound point of the panel
        - width (float): Width of the panel
        - chord (float): Chord of the panel
        - TE_point_1 (np.array): Trailing edge point 1 of the panel
        - TE_point_2 (np.array): Trailing edge point 2 of the panel
        - LE_point_1 (np.array): Leading edge point 1 of the panel
        - LE_point_2 (np.array): Leading edge point 2 of the panel
        - filaments (list): List of filaments of the panel
    """

    def __init__(
        self,
        section_1,
        section_2,
        aerodynamic_center,
        control_point,
        bound_point_1,
        bound_point_2,
        x_airf,
        y_airf,
        z_airf,
    ):
        TE_point_1 = np.array(section_1.TE_point)
        LE_point_1 = np.array(section_1.LE_point)
        TE_point_2 = np.array(section_2.TE_point)
        LE_point_2 = np.array(section_2.LE_point)

        self._TE_point_1 = TE_point_1
        self._LE_point_1 = LE_point_1
        self._TE_point_2 = TE_point_2
        self._LE_point_2 = LE_point_2
        self._chord = np.average(
            [
                jit_norm(TE_point_1 - LE_point_1),
                jit_norm(TE_point_2 - LE_point_2),
            ]
        )
        self._va = None
        self._corner_points = np.array([LE_point_1, TE_point_1, TE_point_2, LE_point_2])

        # Defining panel_aero_model
        if section_1.aero_input[0] != section_2.aero_input[0]:
            raise ValueError(
                "Both sections should have the same aero_input, got"
                + section_1.aero_input[0]
                + " and "
                + section_2.aero_input[0]
            )
        self._panel_aero_model = section_1.aero_input[0]

        # Initializing the panel aerodynamic data dependent on the aero_model
        if self._panel_aero_model == "lei_airfoil_breukels":
            self.instantiate_lei_airfoil_breukels_cl_cd_cm_coefficients(
                section_1, section_2
            )
        elif self._panel_aero_model == "inviscid":
            pass
        elif self._panel_aero_model == "polar_data":
            # Average the polar_data of the two sections
            aero_1 = section_1.aero_input[1]
            aero_2 = section_2.aero_input[1]
            if len(aero_1) != len(aero_2) or aero_1.shape != aero_2.shape:
                raise ValueError(
                    "The polar data of the two sections should have the same shape & length"
                )
            self._panel_polar_data = (aero_1 + aero_2) / 2
        else:
            raise NotImplementedError

        self._aerodynamic_center = aerodynamic_center
        self._control_point = control_point
        self._bound_point_1 = bound_point_1
        self._bound_point_2 = bound_point_2
        self._x_airf = x_airf
        self._y_airf = y_airf
        self._z_airf = z_airf

        # Calculuting width at the bound, should be done averaged over whole panel
        # Conceptually, you should mulitply by the width of the bound vortex and thus take the average width.
        self._width = jit_norm(bound_point_2 - bound_point_1)

        ### Setting up the filaments (order used to reversed for right-to-left input)
        self._filaments = []
        self._filaments.append(BoundFilament(x1=bound_point_2, x2=bound_point_1))
        self._filaments.append(BoundFilament(x1=bound_point_1, x2=TE_point_1))
        self._filaments.append(BoundFilament(x1=TE_point_2, x2=bound_point_2))

    ###########################
    ## GETTER FUNCTIONS
    ###########################

    @property
    def z_airf(self):
        """Unit vector pointing in the airfoil plane, so that is towards left-tip in spanwise direction"""
        return self._z_airf

    @property
    def x_airf(self):
        """Unit vector pointing upwards from the chord-line, perpendicular to the panel"""
        return self._x_airf

    @property
    def y_airf(self):
        """Unit vector pointing parallel to the chord-line, from LE-to-TE"""
        return self._y_airf

    @property
    def va(self):
        return self._va

    @property
    def aerodynamic_center(self):
        """The aerodynamic center of the panel, also LLTpoint, at 1/4c"""
        return self._aerodynamic_center

    @property
    def control_point(self):
        """The control point of the panel, also VSMpoint, at 3/4c"""
        return self._control_point

    @property
    def corner_points(self):
        return self._corner_points

    @property
    def bound_point_1(self):
        return self._bound_point_1

    @property
    def bound_point_2(self):
        return self._bound_point_2

    @property
    def width(self):
        return self._width

    @property
    def chord(self):
        return self._chord

    @property
    def TE_point_1(self):
        return self._TE_point_1

    @property
    def TE_point_2(self):
        return self._TE_point_2

    @property
    def LE_point_1(self):
        return self._LE_point_1

    @property
    def LE_point_2(self):
        return self._LE_point_2

    @property
    def filaments(self):
        return self._filaments

    ###########################
    ## SETTER FUNCTIONS
    ###########################
    @va.setter
    def va(self, value):
        self._va = value

    ###########################
    ## CALCULATE FUNCTIONS      # All this return something
    ###########################

    def calculate_relative_alpha_and_relative_velocity(
        self, induced_velocity: np.array
    ):
        """Calculates the relative angle of attack and relative velocity of the panel

        Args:
            induced_velocity (np.array): Induced velocity at the control point

        Returns:
            alpha (float): Relative angle of attack of the panel
            relative_velocity (np.array): Relative velocity of the panel
        """
        # Calculate relative velocity and angle of attack
        # Constant throughout the iterations: self.va, self.x_airf, self.y_airf
        relative_velocity = self.va + induced_velocity
        v_normal = jit_dot(self.x_airf, relative_velocity)
        v_tangential = jit_dot(self.y_airf, relative_velocity)
        alpha = np.arctan(v_normal / v_tangential)
        return alpha, relative_velocity

    def instantiate_lei_airfoil_breukels_cl_cd_cm_coefficients(
        self, section_1, section_2
    ):
        """Instantiates the Lei Airfoil Breukels Cl, Cd, Cm coefficients

        Args:
            section_1 (Section Object): First section of the panel
            section_2 (Section Object): Second section of the panel

        Returns:
            None"""

        t1, k1 = section_1.aero_input[1]
        t2, k2 = section_2.aero_input[1]
        t = (t1 + t2) / 2
        k = (k1 + k2) / 2

        # cl_coefficients
        C20 = -0.008011
        C21 = -0.000336
        C22 = 0.000992
        C23 = 0.013936
        C24 = -0.003838
        C25 = -0.000161
        C26 = 0.001243
        C27 = -0.009288
        C28 = -0.002124
        C29 = 0.012267
        C30 = -0.002398
        C31 = -0.000274
        C32 = 0
        C33 = 0
        C34 = 0
        C35 = -3.371000
        C36 = 0.858039
        C37 = 0.141600
        C38 = 7.201140
        C39 = -0.676007
        C40 = 0.806629
        C41 = 0.170454
        C42 = -0.390563
        C43 = 0.101966

        S9 = C20 * t**2 + C21 * t + C22
        S10 = C23 * t**2 + C24 * t + C25
        S11 = C26 * t**2 + C27 * t + C28
        S12 = C29 * t**2 + C30 * t + C31
        S13 = C32 * t**2 + C33 * t + C34
        S14 = C35 * t**2 + C36 * t + C37
        S15 = C38 * t**2 + C39 * t + C40
        S16 = C41 * t**2 + C42 * t + C43

        lambda5 = S9 * k + S10
        lambda6 = S11 * k + S12
        lambda7 = S13 * k + S14
        lambda8 = S15 * k + S16

        self._cl_coefficients = [lambda5, lambda6, lambda7, lambda8]

        # cd_coefficients
        C44 = 0.546094
        C45 = 0.022247
        C46 = -0.071462
        C47 = -0.006527
        C48 = 0.002733
        C49 = 0.000686
        C50 = 0.123685
        C51 = 0.143755
        C52 = 0.495159
        C53 = -0.105362
        C54 = 0.033468

        cd_2_deg = (C44 * t + C45) * k**2 + (C46 * t + C47) * k + (C48 * t + C49)
        cd_1_deg = 0
        cd_0_deg = (C50 * t + C51) * k + (C52 * t**2 + C53 * t + C54)

        self._cd_coefficients = [cd_2_deg, cd_1_deg, cd_0_deg]

        # cm_coefficients
        C55 = -0.284793
        C56 = -0.026199
        C57 = -0.024060
        C58 = 0.000559
        C59 = -1.787703
        C60 = 0.352443
        C61 = -0.839323
        C62 = 0.137932

        cm_2_deg = (C55 * t + C56) * k + (C57 * t + C58)
        cm_1_deg = 0
        cm_0_deg = (C59 * t + C60) * k + (C61 * t + C62)

        self._cm_coefficients = [cm_2_deg, cm_1_deg, cm_0_deg]

    def calculate_cl(self, alpha):
        """
        Get the lift coefficient (Cl) for a given angle of attack.

        Args:
            alpha (float): Angle of attack in radians.
            airfoil_data (np.array): Array containing airfoil data.

        Returns:
            float: Interpolated lift coefficient (Cl).
        """
        if self._panel_aero_model == "lei_airfoil_breukels":
            cl = np.polyval(self._cl_coefficients, np.rad2deg(alpha))
            # if outside of 20 degrees which in rad = np.pi/9
            if alpha > (np.pi / 9) or alpha < -(np.pi / 9):
                cl = 2 * np.cos(alpha) * np.sin(alpha) ** 2
            return cl
        elif self._panel_aero_model == "inviscid":
            return 2 * np.pi * alpha
        elif self._panel_aero_model == "polar_data":
            return np.interp(
                alpha,
                self._panel_polar_data[:, 0],
                self._panel_polar_data[:, 1],
            )
        else:
            raise NotImplementedError

    def calculate_cd_cm(self, alpha):
        """
        Get the lift, drag, and moment coefficients (Cl, Cd, Cm) for a given angle of attack.

        Args:
            alpha (float): Angle of attack in radians.
            airfoil_data (np.array): Array containing airfoil data.

        Returns:
            tuple: Interpolated (Cl, Cd, Cm) coefficients.
        """
        if self._panel_aero_model == "lei_airfoil_breukels":
            cd = np.polyval(self._cd_coefficients, np.rad2deg(alpha))
            cm = np.polyval(self._cm_coefficients, np.rad2deg(alpha))
            # if outside of 20 degrees (np.pi/9)
            if alpha > (np.pi / 9) or alpha < -(np.pi / 9):
                cd = 2 * np.sin(alpha) ** 3
            return cd, cm
        elif self._panel_aero_model == "inviscid":
            cd = 0.0
            cm = 0.0
            return cd, cm
        elif self._panel_aero_model == "polar_data":
            cd = np.interp(
                alpha, self._panel_polar_data[:, 0], self._panel_polar_data[:, 2]
            )
            cm = np.interp(
                alpha, self._panel_polar_data[:, 0], self._panel_polar_data[:, 3]
            )
            return cd, cm
        else:
            raise NotImplementedError

    def calculate_velocity_induced_bound_2D(self, evaluation_point):
        """Calculates velocity induced by bound vortex filaments at the control point
            Only needed for VSM, as LLT bound and filament align, thus no induced velocity

        Args:
            self: Panel object

        Returns:
            np.array: Induced velocity at the control point
        """
        ### DIRECTION
        # r3 perpendicular to the bound vortex
        r3 = evaluation_point - (self.bound_point_1 + self.bound_point_2) / 2
        # r0 should be the direction of the bound vortex
        r0 = self.bound_point_1 - self.bound_point_2
        cross = jit_cross(r0, r3)
        return (
            cross
            / (cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
            / 2
            / np.pi
            * jit_norm(r0)
        )

    def calculate_velocity_induced_single_ring_semiinfinite(
        self,
        evaluation_point,
        evaluation_point_on_bound,
        va_norm,
        va_unit,
        gamma,
        core_radius_fraction,
    ):
        """
        Calculates the velocity induced by a ring at a certain controlpoint

        Parameters
        ----------
        ring : List of dictionaries defining the filaments of a vortex ring
        controlpoint : Dictionary defining a controlpoint
        model : VSM: Vortex Step method/ LLT: Lifting Line Theory
        Uinf : Wind speed vector

        Returns
        -------
        velind : Induced velocity

        """
        velind = [0, 0, 0]

        # TODO: ADD option for more wake filaments
        for i, filament in enumerate(self.filaments):
            # bound
            if i == 0:
                if evaluation_point_on_bound:
                    tempvel = [0, 0, 0]
                else:
                    tempvel = filament.velocity_3D_bound_vortex(
                        evaluation_point, gamma, core_radius_fraction
                    )
            # trailing1 or trailing2
            elif i == 1 or i == 2:
                tempvel = filament.velocity_3D_trailing_vortex(
                    evaluation_point, gamma, va_norm
                )
            # trailing_semi_inf1
            elif i == 3:
                tempvel = filament.velocity_3D_trailing_vortex_semiinfinite(
                    va_unit, evaluation_point, gamma, va_norm
                )
            # trailing_semi_inf2
            elif i == 4:
                tempvel = filament.velocity_3D_trailing_vortex_semiinfinite(
                    va_unit, evaluation_point, gamma, va_norm
                )

            velind[0] += tempvel[0]
            velind[1] += tempvel[1]
            velind[2] += tempvel[2]

        return np.array(velind)

    def calculate_filaments_for_plotting(self):
        """Calculates the filaments for plotting
            It calculates right direction, filament length and appends a color

        Args:
            self: Panel object

        Returns:
            list: List of lists containing the filaments for plotting
        """
        filaments = []
        for i, filament in enumerate(self.filaments):
            x1 = filament.x1
            if hasattr(filament, "x2") and filament.x2 is not None:
                x2 = filament.x2
                if i == 0:  # bound
                    color = "magenta"
                else:  # trailing
                    color = "green"
            else:
                # For semi-infinite filaments
                x2 = x1 + 1 * self.chord * (self.va / jit_norm(self.va))
                color = "orange"
                if filament.filament_direction == -1:
                    x1, x2 = x2, x1
                    color = "red"

            filaments.append([x1, x2, color])
        return filaments
