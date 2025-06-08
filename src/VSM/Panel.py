import numpy as np
import logging
from VSM.Filament import BoundFilament
from . import jit_cross, jit_norm, jit_dot


class Panel:
    """
    Class for Panel object

    This class represents a panel defined by two sections, and it calculates various
    aerodynamic properties and induced velocities related to the panel.

    Args:
        section_1 (Section Object): First section of the panel.
        section_2 (Section Object): Second section of the panel.
        aerodynamic_center (np.ndarray): Aerodynamic center of the panel.
        control_point (np.ndarray): Control point of the panel.
        bound_point_1 (np.ndarray): First bound point of the panel.
        bound_point_2 (np.ndarray): Second bound point of the panel.
        x_airf (np.ndarray): Unit vector pointing upwards from the chord-line, perpendicular to the panel.
        y_airf (np.ndarray): Unit vector pointing parallel to the chord-line, from leading edge to trailing edge.
        z_airf (np.ndarray): Unit vector in the airfoil plane, pointing towards the left tip in spanwise direction.

    Attributes / Properties:
        x_airf (np.ndarray): Unit vector pointing upwards from the chord-line.
        y_airf (np.ndarray): Unit vector pointing along the chord-line from LE-to-TE.
        z_airf (np.ndarray): Unit vector in the airfoil plane, towards the left tip.
        va (np.ndarray): Relative velocity of the panel (can be set externally).
        aerodynamic_center (np.ndarray): Aerodynamic center of the panel (typically at 1/4 chord).
        control_point (np.ndarray): Control point of the panel (typically at 3/4 chord).
        corner_points (np.ndarray): Array of the corner points of the panel.
        bound_point_1 (np.ndarray): First bound point of the panel.
        bound_point_2 (np.ndarray): Second bound point of the panel.
        width (float): Width of the panel, computed as the norm of the bound vortex.
        chord (float): Average chord length of the panel.
        TE_point_1 (np.ndarray): Trailing edge point from section 1.
        TE_point_2 (np.ndarray): Trailing edge point from section 2.
        LE_point_1 (np.ndarray): Leading edge point from section 1.
        LE_point_2 (np.ndarray): Leading edge point from section 2.
        filaments (list): List of vortex filament objects defining the panel.
        panel_polar_data (np.ndarray or None): Polar data for aerodynamic properties if provided.

    Methods:
        calculate_relative_alpha_and_relative_velocity(induced_velocity: np.ndarray):
            Calculates the relative angle of attack and relative velocity of the panel by combining
            its own velocity with an induced velocity.

        instantiate_lei_airfoil_breukels_cl_cd_cm_coefficients(section_1, section_2):
            Instantiates the lift (Cl), drag (Cd), and moment (Cm) coefficients for the lei_airfoil_breukels model,
            based on the aerodynamic inputs of the two sections.

        calculate_cl(alpha):
            Returns the lift coefficient (Cl) for a given angle of attack. The calculation method depends on
            the panel aerodynamic model.

        calculate_cd_cm(alpha):
            Returns the drag (Cd) and moment (Cm) coefficients for a given angle of attack. The calculation
            method depends on the panel aerodynamic model.

        calculate_velocity_induced_bound_2D(evaluation_point: np.ndarray):
            Computes the velocity induced by the bound vortex filaments at a specified evaluation point.

        calculate_velocity_induced_single_ring_semiinfinite(
            evaluation_point: np.ndarray,
            evaluation_point_on_bound: bool,
            va_norm: float,
            va_unit: np.ndarray,
            gamma: float,
            core_radius_fraction: float
        ):
            Calculates the velocity induced by a vortex ring (including semi-infinite trailing vortices)
            at a given evaluation point.

        calculate_filaments_for_plotting():
            Prepares and returns the filament data for plotting, including filament start/end points and colors.
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
        self._TE_point_1 = section_1.TE_point
        self._LE_point_1 = section_1.LE_point
        self._TE_point_2 = section_2.TE_point
        self._LE_point_2 = section_2.LE_point
        self._chord = np.average(
            [
                jit_norm(self._TE_point_1 - self._LE_point_1),
                jit_norm(self._TE_point_2 - self._LE_point_2),
            ]
        )
        self._va = None
        self._corner_points = np.array(
            [self._LE_point_1, self._TE_point_1, self._TE_point_2, self._LE_point_2]
        )
        self._panel_polar_data = np.array(
            [
                0.5 * (a1 + a2)
                for a1, a2 in zip(section_1.polar_data, section_2.polar_data)
            ]
        )
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
        self._filaments.append(BoundFilament(x1=bound_point_1, x2=self._TE_point_1))
        self._filaments.append(BoundFilament(x1=self._TE_point_2, x2=bound_point_2))

    ###########################
    ## GETTER FUNCTIONS
    ###########################

    @property
    def x_airf(self):
        """Unit vector pointing upwards from the chord-line, perpendicular to the panel"""
        return self._x_airf

    @property
    def y_airf(self):
        """Unit vector pointing parallel to the chord-line, from LE-to-TE"""
        return self._y_airf

    @property
    def z_airf(self):
        """Unit vector pointing in the airfoil plane, so that is towards left-tip in spanwise direction"""
        return self._z_airf

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

    @property
    def panel_polar_data(self):
        return self._panel_polar_data

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

    def calculate_cl(self, alpha):
        """
        Get the lift coefficient (Cl) for a given angle of attack in radians.
        """
        return np.interp(
            alpha,
            self._panel_polar_data[:, 0],
            self._panel_polar_data[:, 1],
        )

    def calculate_cd_cm(self, alpha):
        """
        Get the drag and moment coefficients (Cd, Cm) for a given angle of attack in radians.
        """
        cd = np.interp(
            alpha, self._panel_polar_data[:, 0], self._panel_polar_data[:, 2]
        )
        cm = np.interp(
            alpha, self._panel_polar_data[:, 0], self._panel_polar_data[:, 3]
        )
        return cd, cm

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
