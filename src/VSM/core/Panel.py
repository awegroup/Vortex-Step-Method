import numpy as np
import logging
from VSM.core.Filament import BoundFilament
from . import jit_cross, jit_norm, jit_dot


class Panel:
    """Panel representing a discrete wing section bounded by two wing sections.

    This class encapsulates geometric properties, aerodynamic characteristics, and vortex
    filament system for computing induced velocities and aerodynamic forces.

    Attributes:
        _TE_point_1 (np.ndarray): Trailing edge point from section 1.
        _LE_point_1 (np.ndarray): Leading edge point from section 1.
        _TE_point_2 (np.ndarray): Trailing edge point from section 2.
        _LE_point_2 (np.ndarray): Leading edge point from section 2.
        _chord (float): Average chord length of the panel.
        _va (np.ndarray): Relative velocity of the panel.
        _corner_points (np.ndarray): Array of corner points defining the panel.
        _panel_polar_data (np.ndarray): Interpolated polar data for the panel.
        _aerodynamic_center (np.ndarray): Aerodynamic center (1/4 chord).
        _control_point (np.ndarray): Control point (3/4 chord).
        _bound_point_1 (np.ndarray): First bound vortex point.
        _bound_point_2 (np.ndarray): Second bound vortex point.
        _x_airf (np.ndarray): Normal unit vector.
        _y_airf (np.ndarray): Chordwise unit vector.
        _z_airf (np.ndarray): Spanwise unit vector.
        _width (float): Panel width at bound vortex.
        _filaments (list): List of vortex filament objects.
    """

    def __init__(
        self,
        section_1,
        section_2,
        aerodynamic_center: np.ndarray,
        control_point: np.ndarray,
        bound_point_1: np.ndarray,
        bound_point_2: np.ndarray,
        x_airf: np.ndarray,
        y_airf: np.ndarray,
        z_airf: np.ndarray,
    ):
        """Initialize panel from two sections and geometric parameters.

        Args:
            section_1: First section object defining panel boundary.
            section_2: Second section object defining panel boundary.
            aerodynamic_center (np.ndarray): Aerodynamic center coordinates.
            control_point (np.ndarray): Control point coordinates.
            bound_point_1 (np.ndarray): First bound vortex point.
            bound_point_2 (np.ndarray): Second bound vortex point.
            x_airf (np.ndarray): Normal unit vector.
            y_airf (np.ndarray): Chordwise unit vector.
            z_airf (np.ndarray): Spanwise unit vector.
        """
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

    def compute_relative_alpha_and_relative_velocity(
        self, induced_velocity: np.ndarray
    ) -> tuple:
        """Calculate relative angle of attack and velocity including induced effects.

        Args:
            induced_velocity (np.ndarray): Induced velocity at the evaluation point.

        Returns:
            tuple: (alpha, relative_velocity)
                - alpha (float): Relative angle of attack in radians.
                - relative_velocity (np.ndarray): Total relative velocity vector.
        """
        # Calculate relative velocity and angle of attack
        # Constant throughout the iterations: self.va, self.x_airf, self.y_airf
        relative_velocity = self.va + induced_velocity
        v_normal = jit_dot(self.x_airf, relative_velocity)
        v_tangential = jit_dot(self.y_airf, relative_velocity)
        alpha = np.arctan2(v_normal, v_tangential)
        return alpha, relative_velocity

    def compute_cl(self, alpha: float) -> float:
        """Get lift coefficient for given angle of attack.

        Args:
            alpha (float): Angle of attack in radians.

        Returns:
            float: Lift coefficient.
        """
        return np.interp(
            alpha,
            self._panel_polar_data[:, 0],
            self._panel_polar_data[:, 1],
        )

    def compute_cd_cm(self, alpha: float) -> tuple:
        """Get drag and moment coefficients for given angle of attack.

        Args:
            alpha (float): Angle of attack in radians.

        Returns:
            tuple: (cd, cm)
                - cd (float): Drag coefficient.
                - cm (float): Moment coefficient.
        """
        cd = np.interp(
            alpha, self._panel_polar_data[:, 0], self._panel_polar_data[:, 2]
        )
        cm = np.interp(
            alpha, self._panel_polar_data[:, 0], self._panel_polar_data[:, 3]
        )
        return cd, cm

    def compute_velocity_induced_bound_2D(
        self, evaluation_point: np.ndarray
    ) -> np.ndarray:
        """Calculate 2D bound vortex induced velocity for VSM correction.

        Only needed for VSM, as LLT bound and filament align, thus no induced velocity.

        Args:
            evaluation_point (np.ndarray): Point where velocity is evaluated.

        Returns:
            np.ndarray: Induced velocity vector from 2D bound vortex.
        """
        # r3 is the vector from evaluation point to bound vortex mid-point
        # r0 is the direction of the bound vortex, so we take the unit
        # we take the cross product, r0 x r3,
        # because of it we do not need to compute the perpendicular distance:
        #   We do not need to compute the perpendicular direction from
        #   the evaluation point to the bound vortex, as when taking the cross product
        #   any contribution that is not perpendicular will fall away.
        # we use the 1/2pi * ( 1/ r) equation for an infinite vortex.
        # literature: Rannenberg, Damiani, Cayon etc.

        r3 = evaluation_point - (self.bound_point_1 + self.bound_point_2) / 2
        r0 = self.bound_point_1 - self.bound_point_2
        r0_unit_X_r3 = jit_cross(r0 / jit_norm(r0), r3)

        return (1 / (2 * np.pi)) * (r0_unit_X_r3 / (jit_norm(r0_unit_X_r3) ** 2))

    def compute_velocity_induced_single_ring_semiinfinite(
        self,
        evaluation_point: np.ndarray,
        evaluation_point_on_bound: bool,
        va_norm: float,
        va_unit: np.ndarray,
        gamma: float,
        core_radius_fraction: float,
    ) -> np.ndarray:
        """Calculate velocity induced by complete vortex ring system.

        Args:
            evaluation_point (np.ndarray): Point where velocity is computed.
            evaluation_point_on_bound (bool): True for LLT, False for VSM treatment.
            va_norm (float): Apparent velocity magnitude.
            va_unit (np.ndarray): Apparent velocity unit vector.
            gamma (float): Circulation strength.
            core_radius_fraction (float): Vortex core radius parameter.

        Returns:
            np.ndarray: Total induced velocity from all filaments.
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

    def compute_filaments_for_plotting(self) -> list:
        """Prepare filament data for 3D visualization.

        Computes filament endpoints, directions, and assigns colors for plotting.

        Returns:
            list: List of [start_point, end_point, color] for each filament.
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
