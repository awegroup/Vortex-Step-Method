from abc import ABC, abstractmethod
import numpy as np
import logging
from . import jit_cross, jit_norm, jit_dot

logging.basicConfig(level=logging.INFO)


class Filament(ABC):
    """Abstract base class for vortex filaments.

    Attributes:
        _alpha0 (float): Oseen parameter for viscous diffusion (1.25643).
        _nu (float): Kinematic viscosity of air in m²/s (1.48e-5).
    """

    @abstractmethod
    def __init__(self):
        """Initialize filament with physical constants."""
        self._alpha0 = 1.25643  # Oseen parameter
        self._nu = 1.48e-5  # Kinematic viscosity of air


class BoundFilament(Filament):
    """Bound vortex filament between two points along quarter-chord line.

    Attributes:
        _x1 (np.ndarray): First endpoint of the filament.
        _x2 (np.ndarray): Second endpoint of the filament.
        _length (float): Filament length.
        _r0 (np.ndarray): Filament vector (x2 - x1).
    """

    def __init__(self, x1: np.ndarray, x2: np.ndarray):
        """Initialize bound filament with two endpoints.

        Args:
            x1 (np.ndarray): First endpoint coordinates.
            x2 (np.ndarray): Second endpoint coordinates.
        """
        self._x1 = np.array(x1)
        self._x2 = np.array(x2)
        self._length = jit_norm(self._x2 - self._x1)
        self._r0 = self._x2 - self._x1
        super().__init__()

    @property
    def x1(self) -> np.ndarray:
        """Get first endpoint of the filament.

        Returns:
            np.ndarray: First endpoint coordinates.
        """
        return self._x1

    @property
    def x2(self) -> np.ndarray:
        """Get second endpoint of the filament.

        Returns:
            np.ndarray: Second endpoint coordinates.
        """
        return self._x2

    def velocity_3D_bound_vortex(
        self, XVP: np.ndarray, gamma: float, core_radius_fraction: float
    ) -> np.ndarray:
        """Calculate velocity induced by bound vortex filament using Vatistas core model.

        Vortex core correction from: Rick Damiani et al. "A vortex step method for nonlinear
        airfoil polar data as implemented in KiteAeroDyn".

        Args:
            XVP (np.ndarray): Evaluation point coordinates.
            gamma (float): Vortex strength (circulation).
            core_radius_fraction (float): Core radius as fraction of filament length.

        Returns:
            np.ndarray: Induced velocity vector [vx, vy, vz].
        """
        XV1 = self.x1
        XV2 = self.x2

        r0 = XV2 - XV1  # Vortex filament
        r1 = XVP - XV1  # Controlpoint to one end of the vortex filament
        r2 = XVP - XV2  # Controlpoint to one end of the vortex filament

        # Cross products used for later computations
        r1Xr0 = jit_cross(r1, r0)
        r2Xr0 = jit_cross(r2, r0)

        epsilon = core_radius_fraction * jit_norm(r0)  # Cut-off radius
        # If point is outside the core radius of filament
        if jit_norm(r1Xr0) / jit_norm(r0) > epsilon:
            # Perpendicular distance from XVP to vortex filament (r0)
            r1Xr2 = jit_cross(r1, r2)
            return (
                gamma
                / (4 * np.pi)
                * r1Xr2
                / (jit_norm(r1Xr2) ** 2)
                * jit_dot(r0, r1 / jit_norm(r1) - r2 / jit_norm(r2))
            )
        # If point is on the filament
        elif jit_norm(r1Xr0) / jit_norm(r0) < 1e-12 * epsilon:
            return np.zeros(3)
        # If point is inside the core radius of filament
        else:
            logging.info(f"inside core radius")
            # logging.info(f"epsilon: {epsilon}")
            logging.info(
                f"distance from control point to filament: {jit_norm(r1Xr0) / jit_norm(r0)}"
            )
            # The control point is placed on the edge of the radius core
            # proj stands for the vectors respect to the new controlpoint
            r1_proj = jit_dot(r1, r0) * r0 / (
                jit_norm(r0) ** 2
            ) + epsilon * r1Xr0 / jit_norm(r1Xr0)
            r2_proj = jit_dot(r2, r0) * r0 / (
                jit_norm(r0) ** 2
            ) + epsilon * r2Xr0 / jit_norm(r2Xr0)
            r1Xr2_proj = jit_cross(r1_proj, r2_proj)
            vel_ind_proj = (
                gamma
                / (4 * np.pi)
                * r1Xr2_proj
                / (jit_norm(r1Xr2_proj) ** 2)
                * jit_dot(
                    r0,
                    r1_proj / jit_norm(r1_proj) - r2_proj / jit_norm(r2_proj),
                )
            )
            return jit_norm(r1Xr0) / (jit_norm(r0) * epsilon) * vel_ind_proj

    def velocity_3D_trailing_vortex(
        self, XVP: np.ndarray, gamma: float, Uinf: float
    ) -> np.ndarray:
        """Calculate velocity induced by trailing vortex filament with viscous core correction.

        Vortex core correction from: Rick Damiani et al. "A vortex step method for nonlinear
        airfoil polar data as implemented in KiteAeroDyn".

        Args:
            XVP (np.ndarray): Evaluation point coordinates.
            gamma (float): Vortex strength.
            Uinf (float): Inflow velocity magnitude.

        Returns:
            np.ndarray: Induced velocity vector.
        """
        XV1 = self.x1
        XV2 = self.x2

        r0 = XV2 - XV1  # Vortex filament
        r1 = XVP - XV1  # Controlpoint to one end of the vortex filament
        r2 = XVP - XV2  # Controlpoint to one end of the vortex filament

        r_perp = (
            jit_dot(r1, r0) * r0 / (jit_norm(r0) ** 2)
        )  # Vector from XV1 to XVP perpendicular to the core radius
        epsilon = np.sqrt(
            4 * self._alpha0 * self._nu * jit_norm(r_perp) / Uinf
        )  # Cut-off radius

        # Cross products used for later computations
        r1Xr0 = jit_cross(r1, r0)
        r2Xr0 = jit_cross(r2, r0)

        # if point is outside the core radius of filament
        if (
            jit_norm(r1Xr0) / jit_norm(r0) > epsilon
        ):  # Perpendicular distance from XVP to vortex filament (r0)
            r1Xr2 = jit_cross(r1, r2)
            return (
                gamma
                / (4 * np.pi)
                * r1Xr2
                / (jit_norm(r1Xr2) ** 2)
                * jit_dot(r0, r1 / jit_norm(r1) - r2 / jit_norm(r2))
            )
        # if point is on the filament
        elif jit_norm(r1Xr0) / jit_norm(r0) < 1e-12 * epsilon:
            return np.zeros(3)
        # if point is inside the core radius of filament
        else:
            # The control point is placed on the edge of the radius core
            # proj stands for the vectors respect to the new controlpoint
            r1_proj = jit_dot(r1, r0) * r0 / (
                jit_norm(r0) ** 2
            ) + epsilon * r1Xr0 / jit_norm(r1Xr0)
            r2_proj = jit_dot(r2, r0) * r0 / (
                jit_norm(r0) ** 2
            ) + epsilon * r2Xr0 / jit_norm(r2Xr0)
            r1Xr2_proj = jit_cross(r1_proj, r2_proj)
            vel_ind_proj = (
                gamma
                / (4 * np.pi)
                * r1Xr2_proj
                / (jit_norm(r1Xr2_proj) ** 2)
                * jit_dot(
                    r0,
                    r1_proj / jit_norm(r1_proj) - r2_proj / jit_norm(r2_proj),
                )
            )
            return jit_norm(r1Xr0) / (jit_norm(r0) * epsilon) * vel_ind_proj


class SemiInfiniteFilament(Filament):
    """Semi-infinite trailing vortex extending to infinity in wake direction.

    Attributes:
        _x1 (np.ndarray): Starting point (trailing edge).
        _direction (np.ndarray): Unit vector of wake direction.
        _vel_mag (float): Wake velocity magnitude.
        _filament_direction (int): ±1 indicating filament orientation.
    """

    def __init__(
        self,
        x1: np.ndarray,
        direction: np.ndarray,
        vel_mag: float,
        filament_direction: int,
    ):
        """Initialize semi-infinite filament.

        Args:
            x1 (np.ndarray): Starting point (trailing edge).
            direction (np.ndarray): Unit vector of apparent wind speed.
            vel_mag (float): Magnitude of apparent wind speed.
            filament_direction (int): -1 or 1, indicating direction relative to apparent wind.
        """
        self._x1 = x1  # the trailing edge point, of which the trailing vortex starts
        # x2 is a point far away from the filament, defined here for plotting purposes
        # self._x2 = x1 + filament_direction * direction * 0.5
        self._direction = direction  # unit vector of apparent wind speed
        self._vel_mag = vel_mag  # the magnitude of the apparent wind speed
        self._filament_direction = filament_direction  # -1 or 1, indicating if its with or against the direction of the apparent wind speed
        super().__init__()

    @property
    def x1(self) -> np.ndarray:
        """Get starting point of the semi-infinite filament.

        Returns:
            np.ndarray: Starting point coordinates.
        """
        return self._x1

    @property
    def filament_direction(self) -> int:
        """Get filament direction multiplier.

        Returns:
            int: Direction multiplier (±1).
        """
        return self._filament_direction

    def velocity_3D_trailing_vortex_semiinfinite(
        self, Vf: np.ndarray, XVP: np.ndarray, GAMMA: float, Uinf: float
    ) -> np.ndarray:
        """Calculate velocity induced by semi-infinite trailing vortex filament.

        Vortex core correction from: Rick Damiani et al. "A vortex step method for nonlinear
        airfoil polar data as implemented in KiteAeroDyn".

        Args:
            Vf (np.ndarray): Wake velocity vector.
            XVP (np.ndarray): Evaluation point.
            GAMMA (float): Circulation strength.
            Uinf (float): Inflow velocity magnitude.

        Returns:
            np.ndarray: Induced velocity vector.
        """
        XV1 = self.x1
        GAMMA = -GAMMA * self.filament_direction

        r1 = XVP - XV1  # Vector from XV1 to XVP
        r1XVf = jit_cross(r1, Vf)

        r_perp = (
            jit_dot(r1, Vf) * Vf
        )  # Vector from XV1 to XVP perpendicular to the core radius
        epsilon = np.sqrt(
            4 * self._alpha0 * self._nu * jit_norm(r_perp) / Uinf
        )  # Cut-off radius

        # if point is outside the core radius of filament
        if jit_norm(r1XVf) / jit_norm(Vf) > epsilon:
            # determine scalar
            K = (
                GAMMA
                / 4
                / np.pi
                / jit_norm(r1XVf) ** 2
                * (1 + jit_dot(r1, Vf) / jit_norm(r1))
            )
            # determine the three velocity components
            return K * r1XVf
        # if point is on the filament
        elif jit_norm(r1XVf) / jit_norm(Vf) < 1e-12 * epsilon:
            return np.zeros(3)
        # else, if point within core
        else:
            r1_proj = jit_dot(r1, Vf) * Vf + epsilon * (
                r1 / jit_norm(r1) - Vf
            ) / jit_norm(r1 / jit_norm(r1) - Vf)
            r1XVf_proj = jit_cross(r1_proj, Vf)
            K = (
                GAMMA
                / 4
                / np.pi
                / jit_norm(r1XVf_proj) ** 2
                * (1 + jit_dot(r1_proj, Vf) / jit_norm(r1_proj))
            )
            # determine the three velocity components
            return K * r1XVf_proj
