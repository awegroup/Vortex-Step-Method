import numpy as np
from .Filament import BoundFilament, SemiInfiniteFilament
from . import jit_norm


class Wake:
    """Wake management class for trailing vortex filament systems.

    Static factory class that handles creation and updates of semi-infinite
    vortex filaments extending from wing trailing edges.
    """

    def __init__(self):
        """Prevent direct instantiation.

        Raises:
            RuntimeError: Always raised to enforce static usage pattern.
        """
        raise RuntimeError("Use Wake.frozen_wake(...) static method.")

    @staticmethod
    def frozen_wake(va_distribution: np.ndarray, panels: list) -> list:
        """Update panel filament systems with semi-infinite wake elements.

        Creates or updates semi-infinite trailing vortex filaments based on local
        flow conditions using the frozen wake assumption.

        Args:
            va_distribution (np.ndarray): Array of velocity vectors at each panel.
            panels (list): List of Panel objects to update.

        Returns:
            list: Updated panel list with wake filaments.

        Raises:
            ValueError: If panel has unexpected number of filaments.
        """
        joint_length_factor = 0.25
        eps = 1e-12
        for i, panel in enumerate(panels):
            va_i = va_distribution[i]
            vel_mag = jit_norm(va_i)
            direction = va_i / jit_norm(va_i)
            z_airf = panel.z_airf

            direction_perp = direction - np.dot(direction, z_airf) * z_airf
            dir_perp_norm = np.linalg.norm(direction_perp)
            if dir_perp_norm < eps:
                chord_dir = panel.y_airf
                direction_perp = chord_dir - np.dot(chord_dir, z_airf) * z_airf
                dir_perp_norm = np.linalg.norm(direction_perp)
            if dir_perp_norm < eps:
                basis = np.array([1.0, 0.0, 0.0])
                if abs(z_airf[0]) > 0.9:
                    basis = np.array([0.0, 1.0, 0.0])
                direction_perp = np.cross(z_airf, basis)
                dir_perp_norm = np.linalg.norm(direction_perp)
            direction_perp = direction_perp / dir_perp_norm

            joint_length = panel.chord * joint_length_factor
            joint_point_1 = panel.TE_point_1 + direction_perp * joint_length
            joint_point_2 = panel.TE_point_2 + direction_perp * joint_length

            # Ensuring that not older runs, their filaments remain present
            if len(panel.filaments) < 3:
                raise ValueError("The panel has an unexpected number of filaments")

            base_filaments = panel.filaments[:3]
            jointed_filaments = [
                BoundFilament(panel.TE_point_1, joint_point_1),
                BoundFilament(panel.TE_point_2, joint_point_2),
                SemiInfiniteFilament(
                    joint_point_1, direction, vel_mag, filament_direction=1
                ),
                SemiInfiniteFilament(
                    joint_point_2, direction, vel_mag, filament_direction=-1
                ),
            ]
            panel.filaments[:] = base_filaments + jointed_filaments
        return panels
