import numpy as np
from .Filament2 import BoundFilament, SemiInfiniteFilament  #TODO: NEW
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
    def frozen_wake(  #TODO: NEW
        va_distribution: np.ndarray, panels: list, use_jointed_wake: bool = False  #TODO: NEW
    ) -> list:  #TODO: NEW
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
        if not use_jointed_wake:  #TODO: NEW
            for i, panel in enumerate(panels):  #TODO: NEW
                va_i = va_distribution[i]  #TODO: NEW
                vel_mag = jit_norm(va_i)  #TODO: NEW
                direction = va_i / jit_norm(va_i)  #TODO: NEW

                # Ensuring that not older runs, their filaments remain present  #TODO: NEW
                if len(panel.filaments) == 3:  #TODO: NEW
                    panel.filaments.append(  #TODO: NEW
                        SemiInfiniteFilament(  #TODO: NEW
                            panel.TE_point_1, direction, vel_mag, filament_direction=1  #TODO: NEW
                        )  #TODO: NEW
                    )  #TODO: NEW
                    panel.filaments.append(  #TODO: NEW
                        SemiInfiniteFilament(  #TODO: NEW
                            panel.TE_point_2, direction, vel_mag, filament_direction=-1  #TODO: NEW
                        )  #TODO: NEW
                    )  #TODO: NEW
                elif len(panel.filaments) == 5:  #TODO: NEW
                    panel.filaments[3] = SemiInfiniteFilament(  #TODO: NEW
                        panel.TE_point_1, direction, vel_mag, filament_direction=1  #TODO: NEW
                    )  #TODO: NEW
                    panel.filaments[4] = SemiInfiniteFilament(  #TODO: NEW
                        panel.TE_point_2, direction, vel_mag, filament_direction=-1  #TODO: NEW
                    )  #TODO: NEW
                else:  #TODO: NEW
                    raise ValueError("The panel has an unexpected number of filaments")  #TODO: NEW
            return panels  #TODO: NEW

        joint_length_factor = 0.25  #TODO: NEW
        eps = 1e-12  #TODO: NEW
        for i, panel in enumerate(panels):
            va_i = va_distribution[i]
            vel_mag = jit_norm(va_i)
            direction = va_i / jit_norm(va_i)
            z_airf = panel.z_airf  #TODO: NEW

            direction_perp = direction - np.dot(direction, z_airf) * z_airf  #TODO: NEW
            dir_perp_norm = np.linalg.norm(direction_perp)  #TODO: NEW
            if dir_perp_norm < eps:  #TODO: NEW
                chord_dir = panel.y_airf  #TODO: NEW
                direction_perp = chord_dir - np.dot(chord_dir, z_airf) * z_airf  #TODO: NEW
                dir_perp_norm = np.linalg.norm(direction_perp)  #TODO: NEW
            if dir_perp_norm < eps:  #TODO: NEW
                basis = np.array([1.0, 0.0, 0.0])  #TODO: NEW
                if abs(z_airf[0]) > 0.9:  #TODO: NEW
                    basis = np.array([0.0, 1.0, 0.0])  #TODO: NEW
                direction_perp = np.cross(z_airf, basis)  #TODO: NEW
                dir_perp_norm = np.linalg.norm(direction_perp)  #TODO: NEW
            direction_perp = direction_perp / dir_perp_norm  #TODO: NEW

            joint_length = panel.chord * joint_length_factor  #TODO: NEW
            joint_point_1 = panel.TE_point_1 + direction_perp * joint_length  #TODO: NEW
            joint_point_2 = panel.TE_point_2 + direction_perp * joint_length  #TODO: NEW

            # Ensuring that not older runs, their filaments remain present  #TODO: NEW
            if len(panel.filaments) < 3:  #TODO: NEW
                raise ValueError("The panel has an unexpected number of filaments")  #TODO: NEW

            base_filaments = panel.filaments[:3]  #TODO: NEW
            jointed_filaments = [  #TODO: NEW
                BoundFilament(panel.TE_point_1, joint_point_1),  #TODO: NEW
                BoundFilament(panel.TE_point_2, joint_point_2),  #TODO: NEW
                SemiInfiniteFilament(  #TODO: NEW
                    joint_point_1, direction, vel_mag, filament_direction=1  #TODO: NEW
                ),  #TODO: NEW
                SemiInfiniteFilament(  #TODO: NEW
                    joint_point_2, direction, vel_mag, filament_direction=-1  #TODO: NEW
                ),  #TODO: NEW
            ]  #TODO: NEW
            panel.filaments[:] = base_filaments + jointed_filaments  #TODO: NEW
        return panels
