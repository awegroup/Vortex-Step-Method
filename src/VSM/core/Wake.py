import numpy as np
from VSM.core.Filament import SemiInfiniteFilament
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
        for i, panel in enumerate(panels):
            va_i = va_distribution[i]
            vel_mag = jit_norm(va_i)
            direction = va_i / jit_norm(va_i)

            # Ensuring that not older runs, their filaments remain present
            if len(panel.filaments) == 3:
                panel.filaments.append(
                    SemiInfiniteFilament(
                        panel.TE_point_1, direction, vel_mag, filament_direction=1
                    )
                )
                panel.filaments.append(
                    SemiInfiniteFilament(
                        panel.TE_point_2, direction, vel_mag, filament_direction=-1
                    )
                )
            elif len(panel.filaments) == 5:
                panel.filaments[3] = SemiInfiniteFilament(
                    panel.TE_point_1, direction, vel_mag, filament_direction=1
                )
                panel.filaments[4] = SemiInfiniteFilament(
                    panel.TE_point_2, direction, vel_mag, filament_direction=-1
                )
            else:
                raise ValueError("The panel has an unexpected number of filaments")
        return panels
