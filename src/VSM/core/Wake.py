import numpy as np
from VSM.core.Filament import SemiInfiniteFilament
from . import jit_norm


class Wake:
    """
    Class to encapsulate 2D airfoil aerodynamic data and interpolation.

    Use AirfoilAerodynamics.from_yaml_entry(...) to instantiate.
    """

    def __init__(self):
        raise RuntimeError(
            "Use AirfoilAerodynamics.from_yaml_entry(...) to instantiate."
        )

    @staticmethod
    def frozen_wake(
        va_distribution,
        panels,
    ):
        """frozen_wake function is used to update the filaments of the panels

            It makes sure to replace older filaments if present, by checking length of the filaments

        Args:
            - va_distribution (np.ndarray): Array of velocity vectors at each panel
            - panels (List[Panel]): List of panels

        Returns:
            - List[Panel]: List of panels with updated filaments

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
