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

        Creates or updates semi-infinite trailing vortex filaments, each pair
        along its own panel's apparent velocity (frozen wake).

        Args:
            va_distribution (np.ndarray): Array of velocity vectors at each panel.
            panels (list): List of Panel objects to update.

        Returns:
            list: Updated panel list with wake filaments.

        Raises:
            ValueError: If panel has unexpected number of filaments.
        """
        va_distribution = np.asarray(va_distribution, dtype=float)
        n_panels = len(panels)
        if va_distribution.shape != (n_panels, 3):
            raise ValueError(
                f"va_distribution must be shape ({n_panels}, 3), got {va_distribution.shape}"
            )

        # One frozen-wake vector per panel: the panel's own apparent velocity,
        # so a rotating body or a distributed inflow gets a locally aligned
        # wake (the same rule as the AIC assembly). A uniform inflow gives the
        # classical single straight wake.
        speeds = np.linalg.norm(va_distribution, axis=1)
        if np.any(speeds <= 0.0):
            raise ValueError("Every panel needs a non-zero apparent velocity.")
        wake_directions = va_distribution / speeds[:, None]

        for i, panel in enumerate(panels):
            # Ensuring that not older runs, their filaments remain present
            if len(panel.filaments) == 3:
                panel.filaments.append(
                    SemiInfiniteFilament(
                        panel.TE_point_1,
                        wake_directions[i],
                        speeds[i],
                        filament_direction=1,
                    )
                )
                panel.filaments.append(
                    SemiInfiniteFilament(
                        panel.TE_point_2,
                        wake_directions[i],
                        speeds[i],
                        filament_direction=-1,
                    )
                )
            elif len(panel.filaments) == 5:
                panel.filaments[3] = SemiInfiniteFilament(
                    panel.TE_point_1,
                    wake_directions[i],
                    speeds[i],
                    filament_direction=1,
                )
                panel.filaments[4] = SemiInfiniteFilament(
                    panel.TE_point_2,
                    wake_directions[i],
                    speeds[i],
                    filament_direction=-1,
                )
            else:
                raise ValueError("The panel has an unexpected number of filaments")
        return panels
