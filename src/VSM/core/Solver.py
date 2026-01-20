import numpy as np
import logging
from . import jit_cross


class Solver:
    """Solver for aerodynamic circulation distribution and force computation.

    Implements iterative algorithms to determine circulation distribution that satisfies
    boundary conditions for VSM and LLT aerodynamic models.

    Attributes:
        aerodynamic_model_type (str): Aerodynamic model type ('VSM' or 'LLT').
        max_iterations (int): Maximum number of iterations for convergence.
        allowed_error (float): Convergence tolerance for normalized error.
        relaxation_factor (float): Under-relaxation factor for stability.
        core_radius_fraction (float): Vortex core radius fraction.
        gamma_loop_type (str): Iterative algorithm type.
        gamma_initial_distribution_type (str): Initial circulation distribution method.
        is_only_f_and_gamma_output (bool): Return only forces and circulation if True.
        is_with_viscous_drag_correction (bool): Enable viscous drag correction.
        reference_point (list): Reference point for moment calculations.
        mu (float): Dynamic viscosity of fluid.
        rho (float): Fluid density.
        is_smooth_circulation (bool): Apply circulation smoothing.
        smoothness_factor (float): Smoothing strength parameter.
        is_artificial_damping (bool): Enable artificial damping for stall.
        artificial_damping (dict): Artificial damping parameters.
        is_with_simonet_artificial_viscosity (bool): Enable Simonet artificial viscosity.
        _simonet_artificial_viscosity_fva (float): Simonet model parameter.
    """

    def __init__(
        self,
        aerodynamic_model_type: str = "VSM",
        max_iterations: int = 5000,
        allowed_error: float = 1e-6,
        relaxation_factor: float = 0.01,
        core_radius_fraction: float = 0.05,  # Following Damiani et al. (2019) https://docs.nrel.gov/docs/fy19osti/72777.pdf
        gamma_loop_type: str = "base",
        gamma_initial_distribution_type: str = "elliptical",
        is_only_f_and_gamma_output: bool = False,
        is_with_viscous_drag_correction: bool = False,
        reference_point: list = [0, 0, 0],
        mu: float = 1.81e-5,
        rho: float = 1.225,
        is_smooth_circulation: bool = False,
        smoothness_factor: float = 0.08,
        is_artificial_damping: bool = False,
        artificial_damping: dict = {"k2": 0.1, "k4": 0.0},
        is_with_simonet_artificial_viscosity: bool = False,
        simonet_artificial_viscosity_fva: float = None,
        is_aoa_corrected: bool = False,
    ):
        """Initialize solver with configuration parameters.

        Args:
            aerodynamic_model_type (str): Type of aerodynamic model ('VSM' or 'LLT').
            max_iterations (int): Maximum solver iterations.
            allowed_error (float): Convergence tolerance.
            relaxation_factor (float): Under-relaxation factor.
            core_radius_fraction (float): Vortex core radius fraction.
            gamma_loop_type (str): Iterative algorithm type.
            gamma_initial_distribution_type (str): Initial circulation distribution.
            is_only_f_and_gamma_output (bool): Return minimal output if True.
            is_with_viscous_drag_correction (bool): Enable viscous corrections.
            reference_point (list): Reference point for moments.
            mu (float): Dynamic viscosity.
            rho (float): Fluid density.
            is_smooth_circulation (bool): Apply circulation smoothing.
            smoothness_factor (float): Smoothing factor.
            is_artificial_damping (bool): Enable artificial damping.
            artificial_damping (dict): Damping parameters.
            is_with_simonet_artificial_viscosity (bool): Enable Simonet model.
            simonet_artificial_viscosity_fva (float): Simonet parameter.
        """
        self.aerodynamic_model_type = aerodynamic_model_type
        self.max_iterations = int(max_iterations)
        self.allowed_error = allowed_error
        self.relaxation_factor = relaxation_factor
        self.core_radius_fraction = core_radius_fraction
        self.gamma_loop_type = gamma_loop_type
        self.gamma_initial_distribution_type = gamma_initial_distribution_type
        self.is_only_f_and_gamma_output = is_only_f_and_gamma_output
        self.is_with_viscous_drag_correction = is_with_viscous_drag_correction
        self.reference_point = reference_point
        self.is_aoa_corrected = is_aoa_corrected
        # === athmospheric properties ===
        self.mu = mu
        self.rho = rho
        # ===============================
        #       STALL MODELS
        # ===============================
        # === STALL: smooth_circulation ===
        self.is_smooth_circulation = is_smooth_circulation
        self.smoothness_factor = smoothness_factor
        # === STALL: artificial damping ===
        self.is_artificial_damping = is_artificial_damping
        self.artificial_damping = artificial_damping
        # === STALL: simonet_aritificial_viscosity ===
        self.is_with_simonet_artificial_viscosity = is_with_simonet_artificial_viscosity
        self._simonet_artificial_viscosity_fva = simonet_artificial_viscosity_fva

        ## Initializing some empty properties
        self.panels = None
        self.n_panels = None
        self.x_airf_array = None
        self.y_airf_array = None
        self.z_airf_array = None
        self.va_array = None
        self.chord_array = None
        self.width_array = None
        self.y_coords = None

    def solve(self, body_aero, gamma_distribution: np.ndarray = None) -> dict:
        """Solve aerodynamic model for circulation distribution and forces.

        Args:
            body_aero: BodyAerodynamics object with configured geometry and flow conditions.
            gamma_distribution (np.ndarray, optional): Initial circulation guess.

        Returns:
            dict: Comprehensive results dictionary with forces, moments, and distributions.

        Raises:
            ValueError: If inflow conditions are not set.
        """

        if body_aero.va is None:
            raise ValueError("Inflow conditions are not set")

        # Initialize variables here, outside the loop
        self.panels = body_aero.panels
        self.n_panels = body_aero.n_panels
        alpha_array = np.zeros(self.n_panels)
        (
            self.x_airf_array,
            self.y_airf_array,
            self.z_airf_array,
            self.va_array,
            self.chord_array,
            self.width_array,
            self.y_coords,
        ) = (
            np.zeros((self.n_panels, 3)),
            np.zeros((self.n_panels, 3)),
            np.zeros((self.n_panels, 3)),
            np.zeros((self.n_panels, 3)),
            np.zeros(self.n_panels),
            np.zeros(self.n_panels),
            np.zeros(self.n_panels),
        )
        for i, panel in enumerate(self.panels):
            self.x_airf_array[i] = panel.x_airf
            self.y_airf_array[i] = panel.y_airf
            self.z_airf_array[i] = panel.z_airf
            self.va_array[i] = panel.va
            self.chord_array[i] = panel.chord
            self.width_array[i] = panel.width
            self.y_coords[i] = panel.control_point[1]

        va_norm_array = np.linalg.norm(self.va_array, axis=1)
        va_unit_array = self.va_array / va_norm_array[:, None]

        # Calculate the new circulation distribution iteratively
        self.AIC_x, self.AIC_y, self.AIC_z = body_aero.compute_AIC_matrices(
            self.aerodynamic_model_type,
            self.core_radius_fraction,
            va_norm_array,
            va_unit_array,
        )

        if (
            gamma_distribution is not None
            and self.gamma_initial_distribution_type == "previous"
        ):
            gamma_initial = gamma_distribution
        elif (
            gamma_distribution is None
            and self.gamma_initial_distribution_type == "previous"
        ):
            gamma_initial = np.zeros(self.n_panels)
        elif self.gamma_initial_distribution_type == "elliptical":
            gamma_initial = body_aero.compute_circulation_distribution_elliptical_wing()
        elif self.gamma_initial_distribution_type == "cosine":
            gamma_initial = body_aero.compute_circulation_distribution_cosine()
        elif self.gamma_initial_distribution_type == "zero":
            gamma_initial = np.zeros(self.n_panels)
        else:
            raise ValueError(
                "Invalid gamma_initial_distribution_type, should be either: 'previous', 'elliptical', 'cosine' or 'zero'"
            )

        # === run one of the iterative loops ===
        if self.gamma_loop_type == "base":
            converged, gamma_new, alpha_array, Umag_array = self.gamma_loop(
                gamma_initial
            )
            # run again with half the relaxation factor if not converged
            if not converged:
                logging.info(
                    f" ---> Running again with half the relaxation_factor = {self.relaxation_factor / 2}"
                )
                converged, gamma_new, alpha_array, Umag_array = self.gamma_loop(
                    gamma_initial, extra_relaxation_factor=0.5
                )

        elif self.gamma_loop_type == "non_linear":
            converged, gamma_new, alpha_array, Umag_array = self.gamma_loop_non_linear(
                gamma_initial
            )
        else:
            # Instiate the stall_solvers class
            import VSM.StallSolvers as StallSolvers

            stall_solvers = StallSolvers.StallSolvers(self)

            if self.gamma_loop_type == "simonet_stall":
                converged, gamma_new, alpha_array, Umag_array = (
                    stall_solvers.gamma_loop_simonet_stall(gamma_initial)
                )
                # run again with half the relaxation factor if not converged
                if not converged:
                    logging.info(
                        f" ---> Running again with half the relaxation_factor = {self.relaxation_factor / 2}"
                    )
                    converged, gamma_new, alpha_array, Umag_array = (
                        stall_solvers.gamma_loop_simonet_stall(
                            gamma_initial, extra_relaxation_factor=0.5
                        )
                    )
            elif self.gamma_loop_type == "non_linear_simonet_stall":
                converged, gamma_new, alpha_array, Umag_array = (
                    stall_solvers.gamma_loop_non_linear_simonet_stall(gamma_initial)
                )
            elif self.gamma_loop_type == "non_linear_simonet_stall_newton_raphson":
                converged, gamma_new, alpha_array, Umag_array = (
                    stall_solvers.gamma_loop_non_linear_simonet_stall_newton_raphson(
                        gamma_initial
                    )
                )
            else:
                raise ValueError(f"Invalid gamma_loop_type")
        # Calculating results (incl. updating angle of attack for VSM)
        results = body_aero.compute_results(
            gamma_new,
            self.rho,
            self.aerodynamic_model_type,
            self.core_radius_fraction,
            self.mu,
            alpha_array,
            Umag_array,
            self.chord_array,
            self.x_airf_array,
            self.y_airf_array,
            self.z_airf_array,
            self.va_array,
            va_norm_array,
            va_unit_array,
            self.panels,
            self.is_only_f_and_gamma_output,
            self.is_with_viscous_drag_correction,
            self.reference_point,
            self.is_aoa_corrected,
        )
        return results

    def compute_aerodynamic_quantities(self, gamma: np.ndarray) -> tuple:
        """Compute aerodynamic quantities from circulation distribution.

        Args:
            gamma (np.ndarray): Circulation distribution (n x 1).

        Returns:
            tuple: (alpha_array, Umag_array, cl_array, Umagw_array)
                - alpha_array (np.ndarray): Effective angles of attack.
                - Umag_array (np.ndarray): Effective velocity magnitudes.
                - cl_array (np.ndarray): Lift coefficients.
                - Umagw_array (np.ndarray): Reference velocity magnitudes.
        """
        induced_velocity_all = np.array(
            [
                np.matmul(self.AIC_x, gamma),
                np.matmul(self.AIC_y, gamma),
                np.matmul(self.AIC_z, gamma),
            ]
        ).T  # v_ind
        relative_velocity_array = (
            self.va_array + induced_velocity_all
        )  # v_eff = v_inf + v_ind
        relative_velocity_crossz_array = jit_cross(
            relative_velocity_array, self.z_airf_array
        )  # v_eff x z
        Uinfcrossz_array = jit_cross(self.va_array, self.z_airf_array)
        v_normal_array = np.sum(self.x_airf_array * relative_velocity_array, axis=1)
        v_tangential_array = np.sum(self.y_airf_array * relative_velocity_array, axis=1)
        alpha_array = np.arctan(v_normal_array / v_tangential_array)  # alpha_eff
        Umag_array = np.linalg.norm(
            relative_velocity_crossz_array, axis=1
        )  # |v_eff x z|
        Umagw_array = np.linalg.norm(Uinfcrossz_array, axis=1)
        cl_array = np.array(
            [panel.compute_cl(alpha) for panel, alpha in zip(self.panels, alpha_array)]
        )  # cl(alpha_eff)
        return alpha_array, Umag_array, cl_array, Umagw_array

    # should add smooth circulation back
    # could add dynamic relaxation back, although it didnt work
    def gamma_loop(
        self, gamma_initial: np.ndarray, extra_relaxation_factor: float = 1.0
    ) -> tuple:
        """Standard fixed-point iteration with under-relaxation.

        Args:
            gamma_initial (np.ndarray): Initial circulation distribution.
            extra_relaxation_factor (float): Additional relaxation multiplier.

        Returns:
            tuple: (converged, gamma_new, alpha_array, Umag_array)
                - converged (bool): True if converged within tolerance.
                - gamma_new (np.ndarray): Final circulation distribution.
                - alpha_array (np.ndarray): Final angle of attack array.
                - Umag_array (np.ndarray): Final velocity magnitude array.
        """

        # looping untill max_iterations
        converged = False
        gamma_new = np.copy(gamma_initial)
        error_history = []
        for i in range(self.max_iterations):
            gamma = gamma_new
            alpha_array, Umag_array, cl_array, Umagw_array = (
                self.compute_aerodynamic_quantities(gamma)
            )
            gamma_new = (
                0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.chord_array
            )
            gamma_new = (
                1 - self.relaxation_factor * extra_relaxation_factor
            ) * gamma + self.relaxation_factor * extra_relaxation_factor * gamma_new

            # Checking convergence using normalized error
            reference_error = (
                np.amax(np.abs(gamma_new)) if np.amax(np.abs(gamma_new)) != 0 else 1e-4
            )
            normalized_error = np.amax(np.abs(gamma_new - gamma)) / reference_error
            if (normalized_error) < self.allowed_error:
                converged = True
                break

            logging.debug(f"Normalized error at iteration {i}: {normalized_error}")
            # Store error for oscillation detection
            error_history.append(normalized_error)

            # Simple oscillation detection and handling
            if i >= 5 and len(error_history) >= 3:
                if (
                    error_history[-1] > error_history[-2]
                    and error_history[-2] < error_history[-3]
                ):
                    # Oscillation detected, apply additional damping
                    gamma_new = 0.75 * gamma_new + 0.25 * gamma
                    logging.debug(
                        f"Oscillation detected at iteration {i}, applying additional damping"
                    )

        if not converged:
            logging.warning(f"NOT Converged after {self.max_iterations} iterations")
        return converged, gamma_new, alpha_array, Umag_array

    def gamma_loop_non_linear(self, gamma_initial: np.ndarray) -> tuple:
        """Nonlinear solver using robust SciPy optimization methods.

        Solves F(gamma) = gamma_new(gamma) - gamma = 0 using Broyden methods.

        Args:
            gamma_initial (np.ndarray): Initial guess for circulation distribution.

        Returns:
            tuple: (converged, gamma_new, alpha_array, Umag_array)
                - converged (bool): True if converged within tolerance.
                - gamma_new (np.ndarray): Final circulation distribution.
                - alpha_array (np.ndarray): Final angle of attack array.
                - Umag_array (np.ndarray): Final velocity magnitude array.
        """

        def compute_gamma_residual(gamma):
            _, Umag_array, cl_array, Umagw_array = self.compute_aerodynamic_quantities(
                gamma
            )
            gamma_new = (
                0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.chord_array
            )
            # Residual: difference between the computed and current gamma.
            F_val = gamma - gamma_new
            return F_val

        success = False
        if not success:
            try:
                gamma_new = broyden1(
                    lambda x: compute_gamma_residual(x),
                    gamma_initial,
                    f_tol=self.allowed_error,
                    maxiter=self.max_iterations,
                )
                if (
                    np.linalg.norm(compute_gamma_residual(gamma_new), ord=np.inf)
                    < self.allowed_error
                ):
                    success = True
                    logging.info("Converged (non_linear: broyden1)")
                else:
                    logging.warning(
                        "--> broyden1 method did not converge to desired tolerance"
                    )
            except Exception as e:
                logging.warning(f"--> broyden1 failed, running base")
        if not success:
            try:
                gamma_new = broyden2(
                    lambda x: compute_gamma_residual(x),
                    gamma_initial,
                    f_tol=self.allowed_error,
                    maxiter=self.max_iterations,
                )
                if (
                    np.linalg.norm(compute_gamma_residual(gamma_new), ord=np.inf)
                    < self.allowed_error
                ):
                    success = True
                    logging.info("Converged (non_linear: broyden2)")
                else:
                    logging.warning(
                        "--> broyden2 method did not converge to desired tolerance"
                    )
            except Exception as e:
                logging.warning(f"--> broyden2 failed, running base")

        if not success:
            return self.gamma_loop(
                gamma_initial,
            )
        if success:
            alpha_array, Umag_array, cl_array, Umagw_array = (
                self.compute_aerodynamic_quantities(gamma_new)
            )
            return True, gamma_new, alpha_array, Umag_array
