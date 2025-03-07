import numpy as np
import logging
from . import jit_cross
from scipy.optimize import newton_krylov, broyden1, broyden2


# Maurits-tips :)
# call the methods of child-classes, inhereted or composed of
# do not call the attributes of child-classes, call them through getter methods
# only pass the attributes that you need to pass, not the whole object
# only use the methods of level higher/lower, not grabbing methods from higher/lower
# class solve_VSM(Solver)
# class solve_LLM(Solver)


# make abstract class
class Solver:
    """Solver class is used to solve the aerodynamic model

    It is used to solve the circulation distribution of the wing,
    and calculate the aerodynamic forces

    Args:
        aerodynamic_model_type (str): Type of aerodynamic model to use, either 'VSM' or 'LLT' (default: 'VSM')
        density (float): Air density (default: 1.225)
        max_iterations (int): Maximum number of iterations (default: 1500)
        allowed_error (float): Allowed error for convergence (default: 1e-5)
        relaxation_factor (float): Relaxation factor for convergence (default: 0.01)
        is_with_artificial_damping (bool): Whether to apply artificial damping (default: False)
        artificial_damping (dict): Artificial damping parameters (default: {"k2": 0.1, "k4": 0.0})
        gamma_initial_distribution_type (str): Type of initial gamma distribution (default: "elliptic")
        core_radius_fraction (float): Core radius fraction (default: 1e-20)
        mu (float): Dynamic viscosity (default: 1.81e-5)
        is_only_f_and_gamma_output (bool): Whether to only output f and gamma (default: False)

    Returns:
        dict: Results of the aerodynamic model

    Methods:
        solve: Solve the aerodynamic model
        gamma_loop: Loop to calculate the circulation distribution
        calculate_artificial_damping: Calculate the artificial damping
        smooth_circulation: Smooth the circulation
    """

    def __init__(
        self,
        ### Below are all settings, with a default value, that can but don't have to be changed
        aerodynamic_model_type: str = "VSM",
        max_iterations: float = 5000,
        allowed_error: float = 1e-4,  # 1e-5,
        relaxation_factor: float = 0.01,
        core_radius_fraction: float = 1e-20,
        jacobian_eps: float = 1e-3,
        # === athmospheric properties ===
        mu: float = 1.81e-5,
        density: float = 1.225,
        # === other ===
        is_only_f_and_gamma_output: bool = False,
        reference_point: list = [-0.17, 0.00, 9.25],  # roughly the cg of V3
        # === Gamma ===
        gamma_loop_type: str = "base",
        is_with_gamma_feedback: bool = False,
        gamma_initial_distribution_type: str = "elliptical",
        # ===============================
        #       STALL MODELS
        # ===============================
        # === STALL: smooth_circulation ===
        is_smooth_circulation: bool = False,
        smoothness_factor: float = 0.08,  # for smoothing stall model
        # === STALL: artificial damping ===
        is_artificial_damping: bool = False,
        artificial_damping: dict = {"k2": 0.1, "k4": 0.0},
        # === STALL: simonet_aritificial_viscosity ===
        is_with_simonet_artificial_viscosity: bool = False,
        simonet_artificial_viscosity_fva: float = None,
        ## TODO: would be nice to having these defined here instead of inside the panel class?
        # aerodynamic_center_location: float = 0.25,
        # control_point_location: float = 0.75,
        ## TODO: these are hardcoded in the Filament, should be defined here
        # alpha_0 = 1.25643
        # nu = 1.48e-5
    ):
        self.aerodynamic_model_type = aerodynamic_model_type
        self.max_iterations = int(max_iterations)
        self.allowed_error = allowed_error
        self.relaxation_factor = relaxation_factor
        self.core_radius_fraction = core_radius_fraction
        self.jacobian_eps = jacobian_eps
        # === athmospheric properties ===
        self.mu = mu
        self.density = density
        # === other ===
        self.is_only_f_and_gamma_output = is_only_f_and_gamma_output
        self.reference_point = reference_point
        # === Gamma ===
        self.gamma_loop_type = gamma_loop_type
        self.is_with_gamma_feedback = is_with_gamma_feedback
        self.gamma_initial_distribution_type = gamma_initial_distribution_type
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

    def solve(self, body_aero, gamma_distribution=None):
        """Solve the aerodynamic model

        Args:
            body_aero (BodyAerodynamics): BodyAerodynamics object
            gamma_distribution (np.array): Initial gamma distribution (default: None)

        Returns:
            dict: Results of the aerodynamic model"""

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
        self.AIC_x, self.AIC_y, self.AIC_z = body_aero.calculate_AIC_matrices(
            self.aerodynamic_model_type,
            self.core_radius_fraction,
            va_norm_array,
            va_unit_array,
        )

        if self.is_with_gamma_feedback and gamma_distribution is not None:
            gamma_initial = gamma_distribution
        else:
            if self.gamma_initial_distribution_type == "elliptical":
                gamma_initial = (
                    body_aero.calculate_circulation_distribution_elliptical_wing()
                )
            elif self.gamma_initial_distribution_type == "cosine":
                gamma_initial = body_aero.calculate_circulation_distribution_cosine()
            elif self.gamma_initial_distribution_type == "zero":
                gamma_initial = np.zeros(self.n_panels)
            else:
                raise ValueError(
                    "Invalid gamma_initial_distribution_type, should be either: 'elliptical', 'cosine' or 'zero'"
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
        results = body_aero.calculate_results(
            gamma_new,
            self.density,
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
            self.reference_point,
        )
        return results

    def compute_aerodynamic_quantities(self, gamma):
        """Compute the aerodynamic quantities based on the aerodynamic model

        Args:
            - gamma: np.array = Circulation distribution (n x 1)

        Returns:
            alpha_array, Umag_array,cl_array, Umagw_array
                - alpha_array: np.array = Angle of attack array (n x 1)
                - Umag_array: np.array = Relative velocity magnitude array (n x 1)
                - cl_array: np.array = Lift coefficient array (n x 1)
                - Umagw_array: np.array = Relative velocity magnitude array (n x 1)
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
            [
                panel.calculate_cl(alpha)
                for panel, alpha in zip(self.panels, alpha_array)
            ]
        )  # cl(alpha_eff)
        return alpha_array, Umag_array, cl_array, Umagw_array

    # should add smooth circulation back
    # could add dynamic relaxation back
    def gamma_loop(
        self,
        gamma_initial,
        extra_relaxation_factor: float = 1.0,
    ):

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
                    logging.info(
                        f"Oscillation detected at iteration {i}, applying additional damping"
                    )

        if converged:
            logging.info(f"Converged after {i} iterations (base)")
        else:
            logging.warning(f"NOT Converged after {self.max_iterations} iterations")
        return converged, gamma_new, alpha_array, Umag_array

    # def gamma_loop(
    #     self,
    #     gamma_initial,
    #     extra_relaxation_factor: float = 1.0,
    #     tol_threshold: float = 1e-8,
    #     min_iterations: int = 3,
    # ):
    #     """
    #     An improved solver for the gamma iteration loop with adaptive relaxation,
    #     early convergence detection, and better numerical stability.

    #     Parameters:
    #     -----------
    #     gamma_initial : numpy.ndarray
    #         Initial guess for gamma values
    #     extra_relaxation_factor : float, optional
    #         Additional factor to adjust relaxation (default=1.0)
    #     tol_threshold : float, optional
    #         Minimum threshold for reference error (default=1e-8)
    #     min_iterations : int, optional
    #         Minimum number of iterations to perform (default=3)

    #     Returns:
    #     --------
    #     converged : bool
    #         Whether the solution converged
    #     gamma_new : numpy.ndarray
    #         Final computed gamma values
    #     alpha_array : numpy.ndarray
    #         Final alpha values
    #     Umag_array : numpy.ndarray
    #         Final magnitude of velocity
    #     """
    #     # Initialize variables
    #     gamma_new = np.copy(gamma_initial)
    #     converged = False

    #     # Track convergence history for adaptive relaxation
    #     error_history = []
    #     adaptive_relaxation = self.relaxation_factor * extra_relaxation_factor

    #     # Main iteration loop
    #     for i in range(self.max_iterations):
    #         # Store previous gamma for convergence check
    #         gamma_prev = np.copy(gamma_new)

    #         # Compute aerodynamic quantities
    #         alpha_array, Umag_array, cl_array, Umagw_array = (
    #             self.compute_aerodynamic_quantities(gamma_new)
    #         )

    #         # Calculate new gamma values
    #         gamma_raw = (
    #             0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.chord_array
    #         )

    #         # Apply circulation smoothing if enabled
    #         if hasattr(self, "is_smooth_circulation") and self.is_smooth_circulation:
    #             damp, is_damping_applied = self.smooth_circulation(
    #                 circulation=gamma_new,
    #                 smoothness_factor=self.smoothness_factor,
    #                 damping_factor=0.5,
    #             )
    #             if is_damping_applied:
    #                 gamma_raw += damp

    #         # Apply relaxation to update gamma
    #         gamma_new = (
    #             1 - adaptive_relaxation
    #         ) * gamma_new + adaptive_relaxation * gamma_raw

    #         # Calculate error metrics
    #         absolute_error = np.abs(gamma_new - gamma_prev)
    #         max_error = np.amax(absolute_error)
    #         reference_error = max(np.amax(np.abs(gamma_new)), tol_threshold)
    #         normalized_error = max_error / reference_error

    #         # Store error for adaptive relaxation
    #         error_history.append(normalized_error)

    #         # Adaptive relaxation factor adjustment after a few iterations
    #         if i >= 3 and i % 2 == 0:
    #             # If error is increasing, reduce relaxation factor
    #             if len(error_history) >= 3 and error_history[-1] > error_history[-3]:
    #                 adaptive_relaxation = max(0.1, adaptive_relaxation * 0.8)
    #             # If error is decreasing rapidly, gradually increase relaxation
    #             elif (
    #                 len(error_history) >= 3
    #                 and error_history[-1] < 0.5 * error_history[-3]
    #             ):
    #                 adaptive_relaxation = min(1.0, adaptive_relaxation * 1.1)

    #         # Check convergence criteria
    #         if i >= min_iterations and normalized_error < self.allowed_error:
    #             converged = True
    #             break

    #     # Log convergence status
    #     if converged:
    #         logging.info(
    #             f"Converged after {i+1} iterations with normalized error: {normalized_error:.6e}"
    #         )
    #     else:
    #         logging.warning(
    #             f"NOT Converged after {self.max_iterations} iterations. Final error: {normalized_error:.6e}"
    #         )

    #     return converged, gamma_new, alpha_array, Umag_array

    def gamma_loop_non_linear(self, gamma_initial):
        """
        Nonlinear solver to compute the circulation distribution, i.e. solve
        F(gamma) = gamma_new(gamma) - gamma = 0
        using a robust non_linear solver from SciPy.

        Args:
            gamma_initial (np.array): Initial guess for the circulation distribution (n,).

        Returns:
            tuple: (converged (bool), final gamma distribution (np.array),
                    angle of attack array (np.array), relative velocity magnitude array (np.array))
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
