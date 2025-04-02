import numpy as np
import logging
from . import jit_cross


class Solver:
    """Solver class for solving the aerodynamic model.

    This class computes the circulation distribution over a wing and calculates
    the resulting aerodynamic forces. It supports several iterative methods for
    solving the nonlinear system associated with the aerodynamic model.

    Attributes:
        aerodynamic_model_type (str): Type of aerodynamic model to use ('VSM' or 'LLT', default: 'VSM').
        max_iterations (int): Maximum number of iterations for convergence (default: 5000).
        allowed_error (float): Allowed normalized error for convergence (default: 1e-6).
        relaxation_factor (float): Base relaxation factor for iterative updates (default: 0.01).
        core_radius_fraction (float): Fraction of core radius used in induced velocity calculations (default: 1e-20).
        gamma_loop_type (str): Type of iterative loop to use ('base', 'non_linear', or stall-related methods; default: 'base').
        gamma_initial_distribution_type (str): Type of initial gamma distribution ('previous', 'elliptical', 'cosine', or 'zero'; default: 'elliptical').
        is_only_f_and_gamma_output (bool): If True, only output force and gamma values (default: False).
        reference_point (list): Reference point in space for aerodynamic calculations (default: [-0.17, 0.00, 9.25]).
        mu (float): Dynamic viscosity of the fluid (default: 1.81e-5).
        density (float): Fluid density (default: 1.225).
        is_smooth_circulation (bool): If True, applies smoothing to the circulation distribution (default: False).
        smoothness_factor (float): Smoothing factor for circulation (default: 0.08).
        is_artificial_damping (bool): Defines if artificial damping is applied for stall modeling (default: False).
        artificial_damping (dict): Parameters for artificial damping (default: {"k2": 0.1, "k4": 0.0}).
        is_with_simonet_artificial_viscosity (bool): If True, uses Simonet's artificial viscosity model (default: False).
        _simonet_artificial_viscosity_fva (float): Parameter for Simonet's artificial viscosity (default: None).

    Methods:
        solve(body_aero, gamma_distribution=None):
            Solves the aerodynamic model by computing the circulation distribution and
            resulting forces.
        compute_aerodynamic_quantities(gamma):
            Computes aerodynamic quantities such as angle of attack, relative velocity,
            and lift coefficient based on a given circulation distribution.
        gamma_loop(gamma_initial, extra_relaxation_factor=1.0):
            Iterative solver for updating the circulation distribution using a relaxation scheme.
        gamma_loop_non_linear(gamma_initial):
            Nonlinear solver that applies robust methods (Broyden) to compute the circulation distribution.
    """

    def __init__(
        self,
        aerodynamic_model_type: str = "VSM",
        max_iterations: float = 5000,
        allowed_error: float = 1e-6,  # 1e-5,
        relaxation_factor: float = 0.01,
        core_radius_fraction: float = 1e-20,
        gamma_loop_type: str = "base",
        gamma_initial_distribution_type: str = "elliptical",
        is_only_f_and_gamma_output: bool = False,
        reference_point: list = [-0.17, 0.00, 9.25],  # roughly the cg of V3
        # === athmospheric properties ===
        mu: float = 1.81e-5,
        density: float = 1.225,
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
        self.gamma_loop_type = gamma_loop_type
        self.gamma_initial_distribution_type = gamma_initial_distribution_type
        self.is_only_f_and_gamma_output = is_only_f_and_gamma_output
        self.reference_point = reference_point
        # === athmospheric properties ===
        self.mu = mu
        self.density = density
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
            gamma_initial = (
                body_aero.calculate_circulation_distribution_elliptical_wing()
            )
        elif self.gamma_initial_distribution_type == "cosine":
            gamma_initial = body_aero.calculate_circulation_distribution_cosine()
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
    # could add dynamic relaxation back, although it didnt work
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
                    logging.debug(
                        f"Oscillation detected at iteration {i}, applying additional damping"
                    )

        if converged:
            logging.info(f"Converged after {i} iterations (base)")
        else:
            logging.warning(f"NOT Converged after {self.max_iterations} iterations")
        return converged, gamma_new, alpha_array, Umag_array

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
        if success:
            alpha_array, Umag_array, cl_array, Umagw_array = (
                self.compute_aerodynamic_quantities(gamma_new)
            )
            return True, gamma_new, alpha_array, Umag_array
