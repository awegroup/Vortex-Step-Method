import numpy as np
import logging
from . import jit_cross

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
        allowed_error: float = 1e-5,  # 1e-5,
        relaxation_factor: float = 0.01,
        core_radius_fraction: float = 1e-20,
        jacobian_eps: float = 1e-3,
        # === athmospheric properties ===
        mu: float = 1.81e-5,
        density: float = 1.225,
        # === other ===
        is_only_f_and_gamma_output: bool = False,
        reference_point: list = [-0.17, 0.00, 9.25],
        # === Gamma initialisation ===
        is_with_gamma_feedback: bool = False,
        gamma_initial_distribution_type: str = "elliptical",
        # === STALL: smooth_circulation ===
        is_smooth_circulation: bool = False,
        smoothness_factor: float = 0.08,  # for smoothing stall model
        # === STALL: artificial damping ===
        is_artificial_damping: bool = False,
        artificial_damping: dict = {"k2": 0.1, "k4": 0.0},
        # === STALL: simonet_aritificial_viscosity ===
        is_with_simonet_artificial_viscosity: bool = False,
        simonet_artificial_viscosity_fva: float = None,
        # === gamma_loop_type ===
        gamma_loop_type: str = "base",
        ## TODO: would be nice to having these defined here instead of inside the panel class?
        # aerodynamic_center_location: float = 0.25,
        # control_point_location: float = 0.75,
        ## TODO: these are hardcoded in the Filament, should be defined here
        # alpha_0 = 1.25643
        # nu = 1.48e-5
    ):
        self.aerodynamic_model_type = aerodynamic_model_type
        self.density = density
        self.max_iterations = int(max_iterations)
        self.allowed_error = allowed_error
        self.relaxation_factor = relaxation_factor
        self.jacobian_eps = jacobian_eps

        self.gamma_initial_distribution_type = gamma_initial_distribution_type
        self.core_radius_fraction = core_radius_fraction
        self.mu = mu
        self.is_only_f_and_gamma_output = is_only_f_and_gamma_output
        self.is_with_gamma_feedback = is_with_gamma_feedback
        self.reference_point = reference_point
        self.smoothness_factor = smoothness_factor
        # stall model things
        self.is_smooth_circulation = is_smooth_circulation
        self.is_artificial_damping = is_artificial_damping
        self.artificial_damping = artificial_damping
        self.is_with_simonet_artificial_viscosity = is_with_simonet_artificial_viscosity
        self._simonet_artificial_viscosity_fva = simonet_artificial_viscosity_fva
        # gamma_loop type
        self.gamma_loop_type = gamma_loop_type

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

    ### properties
    @property
    def simonet_artificial_viscosity_fva(self):
        return self._simonet_artificial_viscosity_fva

    ### setter functions
    @simonet_artificial_viscosity_fva.setter
    def simonet_artificial_viscosity_fva(self, value):
        """
        Set the simonet_artificial_viscosity_fva value with optional validation.

        Parameters:
            value (float or int): The new value for simonet_artificial_viscosity_fva.

        Raises:
            ValueError: If the provided value is not a numeric type.
        """
        # Example validation: ensure the value is numeric.
        if not isinstance(value, (int, float)):
            raise ValueError(
                "simonet_artificial_viscosity_fva must be a numeric value."
            )
        self._simonet_artificial_viscosity_fva = value

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

        # initialize gamma distribution inside
        # if (
        #     gamma_distribution is None
        #     and self.gamma_initial_distribution_type == "elliptic"
        # ) or not self.is_with_gamma_feedback:
        #     gamma_initial = (
        #         body_aero.calculate_circulation_distribution_elliptical_wing()
        #     )

        # elif len(gamma_distribution) == n_panels:
        #     gamma_initial = gamma_distribution

        if self.is_with_gamma_feedback and gamma_distribution is not None:
            gamma_initial = gamma_distribution
        else:
            # gamma_0 = (
            #     self.density
            #     * np.average(va_norm_array)
            #     * np.max(self.chord_array)
            #     / self.mu
            # ) / 1e5
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

        logging.debug(
            f"Initial gamma_new: {gamma_initial} . is_with_gamma_feedback: {self.is_with_gamma_feedback}",
        )

        ##TODO: remove
        # defining upfront for simonet_artificial_viscosity
        if (
            self.is_with_simonet_artificial_viscosity
            and self.simonet_artificial_viscosity_fva is None
        ):
            root_chord = np.max(self.chord_array)
            wing = body_aero.wings[0]
            projected_area = wing.calculate_projected_area()
            self.simonet_artificial_viscosity_fva = 0.16 * root_chord * projected_area
            print(
                f"Simonet Artificial Viscosity fva: {self.simonet_artificial_viscosity_fva:.4f} (0.005, 5e-4 was optimal before)"
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
        elif self.gamma_loop_type == "simonet_stall":
            converged, gamma_new, alpha_array, Umag_array = (
                self.gamma_loop_simonet_stall(gamma_initial)
            )
            # run again with half the relaxation factor if not converged
            if not converged:
                logging.info(
                    f" ---> Running again with half the relaxation_factor = {self.relaxation_factor / 2}"
                )
                converged, gamma_new, alpha_array, Umag_array = (
                    self.gamma_loop_simonet_stall(
                        gamma_initial, extra_relaxation_factor=0.5
                    )
                )
        elif self.gamma_loop_type == "non_linear":
            converged, gamma_new, alpha_array, Umag_array = self.gamma_loop_non_linear(
                gamma_initial
            )
        elif self.gamma_loop_type == "non_linear_simonet_stall":
            converged, gamma_new, alpha_array, Umag_array = (
                self.gamma_loop_non_linear_simonet_stall(gamma_initial)
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
            self.reference_point,  # roughly the cg of V3
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
        )
        Uinfcrossz_array = jit_cross(self.va_array, self.z_airf_array)
        v_normal_array = np.sum(self.x_airf_array * relative_velocity_array, axis=1)
        v_tangential_array = np.sum(self.y_airf_array * relative_velocity_array, axis=1)
        alpha_array = np.arctan(v_normal_array / v_tangential_array)  # alpha_eff
        Umag_array = np.linalg.norm(relative_velocity_crossz_array, axis=1)
        Umagw_array = np.linalg.norm(Uinfcrossz_array, axis=1)
        cl_array = np.array(
            [
                panel.calculate_cl(alpha)
                for panel, alpha in zip(self.panels, alpha_array)
            ]
        )  # cl(alpha_eff)
        return alpha_array, Umag_array, cl_array, Umagw_array

    def compute_gamma_new(
        self,
        gamma,
    ):
        """Compute the new gamma distribution based on the aerodynamic model

        Args:
            - gamma: np.array = Initial gamma distribution (n x 1)

        Returns:
            - gamma_new: np.array = New gamma distribution (n x 1)

        """
        alpha_array, Umag_array, cl_array, Umagw_array = (
            self.compute_aerodynamic_quantities(gamma)
        )
        gamma_new = 0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.chord_array
        return gamma_new, alpha_array, Umag_array

    # ===============================
    #           Gamma Loop
    # ===============================
    def gamma_loop(
        self,
        gamma_initial,
        extra_relaxation_factor: float = 1.0,
    ):
        """Loop to calculate the circulation distribution

        Args:

        """

        # looping untill max_iterations
        converged = False
        gamma_new = np.copy(gamma_initial)
        for i in range(self.max_iterations):

            gamma = np.array(gamma_new)
            alpha_array, Umag_array, cl_array, Umagw_array = (
                self.compute_aerodynamic_quantities(gamma)
            )
            gamma_new = (
                0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.chord_array
            )

            if self.is_smooth_circulation:
                damp, is_damping_applied = self.smooth_circulation(
                    circulation=gamma,
                    smoothness_factor=self.smoothness_factor,
                    damping_factor=0.5,
                )
            else:
                damp = 0
                is_damping_applied = False

            gamma_new = (
                (1 - self.relaxation_factor * extra_relaxation_factor) * gamma
                + self.relaxation_factor * extra_relaxation_factor * gamma_new
                + damp
            )

            # TODO: could add a dynamic relaxation factor here, although first tries failed, so not super easy

            # Checking Convergence
            reference_error = np.amax(np.abs(gamma_new))
            if reference_error == 0:
                reference_error = 1e-4
            error = np.amax(np.abs(gamma_new - gamma))
            normalized_error = error / reference_error

            logging.debug(
                "Iteration: %d, normalized_error: %f, is_damping_applied: %s",
                i,
                normalized_error,
                is_damping_applied,
            )

            # relative error
            if normalized_error < self.allowed_error:
                # if error smaller than limit, stop iteration cycle
                converged = True
                break

        if converged:
            logging.info(f"Converged after {i} iterations")
        else:
            logging.warning(f"NO convergences after {self.max_iterations} iterations")
        return converged, gamma_new, alpha_array, Umag_array

    # ===============================
    #   Gamma Loop Simonet Stall
    # ===============================
    def gamma_loop_simonet_stall(
        self,
        gamma_initial,
        extra_relaxation_factor: float = 1.0,
    ):
        """Loop to calculate the circulation distribution

        Args:

        """

        # ===============================
        # Helper: compute finite-difference coefficients and second derivative
        # ===============================
        def compute_abc_prime():
            """
            Compute the finite-difference coefficients a', b', c' for the boundary nodes.
            """
            # Boundary nodes
            d1 = self.y_coords[1] - self.y_coords[0]
            d2 = self.y_coords[2] - self.y_coords[0]
            a_prime = 2.0 / (d1 * d2)
            b_prime = 2.0 / ((d1 - d2) * d1)
            c_prime = -2.0 / ((d1 - d2) * d2)
            return a_prime, b_prime, c_prime

        def compute_abc(i):
            """
            Compute the finite-difference coefficients a, b, c for the interior nodes.
            """
            d1 = self.y_coords[i + 1] - self.y_coords[i]
            d_1 = self.y_coords[i] - self.y_coords[i - 1]
            a = 2.0 / ((d1 + d_1) * d_1)
            b = -2.0 / (d1 * d_1)
            c = 2.0 / ((d1 + d_1) * d1)
            return a, b, c

        def compute_abc_dbl_prime():
            """
            Compute the finite-difference coefficients a'', b'', c'' for the boundary nodes.
            """
            # Boundary nodes
            d1 = self.y_coords[-1] - self.y_coords[-2]
            d2 = self.y_coords[-1] - self.y_coords[-3]
            a_dbl_prime = 2.0 / (d1 * d2)
            b_dbl_prime = 2.0 / ((d1 - d2) * d1)
            c_dbl_prime = -2.0 / ((d1 - d2) * d2)
            return a_dbl_prime, b_dbl_prime, c_dbl_prime

        def compute_second_derivative_of_gamma(gamma):
            """
            Compute the second derivative of gamma along the span using the
            exact boundary stencils from eq. (21) in your reference, plus the
            interior formulas.

            In the reference, i=1 and i=n are the boundary nodes. Here, we use
            0-based Python indexing, so:
            * i=0 in Python corresponds to i=1 in the reference
            * i=n-1 in Python corresponds to i=n in the reference

            For interior nodes (1 <= i <= n-2 in Python), we use:
                (∂²Γ/∂y²)_i = a_i Γ[i-1] + b_i Γ[i] + c_i Γ[i+1]

            For i=0 (the first node in Python, i=1 in reference):
                (∂²Γ/∂y²)_0 = a' Γ[0] + b' Γ[1] + c' Γ[2]

            For i=n-1 (the last node in Python, i=n in reference):
                (∂²Γ/∂y²)_(n-1) = a'' Γ[n-1] + b'' Γ[n-2] + c'' Γ[n-3]

            NOTE: The definitions of a', b', c', a'', b'', c'' come from eq. (22).
                We also assume y_coords is sorted along the span (y[0] < y[1] < ...).
            """

            n = self.n_panels
            second_deriv = np.zeros(n)

            # Quick exit if too few points
            if n < 3:
                raise ValueError(
                    "At least 3 panels are required to compute the second derivative."
                )

            # ===============================
            # 1) Boundary formula for i=0 => eq. (21) at i=1 in reference
            # ===============================
            a_prime, b_prime, c_prime = compute_abc_prime()
            second_deriv[0] = (
                a_prime * gamma[0] + b_prime * gamma[1] + c_prime * gamma[2]
            )

            # ===============================
            # 2) Interior points: i = 1..(n-2)
            # ===============================
            for i in range(1, n - 1):
                a, b, c = compute_abc(i)
                second_deriv[i] = a * gamma[i - 1] + b * gamma[i] + c * gamma[i + 1]

            # ===============================
            # 3) Boundary formula for i=n-1 => eq. (21) at i=n in reference
            # ===============================
            # here it is not y_coords[n], because of python 0-based indexing
            a_dbl_prime, b_dbl_prime, c_dbl_prime = compute_abc_dbl_prime()
            second_deriv[n - 1] = (
                a_dbl_prime * gamma[n - 1]
                + b_dbl_prime * gamma[n - 2]
                + c_dbl_prime * gamma[n - 3]
            )

            return second_deriv

        def compute_G_residual(
            gamma,
            mu_array,
        ):
            """Computes the residual vector G for the non_linear system, including the artificial viscosity term.

            This function follows eq. (20) of Simonet et al., computing

                G = gamma - gamma_new - (mu_array * (∂²gamma/∂y²))

            Args:
                gamma (np.ndarray): Current circulation distribution (n x 1).
                mu_array (np.ndarray): Artificial viscosity coefficients (n x 1).

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]:
                    - Residual vector G (n x 1).
                    - Angle of attack array (n x 1).
                    - Relative velocity magnitude array (n x 1).
            """
            alpha_array, Umag_array, cl_array, Umagw_array = (
                self.compute_aerodynamic_quantities(gamma)
            )
            gamma_new = (
                0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.chord_array
            )
            second_derivative_of_gamma = compute_second_derivative_of_gamma(gamma)
            # G_residual = gamma - gamma_new - (mu_array * second_derivative_of_gamma)
            G_residual = (gamma_new + mu_array * second_derivative_of_gamma) - gamma
            return G_residual, alpha_array, Umag_array

        def compute_jacobian_K(F_func, gamma, *args):
            """
            Computes the Jacobian matrix K using finite differences.

            Args:
                F_func (callable): Function to compute the residual vector.
                gamma (np.array): Current circulation distribution.
                eps (float): Finite difference step size.
                args (tuple): Additional arguments for the residual function.

            Returns:
                np.ndarray: Jacobian matrix (n x n).
            """
            n = self.n_panels
            K = np.zeros((n, n))
            F0, _, _ = F_func(gamma, *args)
            for i in range(n):
                gamma_eps = gamma.copy()
                gamma_eps[i] += self.jacobian_eps
                F1, _, _ = F_func(gamma_eps, *args)
                K[:, i] = (F1 - F0) / self.jacobian_eps
            return K

        # ===============================
        # Helper: compute artificial viscosity coefficients mu_tilde_i (eq. 25)
        # -------------------------------
        def compute_mu_tilde_array(K):
            """Computes μ_i from the diagonal of the Jacobian and the central finite-difference coefficient.

            Args:
                - K (np.ndarray): Jacobian matrix (n x n).
                - y_coords (np.array): Spanwise coordinates of the panels.
            Returns
                - mu_tilde_array (np.array): Artificial viscosity coefficients.
            """
            n = self.n_panels
            mu_tilde_array = np.zeros(n)

            # first panel
            a_prime, _, _ = compute_abc_prime()
            mu_tilde_array[0] = np.minimum(0.0, K[0, 0] / a_prime)

            for i in range(1, n - 1):
                _, b, _ = compute_abc(i)
                mu_tilde_array[i] = np.maximum(0.0, K[i, i] / b)

            # last panel
            a_dbl_prime, _, _ = compute_abc_dbl_prime()
            mu_tilde_array[n - 1] = np.minimum(0.0, K[n - 1, n - 1] / a_dbl_prime)
            return mu_tilde_array

        # ===============================
        #   Initializing the mu_array
        # ===============================
        mu_array = np.zeros(self.n_panels)
        gamma = np.copy(gamma_initial)
        K = compute_jacobian_K(compute_G_residual, gamma, mu_array)
        logging.debug(f"Diagonal of K: {np.diag(K)}")
        mu_tilde_array = compute_mu_tilde_array(K)
        logging.debug(f"Mu_tilde_array: {mu_tilde_array}")

        # Using eq 26. and interpreting b as the local spanwise panel width
        mu_array = self.simonet_artificial_viscosity_fva * (
            mu_tilde_array / np.sum(mu_tilde_array * self.width_array)
        )
        # looping untill max_iterations
        converged = False
        gamma_new = np.copy(gamma_initial)
        for i in range(self.max_iterations):

            gamma = np.array(gamma_new)
            alpha_array, Umag_array, cl_array, Umagw_array = (
                self.compute_aerodynamic_quantities(gamma)
            )
            gamma_new = (
                0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.chord_array
            )

            if self.is_smooth_circulation:
                damp, is_damping_applied = self.smooth_circulation(
                    circulation=gamma,
                    smoothness_factor=self.smoothness_factor,
                    damping_factor=0.5,
                )
            elif self.is_with_simonet_artificial_viscosity:
                damp = mu_array * compute_second_derivative_of_gamma(gamma)
                is_damping_applied = True
            else:
                damp = 0
                is_damping_applied = False

            gamma_new = (
                (1 - self.relaxation_factor * extra_relaxation_factor) * gamma
                + self.relaxation_factor * extra_relaxation_factor * gamma_new
                + damp
            )

            # TODO: could add a dynamic relaxation factor here, although first tries failed, so not super easy

            # Checking Convergence
            reference_error = np.amax(np.abs(gamma_new))
            if reference_error == 0:
                reference_error = 1e-4
            error = np.amax(np.abs(gamma_new - gamma))
            normalized_error = error / reference_error

            logging.debug(
                "Iteration: %d, normalized_error: %f, is_damping_applied: %s",
                i,
                normalized_error,
                is_damping_applied,
            )

            # relative error
            if normalized_error < self.allowed_error:
                # if error smaller than limit, stop iteration cycle
                converged = True
                break

        if converged:
            logging.info(f"Converged after {i} iterations")
        else:
            logging.warning(f"NO convergences after {self.max_iterations} iterations")
        return converged, gamma_new, alpha_array, Umag_array

    # ===============================
    #   Gamma Loop Nonlinear
    # ===============================
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
        from scipy.optimize import root, newton_krylov

        def compute_gamma_residual(gamma):
            alpha_array, Umag_array, cl_array, Umagw_array = (
                self.compute_aerodynamic_quantities(gamma)
            )
            gamma_new = (
                0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.chord_array
            )
            # Residual: difference between the computed and current gamma.
            F_val = gamma - gamma_new
            return F_val, alpha_array, Umag_array

        # Define the residual function: only return F(gamma)
        def F(gamma):
            # Here, self.compute_gamma_new is assumed to compute:
            #   gamma_new, alpha_array, Umag_array = compute_aerodynamic_quantities(gamma)
            # and the residual is F = gamma_new - gamma.
            F_val, _, _ = compute_gamma_residual(gamma)
            return F_val

        # First, try the Newton–Krylov method.
        try:
            # Set tolerances
            atol = self.allowed_error * 1e-1
            f_tol = self.allowed_error

            # Call newton_krylov without unsupported keywords.
            sol = newton_krylov(
                F,
                gamma_initial,
                method="lgmres",
                f_tol=f_tol,
                x_tol=atol,
                maxiter=self.max_iterations,
                verbose=False,
            )
            gamma_new = sol

            # Compute final aerodynamic quantities.
            alpha_array, Umag_array, cl_array, Umagw_array = (
                self.compute_aerodynamic_quantities(gamma_new)
            )

            converged = True
            logging.info("Converged using newton_krylov method")

            return converged, gamma_new, alpha_array, Umag_array

        except Exception as e:
            logging.warning(
                f"newton_krylov method failed: {str(e)}. \nFalling back to regular gamma_loop."
            )
            return self.gamma_loop(
                gamma_initial,
            )

    # ===============================
    #   Gamma Loop Nonlinear Simonet Stall
    # ===============================
    def gamma_loop_non_linear_simonet_stall(self, gamma_initial):
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
        from scipy.optimize import root, newton_krylov
        from functools import partial

        # ===============================
        # Helper: compute finite-difference coefficients and second derivative
        # ===============================
        def compute_abc_prime():
            """
            Compute the finite-difference coefficients a', b', c' for the boundary nodes.
            """
            # Boundary nodes
            d1 = self.y_coords[1] - self.y_coords[0]
            d2 = self.y_coords[2] - self.y_coords[0]
            a_prime = 2.0 / (d1 * d2)
            b_prime = 2.0 / ((d1 - d2) * d1)
            c_prime = -2.0 / ((d1 - d2) * d2)
            return a_prime, b_prime, c_prime

        def compute_abc(i):
            """
            Compute the finite-difference coefficients a, b, c for the interior nodes.
            """
            d1 = self.y_coords[i + 1] - self.y_coords[i]
            d_1 = self.y_coords[i] - self.y_coords[i - 1]
            a = 2.0 / ((d1 + d_1) * d_1)
            b = -2.0 / (d1 * d_1)
            c = 2.0 / ((d1 + d_1) * d1)
            return a, b, c

        def compute_abc_dbl_prime():
            """
            Compute the finite-difference coefficients a'', b'', c'' for the boundary nodes.
            """
            # Boundary nodes
            d1 = self.y_coords[-1] - self.y_coords[-2]
            d2 = self.y_coords[-1] - self.y_coords[-3]
            a_dbl_prime = 2.0 / (d1 * d2)
            b_dbl_prime = 2.0 / ((d1 - d2) * d1)
            c_dbl_prime = -2.0 / ((d1 - d2) * d2)
            return a_dbl_prime, b_dbl_prime, c_dbl_prime

        def compute_second_derivative_of_gamma(gamma):
            """
            Compute the second derivative of gamma along the span using the
            exact boundary stencils from eq. (21) in your reference, plus the
            interior formulas.

            In the reference, i=1 and i=n are the boundary nodes. Here, we use
            0-based Python indexing, so:
            * i=0 in Python corresponds to i=1 in the reference
            * i=n-1 in Python corresponds to i=n in the reference

            For interior nodes (1 <= i <= n-2 in Python), we use:
                (∂²Γ/∂y²)_i = a_i Γ[i-1] + b_i Γ[i] + c_i Γ[i+1]

            For i=0 (the first node in Python, i=1 in reference):
                (∂²Γ/∂y²)_0 = a' Γ[0] + b' Γ[1] + c' Γ[2]

            For i=n-1 (the last node in Python, i=n in reference):
                (∂²Γ/∂y²)_(n-1) = a'' Γ[n-1] + b'' Γ[n-2] + c'' Γ[n-3]

            NOTE: The definitions of a', b', c', a'', b'', c'' come from eq. (22).
                We also assume y_coords is sorted along the span (y[0] < y[1] < ...).
            """

            n = self.n_panels
            second_deriv = np.zeros(n)

            # Quick exit if too few points
            if n < 3:
                raise ValueError(
                    "At least 3 panels are required to compute the second derivative."
                )

            # ===============================
            # 1) Boundary formula for i=0 => eq. (21) at i=1 in reference
            # ===============================
            a_prime, b_prime, c_prime = compute_abc_prime()
            second_deriv[0] = (
                a_prime * gamma[0] + b_prime * gamma[1] + c_prime * gamma[2]
            )

            # ===============================
            # 2) Interior points: i = 1..(n-2)
            # ===============================
            for i in range(1, n - 1):
                a, b, c = compute_abc(i)
                second_deriv[i] = a * gamma[i - 1] + b * gamma[i] + c * gamma[i + 1]

            # ===============================
            # 3) Boundary formula for i=n-1 => eq. (21) at i=n in reference
            # ===============================
            # here it is not y_coords[n], because of python 0-based indexing
            a_dbl_prime, b_dbl_prime, c_dbl_prime = compute_abc_dbl_prime()
            second_deriv[n - 1] = (
                a_dbl_prime * gamma[n - 1]
                + b_dbl_prime * gamma[n - 2]
                + c_dbl_prime * gamma[n - 3]
            )

            return second_deriv

        def compute_G_residual(
            gamma,
            mu_array,
        ):
            """Computes the residual vector G for the non_linear system, including the artificial viscosity term.

            This function follows eq. (20) of Simonet et al., computing

                G = gamma - gamma_new - (mu_array * (∂²gamma/∂y²))

            Args:
                gamma (np.ndarray): Current circulation distribution (n x 1).
                mu_array (np.ndarray): Artificial viscosity coefficients (n x 1).

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]:
                    - Residual vector G (n x 1).
                    - Angle of attack array (n x 1).
                    - Relative velocity magnitude array (n x 1).
            """
            alpha_array, Umag_array, cl_array, Umagw_array = (
                self.compute_aerodynamic_quantities(gamma)
            )
            gamma_new = (
                0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.chord_array
            )
            second_derivative_of_gamma = compute_second_derivative_of_gamma(gamma)
            G_residual = (gamma_new + mu_array * second_derivative_of_gamma) - gamma
            return G_residual, alpha_array, Umag_array

        def compute_G_residual_newton_krylov(
            gamma,
            mu_array,
        ):
            """Computes the residual vector G for the non_linear system, including the artificial viscosity term.

            This function follows eq. (20) of Simonet et al., computing

                G = gamma - gamma_new - (mu_array * (∂²gamma/∂y²))

            Args:
                gamma (np.ndarray): Current circulation distribution (n x 1).
                mu_array (np.ndarray): Artificial viscosity coefficients (n x 1).

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]:
                    - Residual vector G (n x 1).
                    - Angle of attack array (n x 1).
                    - Relative velocity magnitude array (n x 1).
            """
            _, Umag_array, cl_array, Umagw_array = self.compute_aerodynamic_quantities(
                gamma
            )
            gamma_new = (
                0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.chord_array
            )
            second_derivative_of_gamma = compute_second_derivative_of_gamma(gamma)
            G_residual = (gamma_new + mu_array * second_derivative_of_gamma) - gamma
            return G_residual

        def compute_F_residual(gamma):
            _, Umag_array, cl_array, Umagw_array = self.compute_aerodynamic_quantities(
                gamma
            )
            gamma_new = (
                0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.chord_array
            )
            return (gamma_new) - gamma

        # def compute_jacobian(F_func, gamma):
        #     n = self.n_panels
        #     K = np.zeros((n, n))
        #     F0 = F_func(gamma)
        #     for i in range(n):
        #         gamma_eps = gamma.copy()
        #         gamma_eps[i] += self.jacobian_eps
        #         F1 = F_func(gamma_eps)
        #         K[:, i] = (F0 - F1) / self.jacobian_eps
        #     return K

        def compute_jacobian(F_func, gamma):
            n = len(gamma)
            K = np.zeros((n, n))
            F0 = F_func(gamma)
            # Typical small number:
            # e.g. ~1e-7 or sqrt(machine_epsilon)
            base_eps = 1e-7

            for i in range(n):
                eps = base_eps * max(1.0, abs(gamma[i]))  # relative scale
                gamma_eps = gamma.copy()
                gamma_eps[i] += eps
                F1 = F_func(gamma_eps)
                K[:, i] = (F1 - F0) / eps

            return K

        import casadi as ca

        def compute_jacobian_casadi_numerical(gamma_new):
            n = len(gamma_new)

            # Convert to CasADi symbols
            gamma_sym = ca.SX.sym("gamma", n)

            # Create a CasADi function that wraps your numpy-based function
            F_expr = []

            # For each input, create output symbol
            for i in range(n):
                gamma_perturb = gamma_new.copy()
                # Small perturbation
                eps = 1e-6
                gamma_perturb[i] += eps
                # Forward evaluation
                f_forward = compute_F_residual(gamma_perturb)
                # Backward evaluation
                gamma_perturb[i] = gamma_new[i] - eps
                f_backward = compute_F_residual(gamma_perturb)
                # Central difference
                df = (f_forward - f_backward) / (2 * eps)
                F_expr.append(df)

            # Convert list to matrix
            J = np.column_stack(F_expr)

            return J

        # ===============================
        # Helper: compute artificial viscosity coefficients mu_tilde_i (eq. 25)
        # -------------------------------
        def compute_mu_tilde_array(K):
            """Computes μ_i from the diagonal of the Jacobian and the central finite-difference coefficient.

            Args:
                - K (np.ndarray): Jacobian matrix (n x n).
                - y_coords (np.array): Spanwise coordinates of the panels.
            Returns
                - mu_tilde_array (np.array): Artificial viscosity coefficients.
            """
            n = self.n_panels
            mu_tilde_array = np.zeros(n)

            # first panel
            a_prime, _, _ = compute_abc_prime()
            mu_tilde_array[0] = np.minimum(0.0, K[0, 0] / a_prime)

            for i in range(1, n - 1):
                _, b, _ = compute_abc(i)
                mu_tilde_array[i] = np.maximum(0.0, K[i, i] / b)

            # last panel
            a_dbl_prime, _, _ = compute_abc_dbl_prime()
            mu_tilde_array[n - 1] = np.minimum(0.0, K[n - 1, n - 1] / a_dbl_prime)
            logging.debug(f"a_prime: {a_prime}, b: {b}, a_dbl_prime: {a_dbl_prime}")
            return mu_tilde_array

        # ===============================
        # Doing a single iteration
        # ===============================
        gamma = np.copy(gamma_initial)
        alpha_array, Umag_array, cl_array, Umagw_array = (
            self.compute_aerodynamic_quantities(gamma)
        )
        gamma_new = 0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.chord_array
        gamma_new = (
            1 - self.relaxation_factor
        ) * gamma + self.relaxation_factor * gamma_new

        # ===============================
        #   Initializing the mu_array
        # # ===============================
        # mu_array = np.zeros(self.n_panels)
        # # gamma = np.copy(gamma_initial)
        # J = compute_jacobian(compute_F_residual, gamma_new)
        # logging.info(f"Jacobian J diagonal: {np.diag(J)}")
        # J = compute_jacobian_casadi_numerical(gamma_new)
        # logging.info(f"Casadi Jacobian J diagonal: {np.diag(J)}")
        # mu_tilde_array = compute_mu_tilde_array(J)
        # sum_mu_tilde = np.sum(mu_tilde_array * self.width_array)
        # logging.info(f"Sum of mu_tilde: {sum_mu_tilde}")
        # if sum_mu_tilde != 0:
        #     mu_array = self.simonet_artificial_viscosity_fva * (
        #         mu_tilde_array / sum_mu_tilde
        #     )
        # else:
        #     logging.info(f"Artificial viscosity not activated")

        # ##TODO: remove the below

        # logging.info(f"mu_tilde_array: {mu_tilde_array}")
        # logging.info(f"mu_array: {mu_array}")

        # Get Jacobian diagonal
        def should_activate_artificial_viscosity(J_diagonal):
            """
            Determines if artificial viscosity should be activated based on Jacobian values.
            Returns True if viscosity should be activated, False otherwise.
            """
            # Check if any interior Jacobian values deviate significantly from -1
            # This indicates non-linear behavior that might need artificial viscosity
            interior_values = J_diagonal[1:-1]  # Skip first and last points

            # Calculate how much values deviate from -1
            deviations = np.abs(
                interior_values + 1
            )  # +1 because we expect values around -1

            # Define threshold for significant deviation
            threshold = 0.1  # Adjust based on your needs

            # If any interior point deviates significantly, activate viscosity
            return np.any(deviations > threshold)

        J = -compute_jacobian(compute_F_residual, gamma_new)
        # Check if we should activate artificial viscosity
        if should_activate_artificial_viscosity(np.diag(J)):
            # Calculate mu_tilde_array and mu_array as before
            mu_tilde_array = compute_mu_tilde_array(J)
            sum_mu_tilde = np.sum(mu_tilde_array * self.width_array)
            logging.info(f"sum mu tilde: {sum_mu_tilde:.1e}")

            if np.abs(sum_mu_tilde) > 0:
                mu_array = self.simonet_artificial_viscosity_fva * (
                    mu_tilde_array / sum_mu_tilde
                )
            else:
                logging.info(
                    "Artificial viscosity not activated (sum is zero or negative)"
                )
                mu_array = np.zeros(self.n_panels)

        else:
            logging.info("Artificial viscosity not activated (linear regime detected)")
            mu_array = np.zeros(self.n_panels)

        logging.info(f"diag J: {np.diag(J)}")
        logging.info(f"mu: {mu_array}")
        # First, try the Newton–Krylov method.
        try:
            # Set tolerances
            atol = self.allowed_error * 1e-1
            f_tol = self.allowed_error

            # Call newton_krylov without unsupported keywords.
            sol = newton_krylov(
                partial(compute_G_residual_newton_krylov, mu_array=mu_array),
                gamma_new,
                method="lgmres",
                f_tol=f_tol,
                x_tol=atol,
                maxiter=self.max_iterations,
                verbose=False,
            )
            gamma_new = sol

            # Compute final aerodynamic quantities.
            alpha_array, Umag_array, cl_array, Umagw_array = (
                self.compute_aerodynamic_quantities(gamma_new)
            )

            converged = True
            logging.info("Converged using newton_krylov method")

            return converged, gamma_new, alpha_array, Umag_array

        except Exception as e:
            logging.warning(f"Newton_krylov method failed")
            return (
                False,
                np.zeros(self.n_panels),
                np.zeros(self.n_panels),
                np.zeros(self.n_panels),
            )
            # return self.gamma_loop_simonet_stall(
            #     gamma_initial,
            # )

    # # TODO: could used analytically expressed Jacobians
    # # TODO: could also represent the second derivatives as a matrix
    # # TODO: could probably also use jit for some things
    # def gamma_loop_newton_raphson_simonet(
    #     self,
    #     gamma_new,
    # ):
    #     """Nonlinear Newton Raphson solver with adaptive relaxation to compute the circulation distribution.

    #     The governing equation is modified by adding an artificial viscosity term
    #     Γ_i - ½ V_proj,i c_i Cl(alpha_eff,i) - μ_i (∂²Γ/∂y_s²)_i = 0.

    #     The second derivative is computed using a second order finite difference scheme on an irregular grid.
    #     The viscosity coefficients μ_i are computed (once) from the initial diagonal of the Jacobian.

    #     Args:
    #         gamma_new (np.array): Initial gamma distribution.
    #         AIC_x, AIC_y, AIC_z (np.array): Induced velocity influence matrices.
    #         va_array (np.array): Free-stream velocity array.
    #         chord_array (np.array): Chord length array.
    #         x_airf_array, y_airf_array, z_airf_array (np.array): Local airfoil axes.
    #         panels (list): List of Panel objects (each assumed to have a control-point coordinate, e.g. panel.y).
    #         relaxation_factor (float): Initial relaxation factor.

    #     Returns:
    #         tuple: (converged (bool), final gamma distribution (np.array),
    #                 angle of attack array (np.array), relative velocity magnitude (np.array))
    #     """

    #     # ===============================
    #     # Helper: compute finite-difference coefficients and second derivative
    #     # ===============================
    #     def compute_abc_prime():
    #         """
    #         Compute the finite-difference coefficients a', b', c' for the boundary nodes.
    #         """
    #         # Boundary nodes
    #         d1 = self.y_coords[1] - self.y_coords[0]
    #         d2 = self.y_coords[2] - self.y_coords[0]
    #         a_prime = 2.0 / (d1 * d2)
    #         b_prime = 2.0 / ((d1 - d2) * d1)
    #         c_prime = -2.0 / ((d1 - d2) * d2)
    #         return a_prime, b_prime, c_prime

    #     def compute_abc(i):
    #         """
    #         Compute the finite-difference coefficients a, b, c for the interior nodes.
    #         """
    #         d1 = self.y_coords[i + 1] - self.y_coords[i]
    #         d_1 = self.y_coords[i] - self.y_coords[i - 1]
    #         a = 2.0 / ((d1 + d_1) * d_1)
    #         b = -2.0 / (d1 * d_1)
    #         c = 2.0 / ((d1 + d_1) * d1)
    #         return a, b, c

    #     def compute_abc_dbl_prime():
    #         """
    #         Compute the finite-difference coefficients a'', b'', c'' for the boundary nodes.
    #         """
    #         # Boundary nodes
    #         d1 = self.y_coords[-1] - self.y_coords[-2]
    #         d2 = self.y_coords[-1] - self.y_coords[-3]
    #         a_dbl_prime = 2.0 / (d1 * d2)
    #         b_dbl_prime = 2.0 / ((d1 - d2) * d1)
    #         c_dbl_prime = -2.0 / ((d1 - d2) * d2)
    #         return a_dbl_prime, b_dbl_prime, c_dbl_prime

    #     def compute_second_derivative_of_gamma(gamma):
    #         """
    #         Compute the second derivative of gamma along the span using the
    #         exact boundary stencils from eq. (21) in your reference, plus the
    #         interior formulas.

    #         In the reference, i=1 and i=n are the boundary nodes. Here, we use
    #         0-based Python indexing, so:
    #         * i=0 in Python corresponds to i=1 in the reference
    #         * i=n-1 in Python corresponds to i=n in the reference

    #         For interior nodes (1 <= i <= n-2 in Python), we use:
    #             (∂²Γ/∂y²)_i = a_i Γ[i-1] + b_i Γ[i] + c_i Γ[i+1]

    #         For i=0 (the first node in Python, i=1 in reference):
    #             (∂²Γ/∂y²)_0 = a' Γ[0] + b' Γ[1] + c' Γ[2]

    #         For i=n-1 (the last node in Python, i=n in reference):
    #             (∂²Γ/∂y²)_(n-1) = a'' Γ[n-1] + b'' Γ[n-2] + c'' Γ[n-3]

    #         NOTE: The definitions of a', b', c', a'', b'', c'' come from eq. (22).
    #             We also assume y_coords is sorted along the span (y[0] < y[1] < ...).
    #         """

    #         n = self.n_panels
    #         second_deriv = np.zeros(n)

    #         # Quick exit if too few points
    #         if n < 3:
    #             raise ValueError(
    #                 "At least 3 panels are required to compute the second derivative."
    #             )

    #         # ===============================
    #         # 1) Boundary formula for i=0 => eq. (21) at i=1 in reference
    #         # ===============================
    #         a_prime, b_prime, c_prime = compute_abc_prime()
    #         second_deriv[0] = (
    #             a_prime * gamma[0] + b_prime * gamma[1] + c_prime * gamma[2]
    #         )

    #         # ===============================
    #         # 2) Interior points: i = 1..(n-2)
    #         # ===============================
    #         for i in range(1, n - 1):
    #             a, b, c = compute_abc(i)
    #             second_deriv[i] = a * gamma[i - 1] + b * gamma[i] + c * gamma[i + 1]

    #         # ===============================
    #         # 3) Boundary formula for i=n-1 => eq. (21) at i=n in reference
    #         # ===============================
    #         # here it is not y_coords[n], because of python 0-based indexing
    #         a_dbl_prime, b_dbl_prime, c_dbl_prime = compute_abc_dbl_prime()
    #         second_deriv[n - 1] = (
    #             a_dbl_prime * gamma[n - 1]
    #             + b_dbl_prime * gamma[n - 2]
    #             + c_dbl_prime * gamma[n - 3]
    #         )

    #         return second_deriv

    #     def compute_G_residual(
    #         gamma,
    #         mu_array,
    #     ):
    #         """Computes the residual vector G for the non_linear system, including the artificial viscosity term.

    #         This function follows eq. (20) of Simonet et al., computing

    #             G = gamma - gamma_new - (mu_array * (∂²gamma/∂y²))

    #         Args:
    #             gamma (np.ndarray): Current circulation distribution (n x 1).
    #             mu_array (np.ndarray): Artificial viscosity coefficients (n x 1).

    #         Returns:
    #             Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #                 - Residual vector G (n x 1).
    #                 - Angle of attack array (n x 1).
    #                 - Relative velocity magnitude array (n x 1).
    #         """
    #         alpha_array, Umag_array, cl_array, Umagw_array = (
    #             self.compute_aerodynamic_quantities(gamma)
    #         )
    #         gamma_new = (
    #             0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.chord_array
    #         )
    #         second_derivative_of_gamma = compute_second_derivative_of_gamma(gamma)
    #         G_residual = (gamma_new + mu_array * second_derivative_of_gamma) - gamma
    #         return G_residual, alpha_array, Umag_array

    #     def compute_jacobian_K(F_func, gamma, *args):
    #         """
    #         Computes the Jacobian matrix K using finite differences.

    #         Args:
    #             F_func (callable): Function to compute the residual vector.
    #             gamma (np.array): Current circulation distribution.
    #             eps (float): Finite difference step size.
    #             args (tuple): Additional arguments for the residual function.

    #         Returns:
    #             np.ndarray: Jacobian matrix (n x n).
    #         """
    #         n = self.n_panels
    #         K = np.zeros((n, n))
    #         F0, _, _ = F_func(gamma, *args)
    #         for i in range(n):
    #             gamma_eps = gamma.copy()
    #             gamma_eps[i] += self.jacobian_eps
    #             F1, _, _ = F_func(gamma_eps, *args)
    #             K[:, i] = (F1 - F0) / self.jacobian_eps
    #         return K

    #     # ===============================
    #     # Helper: compute artificial viscosity coefficients mu_tilde_i (eq. 25)
    #     # -------------------------------
    #     def compute_mu_tilde_array(K):
    #         """Computes μ_i from the diagonal of the Jacobian and the central finite-difference coefficient.

    #         Args:
    #             - K (np.ndarray): Jacobian matrix (n x n).
    #             - y_coords (np.array): Spanwise coordinates of the panels.
    #         Returns
    #             - mu_tilde_array (np.array): Artificial viscosity coefficients.
    #         """
    #         n = self.n_panels
    #         mu_tilde_array = np.zeros(n)

    #         # first panel
    #         a_prime, _, _ = compute_abc_prime()
    #         mu_tilde_array[0] = np.minimum(0.0, K[0, 0] / a_prime)

    #         for i in range(1, n - 1):
    #             _, b, _ = compute_abc(i)
    #             mu_tilde_array[i] = np.maximum(0.0, K[i, i] / b)

    #         # last panel
    #         a_dbl_prime, _, _ = compute_abc_dbl_prime()
    #         mu_tilde_array[n - 1] = np.minimum(0.0, K[n - 1, n - 1] / a_dbl_prime)
    #         return mu_tilde_array

    #     # ===============================
    #     # Main Newton–Raphson loop
    #     # ===============================
    #     # Extract spanwise coordinates from panels (assumes each panel has an attribute "y")
    #     gamma = gamma_new.copy()
    #     relaxation = self.relaxation_factor

    #     # Initially, set μ_i = 0 (no viscosity) so that we can compute the first Jacobian.
    #     mu_array = np.zeros_like(gamma)
    #     # D_matrix = build_second_derivative_matrix(y_coords)

    #     for it in range(self.max_iterations):
    #         # Compute residual (F_val) and aerodynamic arrays.
    #         F_val, alpha_array, Umag_array = compute_G_residual(
    #             gamma,
    #             mu_array,
    #         )

    #         # === Checking Convergence ===
    #         # computes the infinity norm, also known as max norm (maximum absolute row sum)
    #         norm_F = np.linalg.norm(F_val, np.inf)
    #         if norm_F < self.allowed_error:
    #             logging.info(
    #                 f"Converged after {it} iterations with residual norm {norm_F:.3e}."
    #             )
    #             return True, gamma, alpha_array, Umag_array

    #         # === Computing Jacobian K (=delta G/ delta Gamma) using finite differences ===
    #         # TODO: this could be done analytically using eqs 23,24 and appendix A.
    #         K = compute_jacobian_K(compute_G_residual, gamma, mu_array)

    #         # On the first iteration, compute and fix the artificial viscosity coefficients.
    #         if it == 0:
    #             mu_tilde_array = compute_mu_tilde_array(K)
    #             # Using eq 26. and interpreting b as the local spanwise panel width
    #             b_array = np.array([panel.width for panel in self.panels])
    #             total_tilde = np.sum(mu_tilde_array * b_array)
    #             mu_array = self.simonet_artificial_viscosity_fva * (
    #                 mu_tilde_array / total_tilde
    #             )

    #             # Recompute the residual with the new μ coefficients.
    #             F_val, alpha_array, Umag_array = compute_G_residual(
    #                 gamma,
    #                 mu_array,
    #             )

    #             norm_F = np.linalg.norm(F_val, np.inf)

    #         # Solve for the Newton step: K * delta = -F_val.
    #         try:
    #             delta = np.linalg.solve(K, -F_val)
    #         except np.linalg.LinAlgError:
    #             logging.error(f"Jacobian is singular at iteration {it}.")
    #             return False, gamma, alpha_array, Umag_array

    #         # Trial update using the current relaxation factor.
    #         gamma_trial = gamma + relaxation * delta
    #         F_trial, alpha_array, Umag_array = compute_G_residual(
    #             gamma_trial,
    #             mu_array,
    #         )

    #         # Adapt the relaxation factor: increase if the trial is successful; otherwise reduce.
    #         if np.linalg.norm(F_trial, np.inf) < norm_F:
    #             gamma = gamma_trial
    #             relaxation = min(relaxation * 1.1, 1.0)
    #         else:
    #             relaxation *= 0.5
    #             if relaxation < self.min_relaxation_factor:
    #                 logging.error(
    #                     "Relaxation coefficient too small; stopping iteration."
    #                 )
    #                 return False, gamma, alpha_array, Umag_array

    #     logging.warning(
    #         f"Did not converge within {self.max_iterations} iterations (final residual {norm_F:.3e})."
    #     )
    #     return False, gamma, alpha_array, Umag_array

    # def gamma_loop_newton_raphson(
    #     self,
    #     gamma_new,
    # ):
    #     """
    #     Nonlinear Newton Raphson solver with adaptive relaxation to compute the circulation distribution.

    #     Args:
    #         gamma_new (np.array): Initial gamma distribution.
    #         AIC_x, AIC_y, AIC_z (np.array): Induced velocity influence matrices.
    #         va_array (np.array): Free-stream velocity array.
    #         chord_array (np.array): Chord length array.
    #         x_airf_array, y_airf_array, z_airf_array (np.array): Airfoil coordinate arrays.
    #         panels (list): List of Panel objects.
    #         relaxation_factor (float): Initial relaxation factor.

    #     Returns:
    #         tuple: (converged (bool), final gamma distribution (np.array),
    #                 angle of attack array (np.array), relative velocity magnitude (np.array))
    #     """

    #     # def compute_jacobian_J_analytic(self, gamma):
    #     #     """
    #     #     Compute the Jacobian (∂F/∂Γ) analytically, using the derivations
    #     #     from Appendix A of your reference (eqs. A19–A21).

    #     #     Args:
    #     #         gamma (np.ndarray): Current circulation distribution (n x 1).

    #     #     Returns:
    #     #         J (np.ndarray): n-by-n matrix of partial derivatives J[i, j] = ∂F_i / ∂Γ_j
    #     #     """

    #     #     n = self.n_panels
    #     #     J = np.zeros((n, n))

    #     #     # 1) Recompute all aerodynamic quantities at the current circulation gamma.
    #     #     #    - alpha_eff[i], the effective angle of attack at panel i
    #     #     #    - V_proj[i], the "projected" velocity magnitude
    #     #     #    - V_eff[i], the "effective" velocity magnitude
    #     #     #    - c[i], the chord length for each panel
    #     #     #    - Cl[i] = Cl(alpha_eff[i]), the 2D lift coefficient
    #     #     #    etc.
    #     #     alpha_array, Umag_array, cl_array, Umagw_array = (
    #     #         self.compute_aerodynamic_quantities(gamma)
    #     #     )
    #     #     alpha_eff = alpha_array
    #     #     V_proj = Umagw_array
    #     #     V_eff = Umag_array

    #     #     # 2) Compute the derivative dCl/dα at each panel.
    #     #     #    (You might have a panel-specific function or a known formula.)
    #     #     dCl_dalpha = np.array(
    #     #         [panel.dCl_dalpha(alpha_eff[i]) for i, panel in enumerate(self.panels)]
    #     #     )

    #     #     # 3) For each pair (i, j), build the Jacobian using eq. (A19–A21).
    #     #     #    In the paper, eq. (A21) typically looks like:
    #     #     #
    #     #     #      J_{ij} = δ_{ij}
    #     #     #               - (1/2) * c_i * (V_eff[i] / V_proj[i])
    #     #     #                 * [ Cl(alpha_eff[i]) + (dCl/dα)[i] * ∂α_eff[i]/∂Γ_j ]
    #     #     #
    #     #     #    or an equivalent expression. The tricky part is ∂α_eff[i]/∂Γ_j.
    #     #     #
    #     #     #    Also, the sign might differ if your F_i is defined as:
    #     #     #        F_i(Γ) = Γ_i - Γ_calc,i(...)
    #     #     #    or as
    #     #     #        F_i(Γ) = Γ_calc,i(...) - Γ_i.
    #     #     #
    #     #     #    Make sure to be consistent with your definition of the residual F.
    #     #     #
    #     #     for i in range(n):
    #     #         for j in range(n):

    #     #             # δ_{ij} (Kronecker delta)
    #     #             delta_ij = 1.0 if (i == j) else 0.0

    #     #             # Compute the coefficients A_ij, B_ij from ~Eq.9,10
    #     #             # v_ij: A 3D induced velocity vector at panel i due to panel (or vortex) j ~Eq.2
    #     #             # ysi: local y-axis chordwise direction vector
    #     #             # ||ysi||^2: squared norm of ysi
    #     #             ##TODO: What you have here below is wrong
    #     #             x1i = self.x_airf_array[i]  # vertical, in airfoil plane
    #     #             x2i = self.y_airf_array[i]  # chordwise, in airfoil plane
    #     #             x3i = self.z_airf_array[i]  # spanwise, in airfoil plane
    #     #             ysi = self.y_coords
    #     #             ysi_sq = np.dot(ysi, ysi)
    #     #             vij = np.array(
    #     #                 [
    #     #                     self.AIC_x[i, j],
    #     #                     self.AIC_y[i, j],
    #     #                     self.AIC_z[i, j],
    #     #                 ]
    #     #             )
    #     #             A_ij = vij * ysi_sq - x2i * np.dot(vij, x2i)
    #     #             B_ij = x1i * np.dot(vij, x3i) - x3i * np.dot(vij, x1i)

    #     #             # eq. A21
    #     #             J[i, j] = delta_ij - 0.5 * self.chord_array[i] * (
    #     #                 V_eff[i] / V_proj[i]
    #     #             ) * (A_ij * cl_array[i] + B_ij * dCl_dalpha[i])

    #     #     return J

    #     # Nested helper function to compute the residual and related aerodynamic arrays.
    #     def compute_gamma_residual(gamma):
    #         alpha_array, Umag_array, cl_array, Umagw_array = (
    #             self.compute_aerodynamic_quantities(gamma)
    #         )
    #         gamma_new = (
    #             0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.chord_array
    #         )
    #         # Residual: difference between the computed and current gamma.
    #         F_val = gamma_new - gamma
    #         return F_val, alpha_array, Umag_array

    #     # Nested helper function to compute the Jacobian using finite differences.
    #     def compute_jacobian_J(F_func, gamma):
    #         n = self.n_panels
    #         K = np.zeros((n, n))
    #         F0, _, _ = F_func(gamma)
    #         for i in range(n):
    #             gamma_eps = gamma.copy()
    #             gamma_eps[i] += self.jacobian_eps
    #             F1, _, _ = F_func(gamma_eps)
    #             K[:, i] = (F1 - F0) / self.jacobian_eps
    #         return K

    #     # Begin the Newton–Raphson iteration.
    #     gamma = gamma_new.copy()
    #     relaxation = self.relaxation_factor

    #     for i in range(self.max_iterations):
    #         F_val, alpha_array, Umag_array = compute_gamma_residual(gamma)

    #         # === checking for convergence ===
    #         norm_F = np.linalg.norm(F_val, np.inf)
    #         if norm_F < self.allowed_error:
    #             logging.info(
    #                 f"Converged after {i} iterations with residual norm {norm_F:.3e}."
    #             )
    #             return True, gamma, alpha_array, Umag_array

    #         # TODO: this could be replaced using eq. 23,24 and Appendix A
    #         # TODO: it could be written in analytical form
    #         # Compute the Jacobian (K = delta G/ delta Gamma)
    #         J = compute_jacobian_J(compute_gamma_residual, gamma)
    #         # === use newton_raphson to solve the linear system ===
    #         try:
    #             # Compute the Newton step: J * delta = -F_val.
    #             delta = np.linalg.solve(J, -F_val)
    #         except np.linalg.LinAlgError:
    #             logging.error(f"Jacobian is singular at iteration {i}.")
    #             return False, gamma, alpha_array, Umag_array

    #         # === dynamic relaxation part ===
    #         # Trial update with the current relaxation factor.
    #         gamma_trial = gamma + relaxation * delta
    #         F_trial, _, _ = compute_gamma_residual(gamma_trial)

    #         # Adapt the relaxation factor based on the trial result.
    #         if np.linalg.norm(F_trial, np.inf) < norm_F:
    #             gamma = gamma_trial
    #             relaxation = min(relaxation * 1.1, 1.0)
    #         else:
    #             relaxation *= 0.5
    #             if relaxation < self.min_relaxation_factor:
    #                 logging.error(
    #                     "Relaxation coefficient too small; stopping iteration."
    #                 )
    #                 return False, gamma, alpha_array, Umag_array

    #     logging.warning(
    #         f"Did not converge within {self.max_iterations} iterations (final residual {norm_F:.3e})."
    #     )
    #     return False, gamma, alpha_array, Umag_array

    ####################
    ### STALL MODELS ###
    ####################
    # def compute_G_vector(
    #     self,
    #     gamma,
    #     AIC_x,
    #     AIC_y,
    #     AIC_z,
    #     va_array,
    #     chord_array,
    #     x_airf_array,
    #     y_airf_array,
    #     z_airf_array,
    #     panels,
    # ):
    #     """
    #     Return the G_i(gamma) vector as described in your eq. (19)-(20),
    #     ignoring for the moment any -mu_i*(d^2Gamma/dy^2) if you want
    #     that inside G. Or you can put the artificial-viscosity piece in here
    #     if your method directly modifies G.
    #     """
    #     # 1) Compute induced velocities
    #     induced_vel_x = AIC_x @ gamma
    #     induced_vel_y = AIC_y @ gamma
    #     induced_vel_z = AIC_z @ gamma
    #     induced_velocity_all = np.vstack(
    #         (induced_vel_x, induced_vel_y, induced_vel_z)
    #     ).T

    #     # 2) Sum with free stream -> get alpha, etc.
    #     relative_velocity_array = va_array + induced_velocity_all
    #     v_normal_array = np.sum(x_airf_array * relative_velocity_array, axis=1)
    #     v_tangential_array = np.sum(y_airf_array * relative_velocity_array, axis=1)
    #     alpha_array = np.arctan2(v_normal_array, v_tangential_array)
    #     cl_array = np.array([p.calculate_cl(a) for p, a in zip(panels, alpha_array)])

    #     # 3) Build G_i
    #     #   G_i = Gamma_i - 0.5 * Umag^2/Umag_infinity * cl_i * chord_i (just for example),
    #     #   or your actual formula from eq. (19)-(20).

    #     # For a typical lifting-line approach:
    #     # let G_i = gamma_i - (1/2) * V_proj_i * c_i * Cl( alpha_eff_i )
    #     # We'll pretend "V_proj_i" = magnitude of relative_velocity_array[i],
    #     # or you can do something more specialized.
    #     Umag_array = np.linalg.norm(relative_velocity_array, axis=1)
    #     G = gamma - 0.5 * Umag_array * chord_array * cl_array  # simplistic example

    #     return G  # shape (n,)

    # def compute_J_diag_finite_diff(
    #     self,
    #     gamma,
    #     AIC_x,
    #     AIC_y,
    #     AIC_z,
    #     va_array,
    #     chord_array,
    #     x_airf_array,
    #     y_airf_array,
    #     z_airf_array,
    #     panels,
    # ):
    #     """
    #     Return an approximate diagonal of the Jacobian by finite difference.
    #     """
    #     n = len(gamma)
    #     # Evaluate G(gamma) once
    #     G_base = self.compute_G_vector(
    #         gamma,
    #         AIC_x,
    #         AIC_y,
    #         AIC_z,
    #         va_array,
    #         chord_array,
    #         x_airf_array,
    #         y_airf_array,
    #         z_airf_array,
    #         panels,
    #     )
    #     J_diag = np.zeros(n)

    #     for i in range(n):
    #         # pick a small perturbation
    #         eps = 1e-6 * (1.0 + abs(gamma[i]))  # for example
    #         gamma_pert = gamma.copy()
    #         gamma_pert[i] += eps

    #         G_pert = self.compute_G_vector(
    #             gamma_pert,
    #             AIC_x,
    #             AIC_y,
    #             AIC_z,
    #             va_array,
    #             chord_array,
    #             x_airf_array,
    #             y_airf_array,
    #             z_airf_array,
    #             panels,
    #         )
    #         # approximate derivative wrt gamma[i] = [G_i(gamma+eps) - G_i(gamma)] / eps
    #         J_diag[i] = (G_pert[i] - G_base[i]) / eps

    #     return J_diag

    # def apply_simonet_artificial_viscosity(
    #     self, gamma, y_panel_centers, Jdiag, b_array, fva
    # ):
    #     """
    #     Implements the post-stall artificial viscosity correction described by
    #     Chattot [22] and subsequent authors, following Eqs. (19)-(27) in your notes.

    #     1) Build the second derivative of Gamma wrt y_s, using the finite-difference
    #     coefficients for irregular grids (Eqs. 21-22).

    #     2) Compute the "raw" mu_tilde[i] from Eq. (25), which depends on the sign of
    #     the diagonal Jacobian J_{ii}.

    #     3) Scale mu_tilde[i] to obtain mu[i] by Eq. (26), ensuring the "area"
    #     under mu is f_va.

    #     4) Return the array of artificial-viscosity corrections:
    #         correction[i] = - mu[i] * ( d^2 Gamma / d y_s^2 )_i
    #     You can then add that to your new circulation in the iteration.

    #     Args:
    #         gamma (np.ndarray): Current circulation distribution (size n).
    #         y_panel_centers (np.ndarray): Spanwise coordinates of the control
    #             points for each panel (size n).  Must be sorted along the span.
    #         Jdiag (np.ndarray): The diagonal of your lifting-line Jacobian, size n.
    #             We use its sign to decide whether to "activate" viscosity at panel i.
    #         b_array (np.ndarray): The "section span vector" b, i.e. the local
    #             panel width or at least a set of weights used in eq. (26).
    #         fva (float): The artificial-viscosity parameter (a "volume" dimension).
    #             Controls the total strength of the artificial viscosity.

    #     Returns:
    #         viscosity_correction (np.ndarray): length-n array of increments
    #             to be added to gamma, i.e.  gamma_new[i] += viscosity_correction[i].
    #         mu (np.ndarray): The final artificial-viscosity coefficients mu_i.
    #     """

    #     n = len(gamma)
    #     if n < 3:
    #         # With fewer than 3 panels, just return zero correction
    #         return np.zeros_like(gamma), np.zeros_like(gamma)

    #     # ----------------------------------------------------------------------
    #     # 1) Build the second-derivative operator D2 (an n x n matrix) using eq. (21)-(22).
    #     #    Then d2_gamma = D2 @ gamma.
    #     # ----------------------------------------------------------------------

    #     D2 = np.zeros((n, n))

    #     # Helper function to compute the standard interior coefficients: a,b,c
    #     # from eq. (22), for i in 1..n-1 (1-based).  In Python, that's i=1..n-2 (0-based).
    #     def interior_coeffs(i):
    #         # i is 0-based interior index, so d_im1 = y[i] - y[i-1], d_i = y[i+1] - y[i]
    #         d_im1 = y_panel_centers[i] - y_panel_centers[i - 1]
    #         d_i = y_panel_centers[i + 1] - y_panel_centers[i]
    #         a = 2.0 / ((d_im1 + d_i) * d_im1)
    #         b = -2.0 / (d_im1 * d_i)
    #         c = 2.0 / ((d_im1 + d_i) * d_i)
    #         return a, b, c

    #     # Fill interior rows: i=1..n-2 (0-based)
    #     for i in range(1, n - 1):
    #         a, b, c = interior_coeffs(i)
    #         D2[i, i - 1] = a
    #         D2[i, i] = b
    #         D2[i, i + 1] = c

    #     # Boundary at i=0: eq. (21) says use a', b', c'.
    #     # Typically: (∂^2 Γ /∂y^2)_1 = a' Γ_1 + b' Γ_2 + c' Γ_3 in 1-based indexing.
    #     # In Python, that means row 0 of D2 => combine gamma[0], gamma[1], gamma[2].
    #     #
    #     # The reference eq. (22) can be read as:
    #     #    a' =  2/( d1 * d2 ),
    #     #    b' =  (a combination of d1, d2) ...
    #     #    c' =  ...
    #     #
    #     # For clarity, let's define them exactly:
    #     if n >= 3:
    #         d1 = y_panel_centers[1] - y_panel_centers[0]
    #         d2 = y_panel_centers[2] - y_panel_centers[1]
    #         # A typical 3-pt forward difference for an irregular grid per eq. (22) can be arranged as:
    #         a_p = 2.0 / (d1 * d2)
    #         b_p = -((d1 + d2) / (d1 * d2))  # or another expression from the reference
    #         c_p = 2.0 / ((d1 + d2) * d2)  # check carefully with your eq. (22)
    #         #
    #         # You will want to verify these three with your own notes.  As an example:
    #         # eq. (22) might read:
    #         #    a' =  2/( d1*d2 )
    #         #    b' = -2/( d1*(d1 + d2) )
    #         #    c' =  2/( d2*(d1 + d2) )
    #         #
    #         # Please adjust if your reference has them differently.  For demonstration:
    #         D2[0, 0] = a_p
    #         D2[0, 1] = b_p
    #         D2[0, 2] = c_p

    #     # Boundary at i=n-1: eq. (21) says use a'', b'', c''.
    #     # (∂^2 Γ /∂y^2)_n = a'' Γ_n + b'' Γ_(n-1) + c'' Γ_(n-2)
    #     # in 1-based indexing.  In Python, that means row n-1 => combine gamma[n-1], gamma[n-2], gamma[n-3].
    #     if n >= 3:
    #         d_nm1 = y_panel_centers[n - 1] - y_panel_centers[n - 2]
    #         d_nm2 = y_panel_centers[n - 2] - y_panel_centers[n - 3] if n > 3 else d_nm1
    #         # Example coefficient set from eq. (22):
    #         a_pp = 2.0 / (d_nm2 * d_nm1)
    #         b_pp = -((d_nm1 + d_nm2) / (d_nm2 * d_nm1))
    #         c_pp = 2.0 / ((d_nm2 + d_nm1) * d_nm2)
    #         #
    #         # or whichever your reference states exactly.  Insert below:
    #         D2[n - 1, n - 1] = a_pp
    #         D2[n - 1, n - 2] = b_pp
    #         if n >= 3:
    #             D2[n - 1, n - 3] = c_pp

    #     # Now multiply:
    #     d2_gamma = D2.dot(gamma)

    #     # ----------------------------------------------------------------------
    #     # 2) Compute mu_tilde[i] from eq. (25).
    #     #
    #     #   mu_tilde_i =   max(0, J_{ii}^b)   for i in the interior
    #     #                or min(0, J_{11}^f)  at i=0
    #     #                or min(0, J_{nn}^{a''}) at i=n-1
    #     #
    #     # The exact exponents (b, f, a'') come from the boundary conditions in your notes.
    #     # Commonly we check the sign of J_{ii}.  If J_{ii}<0, we turn on viscosity.
    #     # ----------------------------------------------------------------------
    #     mu_tilde = np.zeros(n)

    #     for i in range(n):
    #         if i == 0:
    #             # boundary
    #             # eq. (25) says mu_tilde_1 = min(0, J_{11}^f).  Typically that means:
    #             # if J_{00} (Python) is negative => mu_tilde[0] = J_{00}, else 0
    #             mu_tilde[i] = min(0.0, Jdiag[i])
    #         elif i == n - 1:
    #             # other boundary
    #             mu_tilde[i] = min(0.0, Jdiag[i])
    #         else:
    #             # interior
    #             # eq. (25) says mu_tilde_i = max(0, J_{ii}^b).  Interpreted as
    #             # if J_{ii}<0 => we want a positive mu_tilde, so we might use mu_tilde = - J_{ii}.
    #             # or simpler: mu_tilde = max(0, -J_{ii}).
    #             # (That “-” depends on your sign convention.  Adjust if needed.)
    #             mu_tilde[i] = max(0.0, -Jdiag[i])

    #     # ----------------------------------------------------------------------
    #     # 3) Scale mu_tilde to get mu by eq. (26):
    #     #       mu = f_va * ( mu_tilde / ( mu_tilde · b ) ) * b
    #     # i.e. each mu[i] = f_va * mu_tilde[i] * b[i] / ( sum_j mu_tilde[j]*b[j] ).
    #     # ----------------------------------------------------------------------
    #     mu = np.zeros(n)
    #     dot_tilde_b = np.sum(mu_tilde * b_array)
    #     if abs(dot_tilde_b) > 1e-14:
    #         scale = fva / dot_tilde_b
    #         mu = scale * mu_tilde * b_array  # elementwise

    #     # ----------------------------------------------------------------------
    #     # 4) Form the final viscosity-correction array:
    #     #       viscosity_correction[i] = - mu[i] * (d^2 Gamma / d y_s^2)[i]
    #     # which you can then add to your gamma iteration as in eq. (19).
    #     # ----------------------------------------------------------------------
    #     viscosity_correction = -mu * d2_gamma

    #     # ----------------------------------------------------------------------
    #     # (Optional) 5) You can also compute the artificial viscosity "deviation"
    #     #     P = 100 x (1 / b) sum_{i=1..n} [ mu_i ( d^2 Γ / dy^2 )_i / ( (1/2)*V_proj_i*c_i*Cl(α_eff_i ) ) ] * b_i
    #     # from eq. (27).  You would need c_i, Cl_i, V_proj_i, etc.  Not shown here.
    #     # ----------------------------------------------------------------------
    #     return viscosity_correction

    def calculate_artificial_damping(self, gamma, alpha, stall_angle_list):
        """Calculate the artificial damping

        Args:
            - gamma (np.array): Circulation distribution array
            - alpha (np.array): Angle of attack array
            - stall_angle_list (np.array): Stall angle list

        Returns:
            - np.array: Damping array
            - bool: Whether the damping is applied
        """
        # Determine if there is a stalled case
        is_stalled = False
        for ia, alpha_i in enumerate(alpha):
            if self.aerodynamic_model_type == "LLT" or (
                self.artificial_damping["k2"] == 0
                and self.artificial_damping["k4"] == 0
            ):
                is_stalled = False
                break
            elif alpha_i > stall_angle_list[ia]:
                is_stalled = True
                break
        if not is_stalled:
            damp = 0
            return damp, is_stalled

        # If there is a stalled case, calculate the artificial damping
        n_gamma = len(gamma)
        damp = np.zeros(n_gamma)
        for ig, gamma_ig in enumerate(gamma):
            if ig == 0:
                gim2 = gamma[0]
                gim1 = gamma[0]
                gi = gamma[0]
                gip1 = gamma[1]
                gip2 = gamma[2]
            elif ig == 1:
                gim2 = gamma[0]
                gim1 = gamma[0]
                gi = gamma[1]
                gip1 = gamma[2]
                gip2 = gamma[3]
            elif ig == n_gamma - 2:
                gim2 = gamma[n_gamma - 4]
                gim1 = gamma[n_gamma - 3]
                gi = gamma[n_gamma - 2]
                gip1 = gamma[n_gamma - 1]
                gip2 = gamma[n_gamma - 1]
            elif ig == n_gamma - 1:
                gim2 = gamma[n_gamma - 3]
                gim1 = gamma[n_gamma - 2]
                gi = gamma[n_gamma - 1]
                gip1 = gamma[n_gamma - 1]
                gip2 = gamma[n_gamma - 1]
            else:
                gim2 = gamma[ig - 2]
                gim1 = gamma[ig - 1]
                gi = gamma[ig]
                gip1 = gamma[ig + 1]
                gip2 = gamma[ig + 2]

            dif2 = (gip1 - gi) - (gi - gim1)
            dif4 = (gip2 - 3.0 * gip1 + 3.0 * gi - gim1) - (
                gip1 - 3.0 * gi + 3.0 * gim1 - gim2
            )
            damp[ig] = (
                self.artificial_damping["k2"] * dif2
                - self.artificial_damping["k4"] * dif4
            )
        return damp, is_stalled

    def smooth_circulation(self, circulation, smoothness_factor, damping_factor):
        """
        Check if a circulation curve is smooth and apply damping if necessary.

        Args:
            - circulation (np.array): Circulation strength array of shape (n_points, 1)
            - smoothness_factor (float): Factor to determine the smoothness threshold
            - damping_factor (float): Factor to control the strength of smoothing (0 to 1)

        Returns:
            - np.array: Smoothed circulation array
            - bool: Whether damping was applied
        """

        # Calculate the mean circulation, excluding first and last points
        circulation_mean = np.mean(circulation[1:-1])

        # Calculate the smoothness threshold based on the mean and factor
        smoothness_threshold = smoothness_factor * circulation_mean

        # Calculate the difference between adjacent points, excluding first and last
        differences = np.diff(circulation[1:-1], axis=0)
        logging.debug("circulation_mean: %s, diff: %s", circulation_mean, differences)

        # Check if the curve is smooth based on the maximum difference
        if len(differences) == 0:
            is_smooth = True
        else:
            is_smooth = np.max(np.abs(differences)) <= smoothness_threshold

        if is_smooth:
            return np.zeros(len(circulation)), False

        # Apply damping to smooth the curve
        smoothed = np.copy(circulation)
        for i in range(1, len(circulation) - 1):
            left = circulation[i - 1]
            center = circulation[i]
            right = circulation[i + 1]

            # Calculate the average of neighboring points
            avg = (left + right) / 2

            # Apply damping
            smoothed[i] = center + damping_factor * (avg - center)

        # Ensure the total circulation remains unchanged
        total_original = np.sum(circulation)
        total_smoothed = np.sum(smoothed)
        smoothed *= total_original / total_smoothed

        damp = smoothed - circulation
        return damp, True

    # # =========================================
    # #           Newton-Raphson Jax Method
    # # =========================================
    # --> Method runs nicely, but does not lead to a converged solution.
    # def gamma_loop_newton_raphson_jax(
    #     self,
    #     gamma_new,
    # ):
    #     """
    #     Nonlinear Newton Raphson solver with adaptive relaxation to compute the circulation distribution.

    #     Args:
    #         gamma_new (np.array): Initial gamma distribution.
    #         AIC_x, AIC_y, AIC_z (np.array): Induced velocity influence matrices.
    #         va_array (np.array): Free-stream velocity array.
    #         chord_array (np.array): Chord length array.
    #         x_airf_array, y_airf_array, z_airf_array (np.array): Airfoil coordinate arrays.
    #         panels (list): List of Panel objects.
    #         relaxation_factor (float): Initial relaxation factor.

    #     Returns:
    #         tuple: (converged (bool), final gamma distribution (np.array),
    #                 angle of attack array (np.array), relative velocity magnitude (np.array))
    #     """

    #     import jax
    #     import jax.numpy as jnp

    #     def compute_gamma_residual_jax(
    #         gamma: jnp.ndarray,
    #         AIC_x: jnp.ndarray,
    #         AIC_y: jnp.ndarray,
    #         AIC_z: jnp.ndarray,
    #         va_array: jnp.ndarray,
    #         chord_array: jnp.ndarray,
    #         x_airf_array: jnp.ndarray,
    #         y_airf_array: jnp.ndarray,
    #         z_airf_array: jnp.ndarray,
    #         polar_data_array: jnp.ndarray,
    #     ) -> tuple:
    #         """
    #         JAX-compatible version of the aerodynamic residual function.

    #         Computes:
    #             induced_velocity_all = [AIC_x @ gamma, AIC_y @ gamma, AIC_z @ gamma]^T,
    #             relative_velocity_array = va_array + induced_velocity_all,
    #             effective angle of attack (alpha) via:
    #                 alpha = arctan( (x_airf · v_eff) / (y_airf · v_eff) ),
    #             and then gamma_new = 0.5 * Umag^2/Umagw * cl * chord_array.

    #         The residual is then defined as:
    #             F_val = gamma_new - gamma.

    #         Args:
    #             gamma (jnp.ndarray): Circulation distribution (n,).
    #             AIC_x, AIC_y, AIC_z (jnp.ndarray): Influence matrices (n, n).
    #             va_array (jnp.ndarray): Free-stream velocity vectors (n, 3).
    #             chord_array (jnp.ndarray): Chord lengths (n,).
    #             x_airf_array, y_airf_array, z_airf_array (jnp.ndarray): Local airfoil axes (n, 3).
    #             calculate_cl_func (callable): Function that computes the lift coefficient for a given alpha.
    #                                         Must be JAX-compatible.

    #         Returns:
    #             tuple: A tuple containing:
    #                 - F_val (jnp.ndarray): Residual vector (n,).
    #                 - alpha_array (jnp.ndarray): Angle of attack array (n,).
    #                 - Umag_array (jnp.ndarray): Relative velocity magnitude (n,).
    #         """
    #         # Compute induced velocity at each panel: shape (n,)
    #         induced_velocity_x = jnp.matmul(AIC_x, gamma)
    #         induced_velocity_y = jnp.matmul(AIC_y, gamma)
    #         induced_velocity_z = jnp.matmul(AIC_z, gamma)
    #         # Stack to form a (n, 3) array of induced velocities.
    #         induced_velocity_all = jnp.stack(
    #             [induced_velocity_x, induced_velocity_y, induced_velocity_z], axis=1
    #         )

    #         # Effective (relative) velocity is free-stream + induced.
    #         relative_velocity_array = va_array + induced_velocity_all  # shape (n, 3)

    #         # Compute cross products using jnp.cross.
    #         relative_velocity_crossz_array = jnp.cross(
    #             relative_velocity_array, z_airf_array
    #         )
    #         Uinfcrossz_array = jnp.cross(va_array, z_airf_array)

    #         # Compute the components along the local airfoil axes.
    #         v_normal_array = jnp.sum(x_airf_array * relative_velocity_array, axis=1)
    #         v_tangential_array = jnp.sum(y_airf_array * relative_velocity_array, axis=1)

    #         # Compute effective angle of attack.
    #         alpha_array = jnp.arctan(v_normal_array / v_tangential_array)

    #         # Compute magnitudes (velocity in the plane)
    #         Umag_array = jnp.linalg.norm(relative_velocity_crossz_array, axis=1)
    #         Umagw_array = jnp.linalg.norm(Uinfcrossz_array, axis=1)

    #         # Compute the lift coefficient array using vectorized evaluation.
    #         # Define a function that computes cl for one panel.
    #         def compute_cl_for_panel(alpha, polar_data):
    #             return jnp.interp(alpha, polar_data[:, 0], polar_data[:, 1])

    #         # Vectorize over the first dimension.
    #         cl_array = jax.vmap(compute_cl_for_panel)(alpha_array, polar_data_array)
    #         # Compute the new circulation (aerodynamic forcing term)
    #         gamma_new_calc = (
    #             0.5 * (Umag_array**2) / Umagw_array * cl_array * chord_array
    #         )

    #         # Residual: difference between computed circulation and current gamma.
    #         F_val = gamma_new_calc - gamma

    #         return F_val, alpha_array, Umag_array

    #     def compute_jacobian_J_jax(
    #         gamma: jnp.ndarray,
    #         AIC_x: jnp.ndarray,
    #         AIC_y: jnp.ndarray,
    #         AIC_z: jnp.ndarray,
    #         va_array: jnp.ndarray,
    #         chord_array: jnp.ndarray,
    #         x_airf_array: jnp.ndarray,
    #         y_airf_array: jnp.ndarray,
    #         z_airf_array: jnp.ndarray,
    #         calculate_cl_func,
    #     ):
    #         """
    #         Compute the Jacobian of the residual with respect to gamma using JAX's automatic differentiation.

    #         Args:
    #         gamma: (n,) JAX array of circulation values.
    #         All other arguments are JAX arrays/constants.

    #         Returns:
    #         J: (n, n) Jacobian matrix, where J[i,j] = ∂F_i/∂γ_j.
    #         """

    #         # Wrap the residual function so that gamma is the only variable.
    #         # We differentiate only the first output (F_val).
    #         def F_wrapped(g):
    #             return compute_gamma_residual_jax(
    #                 g,
    #                 AIC_x,
    #                 AIC_y,
    #                 AIC_z,
    #                 va_array,
    #                 chord_array,
    #                 x_airf_array,
    #                 y_airf_array,
    #                 z_airf_array,
    #                 polar_data_array,
    #             )[0]

    #         # jax.jacobian returns a function; evaluate it at gamma.
    #         J = jax.jacobian(F_wrapped)(gamma)
    #         return J

    #     # Newton-Raphson loop using JAX:
    #     def newton_raphson_jax(
    #         gamma_init: jnp.ndarray,
    #         AIC_x: jnp.ndarray,
    #         AIC_y: jnp.ndarray,
    #         AIC_z: jnp.ndarray,
    #         va_array: jnp.ndarray,
    #         chord_array: jnp.ndarray,
    #         x_airf_array: jnp.ndarray,
    #         y_airf_array: jnp.ndarray,
    #         z_airf_array: jnp.ndarray,
    #         calculate_cl_func,
    #         max_iterations: int,
    #         allowed_error: float,
    #         relaxation_factor_init: float,
    #         min_relaxation_factor: float,
    #     ):
    #         """
    #         Newton-Raphson solver using JAX automatic differentiation.

    #         Returns:
    #         (converged: bool, gamma: jnp.ndarray, alpha_array: jnp.ndarray, Umag_array: jnp.ndarray)
    #         """
    #         gamma = gamma_init
    #         relaxation = relaxation_factor_init

    #         for it in range(max_iterations):
    #             # Compute residual
    #             F_val, alpha_array, Umag_array = compute_gamma_residual_jax(
    #                 gamma,
    #                 AIC_x,
    #                 AIC_y,
    #                 AIC_z,
    #                 va_array,
    #                 chord_array,
    #                 x_airf_array,
    #                 y_airf_array,
    #                 z_airf_array,
    #                 polar_data_array,
    #             )
    #             norm_F = jnp.linalg.norm(F_val, ord=jnp.inf)
    #             if norm_F < allowed_error:
    #                 logging.info(
    #                     f"Converged after {it} iterations with residual norm {norm_F:.3e}."
    #                 )
    #                 return True, gamma, alpha_array, Umag_array

    #             # Compute the Jacobian analytically via JAX.
    #             J = compute_jacobian_J_jax(
    #                 gamma,
    #                 AIC_x,
    #                 AIC_y,
    #                 AIC_z,
    #                 va_array,
    #                 chord_array,
    #                 x_airf_array,
    #                 y_airf_array,
    #                 z_airf_array,
    #                 polar_data_array,
    #             )

    #             # Solve for the Newton step: J * delta = -F_val.
    #             try:
    #                 delta = jnp.linalg.solve(J, -F_val)
    #             except Exception as e:
    #                 logging.error(f"Jacobian is singular at iteration {it}.")
    #                 return False, gamma, alpha_array, Umag_array

    #             # Trial update with current relaxation factor.
    #             gamma_trial = gamma + relaxation * delta
    #             F_trial, _, _ = compute_gamma_residual_jax(
    #                 gamma_trial,
    #                 AIC_x,
    #                 AIC_y,
    #                 AIC_z,
    #                 va_array,
    #                 chord_array,
    #                 x_airf_array,
    #                 y_airf_array,
    #                 z_airf_array,
    #                 polar_data_array,
    #             )

    #             if jnp.linalg.norm(F_trial, ord=jnp.inf) < norm_F:
    #                 gamma = gamma_trial
    #                 relaxation = jnp.minimum(relaxation * 1.1, 1.0)
    #             else:
    #                 relaxation = relaxation * 0.5
    #                 if relaxation < min_relaxation_factor:
    #                     logging.error(
    #                         "Relaxation coefficient too small; stopping iteration."
    #                     )
    #                     return False, gamma, alpha_array, Umag_array

    #         logging.warning(
    #             f"Did not converge within {max_iterations} iterations (final residual {norm_F:.3e})."
    #         )
    #         return False, gamma, alpha_array, Umag_array

    #     # Defining everything as jax
    #     AIC_x = jnp.array(self.AIC_x)
    #     AIC_y = jnp.array(self.AIC_y)
    #     AIC_z = jnp.array(self.AIC_z)
    #     va_array = jnp.array(self.va_array)
    #     chord_array = jnp.array(self.chord_array)
    #     x_airf_array = jnp.array(self.x_airf_array)
    #     y_airf_array = jnp.array(self.y_airf_array)
    #     z_airf_array = jnp.array(self.z_airf_array)
    #     polar_data_list = [panel.panel_polar_data for panel in self.panels]
    #     polar_data_array = jnp.array(polar_data_list)

    #     converged, gamma_new, alpha_array, Umag_array = newton_raphson_jax(
    #         gamma_new,
    #         AIC_x,
    #         AIC_y,
    #         AIC_z,
    #         va_array,
    #         chord_array,
    #         x_airf_array,
    #         y_airf_array,
    #         z_airf_array,
    #         polar_data_array,
    #         self.max_iterations,
    #         self.allowed_error,
    #         self.relaxation_factor,
    #         self.min_relaxation_factor,
    #     )
    #     return (
    #         bool(converged),
    #         np.array(gamma_new),
    #         np.array(alpha_array),
    #         np.array(Umag_array),
    #     )
