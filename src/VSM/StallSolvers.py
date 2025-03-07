import numpy as np
import logging
from scipy.optimize import broyden1, root, newton_krylov


class StallSolvers:

    def __init__(self, solver_instance):
        # Using composition to get all the self-attributes from Solver class
        # The self attributes, can be called using self.solver.attribute_name
        self.solver = solver_instance

    def compute_abc_prime(self):
        """
        Compute the finite-difference coefficients a', b', c' for the boundary nodes.
        """
        # Boundary nodes
        d1 = self.solver.y_coords[1] - self.solver.y_coords[0]
        d2 = self.solver.y_coords[2] - self.solver.y_coords[0]
        a_prime = 2.0 / (d1 * d2)
        b_prime = 2.0 / ((d1 - d2) * d1)
        c_prime = -2.0 / ((d1 - d2) * d2)
        return a_prime, b_prime, c_prime

    def compute_abc(self, i):
        """
        Compute the finite-difference coefficients a, b, c for the interior nodes.
        """
        d1 = self.solver.y_coords[i + 1] - self.solver.y_coords[i]
        d_1 = self.solver.y_coords[i] - self.solver.y_coords[i - 1]
        a = 2.0 / ((d1 + d_1) * d_1)
        b = -2.0 / (d1 * d_1)
        c = 2.0 / ((d1 + d_1) * d1)
        return a, b, c

    def compute_abc_dbl_prime(self):
        """
        Compute the finite-difference coefficients a'', b'', c'' for the boundary nodes.
        """
        # Boundary nodes
        d1 = self.solver.y_coords[-1] - self.solver.y_coords[-2]
        d2 = self.solver.y_coords[-1] - self.solver.y_coords[-3]
        a_dbl_prime = 2.0 / (d1 * d2)
        b_dbl_prime = 2.0 / ((d1 - d2) * d1)
        c_dbl_prime = -2.0 / ((d1 - d2) * d2)
        return a_dbl_prime, b_dbl_prime, c_dbl_prime

    def compute_second_derivative_of_gamma(self, gamma):
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

        n = self.solver.n_panels
        second_deriv = np.zeros(n)

        # Quick exit if too few points
        if n < 3:
            raise ValueError(
                "At least 3 panels are required to compute the second derivative."
            )

        # ===============================
        # 1) Boundary formula for i=0 => eq. (21) at i=1 in reference
        # ===============================
        a_prime, b_prime, c_prime = self.compute_abc_prime()
        second_deriv[0] = a_prime * gamma[0] + b_prime * gamma[1] + c_prime * gamma[2]

        # ===============================
        # 2) Interior points: i = 1..(n-2)
        # ===============================
        for i in range(1, n - 1):
            a, b, c = self.compute_abc(i)
            second_deriv[i] = a * gamma[i - 1] + b * gamma[i] + c * gamma[i + 1]

        # ===============================
        # 3) Boundary formula for i=n-1 => eq. (21) at i=n in reference
        # ===============================
        # here it is not y_coords[n], because of python 0-based indexing
        a_dbl_prime, b_dbl_prime, c_dbl_prime = self.compute_abc_dbl_prime()
        second_deriv[n - 1] = (
            a_dbl_prime * gamma[n - 1]
            + b_dbl_prime * gamma[n - 2]
            + c_dbl_prime * gamma[n - 3]
        )

        return second_deriv

    def compute_G_residual(
        self,
        gamma,
        mu_array,
    ):
        alpha_array, Umag_array, cl_array, Umagw_array = (
            self.solver.compute_aerodynamic_quantities(gamma)
        )
        d2gamma_dy2 = self.compute_second_derivative_of_gamma(gamma)
        return (
            gamma
            - (0.5 * Umag_array * self.solver.chord_array * cl_array)
            - (mu_array * d2gamma_dy2)
        )

    def compute_mu_tilde_array(self, K):
        """Computes μ_i from the diagonal of the Jacobian and the central finite-difference coefficient.

        Args:
            - K (np.ndarray): Jacobian matrix (n x n).
            - y_coords (np.array): Spanwise coordinates of the panels.
        Returns
            - mu_tilde_array (np.array): Artificial viscosity coefficients.
        """
        n = self.solver.n_panels
        mu_tilde_array = np.zeros(n)

        # first panel
        a_prime, _, _ = self.compute_abc_prime()
        mu_tilde_array[0] = np.minimum(0.0, K[0, 0] / a_prime)

        for i in range(1, n - 1):
            _, b, _ = self.compute_abc(i)
            mu_tilde_array[i] = np.maximum(0.0, K[i, i] / b)

        # last panel
        a_dbl_prime, _, _ = self.compute_abc_dbl_prime()
        mu_tilde_array[n - 1] = np.minimum(0.0, K[n - 1, n - 1] / a_dbl_prime)
        return mu_tilde_array

    def compute_jacobian_K_finite_difference(self, F_func, gamma, *args):
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
        n = self.solver.n_panels
        K = np.zeros((n, n))
        F0 = F_func(gamma, *args)
        for i in range(n):
            gamma_eps = gamma.copy()
            gamma_eps[i] += self.solver.jacobian_eps
            F1 = F_func(gamma_eps, *args)
            K[:, i] = (F1 - F0) / self.solver.jacobian_eps
        return K

    def compute_F_residual(self, gamma):
        _, Umag_array, cl_array, Umagw_array = (
            self.solver.compute_aerodynamic_quantities(gamma)
        )
        gamma_new = (
            0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.solver.chord_array
        )
        return (gamma_new) - gamma

    def solve_nonlinear_system(self, gamma_initial, mu_array):
        """
        Attempts to solve the nonlinear system F(gamma)=0 using a sequence of methods:
        1. Newton–Krylov with adaptive relaxation.
        2. least_squares with trf.
        3. Broyden’s method.

        Returns:
        Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
            (converged, gamma_new, alpha_array, Umag_array)
        """
        f_tol = self.solver.allowed_error
        max_iter = self.solver.max_iterations

        # Our residual function; it should take (gamma, mu_array) and return an (n,)-vector.
        solver_used = None
        success = False
        gamma_new = None

        # If still not converged, try Broyden's method.
        if not success:
            try:
                gamma_new = broyden1(
                    lambda x: self.compute_G_residual(x, mu_array),
                    gamma_initial,
                    f_tol=f_tol,
                    maxiter=max_iter,
                )
                if (
                    np.linalg.norm(
                        self.compute_G_residual(gamma_new, mu_array), ord=np.inf
                    )
                    < f_tol
                ):
                    solver_used = "broyden1"
                    success = True
                    logging.info("Converged using Broyden's method")
                else:
                    logging.warning(
                        "Broyden's method did not converge to desired tolerance"
                    )
            except Exception as e:
                logging.warning(f"Broyden's method failed: {e}")

        if not success:
            logging.error("Nonlinear solver failed to converge using all methods")
            return (
                False,
                np.zeros(self.solver.n_panels),
                np.zeros(self.solver.n_panels),
                np.zeros(self.solver.n_panels),
            )
        else:
            # Compute final aerodynamic quantities.
            alpha_array, Umag_array, cl_array, Umagw_array = (
                self.solver.compute_aerodynamic_quantities(gamma_new)
            )
            logging.info(f"Converged using {solver_used} method")
            return True, gamma_new, alpha_array, Umag_array

    # def compute_jacobian_casadi_numerical(gamma_new):
    #       import casadi as ca
    #     n = len(gamma_new)

    #     # Convert to CasADi symbols
    #     gamma_sym = ca.SX.sym("gamma", n)

    #     # Create a CasADi function that wraps your numpy-based function
    #     F_expr = []

    #     # For each input, create output symbol
    #     for i in range(n):
    #         gamma_perturb = gamma_new.copy()
    #         # Small perturbation
    #         eps = 1e-6
    #         gamma_perturb[i] += eps
    #         # Forward evaluation
    #         f_forward = compute_F_residual(gamma_perturb)
    #         # Backward evaluation
    #         gamma_perturb[i] = gamma_new[i] - eps
    #         f_backward = compute_F_residual(gamma_perturb)
    #         # Central difference
    #         df = (f_forward - f_backward) / (2 * eps)
    #         F_expr.append(df)

    #     # Convert list to matrix
    #     J = np.column_stack(F_expr)

    #     return J

    def check_multidim_condition_number(self, F, x0, eps=1e-6):
        """
        For a vector function F: R^n -> R^n, compute the Jacobian via finite differences
        at the point x0 and return its condition number (ratio of largest to smallest singular value).

        Args:
            F (callable): Function from R^n to R^n (expects a 1D NumPy array).
            x0 (np.ndarray): Point in R^n at which to evaluate the Jacobian.
            eps (float): Finite difference step size.

        Returns:
            tuple: (condition_number, J)
            condition_number (float): Ratio of largest to smallest singular value of J.
            J (np.ndarray): The computed Jacobian (n x n).
        """
        n = len(x0)
        J = np.zeros((n, n))
        F0 = F(x0)
        for i in range(n):
            x_eps = x0.copy()
            x_eps[i] += eps
            F1 = F(x_eps)
            J[:, i] = (F1 - F0) / eps
        U, s, Vh = np.linalg.svd(J)
        condition_number = s[0] / s[-1]
        return condition_number, J

    def build_second_derivative_matrix(
        self,
    ):
        """
        Construct the N x N matrix that approximates the second derivative
        d²/dy², using the boundary stencils from eq. (22) and the interior
        3-point stencils for non-uniform grids.
        """
        n = self.solver.n_panels
        D = np.zeros((n, n))

        # 1) Top boundary row => a' * Γ[0] + b' * Γ[1] + c' * Γ[2]
        a_prime, b_prime, c_prime = self.compute_abc_prime()
        D[0, 0] = a_prime
        D[0, 1] = b_prime
        D[0, 2] = c_prime

        # 2) Interior rows: for i in [1..n-2]
        for i in range(1, n - 1):
            a, b, c = self.compute_abc(i)
            D[i, i - 1] = a
            D[i, i] = b
            D[i, i + 1] = c

        # 3) Bottom boundary row => a'' * Γ[n-1] + b'' * Γ[n-2] + c'' * Γ[n-3]
        a_dbl_prime, b_dbl_prime, c_dbl_prime = self.compute_abc_dbl_prime()
        D[n - 1, n - 1] = a_dbl_prime
        D[n - 1, n - 2] = b_dbl_prime
        D[n - 1, n - 3] = c_dbl_prime

        return D

    def compute_analytic_jacobian_J(self, gamma):
        n = self.solver.n_panels
        J = np.zeros((n, n))

        alpha_eff_array, V_proj_array, cl_array, _ = (
            self.solver.compute_aerodynamic_quantities(gamma)
        )
        induced_velocity_all = np.array(
            [
                np.matmul(self.solver.AIC_x, gamma),
                np.matmul(self.solver.AIC_y, gamma),
                np.matmul(self.solver.AIC_z, gamma),
            ]
        ).T  # v_ind
        V_eff_array = (
            self.solver.va_array + induced_velocity_all
        )  # v_eff = v_inf + v_ind

        delta_alpha = 1e-4  # 1e-3, 1e-6 were all tested, giving the same results
        for i in range(n):
            x_i = self.solver.x_airf_array[i]
            y_i = self.solver.y_airf_array[i]
            z_i = self.solver.z_airf_array[i]

            # finite difference to find dcl_dalpha
            cl_alpha_plus = self.solver.panels[i].calculate_cl(
                alpha_eff_array[i] + delta_alpha
            )
            dcl_dalpha_i = (cl_alpha_plus - cl_array[i]) / delta_alpha
            # logging.info(
            # f"cl[i]: {cl_array[i]}, cl_alpha_plus: {cl_alpha_plus}, dcl_dalpha: {dcl_dalpha_i}"
            # )
            for j in range(n):
                # δ_{ij} (Kronecker delta)
                delta_ij = 1.0 if (i == j) else 0.0

                # Compute the coefficients A_ij, B_ij from ~Eq.9,10, v_ij is represent in ~Eq.2
                v_ij = np.array(
                    [
                        self.solver.AIC_x[i, j],
                        self.solver.AIC_y[i, j],
                        self.solver.AIC_z[i, j],
                    ]
                )
                A_ij = v_ij * (z_i**2) - (z_i**2) * np.dot(v_ij, z_i)
                B_ij = y_i * np.dot(v_ij, x_i) - x_i * np.dot(v_ij, y_i)

                # logging.info(f"x_ij: {x_i}")
                # logging.info(f"y_ij: {y_i}")
                # logging.info(f"z_ij: {z_i}")
                # logging.info(f"v_ij: {v_ij}")
                # logging.info(f"cl_array[i]: {cl_array[i]}")
                # logging.info(
                #     f"dcl_dalpha[i]: {dcl_dalpha_i}, per deg: {dcl_dalpha_i / (57.3)})"
                # )
                # logging.info(f"delta_ij: {delta_ij}")
                # logging.info(f"self.solver.chord_array[i]: {self.solver.chord_array[i]}")
                # logging.info(f"V_eff[i]: {V_eff_array[i]}")
                # logging.info(f"V_proj[i]: {V_proj_array[i]}")
                # logging.info(f"A_ij: {A_ij}")
                # logging.info(f"B_ij: {B_ij}")
                # logging.info(
                #     f"Aijcl(alpha)+Bijdcl_dalpha: {A_ij * cl_array[i] + B_ij * dcl_dalpha_i}"
                # )
                # logging.info(
                #     f"V_eff[i] / V_proj[i]: {V_eff_array[i] / V_proj_array[i]}"
                # )
                # logging.info(
                #     f"dot-product: {np.dot((V_eff_array[i] / V_proj_array[i]), (A_ij * cl_array[i] + B_ij * dcl_dalpha_i))}"
                # )
                # breakpoint()

                # eq. 8 or A21
                J[i, j] = delta_ij - 0.5 * self.solver.chord_array[i] * np.dot(
                    (V_eff_array[i] / V_proj_array[i]),
                    (A_ij * cl_array[i] + B_ij * dcl_dalpha_i),
                )
                # logging.info(
                # f"[i,j: {i},{j}, delta_ij: {delta_ij}, J[i,j]: {J[i,j]:.3f}"
                # )
        return J

    def compute_analytical_jacobian_K(self, gamma, mu_array):
        J = self.compute_analytic_jacobian_J(gamma)
        mu_matrix = np.diag(mu_array)
        D = self.build_second_derivative_matrix()
        # logging.info(f"J: {J}")
        # logging.info(f"K: {J - mu_matrix @ D}")
        return J - np.matmul(mu_matrix, D)

    # ===============================
    #       Stall Solvers
    # ===============================
    def gamma_loop_simonet_stall(
        self,
        gamma_initial,
        extra_relaxation_factor: float = 1.0,
    ):
        """Loop to calculate the circulation distribution

        Args:

        """

        # Initializing the mu_array
        mu_array = np.zeros(self.solver.n_panels)
        gamma = np.copy(gamma_initial)
        K = self.compute_jacobian_K_finite_difference(
            self.compute_G_residual, gamma, mu_array
        )
        mu_tilde_array = self.compute_mu_tilde_array(K)

        # Using eq 26. and interpreting b as the local spanwise panel width
        mu_array = self.solver.simonet_artificial_viscosity_fva * (
            mu_tilde_array / np.sum(mu_tilde_array * self.solver.width_array)
        )
        # looping untill max_iterations
        converged = False
        gamma_new = np.copy(gamma_initial)
        for i in range(self.solver.max_iterations):

            gamma = np.array(gamma_new)
            alpha_array, Umag_array, cl_array, Umagw_array = (
                self.solver.compute_aerodynamic_quantities(gamma)
            )
            gamma_new = (
                0.5
                * ((Umag_array**2) / Umagw_array)
                * cl_array
                * self.solver.chord_array
            )

            if self.solver.is_smooth_circulation:
                damp, is_damping_applied = self.solver.smooth_circulation(
                    circulation=gamma,
                    smoothness_factor=self.solver.smoothness_factor,
                    damping_factor=0.5,
                )
            elif self.solver.is_with_simonet_artificial_viscosity:
                damp = mu_array * self.compute_second_derivative_of_gamma(gamma)
                is_damping_applied = True
            else:
                damp = 0
                is_damping_applied = False

            gamma_new = (
                (1 - self.solver.relaxation_factor * extra_relaxation_factor) * gamma
                + self.solver.relaxation_factor * extra_relaxation_factor * gamma_new
                + damp
            )

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
            if normalized_error < self.solver.allowed_error:
                # if error smaller than limit, stop iteration cycle
                converged = True
                break

        if converged:
            logging.info(f"Converged after {i} iterations")
        else:
            logging.warning(
                f"NO convergences after {self.solver.max_iterations} iterations"
            )
        return converged, gamma_new, alpha_array, Umag_array

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

        # Doing a single iteration
        gamma = np.copy(gamma_initial)
        alpha_array, Umag_array, cl_array, Umagw_array = (
            self.solver.compute_aerodynamic_quantities(gamma)
        )
        gamma_new = (
            0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.solver.chord_array
        )
        gamma_new = (
            1 - self.solver.relaxation_factor
        ) * gamma + self.solver.relaxation_factor * gamma_new

        logging.info(
            f"condition number: {self.check_multidim_condition_number(compute_F_residual, gamma_new)[0]}"
        )

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

        K = -self.compute_jacobian_K_finite_difference(
            self.compute_F_residual, gamma_new
        )
        # Check if we should activate artificial viscosity
        if should_activate_artificial_viscosity(np.diag(K)):
            # Calculate mu_tilde_array and mu_array as before
            mu_tilde_array = self.compute_mu_tilde_array(K)
            sum_mu_tilde = np.sum(mu_tilde_array * self.solver.width_array)
            # logging.info(f"sum mu tilde: {sum_mu_tilde:.1e}")

            if np.abs(sum_mu_tilde) > 0:
                mu_array = self.solver.simonet_artificial_viscosity_fva * (
                    mu_tilde_array / sum_mu_tilde
                )
            else:
                logging.info(
                    "Artificial viscosity not activated (sum is zero or negative)"
                )
                mu_array = np.zeros(self.solver.n_panels)

        else:
            logging.info("Artificial viscosity not activated (linear regime detected)")
            mu_array = np.zeros(self.solver.n_panels)

            return self.solve_nonlinear_system(gamma_initial, mu_array)

    def gamma_loop_non_linear_simonet_stall_newton_raphson(self, gamma_initial):
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

        # ===============================
        # Doing a single iteration to start with
        # ===============================
        mu_array = np.zeros(self.solver.n_panels)
        relaxation = self.solver.relaxation_factor
        gamma = self.compute_G_residual(gamma_initial, mu_array)
        J = self.compute_analytic_jacobian_J(gamma)
        # logging.info(f"J_matrix: {J}")
        mu_tilde_array = self.compute_mu_tilde_array(J)
        # logging.info(f"mu_tilde_array: {mu_tilde_array}")
        # Using eq 26. and interpreting b as the local spanwise panel width
        b_array = np.array([panel.width for panel in self.solver.panels])
        sum_mu_tilde = np.sum(mu_tilde_array * b_array)
        if np.abs(sum_mu_tilde) > 0:
            mu_array = self.solver.simonet_artificial_viscosity_fva * (
                mu_tilde_array / sum_mu_tilde
            )
        else:
            logging.info(
                f"Artificial viscosity not activated (mu_tilde_sum: {sum_mu_tilde})"
            )
            mu_array = np.zeros(self.solver.n_panels)
        logging.info(f"mu_array: {mu_array}")

        success = False
        solver_used = None
        f_tol = self.solver.allowed_error
        x_tol = self.solver.allowed_error * 1e-1
        max_iter = self.solver.max_iterations
        relaxation = self.solver.relaxation_factor
        init_relax = self.solver.relaxation_factor
        min_relax = 1e-7

        for it in range(self.solver.max_iterations):
            # Compute residual (F_val) and aerodynamic arrays.
            F_val = self.compute_G_residual(
                gamma,
                mu_array,
            )
            # === Checking Convergence ===
            # computes the infinity norm, also known as max norm (maximum absolute row sum)
            norm_F = np.linalg.norm(F_val, np.inf)
            # logging.info(f"it: {it} -- norm_f: {norm_F:.4f}")
            if norm_F < self.solver.allowed_error:
                success = True
                solver_used = "Newton_Raphson"
                break
            K = self.compute_analytical_jacobian_K(gamma, mu_array)
            try:
                delta = np.linalg.solve(K, -F_val)
            except np.linalg.LinAlgError:
                logging.error(f"Jacobian is singular at iteration {it}.")
                break

            # Trial update using the current relaxation factor.
            gamma_new = gamma + relaxation * delta
            F_trial = self.compute_G_residual(
                gamma_new,
                mu_array,
            )
            # Adapt the relaxation factor: increase if the trial is successful; otherwise reduce.
            if np.linalg.norm(F_trial, np.inf) < norm_F:
                gamma = gamma_new
                relaxation = min(relaxation * 1.1, 1.0)
                logging.debug(f"increase relaxation to: {relaxation}")
            else:
                relaxation *= 0.5
                logging.debug(f"decrease relaxation to: {relaxation}")
                if relaxation < min_relax:
                    logging.error(
                        "Relaxation coefficient too small; stopping iteration."
                    )
                    break
        if not success:
            # recompute mu_array
            J = self.compute_analytic_jacobian_J(gamma)
            # logging.info(f"J_matrix: {J}")
            mu_tilde_array = self.compute_mu_tilde_array(J)
            # logging.info(f"mu_tilde_array: {mu_tilde_array}")
            # Using eq 26. and interpreting b as the local spanwise panel width
            b_array = np.array([panel.width for panel in self.solver.panels])
            sum_mu_tilde = np.sum(mu_tilde_array * b_array)
            if np.abs(sum_mu_tilde) > 0:
                mu_array = self.solver.simonet_artificial_viscosity_fva * (
                    mu_tilde_array / sum_mu_tilde
                )
            else:
                logging.info(
                    f"Artificial viscosity not activated (mu_tilde_sum: {sum_mu_tilde})"
                )
                mu_array = np.zeros(self.solver.n_panels)
            logging.info(f"NEW mu_array: {mu_array}")

            def smooth_array(arr, window_size):
                """
                Smooth an array using a simple moving average.

                Parameters:
                arr (list or np.ndarray): Input array of numbers
                window_size (int): Size of the moving average window (must be odd)

                Returns:
                list: Smoothed array
                """
                # Convert to numpy array if it isn't already
                arr = np.array(arr)

                # Check if array is empty
                if arr.size == 0 or window_size < 1:
                    return arr

                # Ensure window_size is odd and not larger than array length
                window_size = min(window_size, len(arr))
                if window_size % 2 == 0:
                    window_size += 1

                smoothed = []
                half_window = (window_size - 1) // 2

                for i in range(len(arr)):
                    # Define window boundaries
                    start = max(0, i - half_window)
                    end = min(len(arr), i + half_window + 1)

                    # Calculate average of window
                    window = arr[start:end]
                    avg = np.mean(window)  # Using np.mean instead of sum/len
                    smoothed.append(avg)

                return smoothed

            gamma = smooth_array(gamma, int(self.solver.n_panels / 10))

            # Broyden's method
            if not success:
                try:
                    gamma_new = broyden1(
                        lambda x: self.compute_G_residual(x, mu_array),
                        gamma,
                        f_tol=f_tol,
                        maxiter=max_iter,
                    )
                    if (
                        np.linalg.norm(
                            self.compute_G_residual(gamma_new, mu_array), ord=np.inf
                        )
                        < f_tol
                    ):
                        solver_used = "Broyden's method"
                        success = True
                        logging.info("Converged using Broyden's method")
                    else:
                        logging.warning(
                            "Broyden's method did not converge to desired tolerance"
                        )
                except Exception as e:
                    logging.warning(f"Broyden's method failed")

            if not success:
                logging.error("Nonlinear solver failed to converge using all methods")
                gamma_new = np.zeros(self.solver.n_panels)
                success = False

            # Compute final aerodynamic quantities.
            alpha_array, Umag_array, cl_array, Umagw_array = (
                self.solver.compute_aerodynamic_quantities(gamma_new)
            )
            if solver_used:
                logging.info(f"Converged using {solver_used} method")
            else:
                logging.info("No solver method succeeded")
            return success, gamma_new, alpha_array, Umag_array

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
            if self.solver.aerodynamic_model_type == "LLT" or (
                self.solver.artificial_damping["k2"] == 0
                and self.solver.artificial_damping["k4"] == 0
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
                self.solver.artificial_damping["k2"] * dif2
                - self.solver.artificial_damping["k4"] * dif4
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


# ##TODO: remove
# # defining upfront for simonet_artificial_viscosity
# if (
#     self.solver.is_with_simonet_artificial_viscosity
#     and self.solver.simonet_artificial_viscosity_fva is None
# ):
#     root_chord = np.max(self.solver.chord_array)
#     wing = body_aero.wings[0]
#     projected_area = wing.calculate_projected_area()
#     self.solver.simonet_artificial_viscosity_fva = 0.16 * root_chord * projected_area
#     print(
#         f"Simonet Artificial Viscosity fva: {self.solver.simonet_artificial_viscosity_fva:.4f} (0.005, 5e-4 was optimal before)"
#     )

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
#     AIC_x = jnp.array(self.solver.AIC_x)
#     AIC_y = jnp.array(self.solver.AIC_y)
#     AIC_z = jnp.array(self.solver.AIC_z)
#     va_array = jnp.array(self.solver.va_array)
#     chord_array = jnp.array(self.solver.chord_array)
#     x_airf_array = jnp.array(self.solver.x_airf_array)
#     y_airf_array = jnp.array(self.solver.y_airf_array)
#     z_airf_array = jnp.array(self.solver.z_airf_array)
#     polar_data_list = [panel.panel_polar_data for panel in self.solver.panels]
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
#         self.solver.max_iterations,
#         self.solver.allowed_error,
#         self.solver.relaxation_factor,
#         self.solver.min_relaxation_factor,
#     )
#     return (
#         bool(converged),
#         np.array(gamma_new),
#         np.array(alpha_array),
#         np.array(Umag_array),
#     )
