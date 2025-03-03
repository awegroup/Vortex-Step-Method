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
        type_initial_gamma_distribution (str): Type of initial gamma distribution (default: "elliptic")
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
        density: float = 1.225,
        max_iterations: int = 5000,
        allowed_error: float = 1e-12,  # 1e-5,
        relaxation_factor: float = 0.01,
        type_initial_gamma_distribution: str = "elliptic",
        is_with_gamma_feedback: bool = True,
        core_radius_fraction: float = 1e-20,
        mu: float = 1.81e-5,
        is_only_f_and_gamma_output: bool = False,
        is_new_vector_definition: bool = True,
        reference_point: list = [-0.17, 0.00, 9.25],
        # --- STALL: smooth_circulation ---
        is_smooth_circulation: bool = False,
        smoothness_factor: float = 0.08,  # for smoothing stall model
        # --- STALL: artificial damping ---
        is_artificial_damping: bool = False,
        artificial_damping: dict = {"k2": 0.1, "k4": 0.0},
        ## TODO: would be nice to having these defined here instead of inside the panel class?
        # aerodynamic_center_location: float = 0.25,
        # control_point_location: float = 0.75,
        ## TODO: these are hardcoded in the Filament, should be defined here
        # alpha_0 = 1.25643
        # nu = 1.48e-5
    ):
        self.aerodynamic_model_type = aerodynamic_model_type
        self.density = density
        self.max_iterations = max_iterations
        self.allowed_error = allowed_error
        self.relaxation_factor = relaxation_factor

        self.type_initial_gamma_distribution = type_initial_gamma_distribution
        self.core_radius_fraction = core_radius_fraction
        self.mu = mu
        self.is_only_f_and_gamma_output = is_only_f_and_gamma_output
        self.is_with_gamma_feedback = is_with_gamma_feedback
        self.is_new_vector_definition = is_new_vector_definition
        self.reference_point = reference_point
        self.smoothness_factor = smoothness_factor
        # stall model things
        self.is_smooth_circulation = is_smooth_circulation
        self.is_artificial_damping = is_artificial_damping
        self.artificial_damping = artificial_damping

    def solve(self, wing_aero, gamma_distribution=None):
        """Solve the aerodynamic model

        Args:
            wing_aero (WingAerodynamics): WingAerodynamics object
            gamma_distribution (np.array): Initial gamma distribution (default: None)

        Returns:
            dict: Results of the aerodynamic model"""

        if wing_aero.va is None:
            raise ValueError("Inflow conditions are not set")

        # Initialize variables here, outside the loop
        panels = wing_aero.panels
        n_panels = wing_aero.n_panels
        alpha_array = np.zeros(n_panels)
        relaxation_factor = self.relaxation_factor
        (
            x_airf_array,
            y_airf_array,
            z_airf_array,
            va_array,
            chord_array,
        ) = (
            np.zeros((n_panels, 3)),
            np.zeros((n_panels, 3)),
            np.zeros((n_panels, 3)),
            np.zeros((n_panels, 3)),
            np.zeros(n_panels),
        )
        for i, panel in enumerate(panels):
            x_airf_array[i] = panel.x_airf
            y_airf_array[i] = panel.y_airf
            z_airf_array[i] = panel.z_airf
            va_array[i] = panel.va
            chord_array[i] = panel.chord

        va_norm_array = np.linalg.norm(va_array, axis=1)
        va_unit_array = va_array / va_norm_array[:, None]

        # Calculate the new circulation distribution iteratively
        AIC_x, AIC_y, AIC_z = wing_aero.calculate_AIC_matrices(
            self.aerodynamic_model_type,
            self.core_radius_fraction,
            va_norm_array,
            va_unit_array,
        )

        # initialize gamma distribution inside
        if (
            gamma_distribution is None
            and self.type_initial_gamma_distribution == "elliptic"
        ) or not self.is_with_gamma_feedback:
            gamma_initial = (
                wing_aero.calculate_circulation_distribution_elliptical_wing()
            )

        elif len(gamma_distribution) == n_panels:
            gamma_initial = gamma_distribution

        logging.debug(
            f"Initial gamma_new: {gamma_initial} . is_with_gamma_feedback: {self.is_with_gamma_feedback}",
        )

        # Run the iterative loop
        converged, gamma_new, alpha_array, Umag_array = self.gamma_loop(
            gamma_initial,
            AIC_x,
            AIC_y,
            AIC_z,
            va_array,
            chord_array,
            x_airf_array,
            y_airf_array,
            z_airf_array,
            panels,
            relaxation_factor,
        )
        # run again with half the relaxation factor if not converged
        if not converged and relaxation_factor > 1e-3:
            logging.info(
                f" ---> Running again with half the relaxation_factor = {relaxation_factor / 2}"
            )
            relaxation_factor = relaxation_factor / 2
            converged, gamma_new, alpha_array, Umag_array = self.gamma_loop(
                gamma_initial,
                AIC_x,
                AIC_y,
                AIC_z,
                va_array,
                chord_array,
                x_airf_array,
                y_airf_array,
                z_airf_array,
                panels,
                relaxation_factor,
            )

        # Calculating results (incl. updating angle of attack for VSM)
        results = wing_aero.calculate_results(
            gamma_new,
            self.density,
            self.aerodynamic_model_type,
            self.core_radius_fraction,
            self.mu,
            alpha_array,
            Umag_array,
            chord_array,
            x_airf_array,
            y_airf_array,
            z_airf_array,
            va_array,
            va_norm_array,
            va_unit_array,
            panels,
            self.is_only_f_and_gamma_output,
            self.is_new_vector_definition,
            self.reference_point,  # roughly the cg of V3
        )

        return results

    def gamma_loop(
        self,
        gamma_new,
        AIC_x,
        AIC_y,
        AIC_z,
        va_array,
        chord_array,
        x_airf_array,
        y_airf_array,
        z_airf_array,
        panels,
        relaxation_factor,
    ):
        """Loop to calculate the circulation distribution

        Args:
            - gamma_new (np.array): Initial gamma distribution
            - AIC_x (np.array): Induced velocity matrix in x-direction
            - AIC_y (np.array): Induced velocity matrix in y-direction
            - AIC_z (np.array): Induced velocity matrix in z-direction
            - va_array (np.array): Free-stream velocity array
            - chord_array (np.array): Chord length array
            - x_airf_array (np.array): Airfoil x-coordinates array
            - y_airf_array (np.array): Airfoil y-coordinates array
            - z_airf_array (np.array): Airfoil z-coordinates array
            - panels (list): List of Panel objects
            - relaxation_factor (float): Relaxation factor for convergence

        Returns:
            - bool: Whether the convergence is reached
            - np.array: Final gamma distribution
            - np.array: Angle of attack array
            - np.array: Relative velocity magnitude array
        """

        # looping untill max_iterations
        converged = False
        for i in range(self.max_iterations):

            gamma = np.array(gamma_new)
            induced_velocity_all = np.array(
                [
                    np.matmul(AIC_x, gamma),
                    np.matmul(AIC_y, gamma),
                    np.matmul(AIC_z, gamma),
                ]
            ).T
            relative_velocity_array = va_array + induced_velocity_all
            relative_velocity_crossz_array = jit_cross(
                relative_velocity_array, z_airf_array
            )
            Uinfcrossz_array = jit_cross(va_array, z_airf_array)
            v_normal_array = np.sum(x_airf_array * relative_velocity_array, axis=1)
            v_tangential_array = np.sum(y_airf_array * relative_velocity_array, axis=1)
            alpha_array = np.arctan(v_normal_array / v_tangential_array)
            Umag_array = np.linalg.norm(relative_velocity_crossz_array, axis=1)
            Umagw_array = np.linalg.norm(Uinfcrossz_array, axis=1)
            cl_array = np.array(
                [panel.calculate_cl(alpha) for panel, alpha in zip(panels, alpha_array)]
            )
            gamma_new = 0.5 * Umag_array**2 / Umagw_array * cl_array * chord_array

            if self.is_smooth_circulation:
                damp, is_damping_applied = self.smooth_circulation(
                    circulation=gamma,
                    smoothness_factor=self.smoothness_factor,
                    damping_factor=0.5,
                )

            # elif self.is_artificial_damping:
            # damp, is_damping_applied = self.smooth_circulation(
            #     circulation=gamma, smoothness_factor=0.1, damping_factor=0.5
            # )
            ## below works well for "split-provided" n_panel=105 V3
            # damp, is_damping_applied = self.smooth_circulation(
            # circulation=gamma, smoothness_factor=0.15, damping_factor=0.5
            # )
            ## below works well for "linear" n_panel=130 V3
            # damp, is_damping_applied = self.smooth_circulation(
            #     circulation=gamma,
            #     smoothness_factor=self.smoothness_factor,
            #     damping_factor=0.5,
            # )
            # logging.debug("damp: %s", damp)
            # J_diag = self.compute_J_diag_finite_diff(
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
            # )

            # damp, mu = self.apply_artificial_viscosity_chattot(
            #     gamma=gamma,
            #     y_panel_centers=[panel.control_point[1] for panel in panels],
            #     Jdiag=J_diag,
            #     b_array=[panel.width for panel in panels],
            #     fva=1e-3,  # user-chosen "area" scaling, m^3 units
            # )
            # is_damping_applied = True

            else:
                damp = 0
                is_damping_applied = False

            gamma_new = (
                (1 - relaxation_factor) * gamma + relaxation_factor * gamma_new + damp
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

    def compute_G_vector(
        self,
        gamma,
        AIC_x,
        AIC_y,
        AIC_z,
        va_array,
        chord_array,
        x_airf_array,
        y_airf_array,
        z_airf_array,
        panels,
    ):
        """
        Return the G_i(gamma) vector as described in your eq. (19)-(20),
        ignoring for the moment any -mu_i*(d^2Gamma/dy^2) if you want
        that inside G. Or you can put the artificial-viscosity piece in here
        if your method directly modifies G.
        """
        # 1) Compute induced velocities
        induced_vel_x = AIC_x @ gamma
        induced_vel_y = AIC_y @ gamma
        induced_vel_z = AIC_z @ gamma
        induced_velocity_all = np.vstack(
            (induced_vel_x, induced_vel_y, induced_vel_z)
        ).T

        # 2) Sum with free stream -> get alpha, etc.
        relative_velocity_array = va_array + induced_velocity_all
        v_normal_array = np.sum(x_airf_array * relative_velocity_array, axis=1)
        v_tangential_array = np.sum(y_airf_array * relative_velocity_array, axis=1)
        alpha_array = np.arctan2(v_normal_array, v_tangential_array)
        cl_array = np.array([p.calculate_cl(a) for p, a in zip(panels, alpha_array)])

        # 3) Build G_i
        #   G_i = Gamma_i - 0.5 * Umag^2/Umag_infinity * cl_i * chord_i (just for example),
        #   or your actual formula from eq. (19)-(20).

        # For a typical lifting-line approach:
        # let G_i = gamma_i - (1/2) * V_proj_i * c_i * Cl( alpha_eff_i )
        # We'll pretend "V_proj_i" = magnitude of relative_velocity_array[i],
        # or you can do something more specialized.
        Umag_array = np.linalg.norm(relative_velocity_array, axis=1)
        G = gamma - 0.5 * Umag_array * chord_array * cl_array  # simplistic example

        return G  # shape (n,)

    def compute_J_diag_finite_diff(
        self,
        gamma,
        AIC_x,
        AIC_y,
        AIC_z,
        va_array,
        chord_array,
        x_airf_array,
        y_airf_array,
        z_airf_array,
        panels,
    ):
        """
        Return an approximate diagonal of the Jacobian by finite difference.
        """
        n = len(gamma)
        # Evaluate G(gamma) once
        G_base = self.compute_G_vector(
            gamma,
            AIC_x,
            AIC_y,
            AIC_z,
            va_array,
            chord_array,
            x_airf_array,
            y_airf_array,
            z_airf_array,
            panels,
        )
        J_diag = np.zeros(n)

        for i in range(n):
            # pick a small perturbation
            eps = 1e-6 * (1.0 + abs(gamma[i]))  # for example
            gamma_pert = gamma.copy()
            gamma_pert[i] += eps

            G_pert = self.compute_G_vector(
                gamma_pert,
                AIC_x,
                AIC_y,
                AIC_z,
                va_array,
                chord_array,
                x_airf_array,
                y_airf_array,
                z_airf_array,
                panels,
            )
            # approximate derivative wrt gamma[i] = [G_i(gamma+eps) - G_i(gamma)] / eps
            J_diag[i] = (G_pert[i] - G_base[i]) / eps

        return J_diag

    def apply_artificial_viscosity_chattot(
        self, gamma, y_panel_centers, Jdiag, b_array, fva
    ):
        """
        Implements the post-stall artificial viscosity correction described by
        Chattot [22] and subsequent authors, following Eqs. (19)-(27) in your notes.

        1) Build the second derivative of Gamma wrt y_s, using the finite-difference
        coefficients for irregular grids (Eqs. 21-22).

        2) Compute the "raw" mu_tilde[i] from Eq. (25), which depends on the sign of
        the diagonal Jacobian J_{ii}.

        3) Scale mu_tilde[i] to obtain mu[i] by Eq. (26), ensuring the "area"
        under mu is f_va.

        4) Return the array of artificial-viscosity corrections:
            correction[i] = - mu[i] * ( d^2 Gamma / d y_s^2 )_i
        You can then add that to your new circulation in the iteration.

        Args:
            gamma (np.ndarray): Current circulation distribution (size n).
            y_panel_centers (np.ndarray): Spanwise coordinates of the control
                points for each panel (size n).  Must be sorted along the span.
            Jdiag (np.ndarray): The diagonal of your lifting-line Jacobian, size n.
                We use its sign to decide whether to "activate" viscosity at panel i.
            b_array (np.ndarray): The "section span vector" b, i.e. the local
                panel width or at least a set of weights used in eq. (26).
            fva (float): The artificial-viscosity parameter (a "volume" dimension).
                Controls the total strength of the artificial viscosity.

        Returns:
            viscosity_correction (np.ndarray): length-n array of increments
                to be added to gamma, i.e.  gamma_new[i] += viscosity_correction[i].
            mu (np.ndarray): The final artificial-viscosity coefficients mu_i.
        """

        n = len(gamma)
        if n < 3:
            # With fewer than 3 panels, just return zero correction
            return np.zeros_like(gamma), np.zeros_like(gamma)

        # ----------------------------------------------------------------------
        # 1) Build the second-derivative operator D2 (an n x n matrix) using eq. (21)-(22).
        #    Then d2_gamma = D2 @ gamma.
        # ----------------------------------------------------------------------

        D2 = np.zeros((n, n))

        # Helper function to compute the standard interior coefficients: a,b,c
        # from eq. (22), for i in 1..n-1 (1-based).  In Python, that's i=1..n-2 (0-based).
        def interior_coeffs(i):
            # i is 0-based interior index, so d_im1 = y[i] - y[i-1], d_i = y[i+1] - y[i]
            d_im1 = y_panel_centers[i] - y_panel_centers[i - 1]
            d_i = y_panel_centers[i + 1] - y_panel_centers[i]
            a = 2.0 / ((d_im1 + d_i) * d_im1)
            b = -2.0 / (d_im1 * d_i)
            c = 2.0 / ((d_im1 + d_i) * d_i)
            return a, b, c

        # Fill interior rows: i=1..n-2 (0-based)
        for i in range(1, n - 1):
            a, b, c = interior_coeffs(i)
            D2[i, i - 1] = a
            D2[i, i] = b
            D2[i, i + 1] = c

        # Boundary at i=0: eq. (21) says use a', b', c'.
        # Typically: (∂^2 Γ /∂y^2)_1 = a' Γ_1 + b' Γ_2 + c' Γ_3 in 1-based indexing.
        # In Python, that means row 0 of D2 => combine gamma[0], gamma[1], gamma[2].
        #
        # The reference eq. (22) can be read as:
        #    a' =  2/( d1 * d2 ),
        #    b' =  (a combination of d1, d2) ...
        #    c' =  ...
        #
        # For clarity, let's define them exactly:
        if n >= 3:
            d1 = y_panel_centers[1] - y_panel_centers[0]
            d2 = y_panel_centers[2] - y_panel_centers[1]
            # A typical 3-pt forward difference for an irregular grid per eq. (22) can be arranged as:
            a_p = 2.0 / (d1 * d2)
            b_p = -((d1 + d2) / (d1 * d2))  # or another expression from the reference
            c_p = 2.0 / ((d1 + d2) * d2)  # check carefully with your eq. (22)
            #
            # You will want to verify these three with your own notes.  As an example:
            # eq. (22) might read:
            #    a' =  2/( d1*d2 )
            #    b' = -2/( d1*(d1 + d2) )
            #    c' =  2/( d2*(d1 + d2) )
            #
            # Please adjust if your reference has them differently.  For demonstration:
            D2[0, 0] = a_p
            D2[0, 1] = b_p
            D2[0, 2] = c_p

        # Boundary at i=n-1: eq. (21) says use a'', b'', c''.
        # (∂^2 Γ /∂y^2)_n = a'' Γ_n + b'' Γ_(n-1) + c'' Γ_(n-2)
        # in 1-based indexing.  In Python, that means row n-1 => combine gamma[n-1], gamma[n-2], gamma[n-3].
        if n >= 3:
            d_nm1 = y_panel_centers[n - 1] - y_panel_centers[n - 2]
            d_nm2 = y_panel_centers[n - 2] - y_panel_centers[n - 3] if n > 3 else d_nm1
            # Example coefficient set from eq. (22):
            a_pp = 2.0 / (d_nm2 * d_nm1)
            b_pp = -((d_nm1 + d_nm2) / (d_nm2 * d_nm1))
            c_pp = 2.0 / ((d_nm2 + d_nm1) * d_nm2)
            #
            # or whichever your reference states exactly.  Insert below:
            D2[n - 1, n - 1] = a_pp
            D2[n - 1, n - 2] = b_pp
            if n >= 3:
                D2[n - 1, n - 3] = c_pp

        # Now multiply:
        d2_gamma = D2.dot(gamma)

        # ----------------------------------------------------------------------
        # 2) Compute mu_tilde[i] from eq. (25).
        #
        #   mu_tilde_i =   max(0, J_{ii}^b)   for i in the interior
        #                or min(0, J_{11}^f)  at i=0
        #                or min(0, J_{nn}^{a''}) at i=n-1
        #
        # The exact exponents (b, f, a'') come from the boundary conditions in your notes.
        # Commonly we check the sign of J_{ii}.  If J_{ii}<0, we turn on viscosity.
        # ----------------------------------------------------------------------
        mu_tilde = np.zeros(n)

        for i in range(n):
            if i == 0:
                # boundary
                # eq. (25) says mu_tilde_1 = min(0, J_{11}^f).  Typically that means:
                # if J_{00} (Python) is negative => mu_tilde[0] = J_{00}, else 0
                mu_tilde[i] = min(0.0, Jdiag[i])
            elif i == n - 1:
                # other boundary
                mu_tilde[i] = min(0.0, Jdiag[i])
            else:
                # interior
                # eq. (25) says mu_tilde_i = max(0, J_{ii}^b).  Interpreted as
                # if J_{ii}<0 => we want a positive mu_tilde, so we might use mu_tilde = - J_{ii}.
                # or simpler: mu_tilde = max(0, -J_{ii}).
                # (That “-” depends on your sign convention.  Adjust if needed.)
                mu_tilde[i] = max(0.0, -Jdiag[i])

        # ----------------------------------------------------------------------
        # 3) Scale mu_tilde to get mu by eq. (26):
        #       mu = f_va * ( mu_tilde / ( mu_tilde · b ) ) * b
        # i.e. each mu[i] = f_va * mu_tilde[i] * b[i] / ( sum_j mu_tilde[j]*b[j] ).
        # ----------------------------------------------------------------------
        mu = np.zeros(n)
        dot_tilde_b = np.sum(mu_tilde * b_array)
        if abs(dot_tilde_b) > 1e-14:
            scale = fva / dot_tilde_b
            mu = scale * mu_tilde * b_array  # elementwise

        # ----------------------------------------------------------------------
        # 4) Form the final viscosity-correction array:
        #       viscosity_correction[i] = - mu[i] * (d^2 Gamma / d y_s^2)[i]
        # which you can then add to your gamma iteration as in eq. (19).
        # ----------------------------------------------------------------------
        viscosity_correction = -mu * d2_gamma

        # ----------------------------------------------------------------------
        # (Optional) 5) You can also compute the artificial viscosity "deviation"
        #     P = 100 x (1 / b) sum_{i=1..n} [ mu_i ( d^2 Γ / dy^2 )_i / ( (1/2)*V_proj_i*c_i*Cl(α_eff_i ) ) ] * b_i
        # from eq. (27).  You would need c_i, Cl_i, V_proj_i, etc.  Not shown here.
        # ----------------------------------------------------------------------
        return viscosity_correction, mu

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
