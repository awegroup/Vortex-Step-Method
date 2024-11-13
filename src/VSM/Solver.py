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
        tol_reference_error (float): Tolerance for reference error (default: 0.001)
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
        max_iterations: int = 1500,
        allowed_error: float = 1e-12,  # 1e-5,
        tol_reference_error: float = 0.001,
        relaxation_factor: float = 0.03,
        is_with_artificial_damping: bool = False,
        artificial_damping: dict = {"k2": 0.1, "k4": 0.0},
        type_initial_gamma_distribution: str = "elliptic",
        is_with_gamma_feedback: bool = True,
        core_radius_fraction: float = 1e-20,
        mu: float = 1.81e-5,
        is_only_f_and_gamma_output: bool = False,
        is_new_vector_definition: bool = True,
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
        self.tol_reference_error = tol_reference_error
        self.relaxation_factor = relaxation_factor
        self.is_with_artificial_damping = is_with_artificial_damping
        self.artificial_damping = artificial_damping
        self.type_initial_gamma_distribution = type_initial_gamma_distribution
        self.core_radius_fraction = core_radius_fraction
        self.mu = mu
        self.is_only_f_and_gamma_output = is_only_f_and_gamma_output
        self.is_with_gamma_feedback = is_with_gamma_feedback
        self.is_new_vector_definition = is_new_vector_definition

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
            is_new_vector_definition=self.is_new_vector_definition,
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

            if self.is_with_artificial_damping:
                damp, is_damping_applied = self.smooth_circulation(
                    circulation=gamma, smoothness_factor=0.1, damping_factor=0.5
                )
                logging.debug("damp: %s", damp)
            else:
                damp = 0
                is_damping_applied = False

            gamma_new = (
                (1 - relaxation_factor) * gamma + relaxation_factor * gamma_new + damp
            )

            # TODO: could add a dynamic relaxation factor here, although first tries failed, so not super easy

            # Checking Convergence
            reference_error = np.amax(np.abs(gamma_new))
            reference_error = max(reference_error, self.tol_reference_error)
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
