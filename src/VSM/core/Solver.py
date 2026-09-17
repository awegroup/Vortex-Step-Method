import hashlib
import numpy as np
import logging
from scipy.linalg import solve_banded
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
        is_with_viscous_drag_correction (bool): Add the spanwise-flow viscous
            drag and spanwise friction force of Gaunaa, Sorensen & Li (2024).
        is_aoa_corrected (bool): Take the force directions from the flow at the
            quarter chord (Gaunaa, Li & Pirrung 2026, TAT3). Default False keeps
            the 3/4-chord directions (the LL-3/4 implementation of that paper).
        is_with_attached_trailed_vortex_force (bool): Add the Kutta-Joukowski
            force on the chordwise (attached trailed) vortex segments between the
            bound vortex and the trailing edge (Gaunaa, Li & Pirrung, TORQUE
            2026, Sec. 3). Default True; it matters for swept wings only.
        reference_point (np.ndarray): Reference point for moment calculations.
        mu (float): Dynamic viscosity of fluid.
        rho (float): Fluid density.
        is_with_artificial_viscosity (bool): Enable Li/Gaunaa spanwise artificial
            viscosity (TORQUE 2026) for post-stall stabilization in gamma_loop.
        artificial_viscosity_factor (float): Coefficient k in the viscosity scaling
            (default 0.035, the conservative envelope from the paper).
        newton_max_iterations (int): Iteration budget of one ``casadi_newton``
            attempt (Newton + pseudo-transient phases together).
        newton_pseudo_time_step (float): Pseudo time step the ``casadi_newton``
            loop restarts from, and never drops below on accepted steps, once
            its Newton line search stalls (0.03; base's explicit step is
            ``relaxation_factor``).
        newton_fallback_to_base (bool): Run the base relaxed-Picard loop when a
            ``casadi_newton`` attempt fails (default True).
        last_fallback (bool): Diagnostic -- did the last ``casadi_newton`` solve
            fall back to base?
        last_newton_evaluations (int): Diagnostic -- residual evaluations used
            by the last ``casadi_newton`` solve.
    """

    def __init__(
        self,
        aerodynamic_model_type: str = "VSM",
        max_iterations: int = 5000,
        allowed_error: float = 1e-6,
        relaxation_factor: float = 0.01,
        core_radius_fraction: float = 0.05,  # Following Damiani et al. (2019) https://docs.nrel.gov/docs/fy19osti/72777.pdf
        gamma_loop_type: str = "base",
        gamma_initial_distribution_type: str = "zero",
        is_only_f_and_gamma_output: bool = False,
        is_with_viscous_drag_correction: bool = False,
        reference_point: np.ndarray | list | tuple | None = None,
        mu: float = 1.81e-5,
        rho: float = 1.225,
        is_aoa_corrected: bool = False,
        is_with_attached_trailed_vortex_force: bool = True,
        is_with_artificial_viscosity: bool = False,
        artificial_viscosity_factor: float = 0.035,
        anderson_depth: int = 5,
        anderson_beta: float = 1.0,
        anderson_max_iterations: int = 1000,
        anderson_fallback_to_base: bool = False,
        stagnation_patience: int = 0,
        stagnation_rtol: float = 0.05,
        newton_max_iterations: int = 200,
        newton_fallback_to_base: bool = True,
        newton_pseudo_time_step: float = 0.03,
        polar_interpolation: str = "linear",
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
            is_aoa_corrected (bool): Quarter-chord force directions (TAT3).
            is_with_attached_trailed_vortex_force (bool): Include the force on
                the chordwise attached trailed vortex segments.
            reference_point (array-like, optional): Reference point for moments.
                Must be shape (3,). Defaults to [0, 0, 0].
            mu (float): Dynamic viscosity.
            rho (float): Fluid density.
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
        self.reference_point = self._check_and_force_shape(reference_point)
        self.is_aoa_corrected = is_aoa_corrected
        self.is_with_attached_trailed_vortex_force = (
            is_with_attached_trailed_vortex_force
        )
        # === athmospheric properties ===
        self.mu = mu
        self.rho = rho
        # ===============================
        #       STALL MODEL
        # ===============================
        # === Li/Gaunaa spanwise artificial viscosity (TORQUE 2026) ===
        # Parameter-free post-stall regularization; see gamma_loop.
        self.is_with_artificial_viscosity = is_with_artificial_viscosity
        self.artificial_viscosity_factor = artificial_viscosity_factor
        # === Anderson-accelerated fixed-point loop (gamma_loop_type="anderson") ===
        # Depth m = number of past residuals mixed per step; beta = mixing/damping.
        # anderson_max_iterations bounds the accelerated attempt before solve()
        # falls back to the base relaxed-Picard loop (deep post-stall / stall-knee
        # safety net). Healthy Anderson converges in O(10s) of iterations, so a
        # small cap keeps the wasted work minimal on the rare limit-cycling state
        # (e.g. the stall knee) before the base loop takes over.
        self.anderson_depth = int(anderson_depth)
        self.anderson_beta = float(anderson_beta)
        self.anderson_max_iterations = int(anderson_max_iterations)
        # Whether a non-converged Anderson attempt retries with the base
        # relaxed-Picard loop. OFF by default since 2026-09-03, with the
        # iteration headroom raised to 1000 instead: measured on the AWETrim
        # 2019+2025 steering campaigns, the fallback rescued 99 of ~92,400
        # Anderson failures (0.1%) while costing up to two 1500-iteration
        # base loops per failure. Callers that want the old always-fall-back
        # robustness pass True (and may lower anderson_max_iterations).
        self.anderson_fallback_to_base = bool(anderson_fallback_to_base)

        # Give up on a circulation solve that has stopped improving, rather
        # than grinding out ``max_iterations``. OFF by default (patience 0):
        # a plain solve should keep its full budget, since a slow solve and a
        # hopeless one are only distinguishable by PROGRESS, never by an
        # iteration count -- shrinking a cap to bound the hopeless case kills
        # the slow-but-converging one too.
        #
        # The caller that wants this is a two-stage scheme whose first stage is
        # a PREDICTOR it may throw away (AWETrim's attached-branch finder): on a
        # genuinely stalled state that predictor exhausts the cap and is then
        # rejected regardless, so the whole budget is waste. ``patience``
        # iterations with no improvement better than ``rtol`` ends it.
        self.stagnation_patience = int(stagnation_patience)
        self.stagnation_rtol = float(stagnation_rtol)
        #: Diagnostic: did the last circulation solve stop on stagnation?
        self.last_stagnated = False
        # === CasADi Newton / pseudo-transient loop (gamma_loop_type="casadi_newton") ===
        # Solves the circulation residual R(gamma) = 0 with an EXACT Jacobian
        # from CasADi automatic differentiation instead of iterating the
        # fixed-point map. Quadratic convergence in attached flow: O(3-6)
        # iterations where the relaxed-Picard loop needs O(1000). When the
        # Newton line search stalls (the piecewise-linear polars make the
        # residual kinked around the stall knee, and |R|^2 then has minima that
        # are not roots) the loop switches to pseudo-transient continuation:
        # implicit Euler on the SAME flow the base loop time-steps explicitly,
        # with the pseudo time step grown by switched evolution relaxation back
        # to pure Newton as the residual falls. ``newton_pseudo_time_step`` is
        # the step it restarts from (base's explicit step is
        # ``relaxation_factor``; the implicit scheme is A-stable so ~100x that
        # is safe). ``newton_max_iterations`` bounds one attempt; a failed
        # attempt falls back to the base relaxed-Picard loop when
        # ``newton_fallback_to_base`` is set -- cheap to keep ON, since a Newton
        # failure is detected within a fraction of one base solve. Requires the
        # optional ``casadi`` dependency.
        self.newton_max_iterations = int(newton_max_iterations)
        self.newton_fallback_to_base = bool(newton_fallback_to_base)
        self.newton_pseudo_time_step = float(newton_pseudo_time_step)
        # Polar lookup used by the CasADi panel function (casadi_newton and the
        # AWETrim CasADi trim): "linear" reproduces np.interp (the numpy loops'
        # tables, byte-identical fixed points); "bspline" is a C2 cubic spline
        # through the same nodes -- smooth Jacobian, no corner chattering, but
        # a slightly different Cl between nodes, so a different fixed point.
        if polar_interpolation not in ("linear", "bspline"):
            raise ValueError("polar_interpolation must be 'linear' or 'bspline'.")
        self.polar_interpolation = polar_interpolation
        # A pseudo-transient step whose residual grows by more than 1/this is
        # rejected and the step quartered (0.5 = residual may at most double).
        self._newton_reject_ratio = 0.5
        self._casadi_newton_cache: dict = {}
        #: Diagnostic: did the last casadi_newton solve fall back to base?
        self.last_fallback = False
        #: Diagnostic: residual evaluations used by the last casadi_newton solve.
        self.last_newton_evaluations = 0

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

    @staticmethod
    def _check_and_force_shape(
        reference_point: np.ndarray | list | tuple | None,
    ) -> np.ndarray:
        """Return reference_point as a float array with shape (3,)."""
        rp = (
            np.zeros(3, dtype=float)
            if reference_point is None
            else np.asarray(reference_point, dtype=float)
        )
        if rp.shape != (3,):
            raise ValueError(f"reference_point must be shape (3,), got {rp.shape}")
        return rp

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

        if gamma_distribution is not None:
            gamma_initial = np.asarray(gamma_distribution, dtype=float)
            if gamma_initial.shape != (self.n_panels,):
                raise ValueError(
                    "gamma_distribution must match number of panels in solve()."
                )
        elif self.gamma_initial_distribution_type == "previous":
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

        elif self.gamma_loop_type == "casadi_newton":
            converged, gamma_new, alpha_array, Umag_array = (
                self.gamma_loop_casadi_newton(gamma_initial)
            )
            self.last_fallback = False
            if not converged and self.newton_fallback_to_base:
                logging.info(
                    " ---> casadi_newton did not converge; falling back to the "
                    "base relaxed-Picard loop"
                )
                self.last_fallback = True
                converged, gamma_new, alpha_array, Umag_array = self.gamma_loop(
                    gamma_initial
                )
                if not converged:
                    converged, gamma_new, alpha_array, Umag_array = self.gamma_loop(
                        gamma_initial, extra_relaxation_factor=0.5
                    )

        elif self.gamma_loop_type == "anderson":
            converged, gamma_new, alpha_array, Umag_array = self.gamma_loop_anderson(
                gamma_initial
            )
            # Deep post-stall can trap Anderson in a limit cycle (the regime where
            # VSM is unreliable anyway and only the base loop's viscosity /
            # heavy relaxation converges). Fall back to the base loop — same
            # fixed point, same two-stage half-relaxation retry — so the
            # accelerated path is never less robust than ``base``. Optional
            # (anderson_fallback_to_base): measured rescue rate 0.1%.
            if not converged and self.anderson_fallback_to_base:
                logging.info(
                    " ---> Anderson did not converge; falling back to base "
                    "relaxed-Picard loop"
                )
                converged, gamma_new, alpha_array, Umag_array = self.gamma_loop(
                    gamma_initial
                )
                if not converged:
                    converged, gamma_new, alpha_array, Umag_array = self.gamma_loop(
                        gamma_initial, extra_relaxation_factor=0.5
                    )

        else:
            raise ValueError(
                f"Invalid gamma_loop_type {self.gamma_loop_type!r}; expected "
                "'base', 'non_linear', 'casadi_newton' or 'anderson'."
            )
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
            relative_velocity_array=self.compute_relative_velocity(gamma_new),
            is_with_attached_trailed_vortex_force=self.is_with_attached_trailed_vortex_force,
        )
        results["gamma_converged"] = bool(converged)
        return results

    def compute_aerodynamic_quantities(self, gamma: np.ndarray) -> tuple:
        """Compute aerodynamic quantities from circulation distribution.

        Args:
            gamma (np.ndarray): Circulation distribution (n x 1).

        Returns:
            tuple: (alpha_array, Umag_array, cl_array)
                - alpha_array (np.ndarray): Effective angles of attack.
                - Umag_array (np.ndarray): Span-perpendicular relative speed
                  |v_eff x z_airf| (the inner 2D speed, Crossflow Principle).
                - cl_array (np.ndarray): Lift coefficients.
        """
        relative_velocity_array = self.compute_relative_velocity(gamma)
        relative_velocity_crossz_array = jit_cross(
            relative_velocity_array, self.z_airf_array
        )  # v_eff x z
        v_normal_array = np.sum(self.x_airf_array * relative_velocity_array, axis=1)
        v_tangential_array = np.sum(self.y_airf_array * relative_velocity_array, axis=1)
        alpha_array = np.arctan2(v_normal_array, v_tangential_array)  # alpha_eff
        Umag_array = np.linalg.norm(
            relative_velocity_crossz_array, axis=1
        )  # |v_eff x z|
        cl_array = np.array(
            [panel.compute_cl(alpha) for panel, alpha in zip(self.panels, alpha_array)]
        )  # cl(alpha_eff)
        return alpha_array, Umag_array, cl_array

    def compute_relative_velocity(self, gamma: np.ndarray) -> np.ndarray:
        """Full 3D relative velocity at every control point, ``va + AIC gamma``
        (the section's own 2D bound-vortex induction already removed by the
        VSM AIC). Shape (n_panels, 3)."""
        induced_velocity_all = np.array(
            [
                np.matmul(self.AIC_x, gamma),
                np.matmul(self.AIC_y, gamma),
                np.matmul(self.AIC_z, gamma),
            ]
        ).T  # v_ind
        return self.va_array + induced_velocity_all

    def _build_spanwise_laplacian(self) -> np.ndarray:
        """Discrete spanwise Laplacian ``L`` with second-order tip closures.

        Interior rows use the standard three-point stencil
        ``(L gamma)_i = gamma_{i-1} - 2 gamma_i + gamma_{i+1}``. The tip rows use
        the closures of Li, Gaunaa, Pirrung & Lønbæk (TORQUE 2026, Eq. 15),
        derived from a quadratic variation of circulation near the tip, which
        enforce ``gamma -> 0`` at the wing tips to second order:
        ``(L gamma)_0 = -4 gamma_0 + (4/3) gamma_1`` and
        ``(L gamma)_{N-1} = (4/3) gamma_{N-2} - 4 gamma_{N-1}``.

        Panels are assumed to be ordered consecutively along the span (the
        standard VSM panel ordering) and approximately uniformly spaced. The
        per-panel viscosity coefficient carries the ``1/dz_i^2`` spacing factor.
        """
        n = self.n_panels
        laplacian = np.zeros((n, n))
        if n < 3:
            return laplacian
        for i in range(1, n - 1):
            laplacian[i, i - 1] = 1.0
            laplacian[i, i] = -2.0
            laplacian[i, i + 1] = 1.0
        laplacian[0, 0] = -4.0
        laplacian[0, 1] = 4.0 / 3.0
        laplacian[n - 1, n - 1] = -4.0
        laplacian[n - 1, n - 2] = 4.0 / 3.0
        return laplacian

    def _local_lift_slope(
        self, alpha_array: np.ndarray, delta: float = np.deg2rad(0.5)
    ) -> np.ndarray:
        """Local lift-curve slope ``dCl/dalpha`` per panel via central differences.

        Evaluated from each panel's own 2-D polar at the current effective angle
        of attack. The slope is negative in post-stall, which is what activates
        the artificial-viscosity regularization in :meth:`gamma_loop`.

        Readable reference implementation and test oracle; the iteration hot
        path uses the vectorized :meth:`_lift_slope_from_ctx`, which evaluates
        the identical central difference from tables prepared once per solve.
        """
        slopes = np.empty(self.n_panels)
        for i, (panel, alpha) in enumerate(zip(self.panels, alpha_array)):
            cl_plus = panel.compute_cl(alpha + delta)
            cl_minus = panel.compute_cl(alpha - delta)
            slopes[i] = (cl_plus - cl_minus) / (2.0 * delta)
        return slopes

    def _panel_stall_angles(self) -> np.ndarray:
        """Per-panel stall-onset AoA [rad]: the first local Cl maximum in the
        positive-Cl region of each panel polar (``inf`` if the polar shows no
        peak).

        Used only as a cheap, geometry-fixed gate for the post-stall
        artificial-viscosity branch: the regularization is a no-op while every
        panel is below its stall onset, so this lets ``gamma_loop`` skip both the
        lift-slope evaluation and the linear solve in attached conditions.
        """
        angles = np.full(self.n_panels, np.inf)
        for i, panel in enumerate(self.panels):
            polar = np.asarray(panel.panel_polar_data, dtype=float)
            alpha, cl = polar[:, 0], polar[:, 1]
            pos = np.where(cl > 0)[0]
            for k in pos[1:-1]:  # first interior Cl peak in the positive-Cl region
                if cl[k] > cl[k - 1] and cl[k] > cl[k + 1]:
                    angles[i] = float(alpha[k])
                    break
        return angles

    def _panel_negative_stall_angles(self) -> np.ndarray:
        """Per-panel negative-stall onset AoA [rad]: the local Cl minimum in
        the negative-Cl region closest to alpha = 0 (``-inf`` if none). Below
        it the lift slope is negative again, so Eq. 16 of Li et al. (2026)
        applies there too; this is the mirror gate of
        :meth:`_panel_stall_angles`.
        """
        angles = np.full(self.n_panels, -np.inf)
        for i, panel in enumerate(self.panels):
            polar = np.asarray(panel.panel_polar_data, dtype=float)
            alpha, cl = polar[:, 0], polar[:, 1]
            neg = np.where(cl < 0)[0]
            for k in neg[1:-1][::-1]:  # interior Cl trough nearest alpha = 0
                if cl[k] < cl[k - 1] and cl[k] < cl[k + 1]:
                    angles[i] = float(alpha[k])
                    break
        return angles

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

        # Spanwise artificial-viscosity regularization (Li, Gaunaa, Pirrung &
        # Lønbæk, TORQUE 2026). Stabilizes post-stall (negative lift-slope)
        # circulation distributions that otherwise develop non-physical sawtooth
        # oscillations and never converge. The context (tridiagonal Laplacian
        # diagonals, polar slope tables, planform area, stall-onset gate) is
        # built once since geometry and polars are frozen during the iteration.
        viscosity_ctx = self._build_viscosity_ctx()
        use_viscosity = viscosity_ctx is not None

        relaxation = self.relaxation_factor * extra_relaxation_factor
        self.last_stagnated = False
        for i in range(self.max_iterations):
            gamma = gamma_new
            alpha_array, Umag_array, cl_array = self.compute_aerodynamic_quantities(
                gamma
            )
            # Kutta-Joukowski with the inner (span-perpendicular, induced)
            # velocity: Gamma = 0.5 |V_inner| c Cl (Gaunaa, Li & Pirrung
            # 2026, Eq. 4).
            gamma_target = 0.5 * Umag_array * cl_array * self.chord_array
            if use_viscosity:
                gamma_target = self._regularize_gamma_target(
                    gamma_target, alpha_array, viscosity_ctx
                )
            gamma_new = (1 - relaxation) * gamma + relaxation * gamma_target

            if not np.all(np.isfinite(gamma_new)):
                # A non-finite circulation never recovers: every later
                # iterate is NaN. Return the last finite one, not converged,
                # instead of grinding max_iterations on NaN and handing the
                # caller NaN forces (seen on a doubled-back lifting line,
                # WingGeometry._warn_if_sections_double_back).
                logging.warning(
                    "Circulation loop produced non-finite gamma at iteration "
                    "%s -- stopping (degenerate mesh or expansive map); "
                    "returning the last finite circulation, not converged.",
                    i,
                )
                gamma_new = gamma
                break

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

            if self._stagnated(error_history):
                logging.debug(
                    "Circulation loop stagnated at iteration %s "
                    "(no improvement in %s iterations); stopping.",
                    i,
                    self.stagnation_patience,
                )
                self.last_stagnated = True
                break

            # Simple oscillation detection and handling. Skipped when artificial
            # viscosity is active, since the regularization already suppresses the
            # sawtooth oscillations this heuristic targets.
            if not use_viscosity and i >= 5 and len(error_history) >= 3:
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
        self.last_iterations = i + 1  # diagnostic: iterations used this solve
        return converged, gamma_new, alpha_array, Umag_array

    def _stagnated(self, error_history: list) -> bool:
        """True when the normalized error has stopped improving.

        Compares the best error of the last ``stagnation_patience`` iterations
        against the best of everything before them: if the recent window has
        not beaten the earlier best by at least ``stagnation_rtol``, the
        iteration is not going anywhere. Uses running minima rather than the
        latest value so an oscillating-but-descending solve is not cut off.
        """
        patience = int(getattr(self, "stagnation_patience", 0) or 0)
        if patience <= 0 or len(error_history) <= patience:
            return False
        recent_best = min(error_history[-patience:])
        prior_best = min(error_history[:-patience])
        return recent_best > prior_best * (1.0 - float(self.stagnation_rtol))

    def _build_viscosity_ctx(self) -> dict | None:
        """Pre-build the frozen-geometry objects the post-stall regularization
        needs, or ``None`` when artificial viscosity is disabled. Shared by the
        base and accelerated loops so they target the same regularized fixed
        point.

        Contents: per-panel stall onset (the cheap gate), planform area, the
        three diagonals of the tridiagonal spanwise Laplacian (the dense matrix
        of :meth:`_build_spanwise_laplacian` is tridiagonal, so the implicit
        solve is done banded), and each panel's polar table for the vectorized
        lift-slope evaluation. When all panels share one alpha grid (the normal
        outcome of batch polar generation) the cl columns are stacked into a
        single matrix so the slope evaluation needs no per-panel Python loop.
        """
        if not self.is_with_artificial_viscosity:
            return None
        laplacian = self._build_spanwise_laplacian()
        alpha_tables = [
            np.asarray(panel.panel_polar_data, dtype=float)[:, 0]
            for panel in self.panels
        ]
        cl_tables = [
            np.asarray(panel.panel_polar_data, dtype=float)[:, 1]
            for panel in self.panels
        ]
        shared_grid = all(
            table.shape == alpha_tables[0].shape
            and np.array_equal(table, alpha_tables[0])
            for table in alpha_tables[1:]
        )
        return {
            "stall_angles": self._panel_stall_angles(),
            "stall_angles_neg": self._panel_negative_stall_angles(),
            "planform_area": float(np.sum(self.width_array * self.chord_array)),
            "L_diag": np.diag(laplacian).copy(),
            "L_super": np.diag(laplacian, 1).copy(),
            "L_sub": np.diag(laplacian, -1).copy(),
            "alpha_grid": alpha_tables[0] if shared_grid else None,
            "cl_matrix": np.vstack(cl_tables) if shared_grid else None,
            "alpha_tables": alpha_tables,
            "cl_tables": cl_tables,
        }

    @staticmethod
    def _interp_rows(
        query: np.ndarray, grid: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """Linear interpolation of ``values[i, :]`` at ``query[i]`` on a shared
        ``grid``, matching ``np.interp`` semantics (clamped at both grid ends).
        """
        idx = np.clip(np.searchsorted(grid, query), 1, grid.size - 1)
        x0 = grid[idx - 1]
        x1 = grid[idx]
        weight = np.clip((query - x0) / (x1 - x0), 0.0, 1.0)
        rows = np.arange(values.shape[0])
        y0 = values[rows, idx - 1]
        y1 = values[rows, idx]
        return y0 + weight * (y1 - y0)

    def _lift_slope_from_ctx(
        self,
        alpha_array: np.ndarray,
        viscosity_ctx: dict,
        delta: float = np.deg2rad(0.5),
    ) -> np.ndarray:
        """Vectorized equivalent of :meth:`_local_lift_slope`, evaluating the
        same central difference of each panel's piecewise-linear polar from the
        tables prepared in :meth:`_build_viscosity_ctx` (shared-grid fast path,
        per-panel fallback when panels carry different alpha grids).
        """
        grid = viscosity_ctx["alpha_grid"]
        if grid is not None and grid.size >= 2:
            cl_matrix = viscosity_ctx["cl_matrix"]
            cl_plus = self._interp_rows(alpha_array + delta, grid, cl_matrix)
            cl_minus = self._interp_rows(alpha_array - delta, grid, cl_matrix)
            return (cl_plus - cl_minus) / (2.0 * delta)
        slopes = np.empty(self.n_panels)
        for i, (alpha_table, cl_table) in enumerate(
            zip(viscosity_ctx["alpha_tables"], viscosity_ctx["cl_tables"])
        ):
            cl_plus = np.interp(alpha_array[i] + delta, alpha_table, cl_table)
            cl_minus = np.interp(alpha_array[i] - delta, alpha_table, cl_table)
            slopes[i] = (cl_plus - cl_minus) / (2.0 * delta)
        return slopes

    def _regularize_gamma_target(
        self,
        gamma_target: np.ndarray,
        alpha_array: np.ndarray,
        viscosity_ctx: dict | None,
    ) -> np.ndarray:
        """Apply the Li/Gaunaa implicit spanwise viscosity to the fixed-point
        target: solve ``(I - diag(mu) L) gamma = gamma_target``.

        Implicit fixed point (I - diag(mu) L) gamma = F(gamma): same steady
        solution as the explicit scheme but stable at relaxation factors of
        order one, whereas the explicit stable step shrinks like N^-2 in
        post-stall. The coefficient ``mu_i = max(0, -k S Cl'_i / dz_i^2)`` with
        k = 0.035 reduces to ``mu = max(0, -k N^2/AR Cl')`` for a uniformly
        spaced wing (Eq. 16).

        Returns ``gamma_target`` unchanged (same object, no solve) while no
        panel is past its stall onset or every ``mu`` is zero — the exact no-op
        that keeps attached-flow iterations as cheap as the unregularized loop.
        The system is tridiagonal, so the solve is banded, not dense.
        """
        if viscosity_ctx is None or not (
            np.any(alpha_array > viscosity_ctx["stall_angles"])
            or np.any(alpha_array < viscosity_ctx["stall_angles_neg"])
        ):
            return gamma_target
        lift_slope = self._lift_slope_from_ctx(alpha_array, viscosity_ctx)
        mu_array = np.maximum(
            0.0,
            -self.artificial_viscosity_factor
            * viscosity_ctx["planform_area"]
            * lift_slope
            / self.width_array**2,
        )
        if not np.any(mu_array > 0.0):
            return gamma_target
        n = gamma_target.size
        # Banded storage of (I - diag(mu) L): row i couples only i-1, i, i+1.
        ab = np.zeros((3, n))
        ab[1] = 1.0 - mu_array * viscosity_ctx["L_diag"]
        ab[0, 1:] = -mu_array[:-1] * viscosity_ctx["L_super"]
        ab[2, :-1] = -mu_array[1:] * viscosity_ctx["L_sub"]
        return solve_banded((1, 1), ab, gamma_target)

    def _fixed_point_target(
        self, gamma: np.ndarray, viscosity_ctx: dict | None = None
    ) -> tuple:
        """Single evaluation of the circulation fixed-point map ``G(gamma)``.

        Returns ``(gamma_target, alpha_array, Umag_array)``. The fixed point
        ``gamma*`` satisfies ``gamma* = G(gamma*)`` — the very quantity the base
        :meth:`gamma_loop` relaxes toward with ``gamma_new = (1-w) gamma + w
        G(gamma)``. Sharing this map lets the accelerated loops converge to the
        identical solution. When artificial viscosity is active and any panel is
        past stall, the post-stall regularization (Li, Gaunaa, Pirrung & Lønbæk,
        TORQUE 2026) is folded into the target so ``base`` and ``anderson`` share
        the same regularized fixed point.
        """
        alpha_array, Umag_array, cl_array = self.compute_aerodynamic_quantities(
            gamma
        )
        gamma_target = 0.5 * Umag_array * cl_array * self.chord_array
        gamma_target = self._regularize_gamma_target(
            gamma_target, alpha_array, viscosity_ctx
        )
        return gamma_target, alpha_array, Umag_array

    def gamma_loop_anderson(self, gamma_initial: np.ndarray) -> tuple:
        """Anderson-accelerated fixed-point iteration for the circulation.

        Anderson acceleration is applied to the *under-relaxed* Picard map

            g(gamma) = (1 - w) gamma + w G(gamma),   w = relaxation_factor,

        not to the raw ``G(gamma)``: the raw circulation map is expansive here
        (hence the base loop under-relaxes heavily), and accelerating it
        directly diverges. The relaxed map is a contraction with the *same*
        fixed point, and each step mixes the last ``anderson_depth`` relaxed
        residuals through a small (``m x m``) least-squares problem (Walker & Ni
        2011, SIAM J. Numer. Anal. 49, 1715). This decouples ``w`` from the
        convergence rate — a conservative, robust ``w`` still converges in
        O(10s) of iterations instead of O(100s), removing the fragile speed/
        stability trade-off of the bare relaxation factor. The stopping rule and
        optional post-stall regularization are identical to ``base``, so the
        returned circulation matches it to solver tolerance.

        .. warning::
            Anderson terminates on a *superlinear* (jumpy) residual, so near the
            tolerance boundary a tiny change in the inflow can flip the returned
            circulation by ~one convergence jump (e.g. from a 1e-3 to a 1e-8
            residual). The converged gamma is therefore a slightly *non-smooth*
            function of the inflow. This is invisible for a standalone solve, but
            it corrupts any *outer* finite-difference Jacobian that differentiates
            through this loop (e.g. the AWETrim quasi-steady trim solvers) unless
            ``allowed_error`` is tight (~1e-8), which pushes the jump below the FD
            step. The base loop's slow *linear* convergence keeps its
            loosely-converged gamma smooth, so ``base`` is the safe choice for
            FD-outer-loop use at loose tolerance.

        Args:
            gamma_initial (np.ndarray): Initial circulation distribution.

        Returns:
            tuple: ``(converged, gamma_new, alpha_array, Umag_array)`` matching
            :meth:`gamma_loop`.
        """
        m = max(1, int(self.anderson_depth))
        beta = float(self.anderson_beta)
        w = self.relaxation_factor
        # Relative Tikhonov regularization of the depth-m least-squares problem:
        # damps the extrapolation when the residual-difference columns are
        # near-linearly-dependent, biasing toward the safe relaxed-Picard step
        # rather than an over-large quasi-Newton stride.
        reg = 1e-10
        # Anderson converges superlinearly here (O(10s) of iterations); if it has
        # not converged within this budget it is in a limit cycle (deep post-stall,
        # where VSM itself is unreliable and only the base loop's viscosity /
        # relaxation tames it). The caller (:meth:`solve`) then falls back to the
        # base relaxed-Picard loop, so this just bounds the wasted work.
        max_it = min(self.max_iterations, int(self.anderson_max_iterations))
        viscosity_ctx = self._build_viscosity_ctx()

        def relaxed_step(x):
            # One evaluation of the relaxed fixed-point map g(x) and its residual
            # f(x) = g(x) - x = w (G(x) - x). Same fixed point as G, contractive.
            target, alpha, umag = self._fixed_point_target(x, viscosity_ctx)
            g = (1.0 - w) * x + w * target
            return g, g - x, alpha, umag

        x = np.array(gamma_initial, dtype=float)
        g, f, alpha_array, Umag_array = relaxed_step(x)

        x_hist: list[np.ndarray] = []  # window of iterates (current one included)
        f_hist: list[np.ndarray] = []  # window of relaxed residuals g(x)-x
        converged = False
        last_k = 0
        error_history: list[float] = []
        self.last_stagnated = False

        for k in range(max_it):
            last_k = k
            # Same normalized-error measure as the base loop: |g - gamma| over
            # the peak circulation (g is the relaxed update, matching base's
            # ``max|gamma_new - gamma| / max|gamma_new|``).
            reference_error = np.amax(np.abs(g))
            reference_error = reference_error if reference_error != 0 else 1e-4
            normalized_error = np.amax(np.abs(f)) / reference_error
            if normalized_error < self.allowed_error:
                converged = True
                break
            logging.debug(
                f"Anderson normalized error at iteration {k}: {normalized_error}"
            )

            error_history.append(normalized_error)
            if self._stagnated(error_history):
                logging.debug(
                    "Anderson loop stagnated at iteration %s "
                    "(no improvement in %s iterations); stopping.",
                    k,
                    self.stagnation_patience,
                )
                self.last_stagnated = True
                break

            x_hist.append(x)
            f_hist.append(f)
            if len(f_hist) > m + 1:
                x_hist.pop(0)
                f_hist.pop(0)

            mk = len(f_hist) - 1
            if mk == 0:
                # No history yet: a single (damped) relaxed Picard step to seed.
                x_new = x + beta * f
            else:
                dF = np.stack(
                    [f_hist[j] - f_hist[j - 1] for j in range(1, len(f_hist))], axis=1
                )  # (n_panels, mk)
                dX = np.stack(
                    [x_hist[j] - x_hist[j - 1] for j in range(1, len(x_hist))], axis=1
                )  # (n_panels, mk)
                gram = dF.T @ dF
                lam = reg * float(np.trace(gram)) / dF.shape[1]
                theta = np.linalg.solve(gram + lam * np.eye(dF.shape[1]), dF.T @ f)
                x_new = x + beta * f - (dX + beta * dF) @ theta

            x = x_new
            g, f, alpha_array, Umag_array = relaxed_step(x)
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(g))):
                # Same guard as the base loop: fall back to the last finite
                # iterate and let the caller's base-loop fallback / failure
                # handling take over.
                logging.warning(
                    "Anderson circulation loop produced non-finite gamma at "
                    "iteration %s -- stopping on the last finite iterate.",
                    k,
                )
                x = x_hist[-1] if x_hist else np.array(gamma_initial, dtype=float)
                g, f, alpha_array, Umag_array = relaxed_step(x)
                break

        self.last_iterations = last_k + 1  # diagnostic: iterations used
        if not converged:
            logging.info(
                f"Anderson did not converge in {max_it} iterations "
                "(deep post-stall limit cycle); caller falls back to base loop."
            )
        return converged, x, alpha_array, Umag_array

    # ------------------------------------------------------------------
    # CasADi damped-Newton circulation solve
    # ------------------------------------------------------------------
    def _polar_tables(self) -> tuple[list, list, list, list]:
        """Per-panel ``(alpha, cl, cd, cm)`` columns of ``panel_polar_data``."""
        tables = [np.asarray(panel.panel_polar_data, dtype=float) for panel in self.panels]
        return (
            [t[:, 0] for t in tables],
            [t[:, 1] for t in tables],
            [t[:, 2] for t in tables],
            [t[:, 3] for t in tables],
        )

    def _casadi_newton_key(self, alpha_tables, cl_tables, cd_tables, cm_tables) -> tuple:
        """Cache key: the symbolic residual depends only on the panel count,
        the polar tables and the regularization settings. Geometry, inflow and
        AIC matrices enter as numeric parameters, so one compiled function
        serves every inflow and every deformed shape that keeps its polars."""
        digest = hashlib.blake2b(digest_size=16)
        for tables in zip(alpha_tables, cl_tables, cd_tables, cm_tables):
            for table in tables:
                digest.update(np.ascontiguousarray(table).tobytes())
        return (
            self.n_panels,
            bool(self.is_with_artificial_viscosity),
            float(self.artificial_viscosity_factor),
            self.polar_interpolation,
            digest.hexdigest(),
        )

    def _build_casadi_newton_function(self, alpha_tables, cl_tables, cd_tables=None, cm_tables=None):
        """Build the CasADi function of the PER-PANEL section physics that
        :meth:`gamma_loop_casadi_newton` assembles its residual and exact
        Jacobian from.

        The circulation residual whose root the Newton loop finds is

            R(gamma) = (I - diag(mu(alpha)) L) gamma - G_raw(gamma),

        exactly the fixed point the ``base`` and ``anderson`` loops iterate
        towards (``gamma = (I - diag(mu) L)^-1 G_raw(gamma)``, see
        :meth:`_regularize_gamma_target`), including the Li/Gaunaa viscosity
        ``mu_i = max(0, -k S Cl'_i / dz_i^2)`` gated on any panel being past
        its stall onset (``mu = 0`` without artificial viscosity). Everything
        that is not the dense induction ``v_ind = AIC gamma`` is local to a
        panel: ``G_raw_i``, ``mu_i`` and ``alpha_i`` depend on ``gamma`` only
        through the panel's own relative velocity ``v_rel_i``. So the symbolic
        function takes ``v_rel`` (and ``L gamma``) as INPUTS and returns

            h_i = G_raw_i + mu_i (L gamma)_i,   c_i = dh_i / dv_rel_i  (3 values)

        plus ``mu``, ``alpha`` and ``|v_rel x z|``; the Newton loop then forms
        ``R = gamma - h`` and, by the chain rule through ``v_rel = va + AIC
        gamma`` (with ``dh_i/d(L gamma)_i = mu_i``),

            J = I - diag(mu) L - sum_k diag(c_k) AIC_k

        in NumPy. This keeps the CasADi graph O(n) -- independent of the
        n x n induction matrices -- so it costs ~0.1 ms per evaluation and
        builds in tens of milliseconds, while the dense algebra runs in BLAS.

        The 2-D polars are ``ca.interpolant`` tables. Queries are clamped to
        the table range first, so the lookup reproduces ``np.interp`` (constant
        beyond both ends) and the lift slope is the same central difference of
        the same piecewise-linear table that :meth:`_lift_slope_from_ctx`
        evaluates. When every panel shares one alpha grid (the normal outcome
        of batch polar generation) the lookup is a single 2-D interpolant over
        (alpha, panel index) evaluated exactly at the integer panel nodes;
        otherwise one 1-D interpolant per panel. ``polar_interpolation =
        "bspline"`` swaps the piecewise-linear table for a cubic spline
        through the same nodes (degree 1 along the panel-index axis, so panel
        rows never mix).

        Outputs, in order: ``h``, ``dh/dv_rel`` (n x 3, from the corner-
        averaged polar), ``mu``, ``alpha``, ``umag``, ``gate``, then for
        callers that build FORCES from the same lookups: ``h_avg`` (h with the
        corner-averaged Cl -- differentiate this for a kink-tolerant
        Jacobian), ``cl``, ``cd``, ``cm`` at ``alpha`` and their corner-
        averaged counterparts ``cl_avg``, ``cd_avg``, ``cm_avg``. With
        "bspline" polars the averaged outputs equal the exact ones.
        """
        import casadi as ca

        n = self.n_panels
        for i, alpha_table in enumerate(alpha_tables):
            if alpha_table.size < 2 or np.any(np.diff(alpha_table) <= 0.0):
                raise ValueError(
                    "casadi_newton needs a strictly increasing polar alpha grid "
                    f"(panel {i} violates this)."
                )

        v_rel = ca.MX.sym("v_rel", n, 3)
        x_airf = ca.MX.sym("x_airf", n, 3)
        y_airf = ca.MX.sym("y_airf", n, 3)
        z_airf = ca.MX.sym("z_airf", n, 3)
        chord = ca.MX.sym("chord", n)
        width = ca.MX.sym("width", n)
        stall_angles = ca.MX.sym("stall_angles", n)
        stall_angles_neg = ca.MX.sym("stall_angles_neg", n)
        l_gamma = ca.MX.sym("l_gamma", n)  # (L gamma), a parameter here

        shared_grid = all(
            table.shape == alpha_tables[0].shape
            and np.array_equal(table, alpha_tables[0])
            for table in alpha_tables[1:]
        )
        if cd_tables is None:
            cd_tables = [np.zeros_like(t) for t in alpha_tables]
        if cm_tables is None:
            cm_tables = [np.zeros_like(t) for t in alpha_tables]
        kind = self.polar_interpolation
        smooth = kind == "bspline"

        def make_lookup(name, value_tables):
            """Clamped lookup ``alpha_vec (n) -> values (n)`` of one polar column."""
            if shared_grid:
                grid = alpha_tables[0]
                matrix = np.vstack(value_tables)  # (n, m)
                # Flattened with the alpha axis varying fastest (CasADi convention).
                opts = {"degree": [3, 1]} if smooth else {}
                table = ca.interpolant(
                    f"{name}_2d",
                    kind,
                    [grid, np.arange(n, dtype=float)],
                    matrix.ravel(order="C"),
                    opts,
                )
                index_row = ca.DM(np.arange(n, dtype=float)).T
                lo, hi = float(grid[0]), float(grid[-1])

                def lookup(alpha_vec):
                    query = ca.fmin(ca.fmax(alpha_vec, lo), hi)
                    return table(ca.vertcat(query.T, index_row)).T

                return lookup
            interpolants = [
                ca.interpolant(f"{name}_{i}", kind, [alpha_table], value_table)
                for i, (alpha_table, value_table) in enumerate(
                    zip(alpha_tables, value_tables)
                )
            ]
            limits = [(float(t[0]), float(t[-1])) for t in alpha_tables]

            def lookup(alpha_vec):
                return ca.vertcat(
                    *[
                        interpolants[i](ca.fmin(ca.fmax(alpha_vec[i], lo), hi))
                        for i, (lo, hi) in enumerate(limits)
                    ]
                )

            return lookup

        cl_of = make_lookup("cl", cl_tables)
        cd_of = make_lookup("cd", cd_tables)
        cm_of = make_lookup("cm", cm_tables)

        def cross_rows(a, b):
            return ca.horzcat(
                a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
                a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
            )

        def row_norm(a):
            return ca.sqrt(ca.sum2(a * a))

        # Same algebra as compute_aerodynamic_quantities, in symbols.
        v_normal = ca.sum2(x_airf * v_rel)
        v_tangential = ca.sum2(y_airf * v_rel)
        alpha = ca.atan2(v_normal, v_tangential)
        umag = row_norm(cross_rows(v_rel, z_airf))
        cl = cl_of(alpha)
        gamma_raw = 0.5 * umag * cl * chord  # Gaunaa, Li & Pirrung 2026, Eq. 4

        if self.is_with_artificial_viscosity:
            delta = np.deg2rad(0.5)
            lift_slope = (cl_of(alpha + delta) - cl_of(alpha - delta)) / (2.0 * delta)
            planform_area = ca.sum1(width * chord)
            # The gate is the ONE discontinuity of the residual (everything
            # else is continuous, piecewise linear); it is returned so the
            # step control can recognise a step that crosses it.
            gate = ca.if_else(
                ca.logic_or(
                    ca.mmax(alpha - stall_angles) > 0.0,
                    ca.mmax(stall_angles_neg - alpha) > 0.0,
                ),
                1.0,
                0.0,
            )
            mu = gate * ca.fmax(
                0.0,
                -self.artificial_viscosity_factor
                * planform_area
                * lift_slope
                / width**2,
            )
        else:
            gate = ca.MX(0.0)
            mu = ca.MX.zeros(n, 1)
        h = gamma_raw + mu * l_gamma

        # Jacobian surrogate. The polars are piecewise linear, so dCl/dalpha
        # jumps at every table node -- by ~2 pi + |post-stall slope| at the
        # Cl-max corner -- and a root can sit ON that corner (the attached
        # branch near its end pins the peak panel there). A Newton step taken
        # with either one-sided slope then overshoots to the other side and
        # chatters forever. For the JACOBIAN ONLY, replace Cl by the two-point
        # average 0.5 (Cl(alpha + d) + Cl(alpha - d)): its slope is the mean of
        # the one-sided slopes at a corner -- the midpoint of the generalized
        # Jacobian, the element a sliding root needs -- and the exact slope
        # wherever the window straddles no node. The residual keeps the exact
        # table, so the roots are unchanged. d = the same 0.5 deg the
        # viscosity's lift-slope difference uses.
        delta_jac = np.deg2rad(0.5)

        def averaged(lookup):
            if smooth:  # a C2 spline has no corners: the exact slope is right
                return lookup(alpha)
            return 0.5 * (lookup(alpha + delta_jac) + lookup(alpha - delta_jac))

        cl_averaged = averaged(cl_of)
        h_for_jacobian = 0.5 * umag * cl_averaged * chord + mu * l_gamma

        # dh_i/dv_rel_i: h_i depends on row i of v_rel only, so three forward
        # sweeps seeded with the unit columns give the full (n x 3) derivative.
        seeds = [ca.DM(np.eye(3)[k][None, :].repeat(n, axis=0)) for k in range(3)]
        dh_dvrel = ca.horzcat(*[ca.jtimes(h_for_jacobian, v_rel, seed) for seed in seeds])

        return ca.Function(
            "vsm_panel_residual",
            [v_rel, x_airf, y_airf, z_airf, chord, width, stall_angles, stall_angles_neg, l_gamma],
            [
                h,
                dh_dvrel,
                mu,
                alpha,
                umag,
                gate,
                h_for_jacobian,
                cl,
                cd_of(alpha),
                cm_of(alpha),
                cl_averaged,
                averaged(cd_of),
                averaged(cm_of),
            ],
            ["v_rel", "x_airf", "y_airf", "z_airf", "chord", "width", "stall_angles", "stall_angles_neg", "l_gamma"],
            ["h", "dh_dvrel", "mu", "alpha", "umag", "gate", "h_avg", "cl", "cd", "cm", "cl_avg", "cd_avg", "cm_avg"],
        )

    def _casadi_newton_function(self):
        alpha_tables, cl_tables, cd_tables, cm_tables = self._polar_tables()
        key = self._casadi_newton_key(alpha_tables, cl_tables, cd_tables, cm_tables)
        fn = self._casadi_newton_cache.get(key)
        if fn is None:
            fn = self._build_casadi_newton_function(
                alpha_tables, cl_tables, cd_tables, cm_tables
            )
            # Keep the cache bounded: a sweep over deformed shapes creates a
            # new polar set per shape.
            if len(self._casadi_newton_cache) >= 32:
                self._casadi_newton_cache.pop(next(iter(self._casadi_newton_cache)))
            self._casadi_newton_cache[key] = fn
        return fn

    def gamma_loop_casadi_newton(self, gamma_initial: np.ndarray) -> tuple:
        """Newton solve of the circulation residual with an exact CasADi
        Jacobian, globalised by pseudo-transient continuation
        (``gamma_loop_type="casadi_newton"``).

        Root-finds ``R(gamma) = (I - diag(mu) L) gamma - G_raw(gamma) = 0``
        (see :meth:`_build_casadi_newton_function`), the same fixed point as
        ``base``/``anderson``. Each iteration is one CasADi evaluation of the
        per-panel physics, three ``n x n`` products and one dense solve.

        Two phases, switched automatically:

        * **Newton** (pseudo time step ``dt = inf``): full Newton direction
          with Armijo backtracking on ``|R|^2``. Quadratic convergence, 3-6
          iterations in attached flow.
        * **Pseudo-transient continuation** (Kelley & Keyes 1998, SIAM J.
          Numer. Anal. 35, 508), entered when the line search cannot reduce
          ``|R|`` -- around the stall knee the piecewise-linear polars make
          ``R`` kinked and ``|R|^2`` has local minima that are not roots, where
          any merit-based method stops. Implicit Euler on the flow
          ``d gamma / dt = G_reg(gamma) - gamma`` (the continuous limit of the
          base loop's relaxed Picard step, so the two share attractors and, from
          the same seed, tend to the same branch): ``((I - diag(mu) L)/dt + J)
          d = -R``. The step is A-stable, so ``dt`` starts ~100x the base loop's
          explicit ``relaxation_factor`` and is then adapted by switched
          evolution relaxation, ``dt <- dt |R_k| / |R_k+1|``, growing back to
          pure Newton as the residual falls; a step that doubles ``|R|`` or
          goes non-finite is rejected and ``dt`` quartered.

        Stopping rule: ``max|R| / max|gamma| < allowed_error``. NOTE this is
        the UN-relaxed residual ``|G(gamma) - gamma|``; the base and Anderson
        loops test ``relaxation_factor * |G - gamma|``, i.e. their converged
        residual is ``1/relaxation_factor`` (100x at the default 0.01)
        LOOSER than this one at the same ``allowed_error``. Newton reaches a
        tight residual for free, so keep ``allowed_error`` tight (1e-8) when
        an outer finite-difference loop differentiates through this solve --
        like Anderson, the superlinear termination makes the converged gamma
        a slightly non-smooth function of the inflow at loose tolerance.

        Branch selection is unchanged: past stall the residual has more than
        one root and the loop lands on the one its seed flows to, so seed
        deliberately (``gamma_distribution``) as with the other loops.

        Returns ``(converged, gamma, alpha_array, Umag_array)`` like
        :meth:`gamma_loop`; ``converged`` False leaves the caller's base-loop
        fallback to decide.
        """
        try:
            fn = self._casadi_newton_function()
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "gamma_loop_type='casadi_newton' requires the optional 'casadi' "
                "package (pip install casadi)."
            ) from exc

        n = self.n_panels
        eye = np.eye(n)
        if self.is_with_artificial_viscosity:
            stall_angles = self._panel_stall_angles()
            stall_angles_neg = self._panel_negative_stall_angles()
            laplacian = self._build_spanwise_laplacian()
        else:
            stall_angles = np.full(n, np.inf)
            stall_angles_neg = np.full(n, -np.inf)
            laplacian = np.zeros((n, n))
        params = (
            self.x_airf_array,
            self.y_airf_array,
            self.z_airf_array,
            self.chord_array,
            self.width_array,
            stall_angles,
            stall_angles_neg,
        )
        aic = (self.AIC_x, self.AIC_y, self.AIC_z)

        def evaluate(x):
            # R = gamma - h(v_rel(gamma), L gamma);  J by the chain rule through
            # v_rel = va + AIC gamma (see _build_casadi_newton_function).
            v_rel = self.va_array + np.column_stack([a @ x for a in aic])
            l_gamma = laplacian @ x
            h, dh_dvrel, mu, alpha, umag, gate = fn(v_rel, *params, l_gamma)[:6]
            h = np.asarray(h, dtype=float).ravel()
            dh_dvrel = np.asarray(dh_dvrel, dtype=float)
            mu = np.asarray(mu, dtype=float).ravel()
            precond = eye - mu[:, None] * laplacian  # (I - diag(mu) L)
            jac = precond.copy()
            for k in range(3):
                jac -= dh_dvrel[:, k][:, None] * aic[k]
            return {
                "x": x,
                "residual": x - h,
                "jac": jac,
                "precond": precond,
                "alpha": np.asarray(alpha, dtype=float).ravel(),
                "umag": np.asarray(umag, dtype=float).ravel(),
                "gate": bool(float(gate) > 0.5),
            }

        def normalized_error(state):
            reference = np.amax(np.abs(state["x"]))
            reference = reference if reference != 0 else 1e-4
            return float(np.amax(np.abs(state["residual"])) / reference)

        def norm(state):
            return float(np.linalg.norm(state["residual"]))

        def finite(state):
            return bool(np.all(np.isfinite(state["residual"])))

        current = evaluate(np.array(gamma_initial, dtype=float))
        evaluations = 1
        iteration = 0
        converged = False
        self.last_stagnated = False
        if not finite(current):
            logging.warning("casadi_newton: non-finite residual at the seed.")
            self.last_iterations = 0
            self.last_newton_evaluations = evaluations
            return False, current["x"], current["alpha"], current["umag"]

        dt = np.inf  # pseudo time step; inf = pure Newton
        dt_restart = float(self.newton_pseudo_time_step)
        dt_min = 1e-4 * dt_restart
        dt_newton = 1e6 * dt_restart  # beyond this the step IS Newton
        failure = None

        while iteration < self.newton_max_iterations:
            if normalized_error(current) < self.allowed_error:
                converged = True
                break

            # ---- direction -------------------------------------------------
            system = current["jac"] if np.isinf(dt) else current["jac"] + current["precond"] / dt
            try:
                direction = np.linalg.solve(system, -current["residual"])
            except np.linalg.LinAlgError:
                direction = None
            if direction is None or not np.all(np.isfinite(direction)):
                if np.isinf(dt):
                    dt = dt_restart  # singular Newton system: go transient
                    continue
                dt *= 0.25
                if dt < dt_min:
                    failure = "singular pseudo-transient system"
                    break
                continue

            if np.isinf(dt):
                # ---- Newton phase: Armijo backtracking on 0.5|R|^2 ---------
                merit = 0.5 * norm(current) ** 2
                slope = float(direction @ (current["jac"].T @ current["residual"]))
                step = 1.0
                accepted = None
                while step >= 2.0**-8:
                    trial = evaluate(current["x"] + step * direction)
                    evaluations += 1
                    if finite(trial) and 0.5 * norm(trial) ** 2 <= merit + 1e-4 * step * slope:
                        accepted = trial
                        break
                    step *= 0.5
                if accepted is None or slope >= 0.0:
                    # Kink minimum of |R|^2 (or not a descent direction):
                    # switch to pseudo-transient continuation from here.
                    dt = dt_restart
                    logging.debug(
                        "casadi_newton: Newton line search stalled at iteration "
                        "%s (|R|/|gamma| = %.2e); switching to pseudo-transient "
                        "continuation with dt = %.3g",
                        iteration,
                        normalized_error(current),
                        dt,
                    )
                    continue
                current = accepted
            else:
                # ---- pseudo-transient phase: implicit Euler + SER ----------
                trial = evaluate(current["x"] + direction)
                evaluations += 1
                ratio = norm(current) / norm(trial) if finite(trial) and norm(trial) > 0 else 0.0
                crossed_gate = finite(trial) and trial["gate"] != current["gate"]
                if not finite(trial) or (
                    ratio < self._newton_reject_ratio
                    and not (crossed_gate and dt <= dt_restart)
                ):
                    # Residual grew too much: reject and shrink. EXCEPT across
                    # the viscosity gate, the residual's one discontinuity:
                    # norms on the two sides are not comparable, and the flow
                    # can legitimately SLIDE along the gate surface (base does,
                    # with its tiny explicit steps). A crossing step at the
                    # floor is accepted so the march chatters across the gate
                    # and moves on instead of shrinking dt to nothing.
                    dt *= 0.25
                    if dt < dt_min:
                        failure = "pseudo time step collapsed"
                        break
                    continue
                current = trial
                # Switched evolution relaxation, floored at the restart step:
                # |R| legitimately GROWS for a while along this flow when
                # panels cross the polar's Cl-max corner (it is not a gradient
                # flow), and un-floored SER would shrink dt below the base
                # loop's explicit step and crawl. The rejection rule above is
                # the safety net; accepted steps never fall below the floor.
                dt = dt_restart if crossed_gate else max(dt_restart, dt * min(ratio, 10.0))
                if dt > dt_newton:
                    dt = np.inf

            iteration += 1
            logging.debug(
                "casadi_newton iteration %s: |R|/|gamma| = %.3e, dt %.3g, max alpha %.2f deg",
                iteration,
                normalized_error(current),
                dt,
                np.rad2deg(np.max(current["alpha"])),
            )
        else:
            converged = normalized_error(current) < self.allowed_error
            if not converged:
                failure = f"iteration budget ({self.newton_max_iterations}) exhausted"

        self.last_iterations = iteration
        self.last_newton_evaluations = evaluations
        if not converged:
            logging.info(
                "casadi_newton did not converge: %s at iteration %s (|R|/|gamma| = %.3e).",
                failure or "line search stalled",
                iteration,
                normalized_error(current),
            )
        return converged, current["x"], current["alpha"], current["umag"]

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
            _, Umag_array, cl_array = self.compute_aerodynamic_quantities(gamma)
            gamma_new = 0.5 * Umag_array * cl_array * self.chord_array
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
            alpha_array, Umag_array, cl_array = (
                self.compute_aerodynamic_quantities(gamma_new)
            )
            return True, gamma_new, alpha_array, Umag_array
