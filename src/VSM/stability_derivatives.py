"""Utilities for computing rigid-body aerodynamic stability derivatives.

This module provides a helper function to evaluate force and moment
coefficient sensitivities with respect to kinematic angles (angle of attack,
sideslip) and body rotation rates (roll, pitch, yaw) by repeatedly invoking
the aerodynamic solver with finite-difference perturbations.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


CoeffDict = Dict[str, float]


def compute_rigid_body_stability_derivatives(
    body_aero,
    solver,
    angle_of_attack: float,
    side_slip: float,
    velocity_magnitude: float,
    roll_rate: float = 0.0,
    pitch_rate: float = 0.0,
    yaw_rate: float = 0.0,
    step_sizes: Optional[Dict[str, float]] = None,
    reference_point: Optional[np.ndarray] = None,
    nondimensionalize_rates: bool = True,
) -> Dict[str, float]:
    """Compute rigid-body stability derivatives for the current configuration.

    This function computes aerodynamic derivatives with respect to kinematic angles
    (angle of attack, sideslip) and body rotation rates (roll, pitch, yaw).

    Parameters
    ----------
    body_aero : VSM.core.BodyAerodynamics.BodyAerodynamics
        Aerodynamic model instance that will be updated in-place.
    solver : VSM.core.Solver.Solver
        Solver configured for the analysis.
    angle_of_attack : float
        Baseline angle of attack in degrees.
    side_slip : float
        Baseline sideslip angle in degrees (positive for wind from the left/port side,
        negative for wind from the right/starboard side).
    velocity_magnitude : float
        Magnitude of the freestream velocity (m/s).
    roll_rate : float, optional
        Baseline body roll rate ``p`` in rad/s. Defaults to 0.0.
    pitch_rate : float, optional
        Baseline body pitch rate ``q`` in rad/s. Defaults to 0.0.
    yaw_rate : float, optional
        Baseline body yaw rate ``r`` in rad/s. Defaults to 0.0.
    step_sizes : dict, optional
        Optional overrides for perturbation steps. Supported keys are
        ``{"alpha", "beta", "p", "q", "r"}``.
        Angle steps are in degrees (internally converted to radians for the derivative),
        and rate steps are in rad/s.
    reference_point : np.ndarray, optional
        Reference point for moment calculation [x, y, z]. If None, defaults to
        solver.reference_point if available, otherwise [0, 0, 0].
    nondimensionalize_rates : bool, optional
        If True (default), rate derivatives are non-dimensionalized using:
            hat_p = p * b / (2*V)
            hat_q = q * c_MAC / (2*V)
            hat_r = r * b / (2*V)
        where b is wingspan, c_MAC is mean aerodynamic chord, and V is velocity magnitude.
        This converts derivatives from per rad/s to per hat-rate (dimensionless).
        If False, rate derivatives remain dimensional (per rad/s).

    Returns
    -------
    dict
        Dictionary with keys such as ``"dCx_dalpha"`` and ``"dCMz_dp"`` covering
        stability derivatives with respect to angles and body rates:

        - Angle derivatives (per radian): dC*/dalpha, dC*/dbeta
        - Rate derivatives: dC*/dp, dC*/dq, dC*/dr
          - If nondimensionalize_rates=True: per hat-rate (dimensionless)
          - If nondimensionalize_rates=False: per rad/s (dimensional)

        where C* ∈ {Cx, Cy, Cz, CMx, CMy, CMz}

    Notes
    -----
    Derivatives are evaluated via central finite differences. Angular
    sensitivities are returned per-radian.

    The reference_point parameter is critical for physically correct rotational
    velocity calculations. The rotational velocity at any point r is computed as:
        v_rot(r) = omega × (r - r_ref)
    where omega is the body rate vector and r_ref is the reference point.

    Non-dimensionalization (when nondimensionalize_rates=True):
    ---------------------------------------------------------------
    Rate derivatives are converted to dimensionless form using:
        - hat_p = p * b / (2*V)         [roll rate]
        - hat_q = q * c_MAC / (2*V)     [pitch rate]
        - hat_r = r * b / (2*V)         [yaw rate]

    This converts derivatives from per rad/s to per hat-rate:
        d(Coeff)/d(hat_p) = d(Coeff)/dp * (2*V/b)
        d(Coeff)/d(hat_q) = d(Coeff)/dq * (2*V/c_MAC)
        d(Coeff)/d(hat_r) = d(Coeff)/dr * (2*V/b)

    This is the standard convention used in flight dynamics and control texts,
    making the derivatives directly compatible with 6-DOF equations of motion.
    """

    coeff_names = ("Cx", "Cy", "Cz", "CMx", "CMy", "CMz")
    param_names = ("alpha", "beta", "p", "q", "r")

    default_steps = {
        "alpha": np.rad2deg(0.005),  # degrees
        "beta": np.rad2deg(0.005),  # degrees
        "p": 0.01,  # rad/s
        "q": 0.01,  # rad/s
        "r": 0.01,  # rad/s
    }
    if step_sizes:
        for key, value in step_sizes.items():
            if key not in default_steps:
                raise KeyError(f"Unsupported step key '{key}'. Allowed: {param_names}")
            default_steps[key] = float(value)

    rates = {"p": roll_rate, "q": pitch_rate, "r": yaw_rate}

    # Use reference_point if provided, otherwise try solver.reference_point, else default to [0,0,0]
    if reference_point is None:
        if hasattr(solver, "reference_point"):
            reference_point = np.array(solver.reference_point)
        else:
            reference_point = np.array([0.0, 0.0, 0.0])

    def set_to_baseline() -> None:
        body_aero.va_initialize(
            Umag=velocity_magnitude,
            angle_of_attack=angle_of_attack,
            side_slip=side_slip,
            yaw_rate=rates["r"],
            pitch_rate=rates["q"],
            roll_rate=rates["p"],
            reference_point=reference_point,
        )

    def _solve_and_extract() -> CoeffDict:
        results = solver.solve(body_aero)
        q_inf = float(results.get("q_ref", np.nan))
        if not np.isfinite(q_inf) or q_inf <= 0.0:
            va_vector = np.asarray(body_aero.va, dtype=float)
            if va_vector.ndim == 1 and va_vector.size == 3:
                speed = np.linalg.norm(va_vector)
            elif va_vector.ndim == 2 and va_vector.shape[1] == 3:
                speed = np.linalg.norm(np.mean(va_vector, axis=0))
            else:
                raise ValueError(
                    "Unable to infer reference dynamic pressure from body_aero.va shape "
                    f"{va_vector.shape}."
                )
            if speed <= 0.0:
                raise ValueError(
                    "Freestream speed must be positive to compute derivatives."
                )
            q_inf = 0.5 * solver.rho * speed**2
        reference_area = results["projected_area"]
        if reference_area <= 0.0:
            raise ValueError("Reference area must be positive.")

        coeffs = {
            "Cx": results["Fx"] / (q_inf * reference_area),
            "Cy": results["Fy"] / (q_inf * reference_area),
            "Cz": results["Fz"] / (q_inf * reference_area),
            "CMx": results["cmx"],
            "CMy": results["cmy"],
            "CMz": results["cmz"],
        }
        return coeffs

    def _evaluate_with_angles(alpha_deg: float, beta_deg: float) -> CoeffDict:
        body_aero.va_initialize(
            Umag=velocity_magnitude,
            angle_of_attack=alpha_deg,
            side_slip=beta_deg,
            yaw_rate=rates["r"],
            pitch_rate=rates["q"],
            roll_rate=rates["p"],
            reference_point=reference_point,
        )
        return _solve_and_extract()

    def _central_difference(
        coeff_plus: CoeffDict, coeff_minus: CoeffDict, delta: float
    ) -> CoeffDict:
        if delta == 0.0:
            raise ValueError("Delta for central difference cannot be zero.")
        return {
            name: (coeff_plus[name] - coeff_minus[name]) / (2.0 * delta)
            for name in coeff_names
        }

    derivatives: Dict[str, float] = {}

    # Baseline state
    set_to_baseline()

    # Angle derivatives (per radian)
    for param, step_key in (("alpha", "alpha"), ("beta", "beta")):
        set_to_baseline()
        step_deg = default_steps[step_key]
        delta_rad = np.deg2rad(step_deg)
        if param == "alpha":
            coeff_plus = _evaluate_with_angles(angle_of_attack + step_deg, side_slip)
            coeff_minus = _evaluate_with_angles(angle_of_attack - step_deg, side_slip)
        else:
            coeff_plus = _evaluate_with_angles(angle_of_attack, side_slip + step_deg)
            coeff_minus = _evaluate_with_angles(angle_of_attack, side_slip - step_deg)

        diff = _central_difference(coeff_plus, coeff_minus, delta_rad)

        for coeff_name in coeff_names:
            derivatives[f"d{coeff_name}_d{param}"] = diff[coeff_name]

    # Body rate derivatives (p, q, r)
    for rate_name in ("p", "q", "r"):
        set_to_baseline()
        delta = default_steps[rate_name]

        # Create perturbed rate dictionaries
        rates_plus = rates.copy()
        rates_minus = rates.copy()
        rates_plus[rate_name] += delta
        rates_minus[rate_name] -= delta

        # Evaluate with positive perturbation
        body_aero.va_initialize(
            Umag=velocity_magnitude,
            angle_of_attack=angle_of_attack,
            side_slip=side_slip,
            yaw_rate=rates_plus["r"],
            pitch_rate=rates_plus["q"],
            roll_rate=rates_plus["p"],
            reference_point=reference_point,
        )
        coeff_plus = _solve_and_extract()

        # Evaluate with negative perturbation
        body_aero.va_initialize(
            Umag=velocity_magnitude,
            angle_of_attack=angle_of_attack,
            side_slip=side_slip,
            yaw_rate=rates_minus["r"],
            pitch_rate=rates_minus["q"],
            roll_rate=rates_minus["p"],
            reference_point=reference_point,
        )
        coeff_minus = _solve_and_extract()

        diff = _central_difference(coeff_plus, coeff_minus, delta)

        for coeff_name in coeff_names:
            derivatives[f"d{coeff_name}_d{rate_name}"] = diff[coeff_name]

    # Restore baseline state before returning
    set_to_baseline()

    # Non-dimensionalize rate derivatives if requested
    if nondimensionalize_rates:
        # Get geometric properties from baseline solution
        results_baseline = solver.solve(body_aero)
        b = results_baseline["wing_span"]  # wingspan

        # Compute mean aerodynamic chord from projected area and span
        # Note: c_MAC = S / b is a reasonable approximation for simple geometries
        # For more complex planforms, a weighted average chord should be computed
        S = results_baseline["projected_area"]
        c_mac = S / b
        print(
            f"\n --> Computed c_MAC = {c_mac:.3f} m from S={S:.3f} m² and b={b:.3f} m."
        )

        V = velocity_magnitude

        # Scaling factors to convert from per rad/s to per hat-rate:
        # hat_p = p * b / (2*V)       -->  d()/dp [per rad/s] * (2*V/b) = d()/d(hat_p)
        # hat_q = q * c_MAC / (2*V)   -->  d()/dq [per rad/s] * (2*V/c_MAC) = d()/d(hat_q)
        # hat_r = r * b / (2*V)       -->  d()/dr [per rad/s] * (2*V/b) = d()/d(hat_r)
        scale_p = (2.0 * V) / b
        scale_q = (2.0 * V) / c_mac
        scale_r = (2.0 * V) / b

        # Apply scaling to all rate derivatives
        for coeff_name in coeff_names:
            key_p = f"d{coeff_name}_dp"
            key_q = f"d{coeff_name}_dq"
            key_r = f"d{coeff_name}_dr"

            if key_p in derivatives:
                derivatives[key_p] *= scale_p  # now per hat_p
            if key_q in derivatives:
                derivatives[key_q] *= scale_q  # now per hat_q
            if key_r in derivatives:
                derivatives[key_r] *= scale_r  # now per hat_r

    return derivatives


def map_derivatives_to_aircraft_frame(
    derivatives_vsm: Dict[str, float],
) -> Dict[str, float]:
    """
    Transform stability derivatives from VSM frame (x rearward, y right, z up)
    to aircraft frame (x forward, y right, z down) using the proper rotation
    R_y(pi) = diag(-1, 1, -1).

    Signs:
      s(Cx)=s(Cz)=s(CMx)=s(CMz) = -1
      s(Cy)=s(CMy)             = +1

    Mapping of independent variables:
      alpha' = alpha
      beta'  = -beta
      p' = -p,  q' = q,  r' = -r
    """
    print("\n" + "=" * 60)
    print("REFERENCE FRAME TRANSFORMATION")
    print("=" * 60)

    print("\nVSM Reference Frame (used above):")
    print("  x: rearward (LE → TE)")
    print("  y: right wing")
    print("  z: upward")
    print("  β: positive for wind from LEFT (port, +Vy)")
    print("  α: positive for nose up")
    print("  Body rates (right-hand rule with z-up):")
    print("    p (roll):  positive = LEFT wing DOWN (CCW looking forward)")
    print("    q (pitch): positive = nose UP (CW looking from right wing)")
    print("    r (yaw):   positive = nose LEFT (CCW looking down)")

    print("\n" + "-" * 40)
    print(f"Aircraft Reference Frame (standard aerospace):  ")
    print(f"  x: forward (tail → nose)")
    print(f"  y: right wing")
    print(f"  z: downward")
    print(f"  Mapping used: R_y(pi) = diag(-1, 1, -1)  [proper 180° about +y]")
    print(f"  Implications:")
    print(f"    - Forces:   (Cx', Cy', Cz') = (-Cx,  Cy, -Cz)")
    print(f"    - Moments:  (CMx',CMy',CMz')= (-CMx, CMy, -CMz)")
    print(f"    - Angles:   alpha' = alpha,  beta' = -beta")
    print(f"    - Rates:    (p', q', r') = (-p, q, -r)")

    print("\n" + "=" * 60)
    print("STABILITY DERIVATIVES IN AIRCRAFT FRAME")
    print("=" * 60)
    s = {"Cx": -1, "Cy": +1, "Cz": -1, "CMx": -1, "CMy": +1, "CMz": -1}

    out: Dict[str, float] = {}

    # Angle derivatives
    for coeff, sc in s.items():
        ka = f"d{coeff}_dalpha"  # alpha' = alpha (no flip)
        kb = f"d{coeff}_dbeta"  # beta' = -beta (extra minus)

        if ka in derivatives_vsm:
            out[ka] = sc * derivatives_vsm[ka]
        if kb in derivatives_vsm:
            out[kb] = -sc * derivatives_vsm[kb]

    # Rate derivatives
    for coeff, sc in s.items():
        kp = f"d{coeff}_dp"  # p' = -p (extra minus)
        kq = f"d{coeff}_dq"  # q' =  q (no flip)
        kr = f"d{coeff}_dr"  # r' = -r (extra minus)

        if kp in derivatives_vsm:
            out[kp] = -sc * derivatives_vsm[kp]
        if kq in derivatives_vsm:
            out[kq] = sc * derivatives_vsm[kq]
        if kr in derivatives_vsm:
            out[kr] = -sc * derivatives_vsm[kr]

    return out


import numpy as np
import math
from typing import Dict, Tuple, Optional, Sequence


# --- Reuse the adapter that maps aircraft inputs -> VSM solver and back -------------
def compute_aircraft_frame_coeffs_from_solver(
    body_aero,
    solver,
    alpha_rad: float,
    beta_rad: float,
    V: float,
    p: float = 0.0,
    q: float = 0.0,
    r: float = 0.0,
    reference_point: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], float, float]:
    # map aircraft->VSM
    alpha_deg_vsm = math.degrees(alpha_rad)
    beta_deg_vsm = -math.degrees(beta_rad)
    p_vsm, q_vsm, r_vsm = -p, q, -r

    if reference_point is None:
        if hasattr(solver, "reference_point"):
            reference_point = np.array(solver.reference_point, dtype=float)
        else:
            reference_point = np.array([0.0, 0.0, 0.0], dtype=float)

    body_aero.va_initialize(
        Umag=V,
        angle_of_attack=alpha_deg_vsm,
        side_slip=beta_deg_vsm,
        yaw_rate=r_vsm,
        pitch_rate=q_vsm,
        roll_rate=p_vsm,
        reference_point=reference_point,
    )
    results = solver.solve(body_aero)

    q_inf = float(results.get("q_ref", np.nan))
    if not np.isfinite(q_inf) or q_inf <= 0.0:
        va_vec = np.asarray(body_aero.va, float)
        if va_vec.ndim == 1 and va_vec.size == 3:
            Va = float(np.linalg.norm(va_vec))
        elif va_vec.ndim == 2 and va_vec.shape[1] == 3:
            Va = float(np.linalg.norm(np.mean(va_vec, axis=0)))
        else:
            raise ValueError(
                "Unable to infer reference dynamic pressure from body_aero.va shape "
                f"{va_vec.shape}."
            )
        if Va <= 0:
            raise ValueError("Speed must be >0")
        rho = float(getattr(solver, "rho", 1.225))
        q_inf = 0.5 * rho * Va * Va
    S = float(results["projected_area"])
    if S <= 0:
        raise ValueError("projected_area must be >0")

    Fx, Fy, Fz = float(results["Fx"]), float(results["Fy"]), float(results["Fz"])
    CMx, CMy, CMz = float(results["cmx"]), float(results["cmy"]), float(results["cmz"])

    # VSM -> aircraft mapping: R_y(pi) = diag(-1, +1, -1)
    Cx_vsm = Fx / (q_inf * S)
    Cy_vsm = Fy / (q_inf * S)
    Cz_vsm = Fz / (q_inf * S)
    coeffs_air = {
        "CX": -Cx_vsm,
        "CY": Cy_vsm,
        "CZ": -Cz_vsm,
        "Cl": -CMx,
        "Cm": CMy,
        "Cn": -CMz,
    }
    return coeffs_air, q_inf, S


# --- Utilities ---------------------------------------------------------------------
def fit_quad_alpha(
    alpha_vec: np.ndarray, y_alpha: np.ndarray
) -> Tuple[float, float, float]:
    """Least-squares fit y(α) ≈ c2 α^2 + c1 α + c0; α in radians."""
    A = np.column_stack([alpha_vec**2, alpha_vec, np.ones_like(alpha_vec)])
    c2, c1, c0 = np.linalg.lstsq(A, y_alpha, rcond=None)[0]
    return float(c2), float(c1), float(c0)


def central_diff(f_plus: float, f_minus: float, h: float) -> float:
    return (f_plus - f_minus) / (2.0 * h)


# --- Main builder ------------------------------------------------------------------
def build_malz_coeff_table_from_solver(
    body_aero,
    solver,
    V: float,
    b: float,
    c: float,
    alpha_grid_deg: Sequence[float] = tuple(np.linspace(-15, 15, 13)),
    beta_step_deg: float = 1.0,
    p_hat_step: float = 0.02,  # small dimensionless hat-rate steps
    q_hat_step: float = 0.02,
    r_hat_step: float = 0.02,
    reference_point: Optional[np.ndarray] = None,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Returns a dict like COEF with keys:
      CX0,CXb,CXp,CXq,CXr,  CY0,CYb,CYp,CYq,CYr,  CZ0,CZb,CZp,CZq,CZr,
      Cl0,Clb,Clp,Clq,Clr,  Cm0,Cmb,Cmp,Cmq,Cmr,  Cn0,Cnb,Cnp,Cnq,Cnr
    Each value is a (c2,c1,c0) tuple per Eq. (19), α in radians.

    All evaluations are at controls = 0.
    """
    alpha_grid = np.radians(np.array(alpha_grid_deg, float))
    # small angle/rate steps
    dbeta = math.radians(beta_step_deg)

    # convert hat-rate steps to actual rad/s perturbations at this V
    # p̂ = b p / (2 V) => p = (2 V / b) p̂ ; etc.
    p_step = (2.0 * V / b) * p_hat_step
    q_step = (2.0 * V / c) * q_hat_step
    r_step = (2.0 * V / b) * r_hat_step

    # storage per α for each output we will fit
    names = ["CX", "CY", "CZ", "Cl", "Cm", "Cn"]
    slices_0 = {k: [] for k in names}
    slices_db = {k: [] for k in names}
    slices_dp = {k: [] for k in names}
    slices_dq = {k: [] for k in names}
    slices_dr = {k: [] for k in names}

    beta0 = 0.0
    p0 = q0 = r0 = 0.0

    for a in alpha_grid:
        # baseline
        c0, _, _ = compute_aircraft_frame_coeffs_from_solver(
            body_aero, solver, a, beta0, V, p0, q0, r0, reference_point
        )
        for k in names:
            slices_0[k].append(c0[k])

        # beta derivative at this α
        cp, _, _ = compute_aircraft_frame_coeffs_from_solver(
            body_aero, solver, a, beta0 + dbeta, V, p0, q0, r0, reference_point
        )
        cm, _, _ = compute_aircraft_frame_coeffs_from_solver(
            body_aero, solver, a, beta0 - dbeta, V, p0, q0, r0, reference_point
        )
        for k in names:
            slices_db[k].append(central_diff(cp[k], cm[k], dbeta))

        # p̂ derivative (via p step in rad/s)
        cp_, _, _ = compute_aircraft_frame_coeffs_from_solver(
            body_aero, solver, a, beta0, V, p0 + p_step, q0, r0, reference_point
        )
        cm_, _, _ = compute_aircraft_frame_coeffs_from_solver(
            body_aero, solver, a, beta0, V, p0 - p_step, q0, r0, reference_point
        )
        for k in names:
            # convert from per (rad/s) to per hat_p by multiplying (2V/b)
            slices_dp[k].append(central_diff(cp_[k], cm_[k], p_step) * (2.0 * V / b))

        # q̂ derivative
        cq_, _, _ = compute_aircraft_frame_coeffs_from_solver(
            body_aero, solver, a, beta0, V, p0, q0 + q_step, r0, reference_point
        )
        cmq, _, _ = compute_aircraft_frame_coeffs_from_solver(
            body_aero, solver, a, beta0, V, p0, q0 - q_step, r0, reference_point
        )
        for k in names:
            slices_dq[k].append(central_diff(cq_[k], cmq[k], q_step) * (2.0 * V / c))

        # r̂ derivative
        cr_, _, _ = compute_aircraft_frame_coeffs_from_solver(
            body_aero, solver, a, beta0, V, p0, q0, r0 + r_step, reference_point
        )
        cmr, _, _ = compute_aircraft_frame_coeffs_from_solver(
            body_aero, solver, a, beta0, V, p0, q0, r0 - r_step, reference_point
        )
        for k in names:
            slices_dr[k].append(central_diff(cr_[k], cmr[k], r_step) * (2.0 * V / b))

    # Fit each α-slice to quadratic c2 α^2 + c1 α + c0, α in radians
    COEF = {}

    # Baseline (*0)
    for k in names:
        COEF[k + "0"] = fit_quad_alpha(alpha_grid, np.array(slices_0[k]))

    # Beta (*b)
    for k in names:
        COEF[k + "b"] = fit_quad_alpha(alpha_grid, np.array(slices_db[k]))

    # Rates (*p,*q,*r)
    for k in names:
        COEF[k + "p"] = fit_quad_alpha(alpha_grid, np.array(slices_dp[k]))
        COEF[k + "q"] = fit_quad_alpha(alpha_grid, np.array(slices_dq[k]))
        COEF[k + "r"] = fit_quad_alpha(alpha_grid, np.array(slices_dr[k]))

    return COEF
