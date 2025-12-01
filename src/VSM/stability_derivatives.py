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
        va_vector = np.asarray(body_aero.va, dtype=float)
        if va_vector.ndim != 1 or va_vector.size != 3:
            raise ValueError("Expected a uniform apparent velocity vector of length 3.")
        speed = np.linalg.norm(va_vector)
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
