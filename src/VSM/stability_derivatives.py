"""Utilities for computing rigid-body aerodynamic stability derivatives.

This module provides a helper function to evaluate force and moment
coefficient sensitivities with respect to translational velocity components,
angles, and body rotation rates by repeatedly invoking the aerodynamic
solver with finite-difference perturbations.
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
) -> Dict[str, float]:
    """Compute rigid-body stability derivatives for the current configuration.

    Parameters
    ----------
    body_aero : VSM.core.BodyAerodynamics.BodyAerodynamics
        Aerodynamic model instance that will be updated in-place.
    solver : VSM.core.Solver.Solver
        Solver configured for the analysis.
    angle_of_attack : float
        Baseline angle of attack in degrees.
    side_slip : float
        Baseline sideslip angle in degrees (positive for starboard-to-port flow).
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
        ``{"u", "v", "w", "alpha", "beta", "p", "q", "r"}``.
        Velocity steps are in m/s, angle steps in degrees (internally converted
        to radians for the derivative), and rate steps in rad/s.

    Returns
    -------
    dict
        Dictionary with keys such as ``"dCx_du"`` and ``"dCMz_dp"`` covering
        all rigid-body force and moment derivatives.

    Notes
    -----
    Derivatives are evaluated via central finite differences. Angular
    sensitivities are returned per-radian.
    """

    coeff_names = ("Cx", "Cy", "Cz", "CMx", "CMy", "CMz")
    param_names = ("u", "v", "w", "alpha", "beta", "p", "q", "r")

    default_steps = {
        "u": 0.1,
        "v": 0.1,
        "w": 0.1,
        "alpha": 0.5,  # degrees
        "beta": 0.5,  # degrees
        "p": 0.01,
        "q": 0.01,
        "r": 0.01,
    }
    if step_sizes:
        for key, value in step_sizes.items():
            if key not in default_steps:
                raise KeyError(f"Unsupported step key '{key}'. Allowed: {param_names}")
            default_steps[key] = float(value)

    rates = {"p": roll_rate, "q": pitch_rate, "r": yaw_rate}

    def set_to_baseline() -> None:
        body_aero.va_initialize(
            Umag=velocity_magnitude,
            angle_of_attack=angle_of_attack,
            side_slip=side_slip,
            yaw_rate=rates["r"],
            pitch_rate=rates["q"],
            roll_rate=rates["p"],
        )

    def _effective_component_step(base_value: float, nominal: float) -> float:
        if abs(base_value) > 1e-8:
            return max(nominal, 1e-3 * abs(base_value))
        return nominal

    def _solve_and_extract() -> CoeffDict:
        results = solver.solve(body_aero)
        va_vector = np.asarray(body_aero.va, dtype=float)
        if va_vector.ndim != 1 or va_vector.size != 3:
            raise ValueError(
                "Expected a uniform apparent velocity vector of length 3."
            )
        speed = np.linalg.norm(va_vector)
        if speed <= 0.0:
            raise ValueError("Freestream speed must be positive to compute derivatives.")

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

    def _evaluate_with_vector(vector: np.ndarray, updated_rates: Dict[str, float]) -> CoeffDict:
        body_aero.va = (
            np.asarray(vector, dtype=float),
            updated_rates["r"],
            updated_rates["q"],
            updated_rates["p"],
        )
        return _solve_and_extract()

    def _evaluate_with_angles(alpha_deg: float, beta_deg: float) -> CoeffDict:
        body_aero.va_initialize(
            Umag=velocity_magnitude,
            angle_of_attack=alpha_deg,
            side_slip=beta_deg,
            yaw_rate=rates["r"],
            pitch_rate=rates["q"],
            roll_rate=rates["p"],
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

    # Velocity component derivatives (u, v, w)
    base_velocity = np.asarray(body_aero.va, dtype=float)
    for axis, component in enumerate(("u", "v", "w")):
        set_to_baseline()
        base_velocity = np.asarray(body_aero.va, dtype=float)
        delta = _effective_component_step(base_velocity[axis], default_steps[component])
        velocity_plus = base_velocity.copy()
        velocity_minus = base_velocity.copy()
        velocity_plus[axis] += delta
        velocity_minus[axis] -= delta

        coeff_plus = _evaluate_with_vector(velocity_plus, rates)
        coeff_minus = _evaluate_with_vector(velocity_minus, rates)
        diff = _central_difference(coeff_plus, coeff_minus, delta)

        for coeff_name in coeff_names:
            derivatives[f"d{coeff_name}_d{component}"] = diff[coeff_name]

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
        base_velocity = np.asarray(body_aero.va, dtype=float)
        delta = default_steps[rate_name]
        rates_plus = rates.copy()
        rates_minus = rates.copy()
        rates_plus[rate_name] += delta
        rates_minus[rate_name] -= delta

        coeff_plus = _evaluate_with_vector(base_velocity, rates_plus)
        coeff_minus = _evaluate_with_vector(base_velocity, rates_minus)
        diff = _central_difference(coeff_plus, coeff_minus, delta)

        for coeff_name in coeff_names:
            derivatives[f"d{coeff_name}_d{rate_name}"] = diff[coeff_name]

    # Restore baseline state before returning
    set_to_baseline()

    return derivatives

