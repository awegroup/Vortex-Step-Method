r"""Trim angle computation utilities.

This module provides a helper to find trim angles (where the pitching moment
coefficient about the reference point crosses zero) and to verify dynamic
stability by checking that :math:`\partial C_{My} / \partial \alpha` is
negative at each solution.
"""

from __future__ import annotations

from typing import List, Mapping, MutableMapping

import numpy as np

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver


def compute_trim_angle(
    body_aero: BodyAerodynamics,
    solver: Solver,
    side_slip: float = 0.0,
    velocity_magnitude: float = 10.0,
    roll_rate: float = 0.0,
    pitch_rate: float = 0.0,
    yaw_rate: float = 0.0,
    alpha_min: float = -5.0,
    alpha_max: float = 15.0,
    coarse_step: float = 2.0,
    fine_tolerance: float = 1e-3,
    derivative_step: float = 0.1,
    max_bisection_iter: int = 40,
    reference_point: np.ndarray = None,
) -> List[MutableMapping[str, float]]:
    """Compute trim angles and verify pitch stability.

    The routine performs a coarse sweep in angle of attack to detect sign
    changes in the pitching moment coefficient ``CMy`` about the solver's
    reference point. Each sign change is refined via bisection until the
    pitching moment crosses zero within ``fine_tolerance``. A solution is
    accepted only if the local slope ``dCMy/dalpha`` is negative.

    Parameters
    ----------
    body_aero : BodyAerodynamics
        Instantiated aerodynamic model that will be updated in-place.
    solver : Solver
        Solver instance configured with the desired reference point.
    side_slip : float, optional
        Sideslip angle (degrees). Default is 0.0.
    velocity_magnitude : float, optional
        Freestream velocity magnitude (m/s). Default is 10.0.
    roll_rate : float, optional
        Body roll rate (rad/s). Default is 0.0.
    pitch_rate : float, optional
        Body pitch rate (rad/s). Default is 0.0.
    yaw_rate : float, optional
        Body yaw rate (rad/s). Default is 0.0.
    alpha_min, alpha_max : float, optional
        Bounds (degrees) for the coarse sweep.
    coarse_step : float, optional
        Step size (degrees) for the coarse sweep.
    fine_tolerance : float, optional
        Angular tolerance (degrees) for the bisection refinement.
    derivative_step : float, optional
        Perturbation (degrees) used to evaluate ``dCMy/dalpha`` at the trim
        angle. Must be positive.
    max_bisection_iter : int, optional
        Maximum number of bisection iterations per bracket.
    reference_point : np.ndarray, optional
        Reference point for moment calculation [x, y, z]. If None, defaults to
        solver.reference_point.

    Returns
    -------
    list of dict
        List of dictionaries, each containing the trim angle (degrees), the
        pitching moment derivative (per radian), and a boolean ``stable`` flag
        indicating the sign of the derivative.

    Notes
    -----
    If no sign change is found, the closest-to-zero value from the coarse
    sweep is returned with ``stable=False``.
    """

    if derivative_step <= 0.0:
        raise ValueError("derivative_step must be positive")
    if coarse_step <= 0.0:
        raise ValueError("coarse_step must be positive")
    if alpha_max <= alpha_min:
        raise ValueError("alpha_max must be greater than alpha_min")

    # Use reference_point if provided, otherwise use solver.reference_point
    if reference_point is None:
        if hasattr(solver, "reference_point"):
            reference_point = np.array(solver.reference_point)
        else:
            reference_point = np.array([0.0, 0.0, 0.0])

    alpha_coarse = np.arange(alpha_min, alpha_max + coarse_step, coarse_step)

    def _extract_cmy(results: Mapping[str, float]) -> float:
        for key in ("CMy", "cmy", "cm", "CMY"):
            if key in results:
                return float(results[key])
        raise KeyError("Pitching moment coefficient not present in solver result")

    def _evaluate_cmy(alpha_deg: float) -> float:
        body_aero.va_initialize(
            Umag=velocity_magnitude,
            angle_of_attack=alpha_deg,
            side_slip=side_slip,
            yaw_rate=yaw_rate,
            pitch_rate=pitch_rate,
            roll_rate=roll_rate,
            reference_point=reference_point,
        )
        solved = solver.solve(body_aero)
        return _extract_cmy(solved)

    def _compute_derivative(alpha_deg: float, delta_deg: float) -> float:
        delta_rad = np.deg2rad(delta_deg)
        cmy_plus = _evaluate_cmy(alpha_deg + delta_deg)
        cmy_minus = _evaluate_cmy(alpha_deg - delta_deg)
        return (cmy_plus - cmy_minus) / (2.0 * delta_rad)

    # Perform coarse sweep
    cmy_coarse: List[float] = []
    for alpha in alpha_coarse:
        cmy_value = _evaluate_cmy(float(alpha))
        cmy_coarse.append(cmy_value)

    cmy_coarse_array = np.asarray(cmy_coarse)
    valid_mask = ~np.isnan(cmy_coarse_array)
    valid_alphas = alpha_coarse[valid_mask]
    valid_cmy = cmy_coarse_array[valid_mask]

    results: List[MutableMapping[str, float]] = []

    if valid_cmy.size < 2:
        raise ValueError(
            f"Insufficient valid CMy values in coarse sweep. "
            f"Got {valid_cmy.size} valid points out of {len(alpha_coarse)} angles tested. "
            f"Check that the aerodynamic solver is working correctly."
        )

    # Look for sign changes
    sign_changes = np.where(np.diff(np.sign(valid_cmy)))[0]

    if sign_changes.size == 0:
        # No sign change found, return closest to zero
        closest_idx = int(np.argmin(np.abs(valid_cmy)))
        trim_alpha = float(valid_alphas[closest_idx])
        derivative = _compute_derivative(trim_alpha, derivative_step)
        return {
            "trim_angle": trim_alpha,
            "dCMy_dalpha": derivative,
            "is_stable": bool(derivative < 0.0),
            "notes": "closest to zero",
        }

    # Refine each sign change with bisection
    for idx in sign_changes:
        alpha_low = float(valid_alphas[idx])
        alpha_high = float(valid_alphas[idx + 1])
        cmy_low = float(valid_cmy[idx])
        cmy_high = float(valid_cmy[idx + 1])

        if np.isnan(cmy_low) or np.isnan(cmy_high):
            continue
        if np.sign(cmy_low) == np.sign(cmy_high):
            continue

        low, high = alpha_low, alpha_high
        c_low, c_high = cmy_low, cmy_high
        iteration = 0

        while (high - low) > fine_tolerance and iteration < max_bisection_iter:
            mid = 0.5 * (low + high)
            c_mid = _evaluate_cmy(mid)

            if c_mid == 0.0:
                low, high = mid, mid
                c_low = c_high = 0.0
                break

            if np.sign(c_mid) == np.sign(c_low):
                low, c_low = mid, c_mid
            else:
                high, c_high = mid, c_mid
            iteration += 1

        trim_alpha = 0.5 * (low + high)
        derivative = _compute_derivative(trim_alpha, derivative_step)
        results.append(
            {
                "trim_angle": trim_alpha,
                "dCMy_dalpha": derivative,
                "is_stable": bool(derivative < 0.0),
                "notes": "sign change",
            }
        )

    idx_list = []
    for idx, res in enumerate(results):
        if res["is_stable"]:
            idx_list.append(idx)

    if len(idx_list) == 0:
        raise ValueError(
            "No stable trim point found. All trim candidates have positive dCMy/dalpha (unstable)."
        )

    if len(idx_list) > 1:
        raise ValueError("Multiple stable trim points found. Not supported.")

    return results[idx_list[0]]
