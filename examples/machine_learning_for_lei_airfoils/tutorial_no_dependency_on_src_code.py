"""
Tutorial: LEI Airfoil Geometry + Masure Regression Polars

This script:
  1) Loads the trained Extra Trees "masure_regression" model directly from .pkl files
     (e.g., ET_re1e6.pkl), *without* using VSM.core.AirfoilAerodynamics.
  2) Builds the LEI airfoil geometry using ONLY `utils_masure_regression_lei_parametric`.
  3) Plots a 2-row layout:
       Row 1 (2 cols): (1,1) detailed construction view, (1,2) clean outline
       Row 2 (3 cols): CL–α, CD–α, CM–α (from the ML model)
"""

import warnings
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- Import ONLY your LEI utilities (geometry + outline) ---
from utils_masure_regression_lei_parametric import (
    generate_profile,  # → (all_points, profile_name, seam_a)
    LEI_airfoil,  # → many outputs (control points, splines, fillet, etc.)
)

# ======================================================================================
#                    Minimal ML Loader (copy-paste from AirfoilAerodynamics)
# ======================================================================================

# In-memory cache of loaded models (per Reynolds)
_MASURE_MODEL_CACHE = {}


def _patch_sklearn_compatibility(model):
    """
    Best-effort patch for ExtraTrees regressors loaded across sklearn versions.
    Mirrors the compatibility helper used in AirfoilAerodynamics.
    """

    def patch_estimator(estimator):
        # Add missing monotonic_cst attribute for ExtraTreeRegressor
        if hasattr(estimator, "estimators_"):
            for tree in estimator.estimators_:
                if not hasattr(tree, "monotonic_cst"):
                    tree.monotonic_cst = None
        elif not hasattr(estimator, "monotonic_cst"):
            estimator.monotonic_cst = None

        # Some sklearn versions expect this attribute
        if not hasattr(estimator, "_support_missing_values"):
            estimator._support_missing_values = lambda X: False

    # Handle different model structures
    if hasattr(model, "named_steps"):
        # Pipeline structure
        for _, step in model.named_steps.items():
            if hasattr(step, "estimators_"):
                # MultiOutputRegressor
                for est in step.estimators_:
                    patch_estimator(est)
            else:
                patch_estimator(step)
    elif hasattr(model, "estimators_"):
        # Direct MultiOutputRegressor
        for est in model.estimators_:
            patch_estimator(est)
    else:
        # Single estimator
        patch_estimator(model)

    return model


def load_masure_regression_model(reynolds, ml_models_dir):
    """
    Load the trained masure regression Extra Trees model for a given Reynolds number.

    Parameters
    ----------
    reynolds : float
        Supported: 1e6, 5e6, 2e7
    ml_models_dir : str or Path
        Directory containing ET_re1e6.pkl, ET_re5e6.pkl, ET_re2e7.pkl

    Returns
    -------
    model : sklearn-like estimator
        Expects input rows shaped: [t, eta, kappa, delta, lambda, phi, alpha_deg]
        Predicts columns in order: [CD, CL, CM]
    """
    if reynolds in _MASURE_MODEL_CACHE:
        return _MASURE_MODEL_CACHE[reynolds]

    if reynolds == 1e6:
        model_name = "ET_re1e6.pkl"
    elif reynolds == 5e6:
        model_name = "ET_re5e6.pkl"
    elif reynolds == 2e7:
        model_name = "ET_re2e7.pkl"
    else:
        raise ValueError(f"No masure_regression model available for Re={reynolds}")

    ml_models_dir = Path(ml_models_dir)
    model_path = ml_models_dir / model_name

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        model = _patch_sklearn_compatibility(model)

        # Smoke test
        _ = model.predict(np.array([[0.07, 0.20, 0.95, -2.0, 0.65, 0.25, 10.0]]))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Ensure the pickle exists at this location."
        ) from exc
    except Exception as e:
        raise RuntimeError(
            f"Failed to load or validate model '{model_name}'. "
            f"This is often due to scikit-learn version differences. "
            f"Original error: {e}"
        ) from e

    _MASURE_MODEL_CACHE[reynolds] = model
    return model


def predict_masure_polars(params, alpha_range, reynolds, ml_models_dir):
    """
    Predict CL, CD, CM using the masure regression model WITHOUT AirfoilAerodynamics.

    Parameters
    ----------
    params : dict
        {"t":..., "eta":..., "kappa":..., "delta":..., "lambda":..., "phi":...}
    alpha_range : list
        [alpha_min_deg, alpha_max_deg, step_deg]
    reynolds : float
        1e6 / 5e6 / 2e7 (matches available models)
    ml_models_dir : str or Path
        Directory containing the trained .pkl

    Returns
    -------
    alpha_deg : np.ndarray
    CL : np.ndarray
    CD : np.ndarray
    CM : np.ndarray
    """
    alpha_deg = np.arange(
        alpha_range[0], alpha_range[1] + alpha_range[2], alpha_range[2]
    )
    n_alpha = len(alpha_deg)

    X = np.zeros((n_alpha, 7))
    for i, a in enumerate(alpha_deg):
        X[i, :] = [
            params["t"],
            params["eta"],
            params["kappa"],
            params["delta"],
            params["lambda"],
            params["phi"],
            a,  # IMPORTANT: alpha in DEGREES
        ]

    model = load_masure_regression_model(reynolds, ml_models_dir)
    Y = model.predict(X)  # columns: [CD, CL, CM]

    CD = Y[:, 0]
    CL = Y[:, 1]
    CM = Y[:, 2]
    return alpha_deg, CL, CD, CM


# ======================================================================================
#                                     Tutorial
# ======================================================================================


def main():
    """
    Figure layout:
      Row 1 (2 cols):
        (1,1) Detailed LEI airfoil (tube, splines, control points, fillet, etc.)
        (1,2) Clean outline only
      Row 2 (3 cols):
        (2,1) CL vs α
        (2,2) CD vs α
        (2,3) CM vs α
    """

    # -------------------------
    # User-configurable inputs
    # -------------------------
    ml_models_dir = "data/ml_models"  # where ET_re*.pkl live
    Re = 1e6  # choose one of: 1e6, 5e6, 2e7
    alpha_range = [-10, 25, 1]  # [deg_min, deg_max, step]

    # LEI geometry params (non-dimensional by chord)
    params = {
        "t": 0.10,  # tube diameter / chord
        "eta": 0.20,  # camber position
        "kappa": 0.10,  # camber height
        "delta": -2.0,  # reflex angle (deg)
        "lambda": 0.30,  # camber tension
        "phi": 0.65,  # LE curvature
    }

    # -------------------------------------------------------------
    # 1) Predict polars directly from the ML model (no wrapper class)
    # -------------------------------------------------------------
    alpha_deg, CL, CD, CM = predict_masure_polars(
        params=params,
        alpha_range=alpha_range,
        reynolds=Re,
        ml_models_dir=ml_models_dir,
    )

    # -------------------------------------------------------------
    # 2) Build airfoil shapes (detailed parts + clean outline)
    # -------------------------------------------------------------
    # Clean outline
    all_points, profile_name, seam_a = generate_profile(
        t_val=params["t"],
        eta_val=params["eta"],
        kappa_val=params["kappa"],
        delta_val=params["delta"],
        lambda_val=params["lambda"],
        phi_val=params["phi"],
    )

    # Detailed construction pieces (we'll plot them manually on our own axes)
    (
        LE_tube_points,
        P1,
        P11,
        P12,
        LE_points,
        TE_points,
        P2,
        P21,
        P22,
        P3,
        round_TE_points,
        P4,
        P5,
        P51,
        P52,
        TE_lower_points,
        P6,
        P61,
        P62,
        P63,
        fillet_points,
        Origin_LE_tube,
        round_TE_mid,
        seam_a_full,
        *_extras,
    ) = LEI_airfoil(
        tube_size=params["t"],
        c_x=params["eta"],
        c_y=params["kappa"],
        TE_angle=params["delta"],
        TE_cam_tension=params["lambda"],
        LE_tension=params["phi"],
    )

    # -------------------------------------------------------------
    # 3) Plot – 2 rows grid
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(18, 9))
    outer_gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0], hspace=0.3)
    top_gs = outer_gs[0].subgridspec(1, 2, wspace=0.15)  # row 1: 2 columns
    bot_gs = outer_gs[1].subgridspec(1, 3, wspace=0.25)  # row 2: 3 columns

    ax_detail = fig.add_subplot(top_gs[0, 0])  # Row1-Col1: detailed
    ax_outline = fig.add_subplot(top_gs[0, 1])  # Row1-Col2: outline

    ax_cl = fig.add_subplot(bot_gs[0, 0])  # Row2-Col1: CL
    ax_cd = fig.add_subplot(bot_gs[0, 1])  # Row2-Col2: CD
    ax_cm = fig.add_subplot(bot_gs[0, 2])  # Row2-Col3: CM

    # ---- Row 1, Col 1: Detailed construction geometry ----
    # LE full circle (dashed)
    eta_arr = np.linspace(0, 2 * np.pi, 100)
    radius = -np.min(LE_tube_points[:, 1])
    origin_circle = np.array([radius, 0.0])
    x_circ = origin_circle[0] + radius * np.cos(eta_arr)
    y_circ = origin_circle[1] + radius * np.sin(eta_arr)
    ax_detail.plot(
        x_circ, y_circ, "--", linewidth=2, color="#3776ab", label="Circular tube"
    )

    # Front spline + controls
    ax_detail.plot(
        LE_points[:, 0],
        LE_points[:, 1],
        "-",
        color="#ff7f0e",
        linewidth=2,
        label="Front spline",
    )
    ctrl_front = np.array([P1, P11, P12, P2])
    ax_detail.plot(ctrl_front[:, 0], ctrl_front[:, 1], "--", color="gray", linewidth=2)
    ax_detail.scatter(
        ctrl_front[:, 0], ctrl_front[:, 1], s=30, color="#ff7f0e", label="Control front"
    )

    # Rear spline + controls
    ax_detail.plot(
        TE_points[:, 0],
        TE_points[:, 1],
        "-",
        color="#2CA02C",
        linewidth=2,
        label="Rear spline",
    )
    ctrl_rear = np.array([P2, P21, P22, P3])
    ax_detail.plot(ctrl_rear[:, 0], ctrl_rear[:, 1], "--", color="gray", linewidth=2)
    ax_detail.scatter(
        ctrl_rear[:, 0], ctrl_rear[:, 1], s=30, color="#2CA02C", label="Control rear"
    )

    # Fillet + TE lower + round TE
    ax_detail.plot(
        fillet_points[:, 0],
        fillet_points[:, 1],
        "-",
        color="#D62728",
        linewidth=2,
        label="LE fillet",
    )
    ctrl_fillet = np.array([P6, P61, P62, P63])
    ax_detail.plot(
        ctrl_fillet[:, 0], ctrl_fillet[:, 1], "--", color="gray", linewidth=2
    )
    ax_detail.scatter(
        ctrl_fillet[:, 0],
        ctrl_fillet[:, 1],
        s=30,
        color="#D62728",
        label="Control LE fillet",
    )

    ax_detail.plot(
        TE_lower_points[:, 0],
        TE_lower_points[:, 1],
        "-",
        color="teal",
        linewidth=2,
        label="TE lower",
    )
    ctrl_tel = np.array([P5, P51, P52, P4])
    ax_detail.plot(ctrl_tel[:, 0], ctrl_tel[:, 1], "--", color="gray", linewidth=2)
    ax_detail.scatter(
        ctrl_tel[:, 0], ctrl_tel[:, 1], s=30, color="teal", label="Control TE lower"
    )

    ax_detail.plot(
        round_TE_points[:, 0],
        round_TE_points[:, 1],
        "-",
        color="k",
        linewidth=1.5,
        label="Round TE",
    )

    # Key points (stars)
    ax_detail.scatter(
        origin_circle[0],
        origin_circle[1],
        marker="*",
        color="b",
        s=35,
        label="LE tube centre",
    )
    ax_detail.scatter(
        TE_points[-1, 0],
        TE_points[-1, 1],
        marker="*",
        color="r",
        s=35,
        label="TE position",
    )
    ax_detail.scatter(
        LE_points[0, 0],
        LE_points[0, 1],
        marker="*",
        color="g",
        s=35,
        label="Tube–canopy intersection",
    )
    ax_detail.scatter(
        LE_points[-1, 0],
        LE_points[-1, 1],
        marker="*",
        color="k",
        s=35,
        label="Max. camber position",
    )

    # Axis styling
    ax_detail.set_title("Detailed LEI airfoil (construction geometry)")
    ax_detail.set_xlabel("x / c")
    ax_detail.set_ylabel("y / c")
    ax_detail.set_aspect("equal", "box")
    ax_detail.grid(True, linestyle="--", alpha=0.3)

    # Bounds similar to helper
    y_min = min(np.min(LE_tube_points[:, 1]), np.min(all_points[:, 1]))
    y_max = max(np.max(LE_points[:, 1]), np.max(all_points[:, 1]))
    ax_detail.set_xlim(-0.02, 1.02)
    ax_detail.set_ylim(1.5 * y_min, 1.2 * y_max)

    ax_detail.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fontsize=8,
        ncol=4,
        frameon=False,
    )

    # ---- Row 1, Col 2: Clean outline ----
    ax_outline.plot(all_points[:, 0], all_points[:, 1], linewidth=2.0)
    ax_outline.set_aspect("equal", "box")
    ax_outline.set_xlim(-0.02, 1.02)
    ymin2, ymax2 = float(all_points[:, 1].min()), float(all_points[:, 1].max())
    pad2 = 0.1 * (ymax2 - ymin2 + 1e-6)
    ax_outline.set_ylim(ymin2 - pad2, ymax2 + pad2)
    ax_outline.set_title("Clean airfoil outline")
    ax_outline.set_xlabel("x / c")
    ax_outline.set_ylabel("y / c")
    ax_outline.grid(True, linestyle="--", alpha=0.3)

    # ---- Row 2: Polars (from ML) ----
    ax_cl.plot(alpha_deg, CL, linewidth=1.8)
    ax_cl.set_title("CL vs. α (masure_regression)")
    ax_cl.set_xlabel(r"$\alpha$ [deg]")
    ax_cl.set_ylabel("CL")
    ax_cl.grid(True, linestyle="--", alpha=0.4)

    ax_cd.plot(alpha_deg, CD, linewidth=1.8)
    ax_cd.set_title("CD vs. α (masure_regression)")
    ax_cd.set_xlabel(r"$\alpha$ [deg]")
    ax_cd.set_ylabel("CD")
    ax_cd.grid(True, linestyle="--", alpha=0.4)

    ax_cm.plot(alpha_deg, CM, linewidth=1.8)
    ax_cm.set_title("CM vs. α (masure_regression)")
    ax_cm.set_xlabel(r"$\alpha$ [deg]")
    ax_cm.set_ylabel("CM")
    ax_cm.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        f"LEI Airfoil (Detailed + Outline) and Masure Regression Polars (Re = {Re:.0e})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
