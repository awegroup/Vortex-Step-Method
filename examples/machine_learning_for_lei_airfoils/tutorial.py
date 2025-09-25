# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from VSM.core.AirfoilAerodynamics import AirfoilAerodynamics

# Import the LEI utilities
from utils_masure_regression_lei_parametric import (
    generate_profile,  # (all_points, profile_name, seam_a)
    LEI_airfoil,  # low-level geometry function (many outputs)
)


## USER INPUT
ml_models_dir = "data/ml_models"


def main(ml_models_dir):
    """
    Layout:
      Row 1 (2 cols):
        (1,1) Detailed LEI airfoil (tube, splines, control points, fillet, etc.)
        (1,2) Clean outline only
      Row 2 (3 cols):
        (2,1) CL vs α
        (2,2) CD vs α
        (2,3) CM vs α
    """

    # -------------------------------------------------------------
    # Parameters & alpha sweep
    # -------------------------------------------------------------
    params = {
        "t": 0.1,
        "eta": 0.2,
        "kappa": 0.1,
        "delta": -2.0,
        "lambda": 0.3,
        "phi": 0.65,
    }
    alpha_range = [-10, 25, 1]
    Re = 1e6

    # -------------------------------------------------------------
    # Run the masure_regression model
    # -------------------------------------------------------------
    aero = AirfoilAerodynamics.from_yaml_entry(
        airfoil_type="masure_regression",
        airfoil_params=params,
        alpha_range=alpha_range,
        reynolds=Re,
        ml_models_dir=ml_models_dir,
    )

    # Extract arrays for polars
    alpha_deg = np.rad2deg(aero.alpha)
    CL, CD, CM = aero.CL, aero.CD, aero.CM

    # -------------------------------------------------------------
    # Build airfoil shapes (detailed pieces + clean outline)
    # -------------------------------------------------------------
    # Clean outline (for the right panel of row 1)
    all_points, profile_name, seam_a = generate_profile(
        t_val=params["t"],
        eta_val=params["eta"],
        kappa_val=params["kappa"],
        delta_val=params["delta"],
        lambda_val=params["lambda"],
        phi_val=params["phi"],
    )

    # Detailed parts (to mimic plot_airfoil in our own ax)
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
    # Figure layout: 2 rows (top has 2 cols, bottom has 3 cols)
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

    # -------------------------------------------------------------
    # Row 1, Col 1: Detailed airfoil (mimics plot_airfoil)
    # -------------------------------------------------------------
    # LE full circle (dashed)
    eta = np.linspace(0, 2 * np.pi, 100)
    radius = -np.min(LE_tube_points[:, 1])
    origin_circle = np.array([radius, 0.0])
    x_circ = origin_circle[0] + radius * np.cos(eta)
    y_circ = origin_circle[1] + radius * np.sin(eta)
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

    # Fillet + TE lower + round TE (show all details)
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

    # Nice bounds similar to the original helper
    y_min = min(np.min(LE_tube_points[:, 1]), np.min(all_points[:, 1]))
    y_max = max(np.max(LE_points[:, 1]), np.max(all_points[:, 1]))
    ax_detail.set_xlim(-0.02, 1.02)
    ax_detail.set_ylim(1.5 * y_min, 1.2 * y_max)
    # Place the legend outside the plot (to the right)
    ax_detail.legend(
        loc="center left",
        bbox_to_anchor=(-0.1, -0.6),
        fontsize=8,
        ncol=5,
        frameon=False,
        borderaxespad=0.0,
    )

    # -------------------------------------------------------------
    # Row 1, Col 2: Clean outline
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # Row 2: Polars (CL/CD/CM vs α)
    # -------------------------------------------------------------
    ax_cl.plot(alpha_deg, CL, linewidth=1.8)
    ax_cl.set_title("CL vs. α")
    ax_cl.set_xlabel(r"$\alpha$ [deg]")
    ax_cl.set_ylabel("CL")
    ax_cl.grid(True, linestyle="--", alpha=0.4)

    ax_cd.plot(alpha_deg, CD, linewidth=1.8)
    ax_cd.set_title("CD vs. α")
    ax_cd.set_xlabel(r"$\alpha$ [deg]")
    ax_cd.set_ylabel("CD")
    ax_cd.grid(True, linestyle="--", alpha=0.4)

    ax_cm.plot(alpha_deg, CM, linewidth=1.8)
    ax_cm.set_title("CM vs. α")
    ax_cm.set_xlabel(r"$\alpha$ [deg]")
    ax_cm.set_ylabel("CM")
    ax_cm.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "LEI Airfoil (Detailed + Outline) and Masure Regression Polars (Re = 1e6)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main(ml_models_dir)
