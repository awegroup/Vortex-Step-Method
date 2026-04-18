import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# --- Core geometry helpers (NumPy) ---


def triangle_setup(H: float, B: float):
    """
    Initial symmetric triangle with apex at (0,0).
    Top segment initially horizontal at height H with length B.
    Returns:
        L0  : initial slanted length (both sides)
        uL  : unit direction (left)
        uR  : unit direction (right)
        P0L : initial left top endpoint
        P0R : initial right top endpoint
    """
    P0L = np.array([-B / 2.0, H], dtype=float)
    P0R = np.array([+B / 2.0, H], dtype=float)
    L0 = np.linalg.norm(P0R)  # same as norm of P0L
    uL = P0L / L0
    uR = P0R / L0
    return L0, uL, uR, P0L, P0R


def endpoints_after_delta(L0, uL, uR, delta):
    """
    Make left slanted length L0+delta and right slanted length L0-delta.
    Directions stay fixed.
    """
    PL = (L0 + delta) * uL
    PR = (L0 - delta) * uR
    return PL, PR


def top_segment_angle_deg(PL, PR):
    """
    Angle of top segment PR-PL w.r.t +x, in degrees.
    Positive = counterclockwise.
    """
    d = PR - PL
    return np.degrees(np.arctan2(d[1], d[0]))


def angle_from_delta_deg(H, B, delta):
    """
    Closed-form angle for speed (no endpoints needed).
    """
    L0 = np.hypot(B / 2.0, H)
    # dy = -2 * delta * (H/L0), dx = B
    return np.degrees(-np.arctan2(2.0 * delta * H / L0, B))


def delta_for_target_angle_deg(H, B, target_angle_deg):
    """
    Inverse: delta that produces a desired tilt.
    """
    L0 = np.hypot(B / 2.0, H)
    ang = np.radians(target_angle_deg)
    return -np.tan(ang) * (B * L0) / (2.0 * H)


def sideslip_from_theta_aoa_deg(theta_deg: float, aoa_deg: float) -> float:
    """
    Compute sideslip beta [deg] given:
      theta_deg : spanwise tilt in the plotted plane (angle of top segment)
      aoa_deg   : angle of attack (inflow angle out of the plotted plane)
    Uses: beta = asin( cos(aoa) * sin(theta) ).
    """
    th = np.radians(theta_deg)
    aoa = np.radians(aoa_deg)
    beta = np.arcsin(np.sin(aoa) * np.sin(th))
    return np.degrees(beta)


def sideslip_from_delta_deg(H: float, B: float, delta: float, aoa_deg: float) -> float:
    """
    Convenience: compute sideslip directly from (H, B, delta, aoa).
    """
    # Reuse your closed-form top-segment tilt
    L0 = np.hypot(B / 2.0, H)
    theta_deg = np.degrees(-np.arctan2(2.0 * delta * H / L0, B))
    return sideslip_from_theta_aoa_deg(theta_deg, aoa_deg)


# --- Interactive demo ---

if __name__ == "__main__":
    # Inputs (edit these as you like)
    H = 12  # height of initial horizontal top segment
    B = 8.3  # length of the top segment

    # Build initial geometry
    L0, uL, uR, P0L, P0R = triangle_setup(H, B)

    # Create figure & main axes (one chart per figure as requested)
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.22)  # space for slider/buttons

    # Initial delta
    delta0 = 0.0
    PL, PR = endpoints_after_delta(L0, uL, uR, delta0)

    # Plot initial geometry
    # Initial top (horizontal)
    (top0_line,) = ax.plot([P0L[0], P0R[0]], [P0L[1], P0R[1]], label="Top (initial)")
    # New top after delta
    (top_line,) = ax.plot([PL[0], PR[0]], [PL[1], PR[1]], label="Top (current)")
    # Slanted segments
    (left_line,) = ax.plot(
        [0.0, PL[0]], [0.0, PL[1]], linestyle="--", label="Left segment"
    )
    (rght_line,) = ax.plot(
        [0.0, PR[0]], [0.0, PR[1]], linestyle="--", label="Right segment"
    )
    # Key points
    pts = ax.scatter(
        [0, P0L[0], P0R[0], PL[0], PR[0]], [0, P0L[1], P0R[1], PL[1], PR[1]], s=25
    )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    angle0 = top_segment_angle_deg(PL, PR)
    ax.set_title(f"Tilt angle = {angle0:+.3f}° (delta={delta0:+.3f})")
    ax.legend(loc="best")

    # Set reasonable view limits so the geometry stays in view as delta changes
    pad = max(B, H) * 0.75 + 1.0
    ax.set_xlim(-B / 2 - pad, B / 2 + pad)
    ax.set_ylim(-pad * 0.2, H + pad)

    # Slider for delta
    ax_delta = plt.axes([0.1, 0.12, 0.8, 0.03])
    max_delta = 0.9 * L0  # keep lengths positive
    s_delta = Slider(
        ax=ax_delta, label="delta", valmin=-max_delta, valmax=max_delta, valinit=delta0
    )

    # Optional button to reset delta
    ax_reset = plt.axes([0.1, 0.06, 0.12, 0.05])
    btn_reset = Button(ax_reset, "Reset")

    # Optional button to set delta for a target angle (example: +5°)
    ax_setang = plt.axes([0.24, 0.06, 0.22, 0.05])
    btn_setang = Button(ax_setang, "Set angle +5°")
    # ... after your existing slider setup
    ax_aoa = plt.axes([0.1, 0.17, 0.8, 0.03])
    s_aoa = Slider(ax=ax_aoa, label="AoA (deg)", valmin=-20.0, valmax=20.0, valinit=7.0)

    def update(val):
        d = s_delta.val
        aoa = s_aoa.val
        PL_new, PR_new = endpoints_after_delta(L0, uL, uR, d)
        top_line.set_data([PL_new[0], PR_new[0]], [PL_new[1], PR_new[1]])
        left_line.set_data([0.0, PL_new[0]], [0.0, PL_new[1]])
        rght_line.set_data([0.0, PR_new[0]], [0.0, PR_new[1]])

        pts.set_offsets(
            np.array(
                [
                    [0, 0],
                    [P0L[0], P0L[1]],
                    [P0R[0], P0R[1]],
                    [PL_new[0], PL_new[1]],
                    [PR_new[0], PR_new[1]],
                ]
            )
        )

        theta = top_segment_angle_deg(PL_new, PR_new)
        beta = sideslip_from_theta_aoa_deg(theta, aoa)
        ax.set_title(
            f"Tilt θ = {theta:+.3f}°, AoA α = {aoa:+.3f}°, Sideslip β = {beta:+.3f}° (delta={d:+.3f})"
        )
        fig.canvas.draw_idle()

    s_aoa.on_changed(update)

    def on_reset(event):
        s_delta.reset()

    def on_set_angle(event, target_deg=5.0):
        d = delta_for_target_angle_deg(H, B, target_deg)
        # Clip to slider range just in case
        d = np.clip(d, s_delta.valmin, s_delta.valmax)
        s_delta.set_val(d)

    s_delta.on_changed(update)
    btn_reset.on_clicked(on_reset)
    btn_setang.on_clicked(lambda e: on_set_angle(e, 5.0))

    plt.show()
