"""Figure 14 of docs/figures: what "unrolling the wing onto a panel's tangent"
means in the curved bound-vortex regularisation (report Section 4, eq. 6).

For an evaluation panel i the reference wing is built on panel i's own
tangent line through its aerodynamic centre: every node keeps its arc-length
coordinate along the bound line, the legs are perpendicular to the line in
panel i's chord direction with each panel's own chord, and the wake keeps its
direction. The reference is different for every panel. The figure shows it
for two panels of the arc wing (3-D and front view), for a panel of the
kinked swept wing (top view), and how far the unrolled nodes move from the
actual ones as a function of the distance from the evaluation panel (the
near field cancels in the difference, the far field is the curvature effect).

Usage (from the repository root, seconds)::

    python docs/validation/unrolling_figure.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

import curved_bound_vortex_validation as cbv

logging.disable(logging.CRITICAL)
FIG = Path(__file__).resolve().parents[1] / "figures"


def geometry(body):
    P = body.panels
    bound = np.array([p.bound_point_1 for p in P] + [P[-1].bound_point_2])
    te = np.array([p.TE_point_1 for p in P] + [P[-1].TE_point_2])
    ac = np.array([p.aerodynamic_center for p in P])
    chord_dir = np.array([p.y_airf for p in P])
    chords = np.array([p.chord for p in P])
    return bound, te, ac, chord_dir, chords


def unrolled(bound, te, ac, chord_dir, chords, i, ac_fraction=0.25):
    """Nodes and leg ends of the reference wing of panel i (as the kernel builds it)."""
    seg = bound[1:] - bound[:-1]
    L = np.linalg.norm(seg, axis=1)
    s = np.concatenate(([0.0], np.cumsum(L)))
    t = seg[i] / L[i]
    s_P = 0.5 * (s[i] + s[i + 1])
    nodes = ac[i][None, :] + (s - s_P)[:, None] * t[None, :]
    # leg of node k: panel i's chord direction, the adjacent panel's chord
    c_node = np.concatenate(([chords[0]], 0.5 * (chords[1:] + chords[:-1]), [chords[-1]]))
    ends = nodes + (1.0 - ac_fraction) * c_node[:, None] * chord_dir[i][None, :]
    return nodes, ends, s - s_P


def draw_wing(ax, bound, te, ij=None, color="k", lw=1.2, alpha=1.0, label=None, legs=True):
    if ij is None:
        ax.plot(bound[:, 0], bound[:, 1], bound[:, 2], color=color, lw=lw, alpha=alpha, label=label)
        if legs:
            for a, b in zip(bound, te):
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, lw=0.6, alpha=alpha)
            ax.plot(te[:, 0], te[:, 1], te[:, 2], color=color, lw=0.6, alpha=alpha, ls=":")
    else:
        u, v = ij
        ax.plot(bound[:, u], bound[:, v], color=color, lw=lw, alpha=alpha, label=label)
        if legs:
            for a, b in zip(bound, te):
                ax.plot([a[u], b[u]], [a[v], b[v]], color=color, lw=0.6, alpha=alpha)
            ax.plot(te[:, u], te[:, v], color=color, lw=0.6, alpha=alpha, ls=":")


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9})
    fig = plt.figure(figsize=(15, 9.5))
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    axf = fig.add_subplot(2, 2, 2)
    axt = fig.add_subplot(2, 2, 3)
    axd = fig.add_subplot(2, 2, 4)

    # ---- arc wing, two evaluation panels ----
    body = cbv.arc(20)
    bound, te, ac, cdir, chords = geometry(body)
    picks = [(10, "tab:blue", "panel 10 (mid-span)"), (2, "tab:red", "panel 2 (near the tip)")]
    draw_wing(ax3, bound, te, color="k", label="actual wing (bound line, legs, trailing edge)")
    draw_wing(axf, bound, te, ij=(1, 2), color="k", label="actual wing")
    for i, col, lab in picks:
        nodes, ends, _ = unrolled(bound, te, ac, cdir, chords, i)
        draw_wing(ax3, nodes, ends, color=col, lw=1.4, alpha=0.9, label=f"unrolled reference of {lab}")
        draw_wing(axf, nodes, ends, ij=(1, 2), color=col, lw=1.4, alpha=0.9, label=f"unrolled reference of {lab}")
        ax3.plot([ac[i, 0]], [ac[i, 1]], [ac[i, 2]], "*", color=col, ms=12)
        axf.plot([ac[i, 1]], [ac[i, 2]], "*", color=col, ms=12)
        axf.plot(nodes[:, 1], nodes[:, 2], "o", color=col, ms=2.5)
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    ax3.set_title("(a) arc wing: the reference wing is straight, tangent at the evaluation panel")
    ax3.view_init(elev=22, azim=-60)
    ax3.set_box_aspect((1.0, 3.0, 1.6))
    ax3.legend(fontsize=7, loc="upper left", frameon=False)
    axf.set_aspect("equal"); axf.set_xlabel("y"); axf.set_ylabel("z")
    axf.set_title("(b) front view: nodes keep their arc-length distance from the evaluation panel")
    axf.legend(fontsize=7, frameon=False, loc="lower center")

    # ---- kinked swept wing, one evaluation panel on the right half ----
    body = cbv.swept(20)
    bound, te, ac, cdir, chords = geometry(body)
    i = 5
    nodes, ends, _ = unrolled(bound, te, ac, cdir, chords, i)
    draw_wing(axt, bound, te, ij=(1, 0), color="k", label="actual wing (ribs along the flow)")
    draw_wing(axt, nodes, ends, ij=(1, 0), color="tab:red", lw=1.4, label=f"unrolled reference of panel {i}: legs along its chord direction")
    axt.plot([ac[i, 1]], [ac[i, 0]], "*", color="tab:red", ms=12)
    axt.plot(nodes[:, 1], nodes[:, 0], "o", color="tab:red", ms=2.5)
    axt.invert_yaxis(); axt.set_aspect("equal")
    axt.set_xlabel("y"); axt.set_ylabel("x (flow downwards)")
    axt.set_title("(c) kinked swept wing, top view: the other half is laid along the panel's own line")
    axt.legend(fontsize=7, frameon=False, loc="lower left")

    # ---- how far the unrolled nodes sit from the actual ones ----
    body = cbv.arc(40)
    bound, te, ac, cdir, chords = geometry(body)
    for i, col, lab in ((20, "tab:blue", "panel 20 (mid-span)"), (4, "tab:red", "panel 4 (near the tip)")):
        nodes, ends, ds = unrolled(bound, te, ac, cdir, chords, i)
        gap = np.linalg.norm(nodes - bound, axis=1)
        axd.plot(np.abs(ds), gap, "o-", color=col, ms=3, label=f"arc wing, {lab}")
    body = cbv.v3_inviscid(40)
    bound, te, ac, cdir, chords = geometry(body)
    for i, col, lab in ((20, "tab:green", "V3, panel 20 (mid-span)"), (3, "tab:orange", "V3, panel 3 (tip)")):
        nodes, ends, ds = unrolled(bound, te, ac, cdir, chords, i)
        gap = np.linalg.norm(nodes - bound, axis=1)
        axd.plot(np.abs(ds), gap, "s--", color=col, ms=3, label=lab)
    axd.axvspan(0, 0.75, color="0.9", label="within one leg length (0.75 c)")
    axd.set_xlabel("arc-length distance of node from the evaluation panel")
    axd.set_ylabel("distance between actual and unrolled node")
    axd.set_title("(d) the near field coincides and cancels; the far field is the curvature effect")
    axd.legend(fontsize=7, frameon=False)
    fig.suptitle("Unrolling the wing onto the evaluation panel's tangent: one reference wing per panel", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "14_unrolling_per_panel.png"
    fig.savefig(out, dpi=150)
    print("figure 14 written to", out)


if __name__ == "__main__":
    main()
