"""Figure 16 of docs/figures: why two line vortices meeting at an angle have
no finite mutual velocity and two chordwise sheets do (report Section 4.2).

Line 1 runs along -y from the kink, line 2 starts at the kink turned by the
angle theta out of the plane. P sits on line 1 (at its quarter chord) at
distance r from the kink. Plotted: the chordwise velocity at P induced by a
unit-strength line 2 (single filament at the quarter chord) and by the same
circulation spread over the chord with the thin-airfoil strips, against r.

Usage (from the repository root, seconds)::

    python docs/validation/kink_figure.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from VSM.core.utils import thin_airfoil_strips

FIG = Path(__file__).resolve().parents[1] / "figures"


def seg(a, b, p):
    r1, r2 = p - a, p - b
    cr = np.cross(r1, r2)
    n = np.dot(cr, cr)
    if n < 1e-24:
        return np.zeros(3)
    return cr / (4 * np.pi * n) * np.dot(b - a, r1 / np.linalg.norm(r1) - r2 / np.linalg.norm(r2))


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9})
    x, w = thin_airfoil_strips(24)
    L = 50.0
    rs = np.logspace(-3, 1, 80)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for theta_deg, col in ((2.0, "tab:blue"), (5.0, "tab:red"), (10.0, "tab:green")):
        t = np.radians(theta_deg)
        e2 = np.array([0.0, np.cos(t), np.sin(t)])
        vf, vs = [], []
        for r in rs:
            P = np.array([0.25, -r, 0.0])
            a0 = np.array([0.25, 0.0, 0.0])
            vf.append(abs(seg(a0, a0 + L * e2, P)[0]))
            vs.append(abs(sum(w[q] * seg(np.array([x[q], 0, 0]), np.array([x[q], 0, 0]) + L * e2, P)[0] for q in range(24))))
        ax[0].loglog(rs, vf, "-", color=col, label=f"line, kink {theta_deg:.0f} deg")
        ax[0].loglog(rs, vs, "--", color=col, label=f"sheet, kink {theta_deg:.0f} deg")
        if theta_deg == 5.0:
            ax[1].loglog(rs, vf, "r-", label="line, computed")
            ax[1].loglog(rs, (1 - np.cos(t)) / (4 * np.pi * rs * np.sin(t)), "k:", label="line, (1 - cos theta) / (4 pi r sin theta)")
            ax[1].loglog(rs, vs, "r--", label="sheet, computed")
    ax[0].axvline(0.15, color="0.6", ls=":")
    ax[0].text(0.16, 2.0, "0.15 c", color="0.4")
    ax[0].set_xlabel("distance r of P from the kink, along line 1 (chord = 1)")
    ax[0].set_ylabel("|chordwise velocity at P| per unit circulation of line 2")
    ax[0].set_title("(a) two vortices meeting at a kink: line 2 seen from a point on line 1")
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].set_xlabel("r (chord = 1), kink 5 deg")
    ax[1].set_ylabel("|chordwise velocity at P|")
    ax[1].set_title("(b) kink 5 deg: line, small-angle formula, sheet")
    ax[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = FIG / "16_kink_line_vs_sheet.png"
    fig.savefig(out, dpi=150)
    print("figure 16 written to", out)


if __name__ == "__main__":
    main()
