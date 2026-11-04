"""Figure 15 of docs/figures: the chordwise spreading of the bound vorticity in
the curved bound-vortex regularisation (report Section 4, eqs. 4-6).

(a) Three neighbouring panels of the arc wing with their bound segments
    replaced by the 24 thin-airfoil strips along the chord (line weight and
    shade proportional to the strip weight), each strip with its own legs on
    the node's leg line.
(b) The same panels on the unrolled (straight) reference line of the middle
    panel, with the same strips.
(c) Chordwise velocity induced at the quarter-chord point of the middle panel
    by ONE neighbouring bound segment of unit strength, against its arc-length
    distance, for the single filament and for the spread sheet: far away the
    two coincide, within a chord the sheet stays finite where the filament
    grows.
(d) The running sum of those contributions from the far field inwards, for
    several panel counts: the filament sum keeps rising as the nearest
    neighbours get closer (the logarithm), the sheet sum settles.

Usage (from the repository root, seconds)::

    python docs/validation/chord_spread_figure.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

import curved_bound_vortex_validation as cbv
from VSM.core.utils import thin_airfoil_strips

logging.disable(logging.CRITICAL)
FIG = Path(__file__).resolve().parents[1] / "figures"
AC = 0.25


def seg_velocity(a, b, p):
    """Unit-strength straight segment a->b at p (Biot-Savart), zero on the line."""
    r1, r2 = p - a, p - b
    cr = np.cross(r1, r2)
    n = np.dot(cr, cr)
    if n < 1e-20:
        return np.zeros(3)
    return cr / (4 * np.pi * n) * np.dot(b - a, r1 / np.linalg.norm(r1) - r2 / np.linalg.norm(r2))


def geometry(body):
    P = body.panels
    bound = np.array([p.bound_point_1 for p in P] + [P[-1].bound_point_2])
    te = np.array([p.TE_point_1 for p in P] + [P[-1].TE_point_2])
    ac = np.array([p.aerodynamic_center for p in P])
    chord_dir = np.array([p.y_airf for p in P])
    chords = np.array([p.chord for p in P])
    return bound, te, ac, chord_dir, chords


def strips_of_panel(b1, b2, t1, t2, x, w):
    """Strip segments of one panel: stations along each node's leg line."""
    f = (x - AC) / (1.0 - AC)
    a = b1[None, :] + f[:, None] * (t1 - b1)[None, :]
    b = b2[None, :] + f[:, None] * (t2 - b2)[None, :]
    return a, b, w


def unrolled(bound, te, ac, chord_dir, chords, i):
    seg = bound[1:] - bound[:-1]
    L = np.linalg.norm(seg, axis=1)
    s = np.concatenate(([0.0], np.cumsum(L)))
    t = seg[i] / L[i]
    s_P = 0.5 * (s[i] + s[i + 1])
    nodes = ac[i][None, :] + (s - s_P)[:, None] * t[None, :]
    c_node = np.concatenate(([chords[0]], 0.5 * (chords[1:] + chords[:-1]), [chords[-1]]))
    ends = nodes + (1.0 - AC) * c_node[:, None] * chord_dir[i][None, :]
    return nodes, ends, s - s_P


def draw_strips(ax, bound, te, panels, x, w, ij, color, show_legs=True):
    u, v = ij
    for j in panels:
        a, b, ww = strips_of_panel(bound[j], bound[j + 1], te[j], te[j + 1], x, w)
        for k in range(len(ww)):
            lw = 0.3 + 6.0 * ww[k]
            ax.plot([a[k, u], b[k, u]], [a[k, v], b[k, v]], color=color, lw=lw, alpha=0.35 + 0.65 * ww[k] / ww.max(), solid_capstyle="butt")
        if show_legs:
            for k in range(0, len(ww), 6):
                ax.plot([a[k, u], te[j, u]], [a[k, v], te[j, v]], color=color, lw=0.4, alpha=0.5)
                ax.plot([b[k, u], te[j + 1, u]], [b[k, v], te[j + 1, v]], color=color, lw=0.4, alpha=0.5)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 9})
    fig = plt.figure(figsize=(14, 10))
    ax = np.empty((2, 2), dtype=object)
    ax[0, 0] = fig.add_subplot(2, 2, 1)
    ax[0, 1] = fig.add_subplot(2, 2, 2, projection="3d")
    ax[1, 0] = fig.add_subplot(2, 2, 3)
    ax[1, 1] = fig.add_subplot(2, 2, 4)
    x, w = thin_airfoil_strips(24)

    # ---- (a) real panels near the tip of the arc wing, front-ish view (y, z) with the chord along x shown by projection
    body = cbv.arc(16)
    bound, te, ac, cdir, chords = geometry(body)
    i = 3
    panels = [i - 1, i, i + 1]
    A = ax[0, 0]
    # top view (x along the page downwards, y across): shows the chordwise strips
    for j in range(len(bound) - 1):
        A.plot([bound[j, 1], bound[j + 1, 1]], [bound[j, 0], bound[j + 1, 0]], color="0.6", lw=1)
    for j in panels:
        A.plot([bound[j, 1], bound[j + 1, 1]], [bound[j, 0], bound[j + 1, 0]], color="k", lw=2.2)
        for k in (j, j + 1):
            A.plot([bound[k, 1], te[k, 1]], [bound[k, 0], te[k, 0]], color="k", lw=0.8)
    draw_strips(A, bound, te, panels, x, w, (1, 0), "tab:red")
    A.plot([ac[i, 1]], [ac[i, 0]], "*", color="tab:blue", ms=13, zorder=5)
    A.annotate("P_i (quarter chord of panel i)", (ac[i, 1], ac[i, 0]), textcoords="offset points", xytext=(10, -12), fontsize=8, color="tab:blue")
    A.set_xlim(bound[i - 1, 1] - 0.15, bound[i + 2, 1] + 0.15)
    A.set_ylim(1.08, -0.08)
    A.set_aspect("equal")
    A.set_xlabel("y"); A.set_ylabel("x (chordwise, flow downwards)")
    A.set_title("(a) three real panels of the arc wing, top view: the bound vortex spread into 24 strips")

    # ---- (b) 3-D view: the real strips (red) and the unrolled copies (green) of the same panels
    B = ax[0, 1]
    nodes, ends, _ = unrolled(bound, te, ac, cdir, chords, i)
    B.plot(bound[:, 0], bound[:, 1], bound[:, 2], color="0.6", lw=1)
    B.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], color="0.6", lw=1, ls=":")
    for pts, tes, col in ((bound, te, "tab:red"), (nodes, ends, "tab:green")):
        for j in panels:
            B.plot([pts[j, 0], pts[j + 1, 0]], [pts[j, 1], pts[j + 1, 1]], [pts[j, 2], pts[j + 1, 2]], color="k", lw=2)
            a, b, ww = strips_of_panel(pts[j], pts[j + 1], tes[j], tes[j + 1], x, w)
            for k in range(len(ww)):
                B.plot([a[k, 0], b[k, 0]], [a[k, 1], b[k, 1]], [a[k, 2], b[k, 2]], color=col, lw=0.3 + 5.0 * ww[k], alpha=0.35 + 0.65 * ww[k] / ww.max())
            for k in (j, j + 1):
                B.plot([pts[k, 0], tes[k, 0]], [pts[k, 1], tes[k, 1]], [pts[k, 2], tes[k, 2]], color="k", lw=0.6)
    B.plot([ac[i, 0]], [ac[i, 1]], [ac[i, 2]], "*", color="tab:blue", ms=12)
    B.set_xlabel("x"); B.set_ylabel("y"); B.set_zlabel("z")
    pts_all = np.vstack([bound[i - 1:i + 3], te[i - 1:i + 3], nodes[i - 1:i + 3], ends[i - 1:i + 3]])
    lo, hi = pts_all.min(axis=0) - 0.1, pts_all.max(axis=0) + 0.1
    B.set_xlim(lo[0], hi[0]); B.set_ylim(lo[1], hi[1]); B.set_zlim(lo[2], hi[2])
    B.set_box_aspect(tuple(hi - lo))
    B.view_init(elev=25, azim=-50)
    B.set_title("(b) the same strips on the real (red, curved) and on the unrolled (green, straight) panels")

    # ---- (c) one neighbouring segment's chordwise velocity at P_i against distance: filament vs sheet
    C = ax[1, 0]
    body = cbv.arc(320)
    bound, te, ac, cdir, chords = geometry(body)
    i = 160
    p = ac[i]
    seg = bound[1:] - bound[:-1]
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(seg, axis=1))))
    s_mid = 0.5 * (s[:-1] + s[1:])
    d = np.abs(s_mid - s_mid[i])
    v_fil, v_sheet = np.zeros(len(seg)), np.zeros(len(seg))
    for j in range(len(seg)):
        if j == i:
            continue
        v_fil[j] = seg_velocity(bound[j], bound[j + 1], p)[0]
        a, b, ww = strips_of_panel(bound[j], bound[j + 1], te[j], te[j + 1], x, w)
        v_sheet[j] = sum(ww[k] * seg_velocity(a[k], b[k], p)[0] for k in range(len(ww)))
    o = np.argsort(d)
    m = (d > 0)
    C.loglog(d[m], np.abs(v_fil[m]), ".", color="tab:red", ms=4, label="single filament at the quarter chord")
    C.loglog(d[m], np.abs(v_sheet[m]), ".", color="tab:green", ms=4, label="chord-spread sheet (24 strips)")
    C.axvline(0.15, color="0.5", ls="--", lw=0.8)
    C.text(0.16, np.abs(v_fil[m]).max() * 0.5, "effective cut-off 0.15 c", fontsize=8, color="0.3")
    C.axvline(1.0, color="0.8", ls=":", lw=0.8); C.text(1.05, np.abs(v_fil[m]).max() * 0.5, "one chord", fontsize=8, color="0.5")
    C.set_xlabel("arc-length distance of the neighbouring segment from P_i (chord = 1)")
    C.set_ylabel("|chordwise velocity at P_i| per unit circulation, one segment")
    C.set_title("(c) what one neighbour induces: the sheet is the filament beyond a chord, finite within it")
    C.legend(fontsize=8, frameon=False)

    # ---- (d) running sum from the far field inwards, for several panel counts
    D = ax[1, 1]
    for N, col in ((40, "0.75"), (80, "0.55"), (160, "0.35"), (320, "0.1")):
        body = cbv.arc(N)
        bound, te, ac, cdir, chords = geometry(body)
        i = N // 2
        p = ac[i]
        seg = bound[1:] - bound[:-1]
        s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(seg, axis=1))))
        s_mid = 0.5 * (s[:-1] + s[1:])
        d = np.abs(s_mid - s_mid[i])
        vf, vs = np.zeros(N), np.zeros(N)
        for j in range(N):
            if j == i:
                continue
            vf[j] = seg_velocity(bound[j], bound[j + 1], p)[0]
            a, b, ww = strips_of_panel(bound[j], bound[j + 1], te[j], te[j + 1], x, w)
            vs[j] = sum(ww[k] * seg_velocity(a[k], b[k], p)[0] for k in range(len(ww)))
        o = np.argsort(-d)  # far to near
        dd = d[o][d[o] > 0]
        D.semilogx(dd, np.cumsum(vf[o][d[o] > 0]), "-", color=col, lw=1.2, label=f"filaments, {N} panels")
        D.semilogx(dd, np.cumsum(vs[o][d[o] > 0]), "--", color="tab:green", lw=1.2, alpha=min(1.0, 0.35 + 0.08 * (N // 40)), label=f"sheets, {N} panels")
    D.invert_xaxis()
    D.set_xlabel("distance from P_i of the segments summed so far (far field on the left)")
    D.set_ylabel("running sum of the chordwise velocity at P_i")
    D.set_title("(d) summing the neighbours from far to near: filaments keep growing, sheets settle")
    D.legend(fontsize=7, frameon=False, ncol=2)

    handles = [
        Line2D([], [], color="k", lw=2.2, label="bound vortex (quarter-chord line)"),
        Line2D([], [], color="k", lw=0.8, label="legs to the trailing edge"),
        Line2D([], [], color="tab:red", lw=2, label="strips of the real panels (shade = weight)"),
        Line2D([], [], color="tab:green", lw=2, label="strips of the unrolled panels"),
        Line2D([], [], color="tab:blue", marker="*", ls="", ms=10, label="evaluation point P_i"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Chordwise spreading of the bound vorticity: the sheet that replaces the line", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    out = FIG / "15_chord_spread.png"
    fig.savefig(out, dpi=150)
    print("figure 15 written to", out)


if __name__ == "__main__":
    main()
