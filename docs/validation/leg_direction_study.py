"""Attached-trailed leg direction: section ribs versus perpendicular to the
quarter-chord line or along the flow (``BodyAerodynamics(leg_direction=...)``).

Gaunaa, Li & Pirrung (TORQUE 2026, Sec. 3) run the attached part of the
trailed vortex along the local chordwise direction of the Crossflow
Principle, perpendicular to the quarter-chord line, up to the trailing edge.
The code's default keeps the legs along the (interpolated) input ribs, while
the panel frame, chord and control point already follow CP1. This script
compares the two with the curved bound-vortex regularisation in place:

* on-blade over Trefftz induced drag (inviscid) on a straight wing with ribs
  drawn skewed (the same wing as the unskewed one: perpendicular and
  streamwise legs must give 1.000), the kinked 30 deg swept wing, a yawed
  wing, the arc, the swept-anhedral elliptic wing and the V3 kite, with the
  attached-trailed (AT) force on and off; lifting-surface references for the
  planar wings, the elliptic wing and the inviscid V3 are produced by
  ``vlm_referee.py`` (AeroSandbox);
* the leg geometry on the V3 (how far the perpendicular legs reach along the
  trailing edge at the curved tips);
* V3 viscous polars against RANS (Poland 2025).

An earlier version of this study (commit 118e91d) used perpendicular legs of
fixed length (1 - ac) c, which end on the trailing edge only where it is
parallel to the quarter-chord line; those results are superseded.

Usage (from the repository root, about 4 minutes; ``--figures`` only redraws
figures 10 and 11 from the stored JSON, which needs ``vlm_referee.json``)::

    python docs/validation/leg_direction_study.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver

import curved_bound_vortex_validation as cbv

logging.disable(logging.CRITICAL)
OUT = Path(__file__).resolve().parent
DATA = cbv.DATA
LEGS = ("rib", "streamwise", "perpendicular")


def ratio(body, at=True):
    r = Solver(allowed_error=1e-7, max_iterations=4000, is_with_attached_trailed_vortex_force=at).solve(body)
    return float(r["drag"] / r["drag_induced_trefftz"]), float(r["cl"]), bool(r["gamma_converged"])


def leg_geometry(body):
    """Leg length over (1 - ac) c and the distance between the leg end and the
    section's own trailing-edge point over c, per node (max over the wing)."""
    P = body.panels
    bound = np.array([p.bound_point_1 for p in P] + [P[-1].bound_point_2])
    te = np.array([p.TE_point_1 for p in P] + [P[-1].TE_point_2])
    sec_te = np.array([p.corner_points[1] for p in P] + [P[-1].corner_points[2]])
    chords = np.array([p.chord for p in P])
    c_node = np.concatenate(([chords[0]], 0.5 * (chords[1:] + chords[:-1]), [chords[-1]]))
    leg = np.linalg.norm(te - bound, axis=1) / (0.75 * c_node)
    reach = np.linalg.norm(te - sec_te, axis=1) / c_node
    return float(leg.max()), float(reach.max())


def trefftz_table():
    print("\non-blade / Trefftz induced drag (CL), inviscid, regularised quarter-chord term")
    cases = [
        ("straight AR 8", cbv.straight, (40, 80)),
        ("straight, ribs skewed 22 deg", lambda n, **kw: cbv.straight(n, skew=0.4, **kw), (40, 80)),
        ("swept 30 deg (kinked), streamwise ribs", cbv.swept, (40, 80, 160)),
        ("yawed 30 deg, streamwise ribs", cbv.yawed, (40, 80)),
        ("arc 110 deg, R = 4c", cbv.arc, (40, 80)),
        ("elliptic, 10% sweep, 25% anhedral", cbv.elliptic_swept_anhedral, (40, 80, 160)),
        ("V3 kite, inviscid, 10 deg", cbv.v3_inviscid, (40, 80, 160)),
    ]
    rows = {}
    for case, mk, Ns in cases:
        for legs in LEGS:
            for at in (True, False):
                vals = []
                for N in Ns:
                    try:
                        body = mk(N, leg_direction=legs)
                        rt, cl, conv = ratio(body, at)
                        if not conv:
                            rt = float("nan")
                    except ValueError as err:
                        rt, cl = float("nan"), float("nan")
                        print(f"  [{case}, {legs}, N={N}] {err}")
                    vals.append((N, rt, cl))
                rows[f"{case} | {legs} | AT={at}"] = vals
                print(f"  {case:40s} {legs:13s} AT {'on ' if at else 'off'} ", end="")
                print("  ".join(f"N={N}: {rt:.3f} ({cl:.3f})" for N, rt, cl in vals), flush=True)
    return rows


def geometry_table():
    print("\nleg geometry: max leg length / (0.75 c) and max distance of the leg end from the section's own TE point / c")
    out = {}
    for case, mk in [
        ("straight, ribs skewed 22 deg", lambda n, **kw: cbv.straight(n, skew=0.4, **kw)),
        ("swept 30 deg (kinked)", cbv.swept),
        ("elliptic, 10% sweep, 25% anhedral", cbv.elliptic_swept_anhedral),
        ("V3 kite", cbv.v3_inviscid),
    ]:
        for legs in ("perpendicular", "streamwise"):
            for N in (40, 80):
                try:
                    body = mk(N, leg_direction=legs)
                except ValueError as err:
                    print(f"  {case:36s} {legs:13s} N={N:3d}: {err}")
                    continue
                leg, reach = leg_geometry(body)
                out[f"{case} | {legs} | N={N}"] = {"max_leg_over_075c": leg, "max_reach_over_c": reach}
                print(f"  {case:36s} {legs:13s} N={N:3d}: leg/(0.75c) {leg:.2f}   reach/c {reach:.2f}")
    return out


def v3_vs_rans():
    print("\nV3 viscous polars (CAD, 2D RANS Re 5e5 polars, 50 panels) against RANS")
    cfd = pd.read_csv(DATA / "3D_polars_literature" / "CFD_RANS_Rey_10e5_Poland2025_alpha_sweep_beta_0.csv")
    cfd = cfd[cfd.alpha < 14]
    ref = np.array([0.422646, 0.0, 9.3667])
    out = {}
    for legs in LEGS:
        body = BodyAerodynamics.instantiate(
            n_panels=50,
            file_path=DATA / "CAD_derived_geometry" / "aero_geometry_CAD_CFD_NF_combined.yaml",
            spanwise_panel_distribution="uniform",
            leg_direction=legs,
        )
        for at in (True, False):
            s = Solver(reference_point=ref, is_with_attached_trailed_vortex_force=at)
            rows = []
            for al, cl_ref, cd_ref in zip(cfd.alpha, cfd.CL, cfd.CD):
                body.va_initialize(2.82, float(al), 0.0)
                r = s.solve(body)
                rows.append((float(al), float(r["cl"]), float(r["cd"]), float(cl_ref), float(cd_ref)))
            e_cl = float(np.mean([abs(a[1] - a[3]) for a in rows]))
            e_cd = float(np.mean([abs(a[2] - a[4]) / a[4] for a in rows]))
            e_cd10 = float(np.mean([abs(a[2] - a[4]) / a[4] for a in rows if a[0] <= 10.5]))
            out[f"{legs} | AT={at}"] = {"rows": rows, "mean_abs_dCL": e_cl, "mean_rel_dCD": e_cd, "mean_rel_dCD_to_10deg": e_cd10}
            print(f"  {legs:13s} AT {'on ' if at else 'off'}: " + "  ".join(f"{a[0]:.0f}deg CL {a[1]:.3f}/{a[3]:.3f} CD {a[2]:.4f}/{a[4]:.4f}" for a in rows))
            print(f"  {'':13s}         mean |dCL| {e_cl:.3f}   mean |dCD|/CD {100 * e_cd:.1f}% (to 10 deg {100 * e_cd10:.1f}%)", flush=True)
    return out


# --------------------------------------------------------------------------
# Figures 10 (geometry of the three leg definitions) and 11 (VLM referee)
# --------------------------------------------------------------------------
FIG = OUT.parent / "figures"
COLOR = {"rib": "tab:blue", "streamwise": "tab:green", "perpendicular": "tab:red"}


def _geometry(body):
    P = body.panels
    bound = np.array([p.bound_point_1 for p in P] + [P[-1].bound_point_2])
    te = np.array([p.TE_point_1 for p in P] + [P[-1].TE_point_2])
    ac = np.array([p.aerodynamic_center for p in P])
    cp = np.array([p.control_point for p in P])
    le_sec = np.array([p.corner_points[0] for p in P] + [P[-1].corner_points[3]])
    te_sec = np.array([p.corner_points[1] for p in P] + [P[-1].corner_points[2]])
    return bound, te, ac, cp, le_sec, te_sec


def _vlm_quads(xsecs, n_chord, n_span):
    """Panel corners of the AeroSandbox lattice (None if AeroSandbox is absent)."""
    try:
        import aerosandbox as asb
    except ImportError:
        return None
    wing = asb.Wing(symmetric=False, xsecs=xsecs)
    if n_span > 1:
        wing = wing.subdivide_sections(n_span)
    points, faces = wing.mesh_thin_surface(method="quad", chordwise_resolution=n_chord, add_camber=False)
    return np.asarray(points)[np.asarray(faces)]


def _draw(ax, body, legs, ij, lattice=None, outline=True, lw=1.2):
    i, j = ij
    bound, te, ac, cp, le_sec, te_sec = _geometry(body)
    if lattice is not None:
        for q in lattice:
            q = np.vstack([q, q[:1]])
            ax.plot(q[:, i], q[:, j], color="0.8", lw=0.5, zorder=0)
    if outline:
        ax.plot(le_sec[:, i], le_sec[:, j], color="0.45", lw=0.8, zorder=1)
        ax.plot(te_sec[:, i], te_sec[:, j], color="0.45", lw=0.8, zorder=1)
        for a, b in zip(le_sec, te_sec):
            ax.plot([a[i], b[i]], [a[j], b[j]], color="0.75", lw=0.5, zorder=1)
    ax.plot(bound[:, i], bound[:, j], color="k", lw=2.0, zorder=3)
    for a, b in zip(bound, te):
        ax.plot([a[i], b[i]], [a[j], b[j]], color=COLOR[legs], lw=lw, zorder=2)
    ax.plot(ac[:, i], ac[:, j], "o", color="k", ms=2.5, zorder=4)
    ax.plot(cp[:, i], cp[:, j], "+", color=COLOR[legs], ms=6, mew=1.3, zorder=4)
    ax.set_aspect("equal")


def figure_10():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    try:
        import aerosandbox as asb

        FLAT = asb.Airfoil("naca0010")
        xs = lambda pts, c=1.0: [asb.WingXSec(xyz_le=p, chord=c, twist=0.0, airfoil=FLAT) for p in pts]
    except ImportError:
        asb = None

    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
    tan30 = np.tan(np.radians(30.0))

    # (a) straight wing with skewed ribs, tip region: rib vs streamwise (= perpendicular)
    A = ax[0, 0]
    lat = _vlm_quads(xs([[0, -4, 0], [0, 4, 0]]), 4, 20) if asb else None
    _draw(A, cbv.straight(20, skew=0.4), "rib", (1, 0), lattice=lat)
    _draw(A, cbv.straight(20, skew=0.4, leg_direction="streamwise"), "streamwise", (1, 0), outline=False)
    A.set_xlim(2.0, 4.6); A.set_ylim(1.6, -0.3)
    A.set_title("(a) straight wing, ribs drawn 22 deg skewed: tip")
    A.set_xlabel("y"); A.set_ylabel("x (flow downwards)")

    # (b) kinked swept wing, root region: rib (= streamwise) vs perpendicular
    B = ax[0, 1]
    lat = _vlm_quads(xs([[4 * tan30, -4, 0], [0, 0, 0], [4 * tan30, 4, 0]]), 4, 10) if asb else None
    _draw(B, cbv.swept(40), "rib", (1, 0), lattice=lat)
    _draw(B, cbv.swept(40, leg_direction="perpendicular"), "perpendicular", (1, 0), outline=False)
    B.set_xlim(-1.6, 1.6); B.set_ylim(2.0, -0.4)
    B.set_title("(b) kinked 30 deg swept wing: root")
    B.set_xlabel("y"); B.set_ylabel("x (flow downwards)")

    # (c) elliptic swept-anhedral wing, right half, top view
    C = ax[0, 2]
    s = 8.0 * (np.pi / 4) / 2
    pts, chords = [], []
    for y in np.linspace(-s, s, 41):
        eta = min(abs(y) / s, 0.999999)
        pts.append([0.10 * s * eta**2, y, -0.25 * s * eta**2]); chords.append(np.sqrt(1 - eta**2))
    lat = None
    if asb:
        lat = _vlm_quads([asb.WingXSec(xyz_le=p, chord=c, twist=0.0, airfoil=FLAT) for p, c in zip(pts, chords)], 4, 1)
    for legs in ("rib", "streamwise", "perpendicular"):
        _draw(C, cbv.elliptic_swept_anhedral(40, leg_direction=legs), legs, (1, 0), lattice=lat if legs == "rib" else None, outline=legs == "rib", lw=1.0)
    C.set_xlim(0.0, 3.3); C.set_ylim(1.4, -0.2)
    C.set_title("(c) elliptic, 10% sweep, 25% anhedral: right half, top view")
    C.set_xlabel("y"); C.set_ylabel("x (flow downwards)")

    # (d) arc wing, side view of the upper tip: streamwise legs climb with the flow
    D = ax[1, 0]
    for legs in ("rib", "streamwise"):
        _draw(D, cbv.arc(40, leg_direction=legs), legs, (0, 2), outline=legs == "rib", lw=1.0)
    D.set_xlim(-0.2, 1.4); D.set_ylim(2.1, 4.2)
    D.set_title("(d) arc wing, 8 deg: side view of the tip region (x-z)")
    D.set_xlabel("x"); D.set_ylabel("z")

    # (e) V3 kite, top view, whole wing, with the VLM lattice
    E = ax[1, 1]
    lat = None
    if asb:
        import vlm_referee

        lat = _vlm_quads(vlm_referee.v3_xsecs(), 4, 2)
    v3 = {legs: cbv.v3_inviscid(40, leg_direction=legs) for legs in ("rib", "streamwise", "perpendicular")}
    for legs, body in v3.items():
        _draw(E, body, legs, (1, 0), lattice=lat if legs == "rib" else None, outline=legs == "rib", lw=0.8)
    E.invert_yaxis()
    E.set_title("(e) V3 kite, 10 deg: top view")
    E.set_xlabel("y"); E.set_ylabel("x (flow downwards)")

    # (f) V3 kite, the +y tip, top view
    F = ax[1, 2]
    for legs, body in v3.items():
        _draw(F, body, legs, (1, 0), outline=legs == "rib", lw=1.2)
    bound = _geometry(v3["rib"])[0]
    F.set_xlim(bound[:, 1].max() - 2.2, bound[:, 1].max() + 0.3)
    xr = bound[:4, 0]
    F.set_ylim(xr.max() + 1.8, xr.min() - 0.5)
    F.set_title("(f) V3 kite: tip region")
    F.set_xlabel("y"); F.set_ylabel("x (flow downwards)")

    handles = [
        Line2D([], [], color="k", lw=2, label="bound vortex (quarter-chord line)"),
        Line2D([], [], color="0.45", lw=0.8, label="section outline (LE, TE, ribs)"),
        Line2D([], [], color="0.8", lw=0.6, label="VLM lattice (AeroSandbox, coarse)"),
        Line2D([], [], color="k", marker="o", ls="", ms=3, label="aerodynamic centre"),
        Line2D([], [], color="tab:blue", lw=1.5, marker="+", ms=7, label="rib legs, control point"),
        Line2D([], [], color="tab:green", lw=1.5, marker="+", ms=7, label="streamwise legs, control point"),
        Line2D([], [], color="tab:red", lw=1.5, marker="+", ms=7, label="perpendicular legs, control point"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Attached-trailed legs: along the ribs, along the flow, or perpendicular to the quarter-chord line", y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    fig.savefig(FIG / "10_leg_direction_geometry.png", dpi=150)
    print(f"figure 10 written to {FIG / '10_leg_direction_geometry.png'}")


def figure_11():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = json.load(open(OUT / "vlm_referee.json"))
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

    # top: circulation distributions, finest VLM vs VSM at the finest N
    panels = [("kinked swept 30 deg", "kinked 30 deg swept wing, 5 deg", 20.0),
              ("elliptic", "elliptic, 10% sweep, 25% anhedral, 10 deg", 10.0),
              ("V3", "V3 kite, flat-plate sections, 10 deg", 2.82)]
    for A, (case, title, U) in zip(ax[0], panels):
        row = ref[case]
        vkeys = [k for k in row if k.startswith("VLM")]
        v = row[vkeys[-1]]
        o = np.argsort(v["y"])
        A.plot(np.array(v["y"])[o], np.array(v["gamma"])[o] / U, "k-", lw=2, label=f"VLM ({vkeys[-1][4:]})")
        for legs in LEGS:
            keys = [k for k in row if k.startswith(f"VSM {legs}")]
            if not keys:
                continue
            r = row[keys[-1]]
            o = np.argsort(r["y"])
            A.plot(np.array(r["y"])[o], np.array(r["gamma"])[o] / U, color=COLOR[legs], lw=1.3, label=f"VSM, {legs} legs ({keys[-1].split()[-1]})")
        A.set_title(title); A.set_xlabel("y"); A.set_ylabel("Gamma / U")
        A.legend(fontsize=8, frameon=False)

    # bottom: CL and induced drag relative to the VLM (Trefftz drag of the VLM circulation)
    cases = ["straight AR 8", "yawed 30 deg", "kinked swept 30 deg", "arc", "elliptic", "V3"]
    labels = ["straight", "yawed 30", "kinked 30", "arc", "elliptic", "V3"]
    for A, quantity, ylabel in zip(ax[1, :2], ("CL", "CDi"), ("CL / CL_VLM", "CDi (on-blade) / CDi_VLM (Trefftz)")):
        x = np.arange(len(cases))
        for k, legs in enumerate(LEGS):
            vals = []
            for case in cases:
                row = ref[case]
                vkeys = [kk for kk in row if kk.startswith("VLM")]
                v = row[vkeys[-1]]
                keys = [kk for kk in row if kk.startswith(f"VSM {legs}")]
                if not keys:
                    vals.append(np.nan)
                    continue
                r = row[keys[-1]]
                vals.append(r["CL"] / v["CL"] if quantity == "CL" else r["CDi_blade"] / v["CDi_trefftz"])
            A.bar(x + (k - 1) * 0.26, vals, 0.26, color=COLOR[legs], label=f"{legs} legs")
        A.axhline(1.0, color="k", lw=0.8)
        A.set_xticks(x); A.set_xticklabels(labels)
        A.set_ylim(0.85, 1.15); A.set_ylabel(ylabel); A.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    ax[1, 0].set_title("lift relative to the VLM")
    ax[1, 1].set_title("induced drag relative to the VLM")

    # bottom right: the V3 viscous polars against RANS for the three leg options
    A = ax[1, 2]
    st = json.load(open(OUT / "leg_direction_study.json"))["v3_vs_rans"]
    for legs in LEGS:
        rows = np.array(st[f"{legs} | AT=True"]["rows"])
        A.plot(rows[:, 0], rows[:, 2], "-o", color=COLOR[legs], ms=3, label=f"VSM, {legs} legs")
    A.plot(rows[:, 0], rows[:, 4], "k--s", ms=3, label="RANS (Poland 2025)")
    A.set_title("V3 viscous drag against RANS (2D RANS polars, 50 panels)")
    A.set_xlabel("alpha [deg]"); A.set_ylabel("CD"); A.legend(fontsize=8, frameon=False)
    fig.suptitle("Leg direction against a vortex-lattice referee (AeroSandbox) and RANS", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "11_leg_direction_referee.png", dpi=150)
    print(f"figure 11 written to {FIG / '11_leg_direction_referee.png'}")


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="rib / streamwise / perpendicular attached-trailed legs")
    ap.add_argument("--figures", action="store_true", help="only draw figures 10 and 11 from the stored results")
    args = ap.parse_args(argv)
    if args.figures:
        figure_10()
        figure_11()
        return
    t0 = time.time()
    out = {"trefftz": trefftz_table(), "geometry": geometry_table(), "v3_vs_rans": v3_vs_rans()}
    json.dump(out, open(OUT / "leg_direction_study.json", "w"), indent=1)
    print(f"\nresults written to {OUT / 'leg_direction_study.json'}  ({time.time() - t0:.0f} s)")
    figure_10()
    if (OUT / "vlm_referee.json").exists():
        figure_11()


if __name__ == "__main__":
    main()
