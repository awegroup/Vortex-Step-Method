"""Validation of the curved bound-vortex regularisation at the quarter chord.

Reproduces the tables and figures of docs/validation/curved_bound_vortex_report.tex:

* on-blade induced drag over Trefftz-plane induced drag, inviscid polars, as a
  function of the number of spanwise panels, for a straight wing, a straight
  wing with skewed ribs, a 30 deg swept wing with streamwise ribs, a circular
  arc wing, the swept-anhedral elliptic wing of Gaunaa, Li & Pirrung (TORQUE
  2026) and the TU Delft V3 kite;
* the same ratio for the alternatives that were tried and rejected: no bound
  term at the quarter chord (trailed system only), the other panels' bound
  segments evaluated on the line (Phillips & Snyder, "LL-P&S" in Li et al.
  2020; the develop branch before this work), the literal Li et al. 2020
  recipe (3D minus 2D bound induction at the 3/4 chord reused at the quarter
  chord) and the effective-lifting-line / jointed-vortex construction of Reid
  & Hunsaker (2020) for several joint lengths and blending distances;
* the chordwise self-induction at the mid-span of the arc from the bound
  polyline (log-divergent) and from the chordwise-spread sheet (finite);
* the V3 viscous polars (figure 09) and the mechanism figure (figure 08).

Every alternative is a drop-in replacement of
``assemble_curved_bound_correction_AIC`` installed by monkeypatching the name
in ``VSM.core.BodyAerodynamics``; the solver is otherwise unchanged.

Usage (from the repository root; the full run takes about 20 minutes, most
of it in the pure-Python Reid-Hunsaker loops)::

    python docs/validation/curved_bound_vortex_validation.py            # tables
    python docs/validation/curved_bound_vortex_validation.py --quick    # regularised + on-line + none only
    python docs/validation/curved_bound_vortex_validation.py --figures  # tables + figures 08, 09
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import VSM.core.BodyAerodynamics as BA
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.core.WingGeometry import Wing
from VSM.core.utils import (
    _vel_bound_vortex,
    _vel_semiinfinite,
    _vel_trailing_vortex,
    thin_airfoil_strips,
)

logging.disable(logging.CRITICAL)
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = ROOT / "docs" / "figures"
DATA = ROOT / "data" / "TUDELFT_V3_KITE"
REGULARISED = BA.assemble_curved_bound_correction_AIC

_a = np.radians(np.linspace(-25, 25, 101))
INVISCID = np.column_stack((_a, 2 * np.pi * _a, 0 * _a, 0 * _a))


# --------------------------------------------------------------------------
# Geometries
# --------------------------------------------------------------------------
def straight(n, half_span=4.0, skew=0.0):
    wing = Wing(n_panels=n, spanwise_panel_distribution="uniform")
    for y in np.linspace(-half_span, half_span, n + 1):
        wing.add_section(np.array([0.0, y, 0.0]), np.array([1.0, y + skew, 0.0]), INVISCID)
    body = BodyAerodynamics([wing])
    body.va_initialize(20.0, 5.0, 0.0)
    return body


def swept(n, sweep_deg=30.0, half_span=4.0):
    """Kinked 30 deg sweep, sections (ribs) along the freestream x axis."""
    wing = Wing(n_panels=n, spanwise_panel_distribution="uniform")
    for y in np.linspace(-half_span, half_span, n + 1):
        x = abs(y) * np.tan(np.radians(sweep_deg))
        wing.add_section(np.array([x, y, 0.0]), np.array([x + 1.0, y, 0.0]), INVISCID)
    body = BodyAerodynamics([wing])
    body.va_initialize(20.0, 5.0, 0.0)
    return body


def arc(n, radius=4.0, half_angle_deg=55.0):
    """Circular arc in the y-z plane (kite-like), chord 1 along x."""
    wing = Wing(n_panels=n, spanwise_panel_distribution="uniform")
    for th in np.radians(np.linspace(-half_angle_deg, half_angle_deg, n + 1)):
        le = np.array([0.0, radius * np.sin(th), radius * np.cos(th)])
        wing.add_section(le, le + np.array([1.0, 0.0, 0.0]), INVISCID)
    body = BodyAerodynamics([wing])
    body.va_initialize(10.0, 8.0, 0.0)
    return body


def elliptic_swept_anhedral(n, aspect_ratio=8.0, sweep=0.10, anhedral=0.25):
    """Elliptic chord, root chord 1, tip offsets of 10 % (back) and 25 % (down)
    of the semispan, parabolic; the TORQUE 2026 combined case."""
    s = aspect_ratio * (np.pi / 4) / 2
    wing = Wing(n_panels=n, spanwise_panel_distribution="uniform")
    for y in np.linspace(-s, s, n + 1):
        eta = min(abs(y) / s, 0.999999)
        c = np.sqrt(1 - eta**2)
        x = sweep * s * eta**2
        z = -anhedral * s * eta**2
        wing.add_section(np.array([x, y, z]), np.array([x + c, y, z]), INVISCID)
    body = BodyAerodynamics([wing])
    body.va_initialize(10.0, 10.0, 0.0)
    return body


_V3 = None


def v3_inviscid(n, alpha=10.0):
    global _V3
    if _V3 is None:
        _V3 = yaml.safe_load(open(DATA / "CAD_derived_geometry" / "aero_geometry_CAD_CFD_NF_combined.yaml"))
    wing = Wing(n_panels=n, spanwise_panel_distribution="uniform")
    for row in _V3["wing_sections"]["data"]:
        wing.add_section(np.array(row[1:4], float), np.array(row[4:7], float), INVISCID)
    body = BodyAerodynamics([wing])
    body.va_initialize(2.82, alpha, 0.0)
    return body


# --------------------------------------------------------------------------
# Alternatives for the quarter-chord bound-vortex term (drop-in replacements)
# --------------------------------------------------------------------------
STATE = {"body": None, "U": None, "wake": None, "delta": 0.15, "blend": 1.0}


def _seg(x1, x2, p):
    """Unit-strength straight segment x1 -> x2, zero on the line (to round-off)."""
    r1 = p - x1
    r2 = p - x2
    r0 = x2 - x1
    c = np.cross(r1, r2)
    L = np.linalg.norm(r0)
    n = np.linalg.norm(c)
    if n < 1e-9 * L * max(np.linalg.norm(r1), L):
        return np.zeros(3)
    return c / (4 * np.pi * n * n) * (r0 @ (r1 / np.linalg.norm(r1) - r2 / np.linalg.norm(r2)))


def _none(eval_points, bp1, bp2, *args):
    """Trailed system only at the quarter chord."""
    n = eval_points.shape[0]
    return np.zeros((3, n, n))


def _on_line(eval_points, bp1, bp2, te1, te2, cdir, chords, va_norm, ac, core, sx, sw):
    """Other panels' bound segments evaluated on the quarter-chord line
    (Phillips & Snyder; develop before this work)."""
    n = eval_points.shape[0]
    A = np.zeros((3, n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                A[:, i, j] = _vel_bound_vortex(bp2[j], bp1[j], eval_points[i], 1.0, core)
    return A


def _li2020_literal(eval_points, bp1, bp2, te1, te2, cdir, chords, va_norm, ac, core, sx, sw):
    """3D bound induction minus the section's own 2D infinite vortex, evaluated
    at the 3/4-chord control point and reused at the quarter chord."""
    P = STATE["body"].panels
    cps = np.array([p.control_point for p in P])
    n = eval_points.shape[0]
    A = np.zeros((3, n, n))
    for i in range(n):
        for j in range(n):
            v = _seg(bp2[j], bp1[j], cps[i])
            if i == j:
                r0 = bp1[j] - bp2[j]
                t = r0 / np.linalg.norm(r0)
                r1 = cps[i] - bp2[j]
                dp = r1 - (r1 @ t) * t
                v = v - np.cross(t, r1) / (2 * np.pi * (dp @ dp))
            A[:, i, j] = v
    return A


def _ring_trailed(bp1, bp2, te1, te2, ep, wake, U):
    return (
        _vel_trailing_vortex(bp1, te1, ep, 1.0, U)
        + _vel_trailing_vortex(te2, bp2, ep, 1.0, U)
        + _vel_semiinfinite(te1, wake, ep, 1.0, U, 1.0)
        + _vel_semiinfinite(te2, wake, ep, 1.0, U, -1.0)
    )


def _reid_hunsaker(eval_points, bp1, bp2, te1, te2, cdir, chords, va_norm, ac, core, sx, sw):
    """Reid & Hunsaker (2020): for each control point the lifting line is
    straightened near the point (Gaussian blend of width STATE['blend'] times
    the local chord, or an absolute length if STATE['blend'] is a tuple
    ('abs', metres)), every trailing vortex gets a joint of STATE['delta']
    chords perpendicular to the effective line before turning along the
    freestream, and the bound segments of the other rings are kept."""
    U = STATE["U"]
    wake = STATE["wake"]
    n = eval_points.shape[0]
    seg = bp1 - bp2
    L = np.linalg.norm(seg, axis=1)
    s_node = np.concatenate([[0.0], np.cumsum(L)])
    fwd = n == 1 or np.linalg.norm(bp1[0] - bp2[1]) < np.linalg.norm(bp2[0] - bp1[1])
    sgn = 1.0 if fwd else -1.0
    nodes = np.vstack([bp2, bp1[-1:]]) if fwd else np.vstack([bp1, bp2[-1:]])
    A = np.zeros((3, n, n))
    for i in range(n):
        ep = eval_points[i]
        t = seg[i] / L[i]
        s_P = 0.5 * (s_node[i] + s_node[i + 1])
        blend = STATE["blend"]
        ds = blend[1] if isinstance(blend, tuple) else blend * chords[i]
        w = np.exp(-((s_node - s_P) / ds) ** 2)
        eff = (1 - w)[:, None] * nodes + w[:, None] * (ep + sgn * (s_node - s_P)[:, None] * t)
        for j in range(n):
            e2, e1 = (eff[j], eff[j + 1]) if fwd else (eff[j + 1], eff[j])
            tj = e1 - e2
            tj = tj / np.linalg.norm(tj)
            d = cdir[j] - (cdir[j] @ tj) * tj
            d = d / np.linalg.norm(d)
            j1 = e1 + STATE["delta"] * chords[j] * d
            j2 = e2 + STATE["delta"] * chords[j] * d
            v = (0.0 if i == j else 1.0) * _seg(e2, e1, ep)
            v = v + _ring_trailed(e1, e2, j1, j2, ep, wake, U)
            A[:, i, j] = v - _ring_trailed(bp1[j], bp2[j], te1[j], te2[j], ep, wake, U)
    return A


VARIANTS = {
    "regularised (this work)": REGULARISED,
    "trailed only (no bound term)": _none,
    "bound segments on the line (develop before)": _on_line,
    "Li 2020 literal (3/4c difference reused)": _li2020_literal,
}
RH_PARAMS = [  # (joint length / chord, blending distance / chord)
    (0.25, 1.0),
    (0.5, 1.0),
    (1.0, 1.0),
    (0.5, 0.5),
    (0.5, 2.0),
]


def solve(body, variant, **rh):
    BA.assemble_curved_bound_correction_AIC = variant
    STATE["body"] = body
    va = body.va if np.ndim(body.va) == 1 else body.va[0]
    STATE["U"] = float(np.linalg.norm(va))
    STATE["wake"] = va / STATE["U"]
    STATE.update(rh)
    try:
        r = Solver(allowed_error=1e-7, max_iterations=4000).solve(body)
    finally:
        BA.assemble_curved_bound_correction_AIC = REGULARISED
    return r


def ratio(body, variant, **rh):
    r = solve(body, variant, **rh)
    return float(r["drag"] / r["drag_induced_trefftz"]), float(r["cl"])


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
CASES = [
    ("straight AR 8", straight, (40, 80)),
    ("straight, ribs skewed 22 deg", lambda n: straight(n, skew=0.4), (40, 80)),
    ("swept 30 deg, streamwise ribs", swept, (40, 80, 160)),
    ("arc 110 deg, R = 4c", arc, (20, 40, 80, 160)),
    ("elliptic, 10% sweep, 25% anhedral", elliptic_swept_anhedral, (40, 80, 160)),
    ("V3 kite, inviscid, 10 deg", v3_inviscid, (20, 40, 80, 160)),
]


def table_variants(quick):
    names = ["regularised (this work)", "bound segments on the line (develop before)", "trailed only (no bound term)"]
    if not quick:
        names.append("Li 2020 literal (3/4c difference reused)")
    rows = {}
    print("\non-blade / Trefftz induced drag (CL), inviscid polars")
    for case, mk, Ns in CASES:
        for name in names:
            vals = []
            for N in Ns:
                t0 = time.time()
                try:
                    rt, cl = ratio(mk(N), VARIANTS[name])
                except Exception as e:  # degenerate geometry etc.
                    rt, cl = float("nan"), float("nan")
                vals.append((N, rt, cl))
            rows[f"{case} | {name}"] = vals
            print(f"  {case:36s} {name:44s} " + "  ".join(f"N={N}: {rt:.3f}" for N, rt, _ in vals), flush=True)
    return rows


def table_reid_hunsaker():
    rows = {}
    print("\nReid & Hunsaker construction, on-blade / Trefftz (joint/chord, blending/chord)")
    for case, mk, Ns in CASES:
        if "skewed" in case:
            continue
        N = min(max(Ns), 80)
        for delta, blend in RH_PARAMS:
            try:
                rt, _ = ratio(mk(N), _reid_hunsaker, delta=delta, blend=blend)
            except Exception:
                rt = float("nan")
            rows[f"{case} | RH {delta}/{blend}"] = [(N, rt)]
            print(f"  {case:36s} joint {delta:4.2f} c, blend {blend:3.1f} c  N={N}: {rt:.3f}", flush=True)
        # MachUpX defaults: joint 0.15 c, blending distance one semispan
        body = mk(N)
        y = np.array([p.aerodynamic_center[1] for p in body.panels])
        semispan = 0.5 * (y.max() - y.min())
        try:
            rt, _ = ratio(body, _reid_hunsaker, delta=0.15, blend=("abs", semispan))
        except Exception:
            rt = float("nan")
        rows[f"{case} | RH MachUpX defaults"] = [(N, rt)]
        print(f"  {case:36s} MachUpX defaults (0.15 c, semispan)  N={N}: {rt:.3f}", flush=True)
    return rows


def arc_self_induction():
    """Chordwise induced velocity at the arc mid-span quarter-chord point from
    the other panels' bound vortices: polyline (log-divergent) and chordwise-
    spread sheet (finite). Circulation from the 80-panel solution."""
    b80 = arc(80)
    r80 = solve(b80, REGULARISED)
    gam80 = np.asarray(r80["gamma_distribution"])
    th80 = np.array([np.arctan2(p.aerodynamic_center[1], p.aerodynamic_center[2]) for p in b80.panels])
    order = np.argsort(th80)
    sx, sw = thin_airfoil_strips()
    out = {"N": [], "polyline": [], "sheet": []}
    for N in (10, 20, 40, 80, 160, 320):
        b = arc(N)
        P = b.panels
        i = N // 2
        ep = P[i].aerodynamic_center
        th = np.array([np.arctan2(p.aerodynamic_center[1], p.aerodynamic_center[2]) for p in P])
        gam = np.interp(th, th80[order], gam80[order])
        u_poly = np.zeros(3)
        u_sheet = np.zeros(3)
        for j in range(N):
            if j == i:
                continue
            u_poly += gam[j] * _vel_bound_vortex(P[j].bound_point_2, P[j].bound_point_1, ep, 1.0, 1e-20)
            for x, w in zip(sx, sw):
                sh = np.array([x - 0.25, 0.0, 0.0])
                u_sheet += gam[j] * w * _vel_bound_vortex(P[j].bound_point_2 + sh, P[j].bound_point_1 + sh, ep, 1.0, 1e-20)
        out["N"].append(N)
        out["polyline"].append(float(u_poly[0]))
        out["sheet"].append(float(u_sheet[0]))
    print("\narc mid-span chordwise self-induction u [m/s] (U = 10 m/s):")
    for N, up, us in zip(out["N"], out["polyline"], out["sheet"]):
        print(f"  N={N:4d}: polyline {up:+.4f}   sheet {us:+.4f}")
    return out


# --------------------------------------------------------------------------
# V3 viscous polars (figure 09)
# --------------------------------------------------------------------------
def v3_polars():
    ref = np.array([0.422646, 0.0, 9.3667])
    U = 2.82
    body = BodyAerodynamics.instantiate(
        n_panels=50,
        file_path=DATA / "CAD_derived_geometry" / "aero_geometry_CAD_CFD_NF_combined.yaml",
        spanwise_panel_distribution="uniform",
    )
    alphas = [-3, -1, 1, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 17, 19]
    betas = [0, 4, 8, 12, 14, 20]
    runs = {
        "old34": (dict(is_aoa_corrected=False), REGULARISED),
        "on_line": (dict(is_aoa_corrected=True), _on_line),
        "fix": (dict(is_aoa_corrected=True), REGULARISED),
    }
    pol = {"alpha": {}, "beta": {}}
    for tag, (kw, variant) in runs.items():
        s = Solver(reference_point=ref, **kw)
        BA.assemble_curved_bound_correction_AIC = variant
        rows = []
        for al in alphas:
            body.va_initialize(U, float(al), 0.0)
            r = s.solve(body)
            rows.append((al, float(r["cl"]), float(r["cd"])))
        pol["alpha"][tag] = rows
        rows = []
        for be in betas:
            body.va_initialize(U, 7.4, float(be))
            r = s.solve(body)
            rows.append((be, float(r["cl"]), float(r["cd"]), float(r["cs"])))
        pol["beta"][tag] = rows
        BA.assemble_curved_bound_correction_AIC = REGULARISED
        print(f"  V3 polars {tag} done", flush=True)
    return pol


# --------------------------------------------------------------------------
# Figures 08 and 09
# --------------------------------------------------------------------------
def figure_08(conv, ux):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    A = ax[0, 0]
    R = 4
    th = np.radians(np.linspace(-55, 55, 13))
    y, z = R * np.sin(th), R * np.cos(th)
    A.plot(y, z, "o-", color="k", lw=2, ms=4, label="bound vortex (polyline) on the quarter-chord line")
    i = 6
    P = (0.5 * (y[i] + y[i + 1]), 0.5 * (z[i] + z[i + 1]))
    A.plot(*P, "r*", ms=14, label="quarter-chord point P of panel i")
    for k in (4, 5, 7, 8):
        m = (0.5 * (y[k] + y[k + 1]), 0.5 * (z[k] + z[k + 1]))
        A.annotate("", xy=P, xytext=m, arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2))
    A.text(0.0, 1.55, "neighbouring segments induce a velocity at P\nperpendicular to the arc plane, i.e. along the chord:\nu ~ (Gamma kappa / 4 pi) ln(span / panel width)\n-> grows without bound as the panels shrink",
           ha="center", va="top", fontsize=8.5, bbox=dict(fc="w", ec="0.7"))
    A.set_xlim(-3.6, 3.6); A.set_ylim(0.3, 4.6); A.set_xlabel("y (span)"); A.set_ylabel("z")
    A.legend(loc="upper left", fontsize=8); A.set_title("(a) Front view: curved bound vortex evaluated on itself")
    B = ax[0, 1]
    B.semilogx(ux["N"], ux["polyline"], "o-", color="tab:red", label="polyline segments (bound on the line)")
    B.semilogx(ux["N"], ux["sheet"], "s-", color="tab:green", label="chordwise-spread sheet (thin-airfoil weight)")
    B.axhspan(-0.20, -0.18, color="0.85", label="value for which on-blade drag = Trefftz")
    B.set_xlabel("number of spanwise panels N"); B.set_ylabel("chordwise induced velocity u at mid-span [m/s]  (U = 10 m/s)")
    B.set_title("(b) Arc wing: the self-induction diverges, the sheet value is finite"); B.legend(fontsize=8); B.grid(alpha=0.3)
    C = ax[0, 2]
    al = np.radians(20)
    C.plot([-0.1, 1.1], [0, 0], "k-", lw=3); C.text(1.12, 0, "chord", va="center")
    e = np.array([np.cos(al), np.sin(al)]); nvec = np.array([-np.sin(al), np.cos(al)]); o = np.array([0.25, 0.0])

    def arr(p, v, c, lab, dy=0.02):
        C.add_patch(FancyArrowPatch(p, p + v, arrowstyle="->", mutation_scale=14, color=c, lw=1.8))
        C.text(*(p + v + np.array([0.02, dy])), lab, color=c, fontsize=9)

    arr(o, 0.9 * e, "k", "freestream U\n(angle alpha to the chord)")
    w = 0.12; u = -0.10
    arr(o + 0.9 * e, -w * nvec, "tab:blue", "downwash w (from the trailed wake)")
    arr(o + 0.9 * e - w * nvec, u * e, "tab:red", "chordwise u (curved bound vortex)", dy=-0.06)
    V = 0.9 * e - w * nvec + u * e; arr(o, V, "tab:purple", "quarter-chord flow V", dy=-0.12)
    Vh = V / np.linalg.norm(V); Lv = np.array([-Vh[1], Vh[0]]) * 0.55
    arr(o, Lv, "tab:green", "lift, perpendicular to V", dy=0.0)
    C.text(-0.1, -0.62, "the lift component along U is the\non-blade induced drag:\nD = rho Gamma (w cos(alpha) + u sin(alpha))\n\nw is small (about 1 deg of induced angle),\nso u = 3% of U changes D by about 12%",
           fontsize=8.5, bbox=dict(fc="w", ec="0.7"))
    C.set_xlim(-0.15, 1.75); C.set_ylim(-0.7, 0.9); C.set_aspect("equal"); C.axis("off"); C.set_title("(c) Why a chordwise velocity changes the drag")
    Dd = ax[1, 0]
    x = np.linspace(0.002, 0.998, 400); g = np.sqrt((1 - x) / x); g /= np.trapezoid(g, x)
    Dd.plot(x, g, color="tab:green", lw=2, label="thin-airfoil bound vorticity gamma(x) ~ sqrt((c-x)/x)")
    Dd.fill_between(x, 0, g, color="tab:green", alpha=0.15)
    sx, _ = thin_airfoil_strips()
    for xq in sx:
        Dd.plot([xq, xq], [0, np.interp(xq, x, g)], color="tab:green", lw=0.6, alpha=0.6)
    Dd.plot([0.25], [0], "ro", ms=9, label="single bound vortex at c/4 (lifting line)"); Dd.plot([0, 1], [0, 0], "k-", lw=3)
    Dd.text(0.5, 2.2, "same total circulation, same centroid (c/4);\nspreading it over the chord makes the curved\nsheet self-induction finite (effective cut-off 0.15 c)",
            ha="center", fontsize=8.5, bbox=dict(fc="w", ec="0.7"))
    Dd.set_xlabel("x / c"); Dd.set_ylabel("gamma(x) c / Gamma"); Dd.set_ylim(-0.3, 3.2); Dd.legend(fontsize=8, loc="upper right")
    Dd.set_title("(d) The regularisation: bound vortex spread over the chord")
    E = ax[1, 1]
    s = np.linspace(-1, 1, 9); yy = 3.0 * s; zz = -1.2 * s**2
    E.plot(yy, zz, "o-", color="k", lw=2, ms=4, label="actual curved system (bound + legs), smeared")
    for k in range(9):
        E.plot([yy[k], yy[k]], [zz[k], zz[k] - 0.6], color="k", lw=0.8)
    E.plot(yy, 0 * yy + 0.9, "o-", color="tab:orange", lw=2, ms=4, label="same panels unrolled onto the tangent line of P")
    for k in range(9):
        E.plot([yy[k], yy[k]], [0.9, 0.3], color="tab:orange", lw=0.8)
    E.plot([0], [0], "r*", ms=14); E.plot([0], [0.9], "r*", ms=14)
    E.text(0, -2.1, "V(1/4c) = classical lifting-line downwash of the UNROLLED wing\n            + [ smeared actual system - smeared unrolled system ]\n\nstraight wing: the bracket is exactly zero (nothing changes)\ncurved wing: the bracket is the finite curvature effect",
           ha="center", fontsize=8.5, bbox=dict(fc="w", ec="0.7"))
    E.set_xlim(-3.6, 3.6); E.set_ylim(-4.4, 1.5); E.axis("off"); E.legend(fontsize=8, loc="lower left")
    E.set_title("(e) The fix: reference straight wing + regularised difference")
    F = ax[1, 2]
    for key, c, lab in (("arc 110 deg, R = 4c", "tab:blue", "arc wing"), ("V3 kite, inviscid, 10 deg", "tab:red", "V3 kite (inviscid)")):
        for name, ls, m, lab2 in (("bound segments on the line (develop before)", "--", "o", "bound on the line"), ("regularised (this work)", "-", "s", "regularised")):
            rows = np.array([(N, rt) for N, rt, _ in conv[f"{key} | {name}"]])
            F.semilogx(rows[:, 0], rows[:, 1], ls, marker=m, color=c, label=f"{lab}: {lab2}")
    F.axhline(1.0, color="k", lw=0.8); F.set_xlabel("number of spanwise panels N"); F.set_ylabel("on-blade induced drag / Trefftz-plane induced drag")
    F.set_title("(f) Result: mesh-independent and equal to Trefftz"); F.legend(fontsize=8); F.grid(alpha=0.3); F.set_ylim(0.8, 1.08)
    fig.suptitle("Why the V3 on-blade induced drag disagrees with the Trefftz plane, and the fix", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "08_curved_bound_vortex_regularisation.png", dpi=150)
    plt.close(fig)


def figure_09(pol):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9})
    D = DATA / "3D_polars_literature"
    cfd = pd.read_csv(D / "CFD_RANS_Rey_10e5_Poland2025_alpha_sweep_beta_0.csv")
    wt = pd.read_csv(D / "windtunnel_alpha_sweep_beta_00_0_Poland_2025_Rey_5e5.csv")
    wtb = pd.read_csv(D / "windtunnel_beta_sweep_alpha_07_4_Poland_2025_Rey_5e5.csv")
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    sty = {
        "old34": dict(color="0.5", ls="--", label="develop before this work (3/4-chord directions)"),
        "on_line": dict(color="tab:blue", ls="-", label="quarter-chord directions, bound segments on the line"),
        "fix": dict(color="tab:red", ls="-", lw=2, label="quarter-chord directions, regularised curved bound vortex"),
    }
    for tag in ("old34", "on_line", "fix"):
        r = np.array(pol["alpha"][tag])
        ax[0, 0].plot(r[:, 0], r[:, 1], **sty[tag]); ax[0, 1].plot(r[:, 0], r[:, 2], **sty[tag]); ax[1, 0].plot(r[:, 0], r[:, 1] / r[:, 2], **sty[tag])
        rb = np.array(pol["beta"][tag]); ax[1, 1].plot(rb[:, 0], rb[:, 2], **sty[tag])
    ax[0, 0].plot(cfd.alpha, cfd.CL, "k^", label="RANS Re 1e6 (Poland 2025)"); ax[0, 1].plot(cfd.alpha, cfd.CD, "k^"); ax[1, 0].plot(cfd.alpha, cfd.CL / cfd.CD, "k^")
    m = (wt.alpha > -4) & (wt.alpha < 20)
    ax[0, 0].errorbar(wt.alpha[m], wt.CL[m], yerr=wt.CL_ci[m], fmt="ko", mfc="w", label="wind tunnel Re 5e5 (Poland 2026), 99% CI")
    ax[0, 1].errorbar(wt.alpha[m], wt.CD[m], yerr=wt.CD_ci[m], fmt="ko", mfc="w"); ax[1, 0].plot(wt.alpha[m], wt.CL[m] / wt.CD[m], "ko", mfc="w")
    mb = wtb.beta >= -0.5
    ax[1, 1].errorbar(wtb.beta[mb], wtb.CD[mb], yerr=wtb.CD_ci[mb], fmt="ko", mfc="w", label="wind tunnel, +beta")
    mb2 = wtb.beta <= 0.5
    ax[1, 1].errorbar(-wtb.beta[mb2], wtb.CD[mb2], yerr=wtb.CD_ci[mb2], fmt="kd", mfc="w", label="wind tunnel, -beta (mirrored)")
    ax[0, 0].set_ylabel("CL"); ax[0, 1].set_ylabel("CD"); ax[1, 0].set_ylabel("L/D"); ax[1, 1].set_ylabel("CD at alpha 7.4 deg")
    for a_ in ax[0]:
        a_.set_xlabel("angle of attack [deg]")
    ax[1, 0].set_xlabel("angle of attack [deg]"); ax[1, 1].set_xlabel("sideslip [deg]")
    for a_ in ax.ravel():
        a_.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8); ax[1, 1].legend(fontsize=8); ax[0, 1].set_ylim(0, 0.35)
    fig.suptitle("TU Delft V3 kite, CAD geometry, 2D RANS Re 5e5 polars, 50 panels, 2.82 m/s: effect of the fix", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "09_v3_polars_curved_bound_vortex_fix.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="only the regularised, on-line and trailed-only variants")
    ap.add_argument("--figures", action="store_true", help="also regenerate docs/figures/08 and 09")
    args = ap.parse_args(argv)
    out = {}
    t0 = time.time()
    out["variants"] = table_variants(args.quick)
    out["arc_self_induction"] = arc_self_induction()
    if not args.quick:
        out["reid_hunsaker"] = table_reid_hunsaker()
    if args.figures:
        print("\nV3 viscous polars for figure 09")
        out["v3_polars"] = v3_polars()
        figure_08(out["variants"], out["arc_self_induction"])
        figure_09(out["v3_polars"])
        print("figures 08 and 09 written to docs/figures")
    json.dump(out, open(OUT / "curved_bound_vortex_validation.json", "w"), indent=1)
    print(f"\nresults written to {OUT / 'curved_bound_vortex_validation.json'}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
