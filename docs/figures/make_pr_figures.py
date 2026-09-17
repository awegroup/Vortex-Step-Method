"""Regenerate the figures documenting the consistent lifting-line coupling
(Gaunaa, Li & Pirrung, TORQUE 2026; Li, Gaunaa, Pirrung & Lonbaek, TORQUE
2026; Gaunaa, Sorensen & Li 2024) as implemented in this solver.

Run from the repository root:  python docs/figures/make_pr_figures.py
Takes a few minutes. The pre-change V3 polars cannot be recomputed with the
current code; they are read from v3_polars_before_fixes.json (solver state
before commit 233a238, same geometry, polars, panels and inflow).
"""
import json
import logging
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from VSM.core.BodyAerodynamics import BodyAerodynamics  # noqa: E402
from VSM.core.Solver import Solver  # noqa: E402
from VSM.core.WingGeometry import Wing  # noqa: E402

logging.disable(logging.CRITICAL)
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
V3_GEOM = ROOT / "data/TUDELFT_V3_KITE/CAD_derived_geometry/aero_geometry_CAD_CFD_NF_combined.yaml"
V3_BREUKELS = ROOT / "data/TUDELFT_V3_KITE/CAD_derived_geometry/aero_geometry_CAD_breukels_regression.yaml"
V3_INVISCID = ROOT / "tests/BodyAerodynamics/aero_geometry_CAD_inviscid.yaml"
LIT = ROOT / "data/TUDELFT_V3_KITE/3D_polars_literature"
REF_POINT = np.array([0.422646, 0.0, 9.3667])
RHO = 1.225


def inviscid_polar():
    a = np.radians(np.linspace(-25, 25, 101))
    return np.column_stack((a, 2 * np.pi * a, 0 * a, 0 * a))


def rect_wing(n, aspect_ratio, sweep_deg=0.0, dist="uniform"):
    polar = inviscid_polar()
    w = Wing(n_panels=n, spanwise_panel_distribution=dist)
    for y in np.linspace(-aspect_ratio / 2, aspect_ratio / 2, n + 1):
        x = abs(y) * np.tan(np.radians(sweep_deg))
        w.add_section(np.array([x, y, 0.0]), np.array([x + 1.0, y, 0.0]), polar)
    return BodyAerodynamics([w])


def elliptic_wing(n, aspect_ratio):
    polar = inviscid_polar()
    b = aspect_ratio * np.pi / 4
    w = Wing(n_panels=n, spanwise_panel_distribution="cosine")
    for y in np.linspace(-b / 2, b / 2, n + 1):
        c = np.sqrt(max(1e-6, 1 - (2 * y / b) ** 2))
        w.add_section(np.array([-c / 4, y, 0.0]), np.array([3 * c / 4, y, 0.0]), polar)
    return BodyAerodynamics([w])


def v3(n=50, geom=V3_GEOM):
    return BodyAerodynamics.instantiate(n_panels=n, file_path=geom, spanwise_panel_distribution="uniform")


# --------------------------------------------------------------------------
# 1. V3 polars before / after, against RANS and wind tunnel
# --------------------------------------------------------------------------
def fig_polars():
    before = json.loads((OUT / "v3_polars_before_fixes.json").read_text())
    alphas = list(range(-5, 25, 2))
    betas = list(range(0, 22, 2))
    body = v3()

    def sweep(solver):
        rows_a, rows_b = [], []
        for a in alphas:
            body.va_initialize(2.82, a, 0.0)
            r = solver.solve(body)
            rows_a.append([a, r["cl"], r["cd"], r["cs"], r["cmx"], r["cmy"], r["cmz"]])
        for b in betas:
            body.va_initialize(2.82, 12.5, b)
            r = solver.solve(body)
            rows_b.append([b, r["cl"], r["cd"], r["cs"], r["cmx"], r["cmy"], r["cmz"]])
        return np.array(rows_a), np.array(rows_b)

    series = [
        ("before (3/4c directions, old default)", np.array(before["alpha"]["aoa0"]), np.array(before["beta"]["aoa0"]), dict(color="k", ls="--")),
    ]
    for label, kw, st in (
        ("after, 3/4c directions", dict(is_aoa_corrected=False), dict(color="tab:blue")),
        ("after, 1/4c directions (LL-Gaunaa)", dict(is_aoa_corrected=True), dict(color="tab:red")),
        ("after, 1/4c, no attached-trailed force", dict(is_aoa_corrected=True, is_with_attached_trailed_vortex_force=False), dict(color="tab:red", ls=":")),
    ):
        A, B = sweep(Solver(reference_point=REF_POINT, **kw))
        series.append((label, A, B, st))

    cfd_a = pd.read_csv(LIT / "CFD_RANS_Rey_10e5_Poland2025_alpha_sweep_beta_0.csv")
    wt_a = pd.read_csv(LIT / "windtunnel_alpha_sweep_beta_00_0_Poland_2025_Rey_5e5.csv")
    cfd_b = pd.read_csv(LIT / "CFD_RANS_Re1e6_beta_sweep_alpha_13_Vire2022_CorrectedByPoland2025.csv")
    wt_b = pd.read_csv(LIT / "windtunnel_beta_sweep_alpha_12_5_Poland_2025_Rey_5e5.csv")

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    for j, (ci, name) in enumerate([(1, "CL"), (2, "CD")]):
        for label, A, B, st in series:
            ax[0, j].plot(A[:, 0], A[:, ci], label=label, **st)
        ax[0, j].plot(cfd_a["alpha"], cfd_a[name], "s", color="tab:green", label="RANS (Poland 2025)")
        ax[0, j].errorbar(wt_a["alpha"], wt_a[name], yerr=wt_a[name + "_ci"], fmt="o", color="tab:orange", label="wind tunnel (Poland 2025)")
        ax[0, j].set_xlabel("alpha [deg]"); ax[0, j].set_ylabel(name); ax[0, j].grid(alpha=0.3)
    for label, A, B, st in series:
        ax[0, 2].plot(A[:, 0], A[:, 1] / A[:, 2], label=label, **st)
    ax[0, 2].plot(cfd_a["alpha"], cfd_a["CL"] / cfd_a["CD"], "s", color="tab:green")
    ax[0, 2].plot(wt_a["alpha"], wt_a["CL"] / wt_a["CD"], "o", color="tab:orange")
    ax[0, 2].set_xlabel("alpha [deg]"); ax[0, 2].set_ylabel("CL/CD"); ax[0, 2].grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8, loc="upper left"); ax[0, 0].set_title("alpha sweep, beta = 0")
    for j, (ci, name) in enumerate([(1, "CL"), (2, "CD"), (3, "CS")]):
        for label, A, B, st in series:
            ax[1, j].plot(B[:, 0], B[:, ci], label=label, **st)
        ax[1, j].plot(cfd_b["beta"], cfd_b[name], "s", color="tab:green", label="RANS alpha 13 (Vire 2022)")
        ax[1, j].errorbar(wt_b["beta"].abs(), wt_b[name] * (-1 if name == "CS" else 1), yerr=wt_b[name + "_ci"], fmt="o", color="tab:orange", label="wind tunnel alpha 12.5 (|beta|)")
        ax[1, j].set_xlabel("beta [deg]"); ax[1, j].set_ylabel(name); ax[1, j].grid(alpha=0.3)
    ax[1, 0].set_title("beta sweep, alpha = 12.5"); ax[1, 1].legend(fontsize=8)
    fig.suptitle("TUDELFT V3 kite, CAD geometry, CFD+NeuralFoil polars, 50 panels, 2.82 m/s: solver before and after the consistency fixes")
    fig.tight_layout(); fig.savefig(OUT / "01_v3_polars_before_after.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------
# 2. Panel frame, drag direction and viscous-correction angle on the V3
# --------------------------------------------------------------------------
def fig_frame():
    body = v3(30, V3_BREUKELS)
    P = body.panels
    n = len(P)
    S = body.wings[0].spanwise_direction / np.linalg.norm(body.wings[0].spanwise_direction)
    # old frame: chord along the rib, normal from the LE step
    old_y, old_x, skew, drag_err = [], [], [], []
    a = np.radians(10.0)
    for p in P:
        y_raw = p.control_point - p.aerodynamic_center; y_raw /= np.linalg.norm(y_raw)
        x_raw = np.cross(y_raw, p.LE_point_1 - p.LE_point_2); x_raw /= np.linalg.norm(x_raw)
        old_y.append(y_raw); old_x.append(x_raw)
        skew.append(np.degrees(np.arcsin(abs(y_raw @ p.z_airf))))
        d = np.cos(a) * y_raw + np.sin(a) * x_raw; d /= np.linalg.norm(d)
        L = np.cross(d, p.z_airf); L /= np.linalg.norm(L)
        D_old = np.cross(S, L); D_old /= np.linalg.norm(D_old)
        drag_err.append(np.degrees(np.arccos(np.clip(D_old @ d, -1, 1))))
    # viscous-correction angle beta: old (geometry only) vs new (flow) at 0 and 10 deg sideslip
    betas = {}
    for side in (0.0, 10.0):
        body.va_initialize(20.0, 10.0, side)
        s = Solver(is_aoa_corrected=True)
        r = s.solve(body)
        v_rel = s.compute_relative_velocity(np.asarray(r["gamma_distribution"]))
        b_old = [np.degrees(np.arcsin(np.clip((np.cos(a) * old_y[i] + np.sin(a) * old_x[i]) @ P[i].z_airf, -1, 1))) for i in range(n)]
        b_new = [np.degrees(np.arctan2(v_rel[i] @ P[i].z_airf, np.linalg.norm(v_rel[i] - (v_rel[i] @ P[i].z_airf) * P[i].z_airf))) for i in range(n)]
        betas[side] = (np.array(b_old), np.array(b_new))

    idx = np.arange(n)
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    sel = range(0, 7)
    LE = np.array([p.LE_point_1 for p in P] + [P[-1].LE_point_2]); TE = np.array([p.TE_point_1 for p in P] + [P[-1].TE_point_2])
    ax.plot(*np.vstack([LE[:8], TE[:8][::-1], LE[:1]]).T, color="0.6", lw=0.8)
    for i in sel:
        c = P[i].aerodynamic_center; L_ = 0.25 * P[i].chord
        ax.quiver(*c, *(L_ * old_y[i]), color="tab:blue", arrow_length_ratio=0.15)
        ax.quiver(*c, *(L_ * old_x[i]), color="tab:blue", arrow_length_ratio=0.15, ls="--")
        ax.quiver(*c, *(L_ * P[i].y_airf), color="tab:red", arrow_length_ratio=0.15)
        ax.quiver(*c, *(L_ * P[i].x_airf), color="tab:red", arrow_length_ratio=0.15, ls="--")
        ax.quiver(*c, *(L_ * P[i].z_airf), color="k", arrow_length_ratio=0.15)
    ax.set_title("V3 left tip: panel axes. blue = old (rib chord, LE-step normal), red = new (span-perpendicular), black = span", fontsize=9)
    ax.view_init(elev=20, azim=-35); ax.set_box_aspect((1.4, 1, 1.2))
    ax = fig.add_subplot(2, 2, 2)
    ax.bar(idx - 0.2, skew, 0.4, label="chord-to-span skew of the rib frame [deg]")
    ax.bar(idx + 0.2, drag_err, 0.4, label="angle of the old drag direction to the local flow [deg]")
    ax.set_xlabel("panel (tip to tip)"); ax.set_ylabel("deg"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("old frame skew and old profile-drag misdirection, alpha 10", fontsize=10)
    for k, side in enumerate((0.0, 10.0)):
        ax = fig.add_subplot(2, 2, 3 + k)
        b_old, b_new = betas[side]
        ax.bar(idx - 0.2, b_old, 0.4, label="old: from the airfoil-plane vector (geometry only)")
        ax.bar(idx + 0.2, b_new, 0.4, label="new: from the relative velocity (flow)")
        ax.set_xlabel("panel (tip to tip)"); ax.set_ylabel("beta [deg]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title(f"spanwise-flow angle beta of the Gaunaa 2024 viscous correction, alpha 10, sideslip {side:.0f}", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "02_v3_frame_drag_direction_viscous_beta.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------
# 3. Trefftz-plane check of the on-blade induced drag
# --------------------------------------------------------------------------
def fig_trefftz():
    cases = [
        ("straight AR 4", rect_wing(60, 4), 5.0), ("straight AR 8", rect_wing(60, 8), 5.0), ("straight AR 20", rect_wing(60, 20), 5.0),
        ("elliptic AR 8", elliptic_wing(60, 8), 5.0), ("swept 30 AR 8", rect_wing(60, 8, 30.0), 5.0),
        ("V3 inviscid 5", v3(50, V3_INVISCID), 5.0), ("V3 inviscid 10", v3(50, V3_INVISCID), 10.0), ("V3 inviscid 15", v3(50, V3_INVISCID), 15.0),
    ]
    settings = [
        ("1/4c directions + AT force", dict(is_aoa_corrected=True), "tab:red"),
        ("1/4c directions, no AT force", dict(is_aoa_corrected=True, is_with_attached_trailed_vortex_force=False), "tab:orange"),
        ("3/4c directions (old default)", dict(is_aoa_corrected=False), "tab:blue"),
    ]
    ratios = np.zeros((len(cases), len(settings)))
    for i, (label, body, aoa) in enumerate(cases):
        for j, (_, kw, _) in enumerate(settings):
            body.va_initialize(10.0, aoa, 0.0)
            r = Solver(allowed_error=1e-8, **kw).solve(body)
            ratios[i, j] = r["drag"] / r["drag_induced_trefftz"]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(cases)); w = 0.26
    for j, (label, _, color) in enumerate(settings):
        ax.bar(x + (j - 1) * w, ratios[:, j], w, label=label, color=color)
        for i in range(len(cases)):
            ax.text(x[i] + (j - 1) * w, ratios[i, j] + 0.01, f"{ratios[i, j]:.2f}", ha="center", fontsize=7)
    ax.axhline(1.0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cases], rotation=15)
    ax.set_ylabel("on-blade induced drag / Trefftz-plane induced drag")
    ax.set_title("Inviscid wings: on-blade induced drag against the far-wake (Trefftz) reference on the same circulation")
    ax.legend(); ax.grid(alpha=0.3, axis="y"); ax.set_ylim(0.6, 1.5)
    fig.tight_layout(); fig.savefig(OUT / "03_trefftz_check.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------
# 4. Attached-trailed vortex force on the V3
# --------------------------------------------------------------------------
def fig_at_force():
    body = v3(30, V3_INVISCID)
    body.va_initialize(10.0, 10.0, 0.0)
    r = Solver(is_aoa_corrected=True, allowed_error=1e-8).solve(body)
    F = np.array(r["F_attached_trailed_distribution"])
    P = body.panels
    B = np.array([p.bound_point_1 for p in P] + [P[-1].bound_point_2])
    TE = np.array([p.TE_point_1 for p in P] + [P[-1].TE_point_2])
    pts = B + (2 / 3) * (TE - B)
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    LE = np.array([p.LE_point_1 for p in P] + [P[-1].LE_point_2])
    ax.plot(*np.vstack([LE, TE[::-1], LE[:1]]).T, color="0.6", lw=0.8)
    ax.plot(*B.T, color="tab:red", lw=2, label="bound vortex")
    for j in range(len(B)):
        ax.plot(*np.vstack([B[j], TE[j]]).T, color="tab:blue", lw=0.8)
    scale = 0.4  # metres per newton, for visibility
    ax.quiver(*pts.T, *(scale * F).T, color="tab:green", arrow_length_ratio=0.2, lw=1.5, label="attached-trailed force (0.4 m per N)")
    ax.set_title(f"V3 inviscid, alpha 10: Kutta-Joukowski force on the chordwise legs\n(total {np.linalg.norm(F.sum(axis=0)):.1f} N of {r['lift']:.0f} N lift)", fontsize=10)
    ax.view_init(elev=25, azim=-60); ax.set_box_aspect((1, 2.3, 1.2)); ax.legend(fontsize=8)
    ax = fig.add_subplot(1, 2, 2)
    e_l = np.array([0, 0, 1.0]); e_d = np.array([np.cos(np.radians(10)), 0, np.sin(np.radians(10))])
    ax.bar(np.arange(len(B)) - 0.2, F @ e_l, 0.4, label="vertical component [N]")
    ax.bar(np.arange(len(B)) + 0.2, F @ e_d, 0.4, label="along freestream [N]")
    ax.set_xlabel("section boundary (tip to tip)"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("per-boundary attached-trailed force (up = adds lift): largest over the curved outboard region, zero at mid-span", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "04_v3_attached_trailed_force.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------
# 5. Adaptive relaxation factor
# --------------------------------------------------------------------------
def fig_relaxation():
    body = v3()
    alphas = [-5, 0, 5, 10, 15, 20]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for model, color in (("VSM", "tab:red"), ("LLT", "tab:blue")):
        for rf, ls, label in ((0.01, "--", "fixed 0.01 (old default)"), (None, "-", "adaptive (Li et al. 2026 limit)")):
            its, omegas = [], []
            for a in alphas:
                body.va_initialize(2.82, a, 0.0)
                s = Solver(aerodynamic_model_type=model, relaxation_factor=rf, reference_point=REF_POINT)
                s.solve(body); its.append(s.last_iterations); omegas.append(s.relaxation_factor_used)
            ax[0].plot(alphas, its, ls=ls, color=color, marker="o", label=f"{model}, {label} (omega {omegas[0]:.3f})")
    ax[0].set_xlabel("alpha [deg]"); ax[0].set_ylabel("iterations of the base loop"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[0].set_title("V3 kite, 50 panels: iterations to converge", fontsize=10)
    # ratio of the true limit to the bound on rectangular wings
    rows = []
    for n_, ar in ((20, 5), (50, 5), (100, 5), (50, 20), (50, 2)):
        for model in ("VSM", "LLT"):
            b = rect_wing(n_, ar); b.va_initialize(10.0, 5.0, 0.0)
            s = Solver(aerodynamic_model_type=model, relaxation_factor=0.05, max_iterations=20000, allowed_error=1e-9)
            r = s.solve(b); g = np.asarray(r["gamma_distribution"]); m = g.size
            eps = 1e-6 * max(1.0, np.abs(g).max()); J = np.zeros((m, m))
            for k in range(m):
                e = np.zeros(m); e[k] = eps
                J[:, k] = (s._fixed_point_target(g + e)[0] - s._fixed_point_target(g - e)[0]) / (2 * eps)
            lam = np.linalg.eigvals(np.eye(m) - J).real.max()
            raw_bound = 2.0 / (1.0 + 0.25 * max(p.chord / p.width for p in b.panels) * 2 * np.pi)
            rows.append((f"N{n_} AR{ar}", model, (2.0 / lam) / raw_bound))
    labels = sorted(set(r_[0] for r_ in rows), key=lambda x: rows.index(next(r_ for r_ in rows if r_[0] == x)))
    x = np.arange(len(labels))
    for j, (model, color) in enumerate((("VSM", "tab:red"), ("LLT", "tab:blue"))):
        vals = [next(r_[2] for r_ in rows if r_[0] == lab and r_[1] == model) for lab in labels]
        ax[1].bar(x + (j - 0.5) * 0.35, vals, 0.35, color=color, label=model)
    ax[1].axhline(1.0, color="tab:blue", lw=1, ls=":"); ax[1].axhline(0.5, color="tab:red", lw=1, ls=":")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels); ax[1].set_ylabel("true omega_max / Li et al. 2026 bound")
    ax[1].set_title("measured stability limit against the paper's bound: exact for LLT, half for VSM", fontsize=10)
    ax[1].legend(); ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "05_relaxation_factor.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------
# 6. Wake direction under a yaw rate
# --------------------------------------------------------------------------
def fig_wake_direction():
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for k, (body, rate, title) in enumerate((
        (rect_wing(16, 8), 1.0, "straight wing AR 8, yaw rate 1 rad/s, 10 m/s (exaggerated for visibility)"),
        (v3(30, V3_BREUKELS), 1.0, "V3 kite, yaw rate 1 rad/s about the reference point, 10 m/s"),
    )):
        body.va_initialize(10.0, 5.0, 0.0, body_rates=np.array([0.0, 0.0, rate]), reference_point=REF_POINT if k else None)
        P = body.panels
        TE = np.array([p.TE_point_1 for p in P] + [P[-1].TE_point_2])
        LE = np.array([p.LE_point_1 for p in P] + [P[-1].LE_point_2])
        va = np.array([p.va for p in P]); units = va / np.linalg.norm(va, axis=1)[:, None]
        mean = np.mean(va, axis=0); mean /= np.linalg.norm(mean)
        a = ax[k]
        a.plot(*np.vstack([LE, TE[::-1], LE[:1]])[:, :2].T, color="0.6")
        O = 0.5 * (TE[:-1] + TE[1:])
        a.quiver(O[:, 0], O[:, 1], np.full(len(P), 2.0 * mean[0]), np.full(len(P), 2.0 * mean[1]), angles="xy", scale_units="xy", scale=1, color="tab:blue", width=0.004, label="old: one wake direction, mean freestream")
        a.quiver(O[:, 0], O[:, 1], 2.0 * units[:, 0], 2.0 * units[:, 1], angles="xy", scale_units="xy", scale=1, color="tab:red", width=0.004, label="new: each ring along its panel's apparent velocity")
        a.set_xlim(LE[:, 0].min() - 0.5, TE[:, 0].max() + 2.5)
        a.set_aspect("equal"); a.grid(alpha=0.3); a.legend(fontsize=8, loc="upper left"); a.set_title(title, fontsize=10); a.set_xlabel("x"); a.set_ylabel("y")
    fig.suptitle("Frozen wake direction (top view): identical in uniform inflow, locally aligned under body rates")
    fig.tight_layout(); fig.savefig(OUT / "06_wake_direction_yaw_rate.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------
# 7. Investigated and rejected: chordwise legs perpendicular to the quarter-chord line
# --------------------------------------------------------------------------
def _closest_on_polyline(p0, d, poly):
    best = (None, np.inf)
    for a, b in zip(poly[:-1], poly[1:]):
        u, v, w = d, b - a, p0 - a
        A = np.array([[u @ u, -u @ v], [u @ v, -v @ v]]); rhs = np.array([-(w @ u), -(w @ v)])
        try:
            t, s_ = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            continue
        s_ = np.clip(s_, 0, 1); t = max(t, 0.0)
        q = a + s_ * v; dist = np.linalg.norm(p0 + t * d - q)
        if dist < best[1]:
            best = (q, dist)
    return best


def fig_leg_variants():
    """The 2026 paper aligns the attached legs with the local chord, which its
    CP1 defines perpendicular to the quarter-chord line; this solver keeps them
    along the input ribs. Both perpendicular constructions were tried: on a
    streamwise-ribbed swept wing the legs cross neighbouring control points
    (NaN), on the V3 they move CL and CD away from RANS (mean errors 0.020 /
    1.5% rib, 0.031 / 4.7% fixed-length, 0.057 / 9.4% to-trailing-edge)."""
    def systems(body):
        P = body.panels; n = len(P)
        LE = np.array([p.LE_point_1 for p in P] + [P[-1].LE_point_2]); TE = np.array([p.TE_point_1 for p in P] + [P[-1].TE_point_2])
        B = np.array([p.bound_point_1 for p in P] + [P[-1].bound_point_2])
        legs_perp = np.zeros_like(TE)
        for j in range(n + 1):
            nb = [k for k in (j - 1, j) if 0 <= k < n]
            d = sum(P[k].y_airf for k in nb); d /= np.linalg.norm(d)
            legs_perp[j] = _closest_on_polyline(B[j], d, TE)[0]
        return LE, TE, B, legs_perp

    def draw(ax, geo, proj=None, title=""):
        LE, TE, B, Lp = geo
        pr = (lambda X: X) if proj is None else proj
        ax.plot(*pr(np.vstack([LE, TE[::-1], LE[:1]])).T, color="0.6", lw=0.8, label="rib outline")
        ax.plot(*pr(B).T, color="tab:red", lw=2, label="bound vortex (1/4c)")
        for j in range(len(B)):
            ax.plot(*pr(np.vstack([B[j], TE[j]])).T, color="tab:blue", lw=1.2, label="leg along the rib (kept)" if j == 0 else None)
            ax.plot(*pr(np.vstack([B[j], Lp[j]])).T, color="tab:green", lw=1.2, label="leg perpendicular to the 1/4c line (rejected)" if j == 0 else None)
        ax.set_title(title, fontsize=10)

    g3 = systems(v3(30, V3_BREUKELS))
    fig = plt.figure(figsize=(18, 6.5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    draw(ax, g3, title="V3 kite, 30 panels (3D)"); ax.view_init(elev=28, azim=-60); ax.set_box_aspect((1, 2.3, 1.2)); ax.legend(fontsize=7, loc="upper left")
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    LE, TE, B, Lp = g3
    draw(ax, (LE[:8], TE[:8], B[:7], Lp[:7]), title="V3 left tip zoom: perpendicular legs fan inboard to the trailing edge"); ax.view_init(elev=20, azim=-35); ax.set_box_aspect((1.4, 1, 1.2))
    ax = fig.add_subplot(1, 3, 3)
    draw(ax, systems(rect_wing(16, 8, 30.0)), proj=lambda X: X[:, :2], title="30 deg swept wing, streamwise ribs (top view):
perpendicular legs cross neighbouring panels")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Chordwise (attached-trailed) legs: along the input ribs (kept) versus perpendicular to the quarter-chord line (tried and rejected)")
    fig.tight_layout(); fig.savefig(OUT / "07_leg_variants_rejected.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    import sys
    fns = [fig_polars, fig_frame, fig_trefftz, fig_at_force, fig_relaxation, fig_wake_direction, fig_leg_variants]
    if len(sys.argv) > 1:
        fns = [f for f in fns if f.__name__ in sys.argv[1:]]
    for fn in fns:
        t = time.time(); fn(); print(f"{fn.__name__}: {time.time() - t:.0f} s")
