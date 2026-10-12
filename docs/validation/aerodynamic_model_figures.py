"""Figures 12 and 13 of docs/figures for the aerodynamic model document
(docs/aerodynamic_model/aerodynamic_model.tex):

* 12: classical wings against theory and a vortex-lattice method. Elliptic
  wings of aspect ratio 20 and 4 (straight quarter-chord line) against
  Prandtl's lifting-line result (C_L = 2 pi alpha / (1 + 2/AR), Oswald
  factor 1) and the AeroSandbox VLM, with the LLT and the VSM; the
  rectangular AR 8 wing against the VLM. The JSON also records what the LLT
  and the VSM do when the same planform is drawn with a straight leading
  edge, i.e. a curved quarter-chord line.
* 13: mesh convergence on the inviscid V3 kite and the straight wing: lift,
  on-blade induced drag and the on-blade over Trefftz ratio against the
  number of panels.

Usage (from the repository root, about 3 minutes)::

    python docs/validation/aerodynamic_model_figures.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.core.WingGeometry import Wing

import curved_bound_vortex_validation as cbv

logging.disable(logging.CRITICAL)
OUT = Path(__file__).resolve().parent
FIG = OUT.parent / "figures"
INVISCID = cbv.INVISCID
RHO = 1.225


def elliptic(n, aspect_ratio, root_chord=1.0, straight_quarter_chord=True, **kw):
    """Planar elliptic wing, sections along x. With ``straight_quarter_chord``
    the quarter-chord line is the y axis, as Prandtl's theory assumes; otherwise
    the leading edge is straight and the quarter-chord line is curved."""
    s = aspect_ratio * (np.pi / 4) * root_chord / 2
    wing = Wing(n_panels=n, spanwise_panel_distribution="cosine")
    for y in np.linspace(-s, s, n + 1):
        eta = min(abs(y) / s, 0.999999)
        c = root_chord * np.sqrt(1 - eta**2)
        x_le = -0.25 * c if straight_quarter_chord else 0.0
        wing.add_section(np.array([x_le, y, 0.0]), np.array([x_le + c, y, 0.0]), INVISCID)
    body = BodyAerodynamics([wing], **kw)
    return body, np.pi * root_chord * s / 2


def elliptic_vlm(aspect_ratio, alphas, root_chord=1.0, n_chord=16, n_seg=2):
    """AeroSandbox VLM on the same elliptic wing (None without AeroSandbox)."""
    try:
        import aerosandbox as asb
    except ImportError:
        return None
    s = aspect_ratio * (np.pi / 4) * root_chord / 2
    flat = asb.Airfoil("naca0010")
    xsecs, chords = [], []
    ys = np.linspace(-s, s, 41)
    for y in ys:
        eta = min(abs(y) / s, 0.999999)
        c = root_chord * np.sqrt(1 - eta**2)
        chords.append(c)
        xsecs.append(asb.WingXSec(xyz_le=[-0.25 * c, y, 0.0], chord=c, twist=0.0, airfoil=flat))
    area = float(np.trapz(chords, ys))
    airplane = asb.Airplane(wings=[asb.Wing(symmetric=False, xsecs=xsecs)], s_ref=area, c_ref=1.0, b_ref=1.0)
    cl = []
    for a in alphas:
        r = asb.VortexLatticeMethod(airplane=airplane, op_point=asb.OperatingPoint(velocity=20.0, alpha=float(a)),
                                    spanwise_resolution=n_seg, chordwise_resolution=n_chord,
                                    align_trailing_vortices_with_wind=False).run()
        cl.append(float(r["CL"]))
    return np.array(cl)


def polar(body, area, U, alphas, model="VSM"):
    cl, cd = [], []
    for a in alphas:
        body.va_initialize(U, a, 0.0)
        r = Solver(aerodynamic_model_type=model, allowed_error=1e-7, max_iterations=4000).solve(body)
        q = 0.5 * RHO * U**2 * area
        cl.append(r["lift"] / q)
        cd.append(r["drag"] / q)
    return np.array(cl), np.array(cd)


def figure_12():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    alphas = np.arange(0, 13, 2.0)
    out = {}
    for AR, col in ((20.0, "tab:blue"), (4.0, "tab:red")):
        body, area = elliptic(60, AR)
        cl_th = 2 * np.pi * np.radians(alphas) / (1 + 2 / AR)
        cdi_th = cl_th**2 / (np.pi * AR)
        ax[0].plot(alphas, cl_th, "-", color=col, lw=2, label=f"Prandtl, AR {AR:.0f}")
        ax[1].plot(cl_th**2, cdi_th, "-", color=col, lw=2, label=f"Prandtl, AR {AR:.0f}")
        for model, mk in (("LLT", "s"), ("VSM", "o")):
            cl, cd = polar(body, area, 20.0, alphas, model)
            ax[0].plot(alphas, cl, mk, color=col, mfc="none" if model == "LLT" else col, ms=5, label=f"{model}, AR {AR:.0f}")
            ax[1].plot(cl**2, cd, mk, color=col, mfc="none" if model == "LLT" else col, ms=5, label=f"{model}, AR {AR:.0f}")
            slope = np.polyfit(np.radians(alphas), cl, 1)[0]
            e = np.polyfit(cl[1:] ** 2, cd[1:], 1)[0]
            out[f"elliptic AR {AR:.0f} {model}"] = {"lift_slope_over_theory": float(slope / (2 * np.pi / (1 + 2 / AR))), "oswald": float(1 / (np.pi * AR * e)), "CL": cl.tolist(), "CDi": cd.tolist()}
        vlm_alphas = np.array([4.0, 8.0, 12.0])
        cl_vlm = elliptic_vlm(AR, vlm_alphas)
        if cl_vlm is not None:
            ax[0].plot(vlm_alphas, cl_vlm, "*", color="k", ms=10, label=f"AeroSandbox VLM, AR {AR:.0f}" if AR == 20.0 else None)
            out[f"elliptic AR {AR:.0f} VLM"] = {"alphas": vlm_alphas.tolist(), "CL": cl_vlm.tolist(),
                                              "VSM_CL_over_VLM": (np.interp(vlm_alphas, alphas, out[f"elliptic AR {AR:.0f} VSM"]["CL"]) / cl_vlm).tolist(),
                                              "LLT_CL_over_VLM": (np.interp(vlm_alphas, alphas, out[f"elliptic AR {AR:.0f} LLT"]["CL"]) / cl_vlm).tolist()}
        # the same planform with a straight leading edge: the quarter-chord line is then curved
        body_c, _ = elliptic(60, AR, straight_quarter_chord=False)
        body_c.va_initialize(20.0, 8.0, 0.0)
        for model in ("LLT", "VSM"):
            r = Solver(aerodynamic_model_type=model, allowed_error=1e-7, max_iterations=4000).solve(body_c)
            out[f"elliptic AR {AR:.0f} curved quarter chord {model} on-blade over Trefftz"] = float(r["drag"] / r["drag_induced_trefftz"])
    ax[0].set_xlabel("alpha [deg]"); ax[0].set_ylabel("CL"); ax[0].set_title("elliptic wings: lift against Prandtl")
    ax[1].set_xlabel("CL^2"); ax[1].set_ylabel("CDi"); ax[1].set_title("elliptic wings: induced drag against Prandtl")
    ax[0].legend(fontsize=7, frameon=False, ncol=2); ax[1].legend(fontsize=7, frameon=False, ncol=2)

    # rectangular AR 8 against the VLM referee results (straight AR 8, 5 deg)
    ref = json.load(open(OUT / "vlm_referee.json"))["straight AR 8"]
    vlm = ref["VLM 16x80"]
    alphas_r = np.arange(0, 13, 2.0)
    body = cbv.straight(80)
    cl, cd = polar(body, 8.0, 20.0, alphas_r)
    ax[2].plot(alphas_r, cl, "o-", color="tab:green", ms=4, label="VSM, 80 panels")
    ax[2].plot([5.0], [vlm["CL"]], "k*", ms=11, label="AeroSandbox VLM 16x80, 5 deg")
    ax[2].plot(alphas_r, 2 * np.pi * np.radians(alphas_r) / (1 + 2 / 8), "--", color="0.5", label="Prandtl elliptic, AR 8 (reference slope)")
    ax[2].set_xlabel("alpha [deg]"); ax[2].set_ylabel("CL"); ax[2].set_title("rectangular AR 8 wing")
    ax[2].legend(fontsize=7, frameon=False)
    cl5, cd5 = polar(body, 8.0, 20.0, [5.0])
    out["rectangular AR 8 VSM at 5 deg over VLM"] = {"CL": float(cl5[0] / vlm["CL"]), "CDi": float(cd5[0] / vlm["CDi_trefftz"])}
    fig.tight_layout()
    fig.savefig(FIG / "12_classical_wings.png", dpi=150)
    print("figure 12 written", json.dumps(out, indent=1))
    return out


def figure_13():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    Ns = (10, 20, 40, 80, 160)
    out = {}
    for name, mk, col in (("V3 kite, inviscid, 10 deg", cbv.v3_inviscid, "tab:red"), ("straight AR 8, 5 deg", cbv.straight, "tab:blue")):
        cl, cd, ratio = [], [], []
        for N in Ns:
            body = mk(N)
            r = Solver(allowed_error=1e-7, max_iterations=4000).solve(body)
            cl.append(r["cl"]); cd.append(r["cd"]); ratio.append(r["drag"] / r["drag_induced_trefftz"])
        out[name] = {"N": list(Ns), "CL": cl, "CD": cd, "on_blade_over_trefftz": ratio}
        ax[0].plot(Ns, np.array(cl) / cl[-1], "o-", color=col, label=name)
        ax[1].plot(Ns, np.array(cd) / cd[-1], "o-", color=col, label=name)
        ax[2].plot(Ns, ratio, "o-", color=col, label=name)
    for A, t in zip(ax, ("CL / CL(160 panels)", "CDi / CDi(160 panels)", "on-blade / Trefftz induced drag")):
        A.set_xscale("log", base=2); A.set_xlabel("number of panels"); A.set_ylabel(t); A.legend(fontsize=8, frameon=False)
    ax[2].axhline(1.0, color="k", lw=0.8)
    fig.suptitle("mesh convergence, uniform spacing", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "13_mesh_convergence.png", dpi=150)
    print("figure 13 written", json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    res = {"figure_12": figure_12(), "figure_13": figure_13()}
    json.dump(res, open(OUT / "aerodynamic_model_figures.json", "w"), indent=1)
