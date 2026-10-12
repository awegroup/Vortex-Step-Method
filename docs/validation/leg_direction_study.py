"""Attached-trailed leg direction: section ribs versus span-perpendicular chord.

Gaunaa, Li & Pirrung (TORQUE 2026) align the attached part of the trailed
vortex with the local chordwise direction, i.e. perpendicular to the
quarter-chord line (CP1). The code keeps the legs along the section ribs,
while the panel frame, the chord and the 3/4-chord control point already
follow CP1. This script compares the two leg definitions with the curved
bound-vortex regularisation in place (the September comparison that led to
keeping the ribs was made with the singular bound-on-line term):

* on-blade over Trefftz induced drag (inviscid) on the arc, the swept-anhedral
  elliptic wing, the V3 kite and a kinked 30 deg swept wing, with the
  attached-trailed (AT) force on and off;
* V3 viscous polars against RANS (Poland 2025), rib versus perpendicular legs.

Perpendicular legs are built per node as the mean of the two adjacent panels'
span-perpendicular chord directions, with the mean of their chords as length
(from the bound point, (1 - ac) c). On a kinked planform adjacent
perpendiculars cross near the kink; the solve then fails or degenerates and
is reported as such.

Result (2026-10-12, table in curved_bound_vortex_report.pdf): on the arc both
leg definitions coincide; on the elliptic wing and the V3 the rib legs give
1.015 and 1.008 of the Trefftz drag with the AT force, the perpendicular
legs 1.025 and 1.038; the perpendicular legs lower the V3 lift by 6 % and
move the viscous polars away from RANS (mean |dCL| 0.031 against 0.020);
on the kinked swept wing they cannot be built. The ribs are kept.

Usage (from the repository root, about 3 minutes)::

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
from VSM.core.Filament import BoundFilament
from VSM.core.Solver import Solver

import curved_bound_vortex_validation as cbv

logging.disable(logging.CRITICAL)
OUT = Path(__file__).resolve().parent
DATA = cbv.DATA


def set_perpendicular_legs(body, ac=0.25, cp=0.75):
    """Replace the rib legs by span-perpendicular legs (in place)."""
    P = body.panels
    n = len(P)
    B = np.array([p.bound_point_1 for p in P] + [P[-1].bound_point_2], float)
    ends = np.zeros_like(B)
    for k in range(n + 1):
        nb = [j for j in (k - 1, k) if 0 <= j < n]
        d = sum(P[j].y_airf for j in nb)
        d = d / np.linalg.norm(d)
        c = np.mean([P[j].chord for j in nb])
        ends[k] = B[k] + (1.0 - ac) * c * d
    for i, p in enumerate(P):
        p._TE_point_1 = ends[i]
        p._TE_point_2 = ends[i + 1]
        p._filaments = [
            BoundFilament(x1=p.bound_point_2, x2=p.bound_point_1),
            BoundFilament(x1=p.bound_point_1, x2=p._TE_point_1),
            BoundFilament(x1=p._TE_point_2, x2=p.bound_point_2),
        ]
        p._control_point = p.aerodynamic_center + (cp - ac) * p.chord * p.y_airf


def ratio(body, at=True):
    r = Solver(allowed_error=1e-7, max_iterations=4000, is_with_attached_trailed_vortex_force=at).solve(body)
    return float(r["drag"] / r["drag_induced_trefftz"]), float(r["cl"]), bool(r["gamma_converged"])


def trefftz_table():
    print("\non-blade / Trefftz induced drag (CL), inviscid, regularised quarter-chord term")
    cases = [
        ("arc 110 deg, R = 4c", cbv.arc, (40, 80)),
        ("elliptic, 10% sweep, 25% anhedral", cbv.elliptic_swept_anhedral, (40, 80)),
        ("V3 kite, inviscid, 10 deg", cbv.v3_inviscid, (40, 80)),
        ("swept 30 deg (kinked)", cbv.swept, (40, 80)),
    ]
    rows = {}
    for case, mk, Ns in cases:
        for legs in ("rib", "perpendicular"):
            for at in (True, False):
                vals = []
                for N in Ns:
                    body = mk(N)
                    if legs == "perpendicular":
                        set_perpendicular_legs(body)
                    try:
                        rt, cl, conv = ratio(body, at)
                        if not np.isfinite(rt):
                            rt = float("nan")
                    except Exception:
                        rt, cl = float("nan"), float("nan")
                    vals.append((N, rt, cl))
                rows[f"{case} | {legs} | AT={at}"] = vals
                print(f"  {case:34s} {legs:13s} AT {'on ':3s} " if at else f"  {case:34s} {legs:13s} AT {'off':3s} ", end="")
                print("  ".join(f"N={N}: {rt:.3f} ({cl:.3f})" for N, rt, cl in vals), flush=True)
    return rows


def v3_vs_rans():
    print("\nV3 viscous polars (CAD, 2D RANS Re 5e5 polars, 50 panels) against RANS, rib vs perpendicular legs")
    cfd = pd.read_csv(DATA / "3D_polars_literature" / "CFD_RANS_Rey_10e5_Poland2025_alpha_sweep_beta_0.csv")
    cfd = cfd[cfd.alpha < 14]
    ref = np.array([0.422646, 0.0, 9.3667])
    out = {}
    for legs in ("rib", "perpendicular"):
        body = BodyAerodynamics.instantiate(
            n_panels=50,
            file_path=DATA / "CAD_derived_geometry" / "aero_geometry_CAD_CFD_NF_combined.yaml",
            spanwise_panel_distribution="uniform",
        )
        if legs == "perpendicular":
            set_perpendicular_legs(body)
        for at in (True, False):
            s = Solver(reference_point=ref, is_with_attached_trailed_vortex_force=at)
            rows = []
            for al, cl_ref, cd_ref in zip(cfd.alpha, cfd.CL, cfd.CD):
                body.va_initialize(2.82, float(al), 0.0)
                r = s.solve(body)
                rows.append((float(al), float(r["cl"]), float(r["cd"]), float(cl_ref), float(cd_ref)))
            e_cl = float(np.mean([abs(a[1] - a[3]) for a in rows]))
            e_cd = float(np.mean([abs(a[2] - a[4]) / a[4] for a in rows]))
            out[f"{legs} | AT={at}"] = {"rows": rows, "mean_abs_dCL": e_cl, "mean_rel_dCD": e_cd}
            print(f"  {legs:13s} AT {'on ' if at else 'off'}: " + "  ".join(f"{a[0]:.0f}deg CL {a[1]:.3f}/{a[3]:.3f} CD {a[2]:.4f}/{a[4]:.4f}" for a in rows))
            print(f"  {'':13s}         mean |dCL| {e_cl:.3f}   mean |dCD|/CD {100 * e_cd:.1f}%", flush=True)
    return out


def main():
    t0 = time.time()
    out = {"trefftz": trefftz_table(), "v3_vs_rans": v3_vs_rans()}
    json.dump(out, open(OUT / "leg_direction_study.json", "w"), indent=1)
    print(f"\nresults written to {OUT / 'leg_direction_study.json'}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
