"""Lifting-surface referee for the attached-trailed leg direction.

The Trefftz plane checks the induced drag of a given circulation but cannot
judge the lift. A vortex-lattice method (VLM) can: for inviscid flat-plate
sections it solves the same potential-flow problem as the lifting line with
``Cl = 2 pi alpha``, with full chordwise resolution. This script runs the VLM
of AeroSandbox (Sharpe, MIT, ``pip install aerosandbox``; horseshoe vortices,
trailing legs along the body x axis, near-field forces) and the VSM with the
three leg options (``rib``, ``streamwise``, ``perpendicular``) on

* a straight AR 8 wing, a yawed wing (uniform 30 deg sweep, no kink) and a
  kinked 30 deg swept wing, chord 1, span 8, at 5 deg;
* the circular-arc wing (110 deg, R = 4 c) at 8 deg;
* the TORQUE 2026 swept-anhedral elliptic wing at 10 deg;
* the inviscid V3 kite (CAD leading and trailing edges, flat-plate sections)
  at 10 deg. AeroSandbox takes its sections perpendicular to the local span,
  so the skewed V3 ribs are unskewed there (chord = rib length, twist from
  the rib's x-z angle); the surface differs from the VSM one at the tips.

For the induced drag two VLM numbers are given: AeroSandbox's near-field
value, and the Trefftz-plane drag of the VLM's own strip circulation on the
wake trace. The two agree on planar wings; on the non-planar wings the
near-field value is low (the trailing legs are not aligned with the wind),
so the Trefftz value is the one to compare with.

Results go to ``vlm_referee.json`` and into the report. About 4 minutes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import yaml

import curved_bound_vortex_validation as cbv
from VSM.core.Solver import Solver

try:
    import aerosandbox as asb
except ImportError as err:  # pragma: no cover
    raise SystemExit("vlm_referee.py needs AeroSandbox: pip install aerosandbox") from err

logging.disable(logging.CRITICAL)
OUT = Path(__file__).resolve().parent
FLAT = asb.Airfoil("naca0010")  # symmetric: the VLM sees a flat camber line
LEGS = ("rib", "streamwise", "perpendicular")
RHO = 1.225


def strip_circulation(vlm, n_chord, n_segments):
    """Chordwise sum of the horseshoe strengths per spanwise strip, with the
    strip's trailing-edge corners; handles the three panel orderings
    AeroSandbox produces."""
    G = np.asarray(vlm.vortex_strengths)
    te = np.asarray(vlm.is_trailing_edge, bool)
    n_strip = len(G) // n_chord
    if np.all(te.reshape(n_strip, n_chord)[:, -1]):
        Gs = G.reshape(n_strip, n_chord).sum(axis=1)
        te_idx = np.arange(n_strip) * n_chord + (n_chord - 1)
    elif np.all(te.reshape(n_chord, n_strip)[-1]):
        Gs = G.reshape(n_chord, n_strip).sum(axis=0)
        te_idx = (n_chord - 1) * n_strip + np.arange(n_strip)
    else:
        ns_seg = n_strip // n_segments
        te3 = te.reshape(n_segments, n_chord, ns_seg)
        if not np.all(te3[:, -1, :]):
            raise RuntimeError("AeroSandbox panel ordering not recognised")
        Gs = G.reshape(n_segments, n_chord, ns_seg).sum(axis=1).ravel()
        te_idx = np.array(
            [(g * n_chord + n_chord - 1) * ns_seg + j for g in range(n_segments) for j in range(ns_seg)]
        )
    bl = np.asarray(vlm.back_left_vertices)[te_idx]
    br = np.asarray(vlm.back_right_vertices)[te_idx]
    return Gs, bl, br


def trefftz_drag(G, bl, br, alpha_deg):
    """Trefftz-plane induced drag of strip circulations ``G`` whose wake
    leaves at the strip corners ``bl -> br``, projected onto the plane
    perpendicular to the freestream (sideslip zero). Each strip edge sheds
    a line vortex of strength equal to the jump in ``G``."""
    al = np.radians(alpha_deg)
    ex = np.array([np.cos(al), 0.0, np.sin(al)])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.cross(ex, ey)
    proj = lambda p: np.column_stack((p @ ey, p @ ez))
    L, R = proj(bl), proj(br)
    order = np.argsort(0.5 * (L[:, 0] + R[:, 0]))
    G, L, R = G[order], L[order], R[order]
    edges = np.vstack([L[:1], 0.5 * (R[:-1] + L[1:]), R[-1:]])
    dG = np.concatenate(([G[0]], G[1:] - G[:-1], [-G[-1]]))
    mid = 0.5 * (L + R)
    tang = R - L
    ds = np.linalg.norm(tang, axis=1)
    tang = tang / ds[:, None]
    nrm = np.column_stack((-tang[:, 1], tang[:, 0]))
    drag = 0.0
    for j in range(len(G)):
        r = mid[j][None, :] - edges
        r2 = np.einsum("ij,ij->i", r, r)
        v = (dG / (2 * np.pi * r2))[:, None] * np.column_stack((-r[:, 1], r[:, 0]))
        drag += 0.5 * RHO * G[j] * float(np.sum(v @ nrm[j])) * ds[j]
    return drag


def run_vlm(xsecs, area, U, alpha, n_chord, n_span_per_segment):
    airplane = asb.Airplane(
        wings=[asb.Wing(symmetric=False, xsecs=xsecs)], s_ref=area, c_ref=1.0, b_ref=1.0
    )
    vlm = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(velocity=U, alpha=alpha),
        spanwise_resolution=n_span_per_segment,
        chordwise_resolution=n_chord,
        align_trailing_vortices_with_wind=False,
    )
    r = vlm.run()
    G, bl, br = strip_circulation(vlm, n_chord, len(xsecs) - 1)
    q = 0.5 * RHO * U**2 * area
    return {
        "CL": float(r["CL"]),
        "CDi_near": float(r["CD"]),
        "CDi_trefftz": float(trefftz_drag(G, bl, br, alpha) / q),
        "gamma": G.tolist(),
        "y": (0.5 * (bl[:, 1] + br[:, 1])).tolist(),
    }


def run_vsm(make, N, legs, area, U):
    body = make(N, leg_direction=legs)
    r = Solver(allowed_error=1e-7, max_iterations=4000).solve(body)
    q = 0.5 * RHO * U**2 * area
    return {
        "CL": float(r["cl"]),
        "CDi_blade": float(r["drag"] / q),
        "CDi_trefftz": float(r["drag_induced_trefftz"] / q),
        "converged": bool(r["gamma_converged"]),
        "gamma": np.asarray(r["gamma_distribution"]).ravel().tolist(),
        "y": [float(p.aerodynamic_center[1]) for p in body.panels],
    }


def show(name, key, v):
    if "CDi_near" in v:
        print(f"  {name:22s} {key:22s}: CL {v['CL']:.4f}  CDi near {v['CDi_near']:.5f}  Trefftz {v['CDi_trefftz']:.5f}")
    else:
        print(f"  {name:22s} {key:22s}: CL {v['CL']:.4f}  CDi blade {v['CDi_blade']:.5f}  Trefftz {v['CDi_trefftz']:.5f}")


def v3_xsecs():
    data = yaml.safe_load(open(cbv.DATA / "CAD_derived_geometry" / "aero_geometry_CAD_CFD_NF_combined.yaml"))
    ribs = [(np.array(r[1:4], float), np.array(r[4:7], float)) for r in data["wing_sections"]["data"]]
    ribs.sort(key=lambda t: t[0][1])
    xsecs = []
    for le, te in ribs:
        cvec = te - le
        xsecs.append(asb.WingXSec(xyz_le=le.tolist(), chord=float(np.linalg.norm(cvec)),
                                  twist=float(np.degrees(np.arctan2(-cvec[2], cvec[0]))), airfoil=FLAT))
    return xsecs


def main():
    out = {}
    tan30 = np.tan(np.radians(30.0))
    cases = {
        "straight AR 8": ([[0.0, -4.0, 0.0], [0.0, 4.0, 0.0]], cbv.straight),
        "yawed 30 deg": ([[-4.0 * tan30, -4.0, 0.0], [4.0 * tan30, 4.0, 0.0]], cbv.yawed),
        "kinked swept 30 deg": ([[4.0 * tan30, -4.0, 0.0], [0.0, 0.0, 0.0], [4.0 * tan30, 4.0, 0.0]], cbv.swept),
    }
    print("planar wings, chord 1, span 8, 5 deg, U = 20 m/s")
    for name, (le_points, make) in cases.items():
        xsecs = [asb.WingXSec(xyz_le=p, chord=1.0, twist=0.0, airfoil=FLAT) for p in le_points]
        n_seg = len(le_points) - 1
        row = {}
        for n_chord, n_span in ((8, 40), (16, 80)):
            key = f"VLM {n_chord}x{n_span}"
            row[key] = run_vlm(xsecs, 8.0, 20.0, 5.0, n_chord, n_span // n_seg)
            show(name, key, row[key])
        for legs in LEGS:
            for N in (40, 80):
                key = f"VSM {legs} N={N}"
                row[key] = run_vsm(make, N, legs, 8.0, 20.0)
                show(name, key, row[key])
        out[name] = row

    print("\ncircular arc, 110 deg, R = 4 c, 8 deg, U = 10 m/s")
    xsecs = []
    for th in np.radians(np.linspace(-55.0, 55.0, 23)):
        xsecs.append(asb.WingXSec(xyz_le=[0.0, 4.0 * np.sin(th), 4.0 * np.cos(th)], chord=1.0, twist=0.0, airfoil=FLAT))
    body = cbv.arc(40)
    area = float(body.wings[0].compute_projected_area())
    row = {"projected_area": area}
    for n_chord, n_seg in ((8, 2), (16, 4)):
        key = f"VLM {n_chord}x{22 * n_seg}"
        row[key] = run_vlm(xsecs, area, 10.0, 8.0, n_chord, n_seg)
        show("arc", key, row[key])
    for legs in LEGS:
        for N in (40, 80):
            key = f"VSM {legs} N={N}"
            row[key] = run_vsm(cbv.arc, N, legs, area, 10.0)
            show("arc", key, row[key])
    out["arc"] = row

    print("\nelliptic, 10% sweep, 25% anhedral, 10 deg, U = 10 m/s")
    s = 8.0 * (np.pi / 4) / 2
    ys = np.linspace(-s, s, 41)
    xsecs, chords = [], []
    for y in ys:
        eta = min(abs(y) / s, 0.999999)
        c = np.sqrt(1 - eta**2)
        chords.append(c)
        xsecs.append(asb.WingXSec(xyz_le=[0.10 * s * eta**2, y, -0.25 * s * eta**2], chord=c, twist=0.0, airfoil=FLAT))
    area = float(np.trapz(chords, ys))
    row = {}
    for n_chord, n_seg in ((8, 1), (16, 2), (16, 4)):
        key = f"VLM {n_chord}x{40 * n_seg}"
        row[key] = run_vlm(xsecs, area, 10.0, 10.0, n_chord, n_seg)
        show("elliptic", key, row[key])
    for legs in LEGS:
        for N in (80, 160):
            key = f"VSM {legs} N={N}"
            row[key] = run_vsm(cbv.elliptic_swept_anhedral, N, legs, area, 10.0)
            show("elliptic", key, row[key])
    out["elliptic"] = row

    print("\nV3 kite, flat-plate sections, 10 deg, U = 2.82 m/s")
    xsecs = v3_xsecs()
    area = float(cbv.v3_inviscid(40).wings[0].compute_projected_area())
    row = {"projected_area": area}
    for n_chord, n_seg in ((8, 2), (16, 4)):
        key = f"VLM {n_chord}x{(len(xsecs) - 1) * n_seg}"
        row[key] = run_vlm(xsecs, area, 2.82, 10.0, n_chord, n_seg)
        show("V3", key, row[key])
    for legs in LEGS:
        for N in (40, 80):
            key = f"VSM {legs} N={N}"
            try:
                row[key] = run_vsm(cbv.v3_inviscid, N, legs, area, 2.82)
            except ValueError as err:
                print(f"  V3 {key}: {err}")
                continue
            show("V3", key, row[key])
    out["V3"] = row

    json.dump(out, open(OUT / "vlm_referee.json", "w"), indent=1)
    print(f"\nresults written to {OUT / 'vlm_referee.json'}")


if __name__ == "__main__":
    main()
