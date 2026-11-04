"""Sensitivity of the curved bound-vortex correction to the chordwise
distribution of the strips (report Section 4.2).

The implemented strips carry the flat-plate thin-airfoil loading
gamma(x) ~ sqrt((c - x)/x). Here they are replaced by the loading of real
sections from NeuralFoil, gamma(x) = ue_upper - ue_lower at its 32
boundary-layer stations (the leading-edge peak below x/c = 0.016 is not
sampled, which biases the centroid aft by a few percent of chord): a NACA
0012 as a check, the V3 centre section (LEI profile, data/TUDELFT_V3_KITE)
at 5, 10 and 15 deg, and a NACA 4412. For each distribution the effective
cut-off c_eff = exp <ln|x - 1/4|> and the centroid are printed, then the
on-blade / Trefftz ratio and CD of the arc wing and the V3 (inviscid polars).

Usage (from the repository root, about two minutes)::

    PYTHONPATH=docs/validation python docs/validation/strip_weight_sensitivity.py
"""
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import aerosandbox as asb
import neuralfoil as nf
import yaml

import VSM.core.BodyAerodynamics as BA
import curved_bound_vortex_validation as cbv
from curved_bound_vortex_validation import ratio, arc, v3_inviscid, solve, DATA
from VSM.core.utils import thin_airfoil_strips

XB = np.asarray(nf.bl_x_points)


def nf_weights(airfoil, alpha, Re=5.6e5, **kw):
    out = airfoil.get_aero_from_neuralfoil(alpha=alpha, Re=Re, model_size="xxxlarge", **kw)
    ue_u = np.array([out[f"upper_bl_ue/vinf_{k}"] for k in range(32)]).ravel()
    ue_l = np.array([out[f"lower_bl_ue/vinf_{k}"] for k in range(32)]).ravel()
    g = ue_u + ue_l  # NeuralFoil lower-surface ue is signed negative; the jump is ue_u - (-ue_l)
    dx = np.gradient(XB)
    w = g * dx
    circ = w.sum()  # integral gamma dx / (U c) ~ CL / 2
    return XB.copy(), w / w.sum(), float(out["CL"]), circ


def stats(x, w):
    return float(np.exp(np.sum(w * np.log(np.abs(x - 0.25))))), float(np.sum(w * x))


def main():
    x0, w0 = thin_airfoil_strips()
    dists = {"flat plate, thin airfoil (implemented)": (x0, w0)}
    print(f"{'distribution':48s} {'c_eff':>6s} {'centroid':>9s} {'CL_nf':>6s} {'2*int(gamma)':>12s}")
    print(f"{'flat plate, thin airfoil (implemented)':48s} {stats(x0, w0)[0]:6.3f} {stats(x0, w0)[1]:9.3f}")

    # flat plate through NeuralFoil, as a check of the pipeline
    plate = asb.Airfoil("naca0012")
    x, w, cl, circ = nf_weights(plate, 5.0)
    print(f"{'naca0012 at 5 deg, NeuralFoil':48s} {stats(x, w)[0]:6.3f} {stats(x, w)[1]:9.3f} {cl:6.3f} {2 * circ:12.3f}")

    d = yaml.safe_load(open(DATA / "CAD_derived_geometry" / "aero_geometry_CAD_neuralfoil.yaml"))
    rows = d["wing_sections"]["data"]
    ids = [r[0] for r in rows]
    print("airfoil ids along the span:", ids)
    mid = rows[len(rows) // 2][0]
    af_info = {r[0]: r[2] for r in d["wing_airfoils"]["data"]}[mid]
    dat = DATA / "CAD_derived_geometry" / af_info["dat_file_path"]
    coords = np.loadtxt(dat, delimiter=",")
    lei = asb.Airfoil(name=f"V3 section {mid}", coordinates=coords)
    nf_kw = dict(xtr_lower=af_info.get("xtr_lower", 1.0), xtr_upper=af_info.get("xtr_upper", 1.0), n_crit=af_info.get("n_crit", 9.0))
    for a in (5.0, 10.0, 15.0):
        x, w, cl, circ = nf_weights(lei, a, **nf_kw)
        name = f"V3 mid section {mid} at {a:.0f} deg, NeuralFoil"
        print(f"{name:48s} {stats(x, w)[0]:6.3f} {stats(x, w)[1]:9.3f} {cl:6.3f} {2 * circ:12.3f}")
        dists[name] = (x, w)
    x, w, cl, circ = nf_weights(asb.Airfoil("naca4412"), 10.0)
    print(f"{'naca4412 at 10 deg, NeuralFoil':48s} {stats(x, w)[0]:6.3f} {stats(x, w)[1]:9.3f} {cl:6.3f} {2 * circ:12.3f}")
    dists["naca4412 at 10 deg, NeuralFoil"] = (x, w)

    print("\non-blade / Trefftz (CD) with each distribution as strip weights")
    cases = [("arc 110 deg, R = 4c", arc, (40, 80)), ("V3 kite, inviscid, 10 deg", v3_inviscid, (40, 80))]
    for name, (x, w) in dists.items():
        xx = np.ascontiguousarray(x, dtype=float)
        ww = np.ascontiguousarray(w, dtype=float)
        BA.thin_airfoil_strips = lambda n_strips=24, xx=xx, ww=ww: (xx, ww)
        line = f"  {name:48s}"
        for cname, mk, Ns in cases:
            for N in Ns:
                r = solve(mk(N), cbv.REGULARISED)
                rt = float(r["drag"] / r["drag_induced_trefftz"])
                line += f"  {cname[:3]} N={N}: {rt:.3f} ({float(r['cd']):.4f})"
        print(line, flush=True)
    BA.thin_airfoil_strips = thin_airfoil_strips


if __name__ == "__main__":
    main()
