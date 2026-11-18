"""Map the flipped RVM's best fit over fixed switch redshifts (compressed CMB + DESI DR2 + SNe).

At each (z1, z2) on a grid, minimise χ² over (h, ω_b, ω_bc*, ν) with the switches held fixed, and record the best ν
and the improvement over ΛCDM. Shows how ν trades off against where the running is switched on.

Usage: python3 scripts/flipped_switch_map.py [sn=pantheon+]  →  results/flipped_switch_map_<sn>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from p1.flipped import chi2, fit_simple  # noqa: E402
from p1.inference import load_data  # noqa: E402

args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
SN = args.get("sn", "pantheon+")
d = load_data(sn=SN)
x_l, c_l = fit_simple("lcdm", d)
x_c, c_c = fit_simple("cpl", d)
Z1 = np.round(np.arange(0.6, 3.01, 0.2), 2)
DZ = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0]
out = {"sn": SN, "chi2_lcdm": c_l, "chi2_cpl": c_c, "grid": []}


def fit_point(z):
    z1, z2 = z
    f = lambda q: chi2("flipped", [q[0], q[1], q[2], q[3], z1, z2], d)
    best = None
    for nu0 in (0.02, 0.06, 0.12):  # several starts: the ν direction is shallow for wide windows
        r = minimize(f, [*x_l, nu0], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-4, "maxfev": 4000})
        r = minimize(f, r.x, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-5, "maxfev": 4000})
        if best is None or r.fun < best.fun:
            best = r
    return {"z1": float(z1), "z2": float(z2), "nu": float(best.x[3]), "chi2": float(best.fun),
            "dchi2_lcdm": float(c_l - best.fun)}


if __name__ == "__main__":
    from multiprocessing import Pool
    pts = [(float(z1), round(float(z1) + dz, 2)) for z1 in Z1 for dz in DZ if z1 + dz <= 10]
    t0 = time.time()
    with Pool() as pool:
        for g in pool.imap_unordered(fit_point, pts):
            out["grid"].append(g)
            print(f"z1={g['z1']:.1f} z2={g['z2']:.1f}  ν={g['nu']:.4f}  Δχ²={g['dchi2_lcdm']:6.2f}  ({time.time() - t0:.0f}s)",
                  flush=True)
    out["grid"].sort(key=lambda g: (g["z1"], g["z2"]))
    (ROOT / "results" / f"flipped_switch_map_{SN}.json").write_text(json.dumps(out, indent=1))
