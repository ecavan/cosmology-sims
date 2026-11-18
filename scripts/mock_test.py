"""Noiseless RVM mocks fitted with ΛCDM and w0waCDM: apparent (w0, wa), crossing redshift and Δχ² for each ν.

Usage: python3 scripts/mock_test.py [nus=...] [sn=pantheon+]  →  results/mock_test.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "python"))
from p1.inference import best_fit, load_data  # noqa: E402
from p1.mocks import asimov  # noqa: E402
from p1.models import omega_r, rvm_omega_c0, theory  # noqa: E402

args = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
SN = args.get("sn", "pantheon+")
SUF = "" if SN == "pantheon+" else f"_{SN}"
NUS = [float(x) for x in args.get("nus", "-0.004,-0.003,-0.002,-0.001,-0.0005,0,0.0005,0.001,0.002,0.003").split(",")]
fits = HERE / "results" / f"real_fits{SUF}.json"
real = json.loads(fits.read_text()) if fits.exists() else {}
BASE = real.get("rvm", {}).get("best_fit", [0.6875, 0.02224, 0.14131, -0.0005])[:3]


def crossing_z(w0, wa):
    """Redshift where w0 + wa(1 − a) = −1, if 0 < z < 5."""
    if wa == 0:
        return None
    a = 1 + (1 + w0) / wa
    return 1 / a - 1 if 1 / 6 < a < 1 else None


def effective_anchor(p_true, omega_bc_fit):
    """Redshift where the truth's ρ_m a³ = A a^{3ν} − kω_r/a equals the CPL fit's ω_bc."""
    h, wb, wbc_star, nu = p_true
    if nu == 0:
        return None
    wr = omega_r(h)
    k = 4 * nu / (1 + 3 * nu)
    amp = rvm_omega_c0(h, wb, wbc_star, nu) + wb + k * wr
    f = lambda lna: amp * np.exp(3 * nu * lna) - k * wr * np.exp(-lna) - omega_bc_fit
    lo, hi = np.log(1 / 3000), 0.0
    if f(lo) * f(hi) > 0:
        return None
    return float(1 / np.exp(brentq(f, lo, hi)) - 1)


d = load_data(sn=SN)
results = {"base": BASE, "runs": []}
for nu in NUS:
    p_true = list(BASE) + [nu]
    m = asimov(d, theory("rvm", p_true))
    xl, cl = best_fit("lcdm", m, start=BASE)
    xc, cc = best_fit("cpl", m, start=list(BASE) + [-1.0, 0.0], restarts=6)
    w0, wa = xc[3], xc[4]
    zc = crossing_z(w0, wa)
    o = theory("cpl", xc).cosmo.omegas()
    r = {"nu": nu, "w0": w0, "wa": wa, "z_cross": zc, "chi2_lcdm": cl, "chi2_cpl": cc,
         "dchi2_lcdm_minus_cpl": cl - cc, "omega_bc_cpl": xc[2], "omega_bc_true_star": p_true[2], "h_cpl": xc[0],
         "Omega_m_cpl": o["b"] + o["c"] + o["nu_m"], "cpl_params": xc.tolist(), "lcdm_params": xl.tolist(),
         "z_anchor_eff": effective_anchor(p_true, xc[2])}
    results["runs"].append(r)
    print(f"ν={nu:+.4f}: w0={w0:+.3f} wa={wa:+.3f} z_cross={zc if zc is None else round(zc, 3)}  "
          f"χ²_ΛCDM={cl:.2f} χ²_CPL={cc:.2f} Δχ²={cl - cc:.2f}  z_anchor_eff={r['z_anchor_eff']}", flush=True)
(HERE / "results" / f"mock_test{SUF}.json").write_text(json.dumps(results, indent=1))
print("saved", f"results/mock_test{SUF}.json")
