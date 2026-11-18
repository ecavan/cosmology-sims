"""Fit ΛCDM, w0waCDM and the RVM to DESI DR2 + SNe + compressed CMB: results/real_fits.json and results/chains/.

Usage: python3 scripts/fit_real.py [models=lcdm,cpl,rvm] [steps=4000] [sn=pantheon+]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "python"))
from p1.inference import best_fit, chi2_parts, load_data, sample  # noqa: E402
from p1.models import PARAMS, theory  # noqa: E402

args = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
SN = args.get("sn", "pantheon+")
SUF = "" if SN == "pantheon+" else f"_{SN}"
models = args.get("models", "lcdm,cpl,rvm").split(",")
steps = int(args.get("steps", 4000))
(HERE / "results" / "chains").mkdir(parents=True, exist_ok=True)
out_path = HERE / "results" / f"real_fits{SUF}.json"
out = json.loads(out_path.read_text()) if out_path.exists() else {}

d = load_data(sn=SN)
for m in models:
    chain, x0, tau = sample(m, d, nwalkers=24, nsteps=steps, burn=steps // 4)
    np.save(HERE / "results" / "chains" / f"{m}{SUF}.npy", chain.astype(np.float32))  # float32 keeps every quoted digit
    xb, c2 = best_fit(m, d, start=x0)
    parts = chi2_parts(d, theory(m, xb))
    om = []
    for p in chain[:: max(1, len(chain) // 2000)]:
        o = theory(m, p).cosmo.omegas()
        om.append(o["b"] + o["c"] + o["nu_m"])
    out[m] = {
        "params": PARAMS[m],
        "best_fit": xb.tolist(),
        "chi2_min": c2,
        "chi2_parts": parts,
        "mean": chain.mean(0).tolist(),
        "std": chain.std(0).tolist(),
        "q16": np.percentile(chain, 16, axis=0).tolist(),
        "q84": np.percentile(chain, 84, axis=0).tolist(),
        "Omega_m_mean": float(np.mean(om)),
        "Omega_m_std": float(np.std(om)),
        "autocorr": [float(t) for t in tau],
        "n_samples": int(len(chain)),
    }
    print(m, "best", np.round(xb, 5), "chi2", round(c2, 2), "mean", np.round(chain.mean(0), 5),
          "std", np.round(chain.std(0), 5), "tau", np.round(tau, 1), flush=True)
    out_path.write_text(json.dumps(out, indent=1))
print("saved", out_path)
