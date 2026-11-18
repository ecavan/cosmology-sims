"""Rebuild noisy realisation i of stream (seed, tag) and refit with many restarts to check the minima were global.

Usage: python3 scripts/refit_check.py tag=b seed=22 i=0
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "python"))
from p1.inference import best_fit, load_data  # noqa: E402
from p1.mocks import noise_factors, truth_vectors, with_vectors  # noqa: E402
from p1.models import theory  # noqa: E402

args = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
TAG, SEED, I = args.get("tag", "b"), int(args.get("seed", 22)), int(args.get("i", 0))
P = json.loads((HERE / "results" / "real_fits.json").read_text())["rvm"]["best_fit"]
d = load_data()
bao_true, mu_true, cmb_true = truth_vectors(d, theory("rvm", P))
L_bao, L_sn, L_cmb = noise_factors(d)
rng = np.random.default_rng(SEED)
for k in range(I + 1):
    nb, ns, nc = rng.normal(size=len(bao_true)), rng.normal(size=len(mu_true)), rng.normal(size=3)
m = with_vectors(d, bao_true + L_bao @ nb, mu_true + L_sn @ ns, cmb_true + L_cmb @ nc)
orig = [json.loads(l) for l in open(HERE / "results" / f"noisy_mocks_{TAG}.jsonl") if json.loads(l)["i"] == I][0]
xl, cl = best_fit("lcdm", m, start=P[:3], restarts=10, seed=100 + I)
xr, cr = best_fit("rvm", m, start=list(xl) + [0.0], restarts=10, seed=200 + I)
xr2, cr2 = best_fit("rvm", m, start=P, restarts=6, seed=300 + I)
if cr2 < cr:
    xr, cr = xr2, cr2
xc, cc = best_fit("cpl", m, start=list(xl) + [-1.0, 0.0], restarts=12, seed=400 + I)
xc2, cc2 = best_fit("cpl", m, start=list(xr[:3]) + [orig["w0"], orig["wa"]], restarts=6, seed=500 + I)
if cc2 < cc:
    xc, cc = xc2, cc2
print(f"[{TAG}] i={I}  original: ΛCDM {orig['chi2_lcdm']:.2f}  RVM {orig['chi2_rvm']:.2f}  CPL {orig['chi2_cpl']:.2f}")
print(f"[{TAG}] i={I}  refit:    ΛCDM {cl:.2f}  RVM {cr:.2f}  CPL {cc:.2f}   (ν={xr[3]:+.5f}, w0={xc[3]:.3f}, wa={xc[4]:.3f})")
print(f"   Δχ²(ΛCDM−CPL): {orig['chi2_lcdm'] - orig['chi2_cpl']:.2f} → {cl - cc:.2f};  "
      f"Δχ²(RVM−CPL): {orig['chi2_rvm'] - orig['chi2_cpl']:.2f} → {cr - cc:.2f}")
