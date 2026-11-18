"""Rerun chosen Pantheon+ noisy mocks exactly as scripts/noisy_mocks.py did and compare with the stored records.

Usage: python3 scripts/check_mock_reproduce.py tag=c seed=3 i=0,100,200,300,401
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
TAG, SEED = args.get("tag", "c"), int(args.get("seed", 3))
WANT = [int(x) for x in args.get("i", "0,1,2,3,4").split(",")]
stored = {json.loads(l)["i"]: json.loads(l) for l in open(HERE / "results" / f"noisy_mocks_{TAG}.jsonl")}
P_TRUE = json.loads((HERE / "results" / "real_fits.json").read_text())["rvm"]["best_fit"]
d = load_data()
bao_true, mu_true, cmb_true = truth_vectors(d, theory("rvm", P_TRUE))
L_bao, L_sn, L_cmb = noise_factors(d)
rng = np.random.default_rng(SEED)
for i in range(max(WANT) + 1):
    nb, ns, nc = rng.normal(size=len(bao_true)), rng.normal(size=len(mu_true)), rng.normal(size=3)
    if i not in WANT:
        continue
    m = with_vectors(d, bao_true + L_bao @ nb, mu_true + L_sn @ ns, cmb_true + L_cmb @ nc)
    _, cl = best_fit("lcdm", m, start=P_TRUE[:3], restarts=2, seed=i)
    _, cr = best_fit("rvm", m, start=P_TRUE, restarts=2, seed=i)
    _, cc = best_fit("cpl", m, start=list(P_TRUE[:3]) + [-1.0, 0.0], restarts=4, seed=i)
    s = stored[i]
    diff = max(abs(cl - s["chi2_lcdm"]), abs(cr - s["chi2_rvm"]), abs(cc - s["chi2_cpl"]))
    print(f"[{TAG}] i={i}: stored ΛCDM {s['chi2_lcdm']:.4f} RVM {s['chi2_rvm']:.4f} CPL {s['chi2_cpl']:.4f} | "
          f"rerun {cl:.4f} {cr:.4f} {cc:.4f} | max |diff| {diff:.2e}", flush=True)
