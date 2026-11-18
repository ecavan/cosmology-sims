"""Mocks about a real-data best fit (truth = lcdm | cpl | flipped), each fitted with ΛCDM, w0waCDM and the flipped RVM.

The flipped fit is the real-data procedure (p1.flipped.fit_flipped), so the freedom in z1, z2 is in the distributions.
Usage: python3 scripts/flipped_mocks.py truth=flipped n=50 seed=1 tag=a [noise=1] [sn=pantheon+] [deadline=<unix time>]
noise=0 gives the single noiseless mock. Output: results/flipped_mocks_<truth>[_sn]_<tag|asimov>.jsonl (restartable).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "python"))
from p1.flipped import fit_flipped, fit_simple  # noqa: E402
from p1.inference import load_data  # noqa: E402
from p1.mocks import noise_factors, truth_vectors, with_vectors  # noqa: E402
from p1.models import theory  # noqa: E402

args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
TRUTH = args.get("truth", "flipped")
N, SEED, TAG = int(args.get("n", 50)), int(args.get("seed", 1)), args.get("tag", "a")
NOISE = int(args.get("noise", 1))
SN = args.get("sn", "pantheon+")
DEADLINE = float(args.get("deadline", "inf"))  # stop cleanly before starting a realisation after this time
real = json.loads((HERE / "results" / f"flipped_fit_{SN}.json").read_text())
suffix = "" if SN == "pantheon+" else f"_{SN}"
out = HERE / "results" / f"flipped_mocks_{TRUTH}{suffix}_{TAG if NOISE else 'asimov'}.jsonl"
done = sum(1 for _ in open(out)) if out.exists() else 0
if not NOISE:
    N = 1

d = load_data(sn=SN)
bao_true, mu_true, cmb_true = truth_vectors(d, theory(TRUTH, real[TRUTH]["x"]))
L_bao, L_sn, L_cmb = noise_factors(d)


def slope_and_crossing(w0, wa):
    zc = None
    if wa != 0:
        a = 1 + (1 + w0) / wa
        if 1 / 6 < a < 1:
            zc = 1 / a - 1
    return (wa / (1 + w0) if abs(1 + w0) > 1e-6 else None), zc


rng = np.random.default_rng(SEED)
for i in range(N):
    nb, ns, nc = rng.normal(size=len(bao_true)), rng.normal(size=len(mu_true)), rng.normal(size=3)
    if i < done:
        continue
    if time.time() > DEADLINE:
        print("MOCKS_PAUSED", TRUTH, TAG, "at", i, flush=True)
        sys.exit(0)
    k = float(NOISE)
    t0 = time.time()
    m = with_vectors(d, bao_true + k * (L_bao @ nb), mu_true + k * (L_sn @ ns), cmb_true + k * (L_cmb @ nc))
    rec = {"i": i, "truth": TRUTH, "noise": NOISE}
    xl, rec["chi2_lcdm"] = fit_simple("lcdm", m, start=list(real["lcdm"]["x"]))
    xc, rec["chi2_cpl"] = fit_simple("cpl", m, start=list(real["cpl"]["x"]))
    xc2, c2 = fit_simple("cpl", m, start=list(xl) + [-1.0, 0.0])  # second start from ΛCDM
    if c2 < rec["chi2_cpl"]:
        xc, rec["chi2_cpl"] = xc2, c2
    rec["w0"], rec["wa"] = float(xc[3]), float(xc[4])
    rec["slope"], rec["zcross"] = slope_and_crossing(rec["w0"], rec["wa"])
    xf, rec["chi2_flipped"], _ = fit_flipped(m, xl)
    rec["x_flipped"] = xf.tolist()
    rec["secs"] = time.time() - t0
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{TRUTH}/{TAG}] {i}: Δχ²(Λ−CPL)={rec['chi2_lcdm'] - rec['chi2_cpl']:.2f} w0={rec['w0']:.3f} "
          f"wa={rec['wa']:.3f} Δχ²(Λ−flipped)={rec['chi2_lcdm'] - rec['chi2_flipped']:.2f} "
          f"Δχ²(flipped−CPL)={rec['chi2_flipped'] - rec['chi2_cpl']:+.2f} ({rec['secs']:.0f}s)", flush=True)
print("MOCKS_DONE", TRUTH, TAG)
