"""Noisy realisations about the best-fit RVM, each fitted with ΛCDM, w0waCDM and the RVM (restartable).

Usage: python3 scripts/noisy_mocks.py [n=100] [seed=1] [tag=a] [sn=pantheon+] [truth=rvm|lcdm|nu:-0.001]
       →  results/noisy_mocks[_sn]_<tag>.jsonl
truth (Pantheon+ only): rvm = the RVM best fit (default, the paper's 1,000 mocks); lcdm = the ΛCDM best fit (ν = 0);
nu:<value> = the RVM with ν fixed at <value> and (h, ω_b, ω_bc*) refitted to the real data. The truth and seed are
written to results/noisy_mocks_<tag>.meta.json. Noise model, fits, starts and restarts are the same for every truth;
the fits start from the truth, as for the default.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "python"))
from p1.inference import best_fit, load_data  # noqa: E402
from p1.mocks import noise_factors, truth_vectors, with_vectors  # noqa: E402
from p1.models import theory  # noqa: E402

args = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
N, SEED, TAG, SN = int(args.get("n", 100)), int(args.get("seed", 1)), args.get("tag", "a"), args.get("sn", "pantheon+")
TRUTH = args.get("truth", "rvm")
TRUTH_MODEL = "rvm"
if SN == "pantheon+":
    real = json.loads((HERE / "results" / "real_fits.json").read_text())
    if TRUTH == "rvm":
        P_TRUE = real["rvm"]["best_fit"]
    elif TRUTH == "lcdm":  # ΛCDM best fit; the RVM fit starts at it with ν = 0
        P_TRUE, TRUTH_MODEL = list(real["lcdm"]["best_fit"]) + [0.0], "lcdm"
    elif TRUTH.startswith("nu:"):  # RVM at fixed ν, other parameters refitted to the real data
        from p1.inference import profile_nu  # noqa: E402

        nu_t = float(TRUTH[3:])
        prof = profile_nu(load_data(sn=SN), real["lcdm"]["best_fit"], [nu_t])[0]
        P_TRUE = list(prof["x"]) + [nu_t]
    else:
        raise SystemExit(f"unknown truth={TRUTH}")
    out = HERE / "results" / f"noisy_mocks_{TAG}.jsonl"
    if TRUTH != "rvm" or TAG not in "abcd":
        meta = {"tag": TAG, "seed": SEED, "n": N, "truth": TRUTH, "truth_model": TRUTH_MODEL, "p_true": list(P_TRUE)}
        if TRUTH.startswith("nu:"):
            meta["truth_refit_chi2"] = prof["chi2"]
        (HERE / "results" / f"noisy_mocks_{TAG}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
else:  # truth: RVM best fit to DESI + CMB + this SN sample, started at its profile minimum
    rob = json.loads((HERE / "results" / f"sn_robustness_{SN}.json").read_text())
    pm = min(rob["profile"], key=lambda q: q["chi2"])
    P_TRUE = list(best_fit("rvm", load_data(sn=SN), start=pm["x"] + [pm["nu"]])[0])
    out = HERE / "results" / f"noisy_mocks_{SN}_{TAG}.jsonl"
done = sum(1 for _ in open(out)) if out.exists() else 0

d = load_data(sn=SN)
bao_true, mu_true, cmb_true = truth_vectors(
    d, theory(TRUTH_MODEL, P_TRUE[:3] if TRUTH_MODEL == "lcdm" else P_TRUE))
L_bao, L_sn, L_cmb = noise_factors(d)
rng = np.random.default_rng(SEED)
for i in range(N):
    nb, ns, nc = rng.normal(size=len(bao_true)), rng.normal(size=len(mu_true)), rng.normal(size=3)
    if i < done:  # same RNG stream, so finished realisations are skipped
        continue
    t0 = time.time()
    m = with_vectors(d, bao_true + L_bao @ nb, mu_true + L_sn @ ns, cmb_true + L_cmb @ nc)
    xl, cl = best_fit("lcdm", m, start=P_TRUE[:3], restarts=2, seed=i)
    xr, cr = best_fit("rvm", m, start=P_TRUE, restarts=2, seed=i)
    xc, cc = best_fit("cpl", m, start=list(P_TRUE[:3]) + [-1.0, 0.0], restarts=4, seed=i)
    rec = {"i": i, "chi2_lcdm": cl, "chi2_rvm": cr, "chi2_cpl": cc, "w0": xc[3], "wa": xc[4], "nu": xr[3],
           "secs": time.time() - t0}
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{TAG}] {i}: Δχ²(ΛCDM−CPL)={cl - cc:.2f} Δχ²(RVM−CPL)={cr - cc:.2f} w0={xc[3]:.3f} wa={xc[4]:.3f} "
          f"ν={xr[3]:+.5f} ({time.time() - t0:.0f}s)", flush=True)
