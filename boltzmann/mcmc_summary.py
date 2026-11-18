"""Combine the chains of one model from boltzmann/mcmc_full.py: convergence, summaries and a triangle plot.

Removes the first 30% of each chain (by weight), computes the Gelman–Rubin R−1 of every parameter across chains
((1 + 1/m) × variance of the m chain means / mean within-chain variance, weighted; the large-n
Gelman–Rubin form), the effective sample size (getdist), and the
mean, best fit (lowest χ² sample), 68% and 95% equal-tail intervals.
Usage: python3 boltzmann/mcmc_summary.py model=lcdm [dir=chains_v2/lcdm] [plot=1] [write=1]
  (default dir: chains_v2/<model>, the unseeded run; chains/ is the first run, kept only as a record)
  write=1 → results/chains_summary_<model>.json; plot=1 → out/triangle_<model>.pdf (nu, Omega_m, H0, S8)
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
BURN = 0.3


def read_chain(path: str):
    """Cobaya text chain; skips any line that is incomplete (e.g. a last line still being written)."""
    rows = []
    with open(path) as f:
        names = f.readline().lstrip("#").split()
        for line in f:
            parts = line.split()
            if len(parts) != len(names):
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    return names, np.array(rows, ndmin=2).reshape(-1, len(names))


def load(chain_dir: Path):
    chains = []
    for c in sorted(glob.glob(str(chain_dir / "c*" / "chain.1.txt"))):
        names, a = read_chain(c)
        w = a[:, names.index("weight")]
        cut = np.searchsorted(np.cumsum(w), BURN * w.sum())
        chains.append((names, a[cut:]))
    return chains


def wquantile(x, w, q):
    i = np.argsort(x)
    c = np.cumsum(w[i]) - 0.5 * w[i]
    return np.interp(q, c / w.sum(), x[i])


def summarise(chains, model: str):
    names = chains[0][0]
    skip = {"weight", "minuslogpost", "minuslogprior", "minuslogprior__0", "chi2"}
    params = [n for n in names if n not in skip and not n.startswith(("chi2__", "minuslogprior__"))]
    allc = np.vstack([a for _, a in chains])
    w_all = allc[:, names.index("weight")]
    ibest = int(np.argmin(allc[:, names.index("chi2")]))
    out = {"model": model, "n_chains": len(chains), "burn_in": BURN,
           "n_samples": [int(a.shape[0]) for _, a in chains],
           "total_weight": [float(a[:, names.index("weight")].sum()) for _, a in chains],
           "chi2_best": float(allc[ibest, names.index("chi2")]), "params": {}}
    try:
        from getdist import MCSamples

        gd = MCSamples(samples=[a[:, [names.index(p) for p in params]] for _, a in chains],
                       weights=[a[:, names.index("weight")] for _, a in chains], names=params,
                       loglikes=[a[:, names.index("minuslogpost")] for _, a in chains], ignore_rows=0,
                       settings={"ignore_rows": 0})
    except Exception as e:  # noqa: BLE001
        gd = None
        print("getdist unavailable for ESS:", e)
    for p in params:
        j = names.index(p)
        means, vars_, ws = [], [], []
        for _, a in chains:
            w = a[:, names.index("weight")]
            x = a[:, j]
            m = np.average(x, weights=w)
            means.append(m)
            vars_.append(np.average((x - m) ** 2, weights=w))
            ws.append(w.sum())
        x = allc[:, j]
        mean = float(np.average(x, weights=w_all))
        sd = float(np.sqrt(np.average((x - mean) ** 2, weights=w_all)))
        W = float(np.mean(vars_))
        B = float(np.var(means, ddof=1)) if len(chains) > 1 else np.nan
        rec = {"mean": mean, "sd": sd, "best_fit": float(x[ibest]),
               "lim68": [float(wquantile(x, w_all, q)) for q in (0.16, 0.84)],
               "lim95": [float(wquantile(x, w_all, q)) for q in (0.025, 0.975)],
               "R_minus_1": ((1 + 1 / len(chains)) * B / W if W > 0 else (0.0 if B == 0 else np.inf))}
        if gd is not None:
            try:
                rec["ess"] = float(gd.getEffectiveSamples(params.index(p)))
            except Exception:  # noqa: BLE001
                rec["ess"] = None
        out["params"][p] = rec
    if model == "rvm" and "rvm_nu" in params:  # mass against the prior edge at ν = 0 (D-028)
        out["frac_nu_near_edge"] = float(w_all[allc[:, names.index("rvm_nu")] > -5e-5].sum() / w_all.sum())
    varying = {p: r["R_minus_1"] for p, r in out["params"].items() if r["sd"] > 0}
    out["R_minus_1_max"] = float(max(varying.values())) if varying else None
    out["R_minus_1_worst"] = max(varying, key=varying.get) if varying else None
    return out, gd, params


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    model = args.get("model", "lcdm")
    chain_dir = HERE / args.get("dir", f"chains_v2/{model if model != 'flipped' else 'flipped_1.22_2.16'}")
    chains = load(chain_dir)
    if not chains:
        raise SystemExit(f"no chains in {chain_dir}")
    s, gd, params = summarise(chains, model)
    s["chain_dir"] = str(chain_dir.relative_to(HERE))
    print(f"{model}: {s['n_chains']} chains, samples after burn-in {s['n_samples']}, "
          f"max R-1 = {s['R_minus_1_max']:.4f} ({s['R_minus_1_worst']}), best chi2 = {s['chi2_best']:.2f}")
    if "frac_nu_near_edge" in s:
        print(f"  fraction of post-burn-in weight with rvm_nu > -5e-5: {s['frac_nu_near_edge']:.4f}")
    for p in ("rvm_nu", "w0_fld", "wa_fld", "H0", "Omega_m", "sigma8", "S8", "omega_bc_star", "omega_cdm"):
        if p in s["params"]:
            r = s["params"][p]
            print(f"  {p:14s} mean {r['mean']:.5g} ± {r['sd']:.2g}  68% {np.round(r['lim68'], 5)}  "
                  f"best {r['best_fit']:.5g}  R-1 {r['R_minus_1']:.4f}  ESS {r.get('ess')}")
    if args.get("write"):
        (HERE / "results" / f"chains_summary_{model}.json").write_text(json.dumps(s, indent=1) + "\n")
    if args.get("plot") and gd is not None:
        from getdist import plots

        show = [p for p in ("rvm_nu", "w0_fld", "wa_fld", "Omega_m", "H0", "S8") if p in params]
        g = plots.get_subplot_plotter()
        g.triangle_plot(gd, show, filled=True)
        (HERE / "out").mkdir(exist_ok=True)
        g.export(str(HERE / "out" / f"triangle_{model}.pdf"))


if __name__ == "__main__":
    main()
