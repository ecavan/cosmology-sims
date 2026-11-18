"""Starting proposal covmats and start points for the chains_v2 MCMC run, from the chains of the first run.

For each model: read the four chains in <src>/<model dir>/c*/chain.1.txt, drop the first half of each (by rows), pool
the rest and compute the weighted covariance of the sampled parameters (Cobaya weights). That matrix is used if it is
positive definite and the pool has at least 10 rows per sampled parameter; otherwise the constructed covmat from
boltzmann/make_covmats.py is used. Start points: one row per new chain, chain k taking a random row (fixed seed) from
the second half of old chain k, so the four starts are distinct and spread over the old chains.
Proposal and start points only: Cobaya keeps learning the proposal (learn_proposal) and nothing here enters the
posterior.
Usage: python3 boltzmann/make_v2_start.py [src=chains] [seed=20261115]
Writes boltzmann/covmats_v2/<model dir>.covmat, boltzmann/starts_v2/<model dir>.json, and prints a report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "boltzmann"))
from mcmc_summary import read_chain  # noqa: E402

args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
SRC = HERE / args.get("src", "chains")
SEED = int(args.get("seed", 20261115))
MODELS = ["lcdm", "rvm", "flipped_1.22_2.16"]
COV2, START2 = HERE / "boltzmann" / "covmats_v2", HERE / "boltzmann" / "starts_v2"


def main() -> None:
    COV2.mkdir(exist_ok=True)
    START2.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    for m in MODELS:
        built = HERE / "boltzmann" / "covmats" / f"{m}.covmat"
        pnames = open(built).readline().lstrip("#").split()  # sampled parameters, in the covmat order
        halves, starts = [], []
        for k in range(1, 5):
            names, a = read_chain(str(SRC / m / f"c{k}" / "chain.1.txt"))
            h = a[a.shape[0] // 2:]
            halves.append(h)
            row = h[rng.integers(h.shape[0])]
            starts.append({p: float(row[names.index(p)]) for p in pnames})
        pool = np.vstack(halves)
        w = pool[:, names.index("weight")]
        X = pool[:, [names.index(p) for p in pnames]]
        C = np.cov(X.T, aweights=w)
        C = (C + C.T) / 2
        eig = np.linalg.eigvalsh(C)
        ok = eig.min() > 0 and pool.shape[0] >= 10 * len(pnames)
        src = "pooled second halves" if ok else "constructed (make_covmats.py) fallback"
        if not ok:
            C = np.loadtxt(built, comments="#")
        np.savetxt(COV2 / f"{m}.covmat", C, header=" ".join(pnames), comments="# ")
        (START2 / f"{m}.json").write_text(json.dumps({"seed": SEED, "src": str(SRC.relative_to(HERE)),
                                                       "starts": starts}, indent=1) + "\n")
        sd = dict(zip(pnames, np.sqrt(np.diag(C))))
        print(f"{m}: rows pooled {pool.shape[0]} for {len(pnames)} params ({pool.shape[0] / len(pnames):.0f}x), "
              f"min eigenvalue {eig.min():.2e} -> {src}")
        print("   sd: " + ", ".join(f"{p} {sd[p]:.3g}" for p in ("H0", "omega_b", "omega_cdm", "logA", "n_s", "tau_reio",
                                                               "rvm_nu") if p in sd))
        for k, s in enumerate(starts, 1):
            print(f"   start c{k}: H0 {s['H0']:.3f} omega_cdm {s['omega_cdm']:.5f}"
                  + (f" rvm_nu {s['rvm_nu']:.6f}" if "rvm_nu" in s else ""))


if __name__ == "__main__":
    main()
