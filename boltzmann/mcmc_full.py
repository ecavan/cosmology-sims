"""Full-CMB MCMC: same data and model settings as full_cmb.py, sampled with Cobaya's MCMC instead of profiled.

Data: Planck PR4 CamSpec TTTEEE + 2018 low-ℓ TT/EE + PR4 lensing, DESI DR2 BAO, Pantheon+ (M profiled, as in
full_cmb.py). One process runs one chain; run 4 with chain=1..4 and combine with boltzmann/mcmc_summary.py.
Derived: Omega_m (today, CLASS), sigma8 (linear, total matter, z = 0), S8 = sigma8 sqrt(Omega_m/0.3), z_star, and
omega_bc_star = (ρ_b + ρ_cdm) a³ h²/ρ_crit,0 at z_star (closed form of the patched CLASS background).
Usage: python3 boltzmann/mcmc_full.py model=lcdm|rvm|flipped|cpl chain=1 [z1=1.22 z2=2.16] [test=1]
       [out=chains_v2] [covmat=path|auto]  — resumable (rerun the same command).
Starting proposal: the first of boltzmann/covmats_v2/<chain dir name>.covmat (boltzmann/make_v2_start.py: learned from
the first run) and boltzmann/covmats/<chain dir name>.covmat (boltzmann/make_covmats.py) that exists, else Cobaya's
"auto"; covmat= overrides. Cobaya keeps learning the proposal from the chain either way.
Start point: chain k starts at entry k of boltzmann/starts_v2/<chain dir name>.json if that file exists (one row of the
first run per chain), else near the stored best fit. Unseeded (seed None), so a resumed chain draws fresh random
numbers; the first run (chains/, seeded 1000+chain) is kept only as a record.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
CLASSY_DIR = os.environ.get("CLASSY_RVM", str(HERE / "boltzmann" / "classy_rvm"))
sys.path.insert(0, CLASSY_DIR)
sys.path.insert(0, str(HERE / "boltzmann"))

import classy  # noqa: E402,F401
from cobaya_likes import DesiBao, SnIa  # noqa: E402

assert classy.__file__.startswith(CLASSY_DIR), f"stock classy imported: {classy.__file__}"

args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
MODEL = args.get("model", "lcdm")
CHAIN = int(args.get("chain", 1))
Z1, Z2 = float(args.get("z1", 1.22)), float(args.get("z2", 2.16))
OUTDIR = HERE / args.get("out", "chains_v2") / (MODEL if MODEL != "flipped" else f"flipped_{Z1}_{Z2}")
_covs = [HERE / "boltzmann" / d / f"{OUTDIR.name}.covmat" for d in ("covmats_v2", "covmats")]
COVMAT = args.get("covmat", next((str(c) for c in _covs if c.exists()), "auto"))
_starts = HERE / "boltzmann" / "starts_v2" / f"{OUTDIR.name}.json"
START = json.loads(_starts.read_text())["starts"][CHAIN - 1] if _starts.exists() else None
RESULTS = {"lcdm": "results/full_cmb_pr4.jsonl", "cpl": "results/full_cmb_pr4.jsonl",
           "rvm": "results/full_cmb_pr4.jsonl", "flipped": "results/full_cmb_flipped.jsonl"}

# Ω_r h² of photons + massless ν in CLASS for T_cmb = 2.7255 K and N_ur = 2.0308 (enters the RVM closed form only)
OMEGA_G_H2 = 2.4728e-5
OMEGA_R_H2 = OMEGA_G_H2 * (1 + 2.0308 * 7 / 8 * (4 / 11) ** (4 / 3))


def stored_best() -> dict:
    """Lowest-χ² stored full-CMB fit (PR4 lensing) to start the chains near."""
    recs = [json.loads(line) for line in open(HERE / RESULTS[MODEL])]
    sel = [r for r in recs if r["model"] == MODEL and r.get("lens", "pr4") == "pr4" and not r.get("free_z")
           and (MODEL != "rvm" or r["rad"] == 1) and (MODEL != "flipped" or (r["z1"], r["z2"]) == (Z1, Z2))]
    return min(sel, key=lambda r: r["chi2"])


best = stored_best()
x0 = {**best["slow"], **best["fast"]}
if MODEL in ("rvm", "flipped"):
    x0["rvm_nu"] = best["nu"]

# flat priors, wide enough not to cut the posterior; proposal widths ~ profile errors
PRIORS = {"H0": (55.0, 85.0, 0.5), "omega_b": (0.019, 0.026, 0.00013), "omega_cdm": (0.08, 0.25, 0.001),
          "logA": (2.5, 3.5, 0.013), "n_s": (0.85, 1.1, 0.004), "tau_reio": (0.01, 0.2, 0.006)}
if MODEL == "rvm":  # ν ≤ 0: with rvm_rad=1 and ν > 0, ρ_cdm < 0 at early times and the perturbations are undefined (D-028)
    PRIORS["rvm_nu"] = (-0.004, 0.0, 0.0002)
if MODEL == "flipped":
    PRIORS["rvm_nu"] = (0.0, 0.2, 0.02)
if MODEL == "cpl":
    PRIORS.update({"w0_fld": (-3.0, 1.0, 0.06), "wa_fld": (-3.0, 2.0, 0.25)})

extra = {"N_ur": 2.0308, "N_ncdm": 1, "m_ncdm": 0.06, "T_ncdm": 0.71611, "non_linear": "hmcode",
         "tol_initial_Omega_r": 0.02}
if MODEL == "rvm":
    extra.update({"rvm_rad": 1})
if MODEL == "flipped":
    extra.update({"rvm_flip_z1": Z1, "rvm_flip_z2": Z2})
if MODEL == "cpl":
    extra.update({"Omega_Lambda": 0.0, "use_ppf": "yes"})

params = {}
for k, (lo, hi, w) in PRIORS.items():
    start = x0[k]
    params[k] = {"prior": {"min": lo, "max": hi}, "ref": {"dist": "norm", "loc": start, "scale": w / 2},
                 "proposal": w}
if START:  # fixed start point for every sampled parameter
    from copy import deepcopy

    from cobaya.component import get_component_class

    # a nuisance redefined here must carry its full likelihood definition (prior, proposal, ...) with only ref changed
    nuis = {}
    for lk in ("planckpr4lensing.PlanckPR4Lensing", "planck_NPIPE_highl_CamSpec.TTTEEE"):
        nuis.update(get_component_class(lk, kind="likelihood").get_defaults().get("params") or {})
    for k, v in START.items():
        if k not in params:
            params[k] = deepcopy(nuis[k])
        params[k]["ref"] = v
params["logA"]["drop"] = True
params["A_s"] = {"value": "lambda logA: 1e-10*np.exp(logA)"}
params["Omega_m"] = {"latex": r"\Omega_{\rm m}"}
params["sigma8"] = {"latex": r"\sigma_8"}
params["z_star"] = {"latex": "z_*"}
params["S8"] = {"derived": "lambda sigma8, Omega_m: sigma8*np.sqrt(Omega_m/0.3)", "latex": "S_8"}
if MODEL == "rvm":  # ρ_b + ρ_cdm = H0² (A a^{-3(1-ν)} − k Ω_r a^{-4}), A = Ω_c + Ω_b + k Ω_r, k = 4ν/(1+3ν)
    params["omega_bc_star"] = {"derived": (
        f"lambda omega_b, omega_cdm, rvm_nu, z_star: "
        f"(omega_cdm + omega_b + 4*rvm_nu/(1+3*rvm_nu)*{OMEGA_R_H2}) * (1+z_star)**(-3*rvm_nu)"
        f" - 4*rvm_nu/(1+3*rvm_nu)*{OMEGA_R_H2}*(1+z_star)"), "latex": r"\omega_{\rm bc}^*"}
elif MODEL == "flipped":  # matter per comoving volume before z2: ω_bc,0 (1+z1)^{-6ν} (1+z2)^{3ν}
    params["omega_bc_star"] = {"derived": (
        f"lambda omega_b, omega_cdm, rvm_nu: (omega_cdm + omega_b) * {1 + Z1}**(-6*rvm_nu) * {1 + Z2}**(3*rvm_nu)"),
        "latex": r"\omega_{\rm bc}^*"}
else:
    params["omega_bc_star"] = {"derived": "lambda omega_b, omega_cdm: omega_b + omega_cdm",
                               "latex": r"\omega_{\rm bc}^*"}

info = {
    "theory": {"classy": {"path": "global", "extra_args": extra}},
    "likelihood": {
        "planck_2018_lowl.TT": None,
        "planck_2018_lowl.EE": None,
        "planck_NPIPE_highl_CamSpec.TTTEEE": None,
        "planckpr4lensing.PlanckPR4Lensing": None,
        "desi": {"external": DesiBao},
        "sn": {"external": SnIa, "sample": "pantheon+"},
    },
    "params": params,
    **({"prior": {"w0_plus_wa_negative": "lambda w0_fld, wa_fld: 0. if w0_fld + wa_fld < 0 else -np.inf"}}
       if MODEL == "cpl" else {}),
    "sampler": {"mcmc": {"drag": True, "learn_proposal": True, "Rminus1_stop": 0.001, "Rminus1_cl_stop": 0.02,  # cross-chain check is in mcmc_queue.py
                        
                         "max_tries": 10000, "seed": None, "output_every": "60s",
                         "learn_every": "20d", "covmat": COVMAT}},
    "output": str(OUTDIR / f"c{CHAIN}" / "chain"),
    "resume": True,
}

if args.get("test"):  # time one likelihood evaluation at the stored best fit and list the priors
    from cobaya.model import get_model

    info.pop("sampler"), info.pop("output"), info.pop("resume")
    model = get_model(info)
    pt = {k: x0.get(k, v.get("ref", {}).get("loc") if isinstance(v, dict) else None)
          for k, v in model.parameterization.sampled_params().items()}
    pt = {k: x0[k] if k in x0 else model.prior.reference()[i]
          for i, k in enumerate(model.parameterization.sampled_params())}
    import copy
    names = list(model.parameterization.sampled_params())
    for rep in range(4):
        q = copy.deepcopy(pt)
        q["H0"] += 0.1 * rep
        t0 = time.time()
        lp = model.logposterior(q)
        print(f"slow eval {rep}: {time.time() - t0:.2f} s  chi2(likes)={-2 * sum(lp.loglikes):.3f}  "
              f"derived={dict(zip(model.parameterization.derived_params(), np.round(lp.derived, 5)))}", flush=True)
    q = copy.deepcopy(q)
    q["amp_143"] *= 1.01
    t0 = time.time()
    model.logposterior(q)
    print(f"fast-only eval: {time.time() - t0:.3f} s")
    print("stored best-fit chi2 (likelihoods only):", round(sum(best["parts"].values()), 3))
    print("\nPRIORS (sampled parameters):")
    for k, d in zip(names, model.prior.pdf):
        lo, hi = d.interval(1.0)
        print(f"  {k:12s} {d.dist.name:8s} " + (f"[{lo:g}, {hi:g}]" if d.dist.name == "uniform"
              else f"mean {d.mean():g}, sd {d.std():g}"))
    sys.exit()

from cobaya.run import run  # noqa: E402

OUTDIR.mkdir(parents=True, exist_ok=True)
run(info)
