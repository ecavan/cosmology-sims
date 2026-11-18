"""Full-CMB profile likelihood: Planck PR4 CamSpec TTTEEE + 2018 low-ℓ TT/EE + PR4 (or 2018) lensing, DESI DR2, SNe.

At fixed model (and ν) it minimises χ² over the ΛCDM parameters (+ w0, wa) and the nuisance parameters, alternating a
slow (cosmology, BOBYQA) and a fast (nuisance, Powell) block. Each run appends one JSON line to `out`.
Usage: python3 boltzmann/full_cmb.py model=lcdm|cpl|rvm|flipped [nu=-0.001] [rad=1] [z1=1.22 z2=2.16 [free_z=1]]
       [lens=pr4|2018|actplanck] [warm_from=results/...jsonl warm_lens=pr4] [sn=pantheon+] [maxfun=160] [warm=1 | extrap=nu_a[,nu_b] | set=H0:67.1,...] [out=...] [test=1]
Default out: results/full_cmb_pr4.jsonl (pr4), results/full_cmb.jsonl (2018), results/full_cmb_flipped.jsonl (flipped).
Needs the patched classy (CLASSY_RVM, see README), Cobaya's Planck likelihoods (COBAYA_PACKAGES_PATH) and, for
lens=pr4, the planckpr4lensing package (github.com/carronj/planck_PR4_lensing).
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
sys.path.insert(0, str(HERE / "python"))

import classy  # noqa: E402,F401
from cobaya.likelihood import Likelihood  # noqa: E402
from cobaya.model import get_model  # noqa: E402

from p1.data import load_desi_dr2, load_sn  # noqa: E402
from p1.likelihoods import SnLikelihood  # noqa: E402

assert classy.__file__.startswith(CLASSY_DIR), f"stock classy imported: {classy.__file__}"
C_KMS = 299792.458

args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
MODEL = args.get("model", "lcdm")
NU = float(args.get("nu", 0.0))
RAD = int(args.get("rad", 1))
Z1, Z2 = float(args.get("z1", 1.22)), float(args.get("z2", 2.16))
FREE_Z = MODEL == "flipped" and bool(int(args.get("free_z", 0)))
LENS = args.get("lens", "pr4")
SN = args.get("sn", "pantheon+")
DEFAULT_OUT = ("results/full_cmb_flipped.jsonl" if MODEL == "flipped"
               else "results/full_cmb_pr4.jsonl" if LENS == "pr4"
               else "results/full_cmb_actplanck.jsonl" if LENS == "actplanck" else "results/full_cmb.jsonl")
OUT = HERE / args.get("out", DEFAULT_OUT)
LENS_LIKE = {"pr4": "planckpr4lensing.PlanckPR4Lensing", "2018": "planck_2018_lensing.native",
             "actplanck": "act_dr6_lenslike.ACTDR6LensLike"}[LENS]
# ACT DR6 + Planck PR4 lensing combined, as in DESI DR2's baseline CMB (needs the act_dr6_lenslike package and data)
LENS_OPTS = {"variant": "actplanck_baseline"} if LENS == "actplanck" else None
MAXFUN = int(args.get("maxfun", 160))


from cobaya_likes import DesiBao, SnIa  # noqa: E402  (moved verbatim from this file)


SLOW = {  # name: (start, scale)
    "H0": (68.2, 0.5), "omega_b": (0.02223, 0.00014), "omega_cdm": (0.1186, 0.0010),
    "logA": (3.045, 0.014), "n_s": (0.9672, 0.0040), "tau_reio": (0.058, 0.007),
    "A_planck": (1.0, 0.0025),  # calibration: strongly correlated with A_s, so it goes in the slow block
}
if MODEL == "cpl":
    SLOW.update({"w0_fld": (-0.84, 0.06), "wa_fld": (-0.6, 0.25)})
if FREE_Z:
    SLOW.update({"rvm_flip_z1": (Z1, 0.05), "rvm_flip_z2": (Z2, 0.05)})


def wc_start(wc_early: float = 0.1186) -> float:
    """Today's ω_c matching ΛCDM's early CDM (RVM: ∝ ν ln 1090; flipped: p1.models.flipped_early_factor)."""
    if MODEL == "rvm":
        return wc_early * (1 + 3 * NU * np.log(1090))
    if MODEL == "flipped":
        f = (1 + Z1) ** (-6 * NU) * (1 + Z2) ** (3 * NU)
        return (wc_early + 0.02223) / f - 0.02223
    return wc_early


SLOW["omega_cdm"] = (wc_start(), SLOW["omega_cdm"][1])  # prior box centred on the matching start
SET = {k: float(v) for k, v in (kv.split(":") for kv in args["set"].split(","))} if args.get("set") else {}
SLOW.update({k: (v, SLOW[k][1]) for k, v in SET.items() if k in SLOW})  # explicit start values
FAST0 = {"A_planck": 1.0, "amp_143": 10.0, "amp_217": 20.0, "amp_143x217": 10.0, "n_143": 1.0, "n_217": 1.0,
         "n_143x217": 1.0, "calTE": 1.0, "calEE": 1.0}
BOUNDS = {"amp_143": (0, 50), "amp_217": (0, 50), "amp_143x217": (0, 50), "n_143": (0, 5), "n_217": (0, 5),
          "n_143x217": (0, 5), "calTE": (0.9, 1.1), "calEE": (0.9, 1.1), "A_planck": (0.9, 1.1)}

extra = {"N_ur": 2.0308, "N_ncdm": 1, "m_ncdm": 0.06, "T_ncdm": 0.71611, "non_linear": "hmcode",
         "tol_initial_Omega_r": 0.02}
if MODEL == "rvm":
    extra.update({"rvm_nu": NU, "rvm_rad": RAD})
if MODEL == "flipped":
    extra.update({"rvm_nu": NU} if FREE_Z else {"rvm_nu": NU, "rvm_flip_z1": Z1, "rvm_flip_z2": Z2})
if MODEL == "cpl":
    extra.update({"Omega_Lambda": 0.0, "use_ppf": "yes"})

params = {k: {"prior": {"min": v[0] - 50 * v[1], "max": v[0] + 50 * v[1]}} for k, v in SLOW.items()
          if k != "A_planck"}
params["logA"] = {"prior": {"min": 2.5, "max": 3.5}, "drop": True}
params["A_s"] = {"value": "lambda logA: 1e-10*np.exp(logA)"}
params["tau_reio"] = {"prior": {"min": 0.01, "max": 0.2}}
info = {
    "theory": {"classy": {"path": "global", "extra_args": extra}},
    "likelihood": {
        "planck_2018_lowl.TT": None,
        "planck_2018_lowl.EE": None,
        "planck_NPIPE_highl_CamSpec.TTTEEE": None,
        LENS_LIKE: LENS_OPTS,
        "desi": {"external": DesiBao},
        "sn": {"external": SnIa, "sample": SN},
    },
    "params": params,
}
model = get_model(info)
fast_names = [p for p in model.parameterization.sampled_params() if p not in SLOW]
assert "A_planck" in model.parameterization.sampled_params()
start_fast = {k: FAST0.get(k, 1.0) for k in fast_names}
slow_names = list(SLOW)
x_scale = np.array([SLOW[k][1] for k in slow_names])
GAUSS_PRIORS = {"A_planck": (1.0, 0.0025)}  # the only informative nuisance prior in this setup


def chi2_point(point: dict) -> tuple[float, dict]:
    ll = model.loglikes(point, as_dict=True, return_derived=False)
    parts = {k: -2 * float(v) for k, v in ll.items()}
    tot = sum(parts.values())
    for k, (mu, sig) in GAUSS_PRIORS.items():
        if k in point:
            tot += ((point[k] - mu) / sig) ** 2
    return (tot if np.isfinite(tot) else 1e30), parts


n_slow_evals = 0


def fit_fast(slow: dict, fast: dict) -> tuple[dict, float]:
    """Minimise over nuisance parameters at fixed cosmology (CLASS output is cached)."""
    from scipy.optimize import minimize

    names = list(fast)
    lo = np.array([BOUNDS.get(n, (-np.inf, np.inf))[0] for n in names])
    hi = np.array([BOUNDS.get(n, (-np.inf, np.inf))[1] for n in names])
    f = lambda v: chi2_point({**slow, **dict(zip(names, v))})[0]
    r = minimize(f, np.array([fast[n] for n in names]), method="Powell", bounds=list(zip(lo, hi)),
                 options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 2000})
    return dict(zip(names, r.x)), float(r.fun)


def fit_slow(slow: dict, fast: dict, maxfun: int) -> tuple[dict, float]:
    import pybobyqa

    global n_slow_evals
    x0 = np.array([slow[k] for k in slow_names])

    def f(u):
        global n_slow_evals
        n_slow_evals += 1
        return chi2_point({**dict(zip(slow_names, x0 + u * x_scale)), **fast})[0]

    r = pybobyqa.solve(f, np.zeros(len(x0)), rhobeg=1.0, rhoend=0.01, maxfun=maxfun, seek_global_minimum=False,
                       bounds=(-8 * np.ones(len(x0)), 8 * np.ones(len(x0))))
    return dict(zip(slow_names, x0 + r.x * x_scale)), float(r.f)


if args.get("test"):
    p = {**{k: v[0] for k, v in SLOW.items()}, **start_fast}
    for label, upd in [("first", {}), ("fast-only", {"amp_143": 11.0}), ("slow", {"H0": 68.3})]:
        t = time.time()
        c, parts = chi2_point({**p, **upd})
        print(f"{label}: {time.time() - t:.2f}s  χ²={c:.2f}", {k: round(v, 2) for k, v in parts.items()}, flush=True)
    print("fast params:", fast_names)
    sys.exit(0)

t0 = time.time()
slow = {k: v[0] for k, v in SLOW.items()}
start = OUT.with_name(OUT.stem + "_start.json")  # best ΛCDM fit written to this output's companion file
seeds = [start, OUT.parent / f"full_cmb{'_pr4' * (LENS == 'pr4')}_start.json", OUT.parent / "full_cmb_start.json"]
seed = next((f for f in seeds if f.exists()), None)
if seed:  # start from the best ΛCDM fit of the same lensing setup if available
    s = json.loads(seed.read_text())
    keep_wc = MODEL in ("rvm", "flipped")  # their matching ω_c start is set above
    slow.update({k: v for k, v in s["slow"].items() if k in slow and not (keep_wc and k == "omega_cdm")})
    start_fast.update({k: v for k, v in s["fast"].items() if k in start_fast})
    if "A_planck" in s["fast"]:
        slow["A_planck"] = s["fast"]["A_planck"]
WARM_LENS = args.get("warm_lens", LENS)  # warm-start from a fit with another lensing likelihood


def same_setup(r: dict, nu: float) -> bool:
    """Whether an earlier record is this model, ν, rad, lensing and (fixed) switches."""
    return (r["model"] == MODEL and r["nu"] == nu and r["rad"] == RAD and r.get("lens", "2018") == WARM_LENS
            and (MODEL != "flipped" or (r.get("z1"), r.get("z2"), r.get("free_z", False)) == (Z1, Z2, FREE_Z)))


if args.get("warm"):  # continue an earlier fit of the same model/ν/rad from its end point
    src = HERE / args["warm_from"] if args.get("warm_from") else OUT
    prev = [json.loads(line) for line in src.read_text().splitlines() if line.strip()] if src.exists() else []
    prev = [r for r in prev if same_setup(r, NU)]
    if prev:
        best_prev = min(prev, key=lambda r: r["chi2"])
        slow.update({k: v for k, v in best_prev["slow"].items() if k in slow})
        start_fast.update({k: v for k, v in best_prev["fast"].items() if k in start_fast})
        print(f"warm start from χ²={best_prev['chi2']:.2f}", flush=True)
if args.get("extrap"):  # start from a linear extrapolation of the best fits at two other ν
    nus = ([float(v) for v in args["extrap"].split(",")] * 2)[:2]  # one value: start from that ν's best fit
    recs = [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()]
    ends = []
    for nv in nus:
        rr = [r for r in recs if same_setup(r, nv)]
        ends.append(min(rr, key=lambda r: r["chi2"]))
    lam = 0.0 if nus[0] == nus[1] else (NU - nus[0]) / (nus[1] - nus[0])
    for k in slow:
        if k in ends[0]["slow"] and k in ends[1]["slow"]:
            slow[k] = ends[0]["slow"][k] + lam * (ends[1]["slow"][k] - ends[0]["slow"][k])
    for k in start_fast:
        if k in ends[0]["fast"] and k in ends[1]["fast"]:
            start_fast[k] = ends[0]["fast"][k] + lam * (ends[1]["fast"][k] - ends[0]["fast"][k])
    print(f"extrapolated start from ν={nus} (λ={lam:.2f}): {({k: round(v, 5) for k, v in slow.items()})}", flush=True)
slow.update({k: v for k, v in SET.items() if k in slow})  # explicit start values win
fast = dict(start_fast)
history = []
for it, mf in enumerate([MAXFUN, MAXFUN // 2, MAXFUN // 2]):
    fast, c_fast = fit_fast(slow, fast)
    slow, c_slow = fit_slow(slow, fast, mf)
    history.append([c_fast, c_slow])
    print(f"[{MODEL} ν={NU} rad={RAD} lens={LENS}] round {it}: χ²_fast={c_fast:.2f} χ²_slow={c_slow:.2f} "
          f"({n_slow_evals} CLASS runs, {time.time() - t0:.0f}s)", flush=True)
fast, c_final = fit_fast(slow, fast)
chi2, parts = chi2_point({**slow, **fast})
res = {"model": MODEL, "nu": NU, "rad": RAD, "lens": LENS, "sn": SN, "chi2": chi2, "parts": parts, "slow": slow,
       "fast": fast, "history": history, "class_runs": n_slow_evals, "secs": time.time() - t0}
if MODEL == "flipped":
    res.update({"z1": Z1, "z2": Z2, "free_z": FREE_Z})
OUT.parent.mkdir(exist_ok=True)
with OUT.open("a") as fh:
    fh.write(json.dumps(res) + "\n")
if MODEL == "lcdm":  # keep the best ΛCDM point as the start for the other fits
    prev = json.loads(start.read_text()).get("chi2", np.inf) if start.exists() else np.inf
    if chi2 < prev:
        start.write_text(json.dumps({"slow": slow, "fast": fast, "chi2": chi2}))
print(f"DONE {MODEL} ν={NU} rad={RAD} lens={LENS}: χ²={chi2:.2f}", {k: round(v, 2) for k, v in parts.items()})
