"""Joint χ² (DESI DR2 BAO + SN Ia + compressed CMB), Nelder–Mead best fits and emcee chains."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import BaoData, load_desi_dr2, load_sn
from .likelihoods import SnLikelihood, bao_theory
from .models import PARAMS, Theory, in_prior, theory

CMB_PRIOR = Path(__file__).resolve().parents[2] / "data" / "cmb_prior_massless.npz"  # scripts/make_cmb_prior.py


@dataclass
class Data:
    bao: BaoData
    sn: SnLikelihood
    cmb_mean: np.ndarray   # (θ*, ω_b, ω_bc)
    cmb_icov: np.ndarray


def load_data(sn: str = "pantheon+") -> Data:
    c = np.load(CMB_PRIOR)
    return Data(load_desi_dr2(), SnLikelihood(load_sn(sn)), c["mean"], np.linalg.inv(c["cov"]))


def chi2_parts(d: Data, t: Theory) -> dict:
    r = d.bao.value - bao_theory(t.cosmo, d.bao, t.r_d)
    rc = d.cmb_mean - np.array([t.theta_star, t.omega_b, t.omega_bc_star])
    return {"bao": float(r @ d.bao.inv_cov @ r), "sn": d.sn.chi2(t.cosmo), "cmb": float(rc @ d.cmb_icov @ rc)}


def log_like(p, model: str, d: Data) -> float:
    if not in_prior(model, p):
        return -np.inf
    try:
        t = theory(model, p)
    except Exception:
        return -np.inf
    s = sum(chi2_parts(d, t).values())
    return -0.5 * s if np.isfinite(s) else -np.inf


START = {"lcdm": [0.677, 0.02223, 0.1414], "cpl": [0.675, 0.02223, 0.1414, -0.85, -0.6],
         "rvm": [0.677, 0.02223, 0.1414, 0.0], "flipped": [0.68, 0.02223, 0.1414, 0.03, 1.2, 2.2]}
STEP = {"h": 0.005, "omega_b": 0.00015, "omega_bc": 0.0012, "omega_bc_star": 0.0012, "w0": 0.05, "wa": 0.2,
        "nu": 0.001, "nu_f": 0.01, "z1": 0.3, "z2": 0.5}


def best_fit(model: str, d: Data, start=None, restarts: int = 4, seed: int = 0):
    """Minimum χ² by Nelder–Mead from `restarts` jittered starts; returns (x, χ²)."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    x0 = np.array(start if start is not None else START[model], float)
    scale = np.array([STEP[n] for n in PARAMS[model]])
    best = None
    for k in range(restarts):
        x = x0 if k == 0 else x0 + scale * rng.normal(size=len(x0))
        if not in_prior(model, x):
            x = x0
        res = minimize(lambda q: -2 * log_like(q, model, d) if np.isfinite(log_like(q, model, d)) else 1e30,
                       x, method="Nelder-Mead",
                       options={"xatol": 1e-7, "fatol": 1e-6, "maxiter": 20000, "maxfev": 20000, "adaptive": True})
        if best is None or res.fun < best.fun:
            best = res
    return best.x, best.fun


def sample(model: str, d: Data, nwalkers: int = 24, nsteps: int = 3000, burn: int = 1000, seed: int = 1, start=None):
    """emcee chain started around the best fit; returns (flat chain, best fit, autocorrelation times)."""
    import emcee

    x0, _ = best_fit(model, d, start=start)
    rng = np.random.default_rng(seed)
    scale = np.array([STEP[n] for n in PARAMS[model]]) * 0.1
    p0 = np.array([x0 + scale * rng.normal(size=len(x0)) for _ in range(nwalkers)])
    s = emcee.EnsembleSampler(nwalkers, len(x0), log_like, args=(model, d))
    s.run_mcmc(p0, nsteps, progress=False)
    chain = s.get_chain(discard=burn, flat=True)
    try:
        tau = s.get_autocorr_time(discard=burn, quiet=True)
    except Exception:
        tau = np.full(len(x0), np.nan)
    return chain, x0, tau


def profile_nu(d: Data, x_lcdm, nus):
    """RVM profile: at each ν (in the given order), minimise over (h, ω_b, ω_bc*) from the last minimum and x_lcdm."""
    from scipy.optimize import minimize

    out, start = [], np.array(x_lcdm)
    for nu in nus:
        f = lambda q: -2 * log_like(list(q) + [nu], "rvm", d)
        best = None
        for x0 in (start, x_lcdm):
            r = minimize(lambda q: f(q) if np.isfinite(f(q)) else 1e30, x0, method="Nelder-Mead",
                         options={"xatol": 1e-7, "fatol": 1e-6, "maxfev": 20000, "adaptive": True})
            if best is None or r.fun < best.fun:
                best = r
        start = best.x
        out.append({"nu": nu, "chi2": float(best.fun), "x": best.x.tolist()})
    return out
