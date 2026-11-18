"""Starting proposal covariance matrices for the full-Planck MCMC (boltzmann/mcmc_full.py covmat=...).

Proposal only: it sets the initial step sizes and directions, and Cobaya keeps learning it from the chains
(learn_proposal). It does not enter the posterior.

ΛCDM: the Planck PR4 NPIPE CamSpec TTTEEE + lowl + lowE covariance (Cobaya covmats, CAMB names), with
100θ_MC replaced by H0 through the CLASS derivatives of 100θ_s at the ΛCDM best fit, then combined (Gaussian product)
with the DESI DR2 BAO + Pantheon+ Fisher matrix in (H0, ω_b, ω_c) from the compressed likelihood.
Extended models: the compressed fit (CMB prior + BAO + SN) Hessian gives σ of the extra parameter(s) X and the slope
γ of (H0, ω_b, ω_c) on X; the shared block is the ΛCDM one plus γ Σ_X γᵀ, Cov(s, X) = γ Σ_X, and X is uncorrelated
with A_s, n_s, τ and the nuisances.
Usage: python3 boltzmann/make_covmats.py planck=<path to base_planck_NPIPE_highl_CamSpec_TTTEEE_lowl_lowE.covmat>
Writes boltzmann/covmats/<lcdm|rvm|flipped_1.22_2.16|cpl>.covmat (Cobaya format: '# names' header, then the matrix).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "python"))
sys.path.insert(0, os.environ.get("CLASSY_RVM", str(HERE / "boltzmann" / "classy_rvm")))  # LCDM θ_s only; stock CLASS gives the same
from p1.inference import chi2_parts, load_data  # noqa: E402
from p1.models import flipped_early_factor, rvm_omega_c0, theory  # noqa: E402

args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
OUT = HERE / "boltzmann" / "covmats"
OUT.mkdir(exist_ok=True)
Z1, Z2 = 1.22, 2.16
RENAME = {"logA": "logA", "ns": "n_s", "ombh2": "omega_b", "omch2": "omega_cdm", "tau": "tau_reio"}
SHARED = ["H0", "omega_b", "omega_cdm"]


def read_covmat(path):
    with open(path) as f:
        names = f.readline().lstrip("#").split()
    return names, np.loadtxt(path, comments="#")


def dtheta():
    """∂(100θ_s)/∂(H0, ω_b, ω_c) with CLASS at the ΛCDM best fit (θ_MC ≈ θ_s up to a constant factor)."""
    from classy import Class

    base = dict(H0=68.12, omega_b=0.02225, omega_cdm=0.1182, A_s=2.1e-9, n_s=0.967, tau_reio=0.058,
                N_ur=2.0308, N_ncdm=1, m_ncdm=0.06, T_ncdm=0.71611)

    def th(**kw):
        c = Class()
        c.set({**base, **kw})
        c.compute(["thermodynamics"])
        v = c.get_current_derived_parameters(["100*theta_s"])["100*theta_s"]
        c.struct_cleanup()
        return v

    return np.array([(th(**{k: base[k] + h}) - th(**{k: base[k] - h})) / (2 * h)
                     for k, h in (("H0", 0.1), ("omega_b", 1e-5), ("omega_cdm", 1e-4))])


def hessian(f, x, steps):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.eye(n)[i] * steps[i], np.eye(n)[j] * steps[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * steps[i] * steps[j])
    return H


def jacobian(g, x, steps):
    return np.column_stack([(g(x + np.eye(len(x))[i] * steps[i]) - g(x - np.eye(len(x))[i] * steps[i])) / (2 * steps[i])
                            for i in range(len(x))])


def write(name, names, C):
    np.testing.assert_allclose(C, C.T, atol=0)
    assert np.all(np.linalg.eigvalsh(C) > 0), f"{name}: covmat not positive definite"
    np.savetxt(OUT / f"{name}.covmat", C, header=" ".join(names), comments="# ")
    sd = np.sqrt(np.diag(C))
    print(f"{name}: " + ", ".join(f"{n} {s:.3g}" for n, s in zip(names, sd) if not n.startswith(("amp_", "n_1", "n_2", "cal"))))


def main():
    d = load_data()
    real = json.loads((HERE / "results" / "real_fits.json").read_text())

    # --- Planck block in the sampled parameters
    pnames, Cp = read_covmat(args["planck"])
    it = pnames.index("theta_MC_100")
    tH, tb, tc = dtheta()
    J = np.eye(len(pnames))  # rows: new params (H0 in place of θ), columns: old
    J[it, :] = 0
    J[it, it] = 1 / tH
    J[it, pnames.index("ombh2")] = -tb / tH
    J[it, pnames.index("omch2")] = -tc / tH
    names = [("H0" if n == "theta_MC_100" else RENAME.get(n, n)) for n in pnames]
    Cy = J @ Cp @ J.T
    print(f"Planck alone: σ(H0) = {np.sqrt(Cy[it, it]):.3f}")

    # --- DESI + Pantheon+ Fisher (compressed likelihood, no CMB prior) in (h, ω_b, ω_bc) -> (H0, ω_b, ω_c)
    def late(p):
        c = chi2_parts(d, theory("lcdm", p))
        return c["bao"] + c["sn"]

    p0 = np.array(real["lcdm"]["best_fit"])
    Fp = hessian(late, p0, [0.002, 0.0001, 0.0008]) / 2
    A = np.array([[100, 0, 0], [0, 1, 0], [0, -1, 1]])  # y = A p
    Ai = np.linalg.inv(A)
    Fy = Ai.T @ Fp @ Ai
    idx = [names.index(n) for n in SHARED]
    P = np.linalg.inv(Cy)
    P[np.ix_(idx, idx)] += Fy
    C_lcdm = np.linalg.inv(P)
    C_lcdm = (C_lcdm + C_lcdm.T) / 2
    write("lcdm", names, C_lcdm)

    # --- extended models: slope of the shared parameters on X from the full compressed fit
    def full(model, fixed=()):
        return lambda p: sum(chi2_parts(d, theory(model, list(p) + list(fixed))).values())

    flip_fit = minimize(full("flipped", (Z1, Z2)), [0.68, 0.02223, 0.1414, 0.05], method="Nelder-Mead",
                        options={"xatol": 1e-7, "fatol": 1e-4, "maxiter": 4000})
    print(f"compressed flipped ({Z1}, {Z2}) best fit: {flip_fit.x}, chi2 {flip_fit.fun:.3f}")
    ext = {
        "rvm": (full("rvm"), np.array(real["rvm"]["best_fit"]), [0.002, 0.0001, 0.0008, 0.0002], ["rvm_nu"],
                lambda p: np.array([100 * p[0], p[1], rvm_omega_c0(*p), p[3]])),
        "flipped_1.22_2.16": (full("flipped", (Z1, Z2)), flip_fit.x, [0.002, 0.0001, 0.0008, 0.01], ["rvm_nu"],
                              lambda p: np.array([100 * p[0], p[1], p[2] / flipped_early_factor(p[3], Z1, Z2) - p[1], p[3]])),
        "cpl": (full("cpl"), np.array(real["cpl"]["best_fit"]), [0.002, 0.0001, 0.0008, 0.03, 0.12], ["w0_fld", "wa_fld"],
                lambda p: np.array([100 * p[0], p[1], p[2] - p[1], p[3], p[4]])),
    }
    for name, (f, x0, steps, xnames, to_y) in ext.items():
        H = hessian(f, x0, steps)
        Cp_ext = 2 * np.linalg.inv(H)
        Jy = jacobian(to_y, x0, np.array(steps) / 10)
        Cye = Jy @ Cp_ext @ Jy.T
        nx = len(xnames)
        Sx = Cye[3:, 3:]
        gamma = Cye[:3, 3:] @ np.linalg.inv(Sx)
        n = len(names)
        C = np.zeros((n + nx, n + nx))
        C[:n, :n] = C_lcdm
        C[np.ix_(idx, idx)] += gamma @ Sx @ gamma.T
        C[np.ix_(idx, range(n, n + nx))] = gamma @ Sx
        C[np.ix_(range(n, n + nx), idx)] = (gamma @ Sx).T
        C[n:, n:] = Sx
        write(name, names + xnames, C)


if __name__ == "__main__":
    main()
