"""Reproduce published flat-ΛCDM grid posteriors: DESI DR2 BAO (Ω_m = 0.2975 ± 0.0086, h·r_d = 101.54 ± 0.73 Mpc)
and Pantheon+ (Ω_m = 0.334 ± 0.018)."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import cosmo_bg  # noqa: E402
from p1.data import fetch_all, load_desi_dr2, load_pantheon_plus  # noqa: E402
from p1.likelihoods import SnLikelihood, bao_theory  # noqa: E402

H_FID, OMBH2 = 0.6736, 0.02237


def cosmo(om):
    return cosmo_bg.Cosmology(h=H_FID, ombh2=OMBH2, omch2=om * H_FID**2 - OMBH2)


def summarize(x, p):
    p = p / p.sum()
    mean = (x * p).sum()
    return mean, np.sqrt(((x - mean) ** 2 * p).sum())


def desi():
    data = load_desi_dr2()
    oms = np.linspace(0.25, 0.35, 201)
    hrd = np.linspace(97.0, 106.0, 361)
    icov = data.inv_cov
    chi2 = np.empty((len(oms), len(hrd)))
    for i, om in enumerate(oms):
        t = bao_theory(cosmo(om), data, r_d=1.0) * H_FID  # = D·h  [Mpc]; D/r_d = t / (h r_d)
        for j, x in enumerate(hrd):
            r = data.value - t / x
            chi2[i, j] = r @ icov @ r
    post = np.exp(-0.5 * (chi2 - chi2.min()))
    m_om, s_om = summarize(oms, post.sum(axis=1))
    m_hr, s_hr = summarize(hrd, post.sum(axis=0))
    print(f"DESI DR2 BAO   Ω_m  = {m_om:.4f} ± {s_om:.4f}   (published 0.2975 ± 0.0086)")
    print(f"               h·r_d = {m_hr:.2f} ± {s_hr:.2f} Mpc (published 101.54 ± 0.73)")
    print(f"               χ²_min = {chi2.min():.2f} for {len(data.z)} points")
    return abs(m_om - 0.2975) < 0.2 * 0.0086 and abs(m_hr - 101.54) < 0.2 * 0.73


def pantheon():
    lik = SnLikelihood(load_pantheon_plus())
    oms = np.linspace(0.24, 0.44, 81)
    t0 = time.time()
    chi2 = np.array([lik.chi2(cosmo(om)) for om in oms])
    dt = (time.time() - t0) / len(oms)
    m, s = summarize(oms, np.exp(-0.5 * (chi2 - chi2.min())))
    print(f"Pantheon+ SN   Ω_m  = {m:.4f} ± {s:.4f}   (published 0.334 ± 0.018)   [{dt*1e3:.0f} ms / likelihood]")
    print(f"               χ²_min = {chi2.min():.1f} for {len(lik.d.z_hd)} SNe")
    return abs(m - 0.334) < 0.25 * 0.018


if __name__ == "__main__":
    fetch_all()
    ok = desi()
    ok &= pantheon()
    print("REPRODUCED" if ok else "MISMATCH — investigate before using the pipeline")
    sys.exit(0 if ok else 1)
