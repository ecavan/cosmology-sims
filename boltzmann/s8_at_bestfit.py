"""σ8 and S8 of each full-Planck best fit (no fitting; one CLASS run per row).

For each row and lensing choice it takes the lowest-χ² record in the stored full-CMB results, runs the patched CLASS
once at exactly that record's cosmological ("slow") parameters, with the same model settings as boltzmann/full_cmb.py
plus `output=mPk`, and reads off:
  sigma8     linear σ8 of total matter (CDM + baryons + massive ν) at z = 0 (CLASS `sigma8`); sigma8_cb is also stored
  Omega_m    today's (z = 0) non-relativistic matter density parameter, as CLASS defines Omega0_m:
             (ρ_b + ρ_cdm + ρ_ncdm − 3 p_ncdm)/ρ_crit at z = 0 (baryons, CDM and the non-relativistic part of the
             0.06 eV neutrino). In the RVM and flipped RVM ρ_cdm is the exchanging CDM, so this is today's value after
             the exchange.
  S8         sigma8 * sqrt(Omega_m / 0.3)
  omega_bc_today, omega_bc_star   ω_b + ω_c today, and (ρ_b + ρ_cdm) a³ h²/ρ_crit,0 at z_star (the CMB's matter per
             comoving volume); equal for ΛCDM and w0wa, they differ when matter exchanges energy with the vacuum.
Output: results/s8_bestfits.json (one record per row and lensing choice, with the parameters used).
Usage: python3 boltzmann/s8_at_bestfit.py   (needs CLASSY_RVM, as full_cmb.py)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
CLASSY_DIR = os.environ.get("CLASSY_RVM", str(HERE / "boltzmann" / "classy_rvm"))
sys.path.insert(0, CLASSY_DIR)
import classy  # noqa: E402

assert classy.__file__.startswith(CLASSY_DIR), f"stock classy imported: {classy.__file__}"

FILES = {"pr4": ("results/full_cmb_pr4.jsonl", "results/full_cmb_flipped.jsonl"),
         "actplanck": ("results/full_cmb_actplanck.jsonl", "results/full_cmb_flipped_actplanck.jsonl")}
# profile minima of the minimal RVM (make_numbers.py: Nfcnu, NfcnuR, Nfcanu, NfcanuR), in 10^-3
NU_MIN = {("pr4", 1): -0.49, ("pr4", 0): -0.56, ("actplanck", 1): -0.61, ("actplanck", 0): -0.65}
# same model settings as full_cmb.py
BASE = {"N_ur": 2.0308, "N_ncdm": 1, "m_ncdm": 0.06, "T_ncdm": 0.71611, "non_linear": "hmcode",
        "tol_initial_Omega_r": 0.02, "output": "mPk", "P_k_max_h/Mpc": 10.0}


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in (HERE / path).read_text().splitlines() if line.strip()]


def best(recs: list[dict], **match) -> dict:
    sel = [r for r in recs if all(r.get(k) == v for k, v in match.items()) and not r.get("free_z", False)]
    if not sel:
        raise LookupError(f"no record for {match}")
    return min(sel, key=lambda r: r["chi2"])


def rows(lens: str) -> list[tuple[str, dict]]:
    main, flip = (load(p) for p in FILES[lens])
    out = [("LCDM", best(main, model="lcdm")), ("w0waCDM", best(main, model="cpl"))]
    for rad, name in ((1, "RVM"), (0, "RVM rad=0")):
        nus = sorted({r["nu"] for r in main if r["model"] == "rvm" and r["rad"] == rad})
        nu = min(nus, key=lambda v: abs(v - NU_MIN[(lens, rad)] * 1e-3))
        out.append((name, best(main, model="rvm", rad=rad, nu=nu)))
    for z1, z2 in ((1.22, 2.16), (1.1, 1.8)):
        out.append((f"flipped ({z1}, {z2})", best(flip, model="flipped", z1=z1, z2=z2)))
    for nu in (0.015, 0.03):
        out.append(("flipped (2.58, 6.1)", best(flip, model="flipped", z1=2.58, z2=6.1, nu=nu)))
    return out


def class_params(r: dict) -> dict:
    s = r["slow"]
    p = {**BASE, "H0": s["H0"], "omega_b": s["omega_b"], "omega_cdm": s["omega_cdm"],
         "A_s": 1e-10 * np.exp(s["logA"]), "n_s": s["n_s"], "tau_reio": s["tau_reio"]}
    if r["model"] == "cpl":
        p.update({"Omega_Lambda": 0.0, "use_ppf": "yes", "w0_fld": s["w0_fld"], "wa_fld": s["wa_fld"]})
    if r["model"] == "rvm":
        p.update({"rvm_nu": r["nu"], "rvm_rad": r["rad"]})
    if r["model"] == "flipped":
        p.update({"rvm_nu": r["nu"], "rvm_flip_z1": r["z1"], "rvm_flip_z2": r["z2"]})
    return p


def evaluate(p: dict) -> dict:
    c = classy.Class()
    c.set(p)
    c.compute()
    h = c.h()
    zs = c.get_current_derived_parameters(["z_star"])["z_star"]
    bg = c.get_background()
    z = bg["z"][::-1]
    rho_bc = (bg["(.)rho_b"] + bg["(.)rho_cdm"])[::-1]
    rho_crit0 = bg["(.)rho_crit"][-1]
    wbc_star = float(np.interp(zs, z, rho_bc) / (1 + zs) ** 3 / rho_crit0 * h * h)
    out = {"sigma8": c.sigma8(), "sigma8_cb": c.sigma8_cb(), "Omega_m": c.Omega_m(), "h": h,
           "omega_bc_today": float(rho_bc[0] / rho_crit0 * h * h), "omega_bc_star": wbc_star, "z_star": zs}
    out["S8"] = out["sigma8"] * np.sqrt(out["Omega_m"] / 0.3)
    c.struct_cleanup()
    c.empty()
    return {k: float(v) for k, v in out.items()}


def main() -> None:
    res = []
    for lens in FILES:
        for name, r in rows(lens):
            p = class_params(r)
            try:
                v = evaluate(p)
            except Exception as e:  # report, don't hide
                v = {"error": str(e)}
            rec = {"row": name, "lens": lens, "model": r["model"], "nu": r["nu"], "rad": r.get("rad"),
                   "z1": r.get("z1"), "z2": r.get("z2"), "chi2": r["chi2"], **v,
                   "params": {k: (float(x) if isinstance(x, (int, float, np.floating)) else x) for k, x in p.items()}}
            res.append(rec)
            print(f"{name:22s} {lens:9s} nu={r['nu']:+.5f}  " + ("ERROR " + v["error"] if "error" in v else
                  f"sigma8={v['sigma8']:.4f} Om={v['Omega_m']:.4f} h={v['h']:.4f} S8={v['S8']:.4f} "
                  f"wbc0={v['omega_bc_today']:.5f} wbc*={v['omega_bc_star']:.5f}"), flush=True)
    (HERE / "results" / "s8_bestfits.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
