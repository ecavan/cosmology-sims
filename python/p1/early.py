"""CAMB early-universe quantities (r_*, z_*, r_d, z_d) tabulated on (ω_b, ω_bc): python3 -m p1.early builds the table.

CAMB, not CLASS, because the Planck PR4 θ* behind the CMB prior uses CAMB's z* (CLASS's differs by ≈5).
Neutrinos: one 0.06 eV species plus N_ur = 3.044 − 1.01317, as in the DESI/Planck baseline.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

TABLE = Path(__file__).resolve().parents[2] / "data" / "camb_early_table_mnu006.npz"
M_NU = 0.06
WB = np.linspace(0.0205, 0.0240, 15)
WBC = np.linspace(0.120, 0.165, 19)
KEYS = ("rs_star", "z_star", "rs_d", "z_d")
CAMB_KEYS = {"rs_star": "rstar", "z_star": "zstar", "rs_d": "rdrag", "z_d": "zdrag"}


def camb_early(wb, wbc, h=0.67):
    import camb

    p = camb.set_params(H0=100 * h, ombh2=wb, omch2=wbc - wb, mnu=M_NU, nnu=3.044, num_massive_neutrinos=1,
                        TCMB=2.7255)
    d = camb.get_background(p).get_derived_params()
    return {k: d[v] for k, v in CAMB_KEYS.items()}


def build_table():
    out = {k: np.empty((len(WB), len(WBC))) for k in KEYS}
    for i, wb in enumerate(WB):
        for j, wbc in enumerate(WBC):
            d = camb_early(wb, wbc)
            for k in KEYS:
                out[k][i, j] = d[k]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(TABLE, wb=WB, wbc=WBC, **out)
    print(f"saved {TABLE}")


@lru_cache(maxsize=1)
def _interp():
    t = np.load(TABLE)
    return {k: RegularGridInterpolator((t["wb"], t["wbc"]), t[k], method="cubic") for k in KEYS}


def early(wb: float, wbc: float) -> dict:
    """Interpolated CAMB r_*, z_*, r_d, z_d for ΛCDM early physics."""
    it = _interp()
    return {k: float(it[k]([[wb, wbc]])[0]) for k in KEYS}


if __name__ == "__main__":
    build_table()
    worst = 0.0  # interpolation and h-dependence error at off-grid points
    for wb, wbc in [(0.02237, 0.1430), (0.02180, 0.1376), (0.02301, 0.1525)]:
        d, e = camb_early(wb, wbc, h=0.70), early(wb, wbc)
        worst = max([worst] + [abs(e[k] / d[k] - 1) for k in KEYS])
    print(f"max interpolation (+h-dependence) error vs direct CAMB: {worst:.2e}")
    import camb
    import cosmo_bg
    p = camb.set_params(H0=67.4, ombh2=0.02223, omch2=0.1192, mnu=M_NU, nnu=3.044, num_massive_neutrinos=1,
                        TCMB=2.7255)
    r = camb.get_background(p)
    zs = r.get_derived_params()["zstar"]
    ours = cosmo_bg.Cosmology(h=0.674, ombh2=0.02223, omch2=0.1192, m_nu=M_NU).d_m([zs])[0]
    print(f"D_M(z*) ours/CAMB − 1 = {ours / r.comoving_radial_distance(zs) - 1:.2e}")
