"""Patched CLASS background vs cosmo_bg: minimal RVM (ν in {−2, −1, 0, +1}×1e-3) and the flipped RVM."""
import os
import sys
from pathlib import Path

import numpy as np

CLASSY_DIR = os.environ.get("CLASSY_RVM", str(Path(__file__).resolve().parent / "classy_rvm"))
sys.path.insert(0, CLASSY_DIR)
import classy  # noqa: E402
import cosmo_bg  # noqa: E402

assert classy.__file__.startswith(CLASSY_DIR), f"stock classy imported: {classy.__file__}"
H, WB, WC, MNU = 0.6875, 0.02224, 0.1188, 0.06
NEFF = 3.044


def run(nu, rad=1, **flip):
    c = classy.Class()
    c.set({"h": H, "omega_b": WB, "omega_cdm": WC, "N_ur": NEFF - 1.0132, "N_ncdm": 1, "m_ncdm": MNU,
           "T_ncdm": 0.71611, "T_cmb": 2.7255, "rvm_nu": nu, "rvm_rad": rad, "YHe": 0.2454,
           "tol_initial_Omega_r": 0.02, **flip})  # CDM carries a 4|ν| radiation-like piece early on
    c.compute()
    return c


def compare(c, ours):
    """Max relative D_M and H(z) differences over zs."""
    dm_c = np.array([c.angular_distance(z) * (1 + z) for z in zs])
    hz_c = np.array([c.Hubble(z) for z in zs]) * 299792.458
    return (np.max(np.abs(dm_c / np.array(ours.d_m(list(zs))) - 1)),
            np.max(np.abs(hz_c / (100 * H * np.array(ours.e_of_z(list(zs)))) - 1)))


zs = np.array([0.1, 0.5, 1.0, 2.33, 10, 100, 1089.0])
worst = 0.0
for nu in [-2e-3, -1e-3, 0.0, 1e-3]:
    c = run(nu)
    ours = cosmo_bg.Cosmology(H, WB, WC, NEFF, 2.7255, "rvm", -1.0, 0.0, nu, m_nu=MNU)
    r1, r2 = compare(c, ours)
    worst = max(worst, r1, r2)
    bg = c.get_background()
    print(f"nu={nu:+.0e}: max|ΔD_M/D_M|={r1:.1e}  max|ΔH/H|={r2:.1e}  100θ*={c.theta_star_100():.6f}"
          f"  rho_cdm>0 everywhere: {bool(np.all(bg['(.)rho_cdm'] > 0))}")
lcdm = run(0.0)
for nu, z1, z2 in [(0.03, 1.1, 1.8), (0.12, 1.1, 1.8), (0.03, 1.22, 2.16), (0.12, 1.22, 2.16), (0.0, 1.22, 2.16)]:
    c = run(nu, rvm_flip_z1=z1, rvm_flip_z2=z2)
    ours = cosmo_bg.Cosmology(H, WB, WC, NEFF, 2.7255, "flipped", -1.0, 0.0, nu, m_nu=MNU, z1=z1, z2=z2)
    r1, r2 = compare(c, ours)
    worst = max(worst, r1, r2)
    same = abs(c.theta_star_100() / lcdm.theta_star_100() - 1) if nu == 0 else float("nan")
    print(f"flipped nu={nu:.2f} z1={z1} z2={z2}: max|ΔD_M/D_M|={r1:.1e}  max|ΔH/H|={r2:.1e}"
          f"  100θ*={c.theta_star_100():.6f}" + (f"  |Δθ*/θ*| vs ΛCDM={same:.0e}" if nu == 0 else ""))
print("background agreement OK" if worst < 2e-5 else f"MISMATCH {worst}")
