"""Parameter vectors → background (Rust core) and CMB/BAO early-universe quantities.

All models are flat with one 0.06 eV neutrino (N_eff = 3.044):
  lcdm:    [h, ω_b, ω_bc]
  cpl:     [h, ω_b, ω_bc, w0, wa]
  rvm:     [h, ω_b, ω_bc*, ν]           ω_bc* = CDM+baryon density at z = 1090 scaled by a³ (what the CMB measures)
  flipped: [h, ω_b, ω_bc*, ν, z1, z2]   ω_bc* = matter density for z > z2 scaled by a³ (no running there)
"""
from __future__ import annotations

from dataclasses import dataclass

import cosmo_bg

from .early import M_NU, early

Z_REF = 1090.0
PARAMS = {
    "lcdm": ["h", "omega_b", "omega_bc"],
    "cpl": ["h", "omega_b", "omega_bc", "w0", "wa"],
    "rvm": ["h", "omega_b", "omega_bc_star", "nu"],
    "flipped": ["h", "omega_b", "omega_bc_star", "nu_f", "z1", "z2"],
}
PRIORS = {  # flat; ω ranges are the early-table bounds; flipped also needs z2 ≥ z1 + 0.1
    "h": (0.50, 0.90), "omega_b": (0.0206, 0.0239), "omega_bc": (0.121, 0.164), "omega_bc_star": (0.121, 0.164),
    "w0": (-3.0, 1.0), "wa": (-3.0, 2.0), "nu": (-0.05, 0.05),
    "nu_f": (-0.5, 0.5), "z1": (0.05, 10.0), "z2": (0.05, 10.0),
}


@dataclass
class Theory:
    cosmo: "cosmo_bg.Cosmology"
    omega_b: float
    omega_bc_star: float   # CDM+baryon density seen by the CMB, scaled to today with a³
    theta_star: float
    r_d: float
    z_star: float


def cosmology(h, wb, wc, **kw):
    return cosmo_bg.Cosmology(h=h, ombh2=wb, omch2=wc, m_nu=M_NU, **kw)


def omega_r(h: float) -> float:
    """Physical density of photons plus massless neutrinos."""
    return cosmology(h, 0.022, 0.12).omegas()["r"] * h * h


def rvm_omega_c0(h, wb, wbc_star, nu, z_ref=Z_REF):
    """Today's ω_c from ω_bc* by inverting ρ_c a³ = A a^{3ν} − ω_b − kω_r/a at a = 1/(1+z_ref)."""
    a = 1.0 / (1.0 + z_ref)
    wr = omega_r(h)
    k = 4 * nu / (1 + 3 * nu)
    wc_star = wbc_star - wb
    return (wc_star + wb + k * wr / a) / a ** (3 * nu) - wb - k * wr


def in_prior(model: str, p) -> bool:
    for name, v in zip(PARAMS[model], p):
        lo, hi = PRIORS[name]
        if not lo <= v <= hi:
            return False
    if model == "cpl" and p[3] + p[4] >= 0:  # early matter domination
        return False
    if model == "flipped" and not p[5] >= p[4] + 0.1:
        return False
    return True


def flipped_early_factor(nu, z1, z2):
    """ρ_m a³ for z > z2 divided by Ω_m0."""
    a1, a2 = 1 / (1 + z1), 1 / (1 + z2)
    return a1 ** (6 * nu) * a2 ** (-3 * nu)


def theory(model: str, p) -> Theory:
    if model == "lcdm":
        h, wb, wbc = p
        c = cosmology(h, wb, wbc - wb)
        wbc_star = wbc
    elif model == "cpl":
        h, wb, wbc, w0, wa = p
        c = cosmology(h, wb, wbc - wb, model="cpl", w0=w0, wa=wa)
        wbc_star = wbc
    elif model == "rvm":
        h, wb, wbc_star, nu = p
        c = cosmology(h, wb, rvm_omega_c0(h, wb, wbc_star, nu), model="rvm", nu=nu)
    elif model == "flipped":
        h, wb, wbc_star, nu, z1, z2 = p
        wm0 = wbc_star / flipped_early_factor(nu, z1, z2)
        c = cosmology(h, wb, wm0 - wb, model="flipped", nu=nu, z1=z1, z2=z2)
    else:
        raise ValueError(model)
    # CAMB early quantities, rescaled by the model/ΛCDM ratio of sound-horizon integrals
    e = early(wb, wbc_star)
    ref = cosmology(p[0], wb, wbc_star - wb)
    corr_star = c.sound_horizon(e["z_star"]) / ref.sound_horizon(e["z_star"])
    corr_d = c.sound_horizon(e["z_d"]) / ref.sound_horizon(e["z_d"])
    theta = e["rs_star"] * corr_star / c.d_m([e["z_star"]])[0]
    return Theory(c, wb, wbc_star, theta, e["rs_d"] * corr_d, e["z_star"])
