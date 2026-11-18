"""BAO and SN Ia likelihoods; distances come from the Rust core."""
from __future__ import annotations

import numpy as np
import cosmo_bg

from .data import BaoData, SnData


def bao_theory(c: "cosmo_bg.Cosmology", data: BaoData, r_d: float) -> np.ndarray:
    """Predicted D/r_d in the data's ordering."""
    out = np.empty(len(data.z))
    for kind, f in (("DM_over_rs", c.d_m), ("DH_over_rs", c.d_h), ("DV_over_rs", c.d_v)):
        m = data.kind == kind
        if m.any():
            out[m] = np.array(f(list(data.z[m]))) / r_d
    return out


class SnLikelihood:
    """SN Ia χ² minimised analytically over the absolute magnitude M."""

    def __init__(self, data: SnData):
        self.d = data
        self.icov = data.inv_cov if data.inv_cov is not None else np.linalg.inv(data.cov)
        self.ones_icov = self.icov.sum(axis=0)
        self.s0 = self.ones_icov.sum()

    def chi2(self, c) -> float:
        d = self.d
        dl = np.array(c.d_l(list(d.z_hd))) * (1 + d.z_hel) / (1 + d.z_hd)
        r = d.mu_obs - (5 * np.log10(dl) + 25)
        a = r @ self.icov @ r
        b = r @ self.ones_icov
        return float(a - b * b / self.s0)
