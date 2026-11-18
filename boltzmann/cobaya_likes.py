"""Cobaya likelihoods for DESI DR2 BAO and SNe Ia with distances from CLASS (shared by full_cmb.py and mcmc_full.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from cobaya.likelihood import Likelihood

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from p1.data import load_desi_dr2, load_sn  # noqa: E402
from p1.likelihoods import SnLikelihood  # noqa: E402

C_KMS = 299792.458


class DesiBao(Likelihood):
    """DESI DR2 BAO as in python/p1, distances from CLASS."""

    def initialize(self):
        self.d = load_desi_dr2()
        self.icov = self.d.inv_cov
        self.zs = np.unique(self.d.z)

    def get_requirements(self):
        return {"angular_diameter_distance": {"z": self.zs}, "Hubble": {"z": self.zs}, "rdrag": None}

    def logp(self, **_):
        p = self.provider
        rd = p.get_param("rdrag")
        th = []
        for z, kind in zip(self.d.z, self.d.kind):
            dm = p.get_angular_diameter_distance(z)[0] * (1 + z)
            dh = C_KMS / p.get_Hubble(z)[0]
            th.append({"DM_over_rs": dm, "DH_over_rs": dh,
                       "DV_over_rs": (z * dm * dm * dh) ** (1 / 3)}[kind] / rd)
        r = self.d.value - np.array(th)
        return -0.5 * float(r @ self.icov @ r)


class SnIa(Likelihood):
    """SN Ia with M profiled (p1.likelihoods.SnLikelihood), distances from CLASS."""

    sample: str = "pantheon+"

    def initialize(self):
        self.lk = SnLikelihood(load_sn(self.sample))
        self.zs = np.asarray(self.lk.d.z_hd)

    def get_requirements(self):
        return {"angular_diameter_distance": {"z": self.zs}}

    def logp(self, **_):
        da = self.provider.get_angular_diameter_distance(self.zs)
        dl = da * (1 + self.zs) ** 2

        class _C:  # SnLikelihood expects .d_l(z_hd)
            @staticmethod
            def d_l(_z):
                return dl

        return -0.5 * self.lk.chi2(_C)
