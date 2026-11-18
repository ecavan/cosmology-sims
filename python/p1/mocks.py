"""Mock data sets: a model's theory vectors with the real covariances, plus optional Gaussian noise."""
from __future__ import annotations

import copy

import numpy as np

from .inference import Data
from .likelihoods import bao_theory
from .models import Theory


def truth_vectors(d: Data, t: Theory):
    """Noiseless BAO, SN μ (with M = −19.25, profiled out anyway) and CMB vectors of theory t."""
    s = d.sn.d
    dl = np.array(t.cosmo.d_l(list(s.z_hd))) * (1 + s.z_hel) / (1 + s.z_hd)
    return bao_theory(t.cosmo, d.bao, t.r_d), 5 * np.log10(dl) + 25 - 19.25, np.array(
        [t.theta_star, t.omega_b, t.omega_bc_star])


def noise_factors(d: Data):
    """Cholesky factors of the BAO, SN and CMB covariances."""
    return (np.linalg.cholesky(d.bao.cov), np.linalg.cholesky(d.sn.d.cov),
            np.linalg.cholesky(np.linalg.inv(d.cmb_icov)))


def with_vectors(d: Data, bao, mu, cmb) -> Data:
    """Copy of d with new data vectors and the same covariances."""
    m = copy.copy(d)
    m.bao = copy.copy(d.bao)
    m.bao.value = bao
    m.sn = copy.copy(d.sn)
    m.sn.d = copy.copy(d.sn.d)
    m.sn.d.mu_obs = mu
    m.cmb_mean = cmb
    return m


def asimov(d: Data, t: Theory) -> Data:
    """Noiseless mock of theory t."""
    return with_vectors(d, *truth_vectors(d, t))
