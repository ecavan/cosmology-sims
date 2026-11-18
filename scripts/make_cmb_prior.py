"""Write data/cmb_prior_massless.npz: DESI DR2 App. A (Eqs. A1–A2) prior on (θ*, ω_b, ω_bc), minus ω_ν = m_ν/93.14 eV.

DESI's 0.14208 is ω_b + ω_c + ω_ν of the Planck PR4 chains of Lemos & Lewis (2023); the CMB fixes ω_b + ω_c here.
"""
from pathlib import Path

import numpy as np

M_NU = 0.06
MEAN = np.array([0.01041, 0.02223, 0.14208 - M_NU / 93.14])
COV = 1e-9 * np.array([[0.006621, 0.12444, -1.1929],
                       [0.12444, 21.344, -94.001],
                       [-1.1929, -94.001, 1488.4]])
out = Path(__file__).resolve().parents[1] / "data" / "cmb_prior_massless.npz"
out.parent.mkdir(exist_ok=True)
np.savez(out, mean=MEAN, cov=COV)
print(f"wrote {out}: mean {MEAN}, sigma {np.sqrt(np.diag(COV))}")
assert np.all(np.linalg.eigvalsh(COV) > 0)
