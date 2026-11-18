# Data

Committed: `cmb_prior_massless.npz` (from `scripts/make_cmb_prior.py`) and `camb_early_table_mnu006.npz`
(from `python3 -m p1.early`, needs CAMB).

Downloaded at pinned versions by `python3 -m p1.data` (run from `python/`; not committed):

| Dataset | Source | Notes |
|---|---|---|
| DESI DR2 BAO (Gaussian) | [CobayaSampler/bao_data](https://github.com/CobayaSampler/bao_data) @ `bb0c1c9` | 13 measurements, z = 0.295–2.33 (arXiv:2503.14738) |
| Pantheon+ | [PantheonPlusSH0ES/DataRelease](https://github.com/PantheonPlusSH0ES/DataRelease) @ `c447f0f` | STAT+SYS covariance, 33 MB (arXiv:2202.04077) |
| DES-Y5, Union3, DES-Dovekie | [CobayaSampler/sn_data](https://github.com/CobayaSampler/sn_data) @ `61d9643` | SN robustness checks |
