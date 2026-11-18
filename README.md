# cosmology-sims

Code and results for **Can a running vacuum explain DESI's phantom crossing?** (Elijah Cavan, 2026).

The paper compares four flat models on DESI DR2 BAO, Pantheon+ (and three other SN Ia samples) and the CMB:

| Model | Parameters | Code name |
|---|---|---|
| ΛCDM | h, ω_b, ω_bc | `lcdm` |
| w0waCDM (CPL) | + w0, wa | `cpl` |
| Running vacuum, ρ_vac = ρ_vac0 + (3ν/8πG)(H² − H0²), exchanging energy with CDM | + ν | `rvm` |
| Flipped running vacuum (arXiv:2512.20616 §II.E), running only for z1 < z < z2 | + ν, z1, z2 | `flipped` |

## Layout

| Path | Contents |
|---|---|
| `rust/cosmo-bg` | Background solver: E(z), distances, sound horizon, the observer's effective w(z) and its CPL projection |
| `rust/cosmo-bg-py` | Python bindings (`import cosmo_bg`) |
| `python/p1` | Data loaders, likelihoods (BAO, SN with M profiled, compressed CMB), fits, mocks |
| `scripts/` | One script per analysis step; each writes to `results/` |
| `results/` | The outputs the paper quotes (fits, chains, mocks, profiles) |
| `boltzmann/` | CLASS v3.4.1 patch for the running vacuum, the full-Planck profile likelihood and the full-Planck MCMC (`mcmc_full.py`, `mcmc_queue.py`, `mcmc_summary.py`) |
| `chains_v2/` | The final full-Planck MCMC chains (ΛCDM, RVM, flipped RVM at switches 1.22 and 2.16; four unseeded chains each). `results/chains_summary_*.json` are computed from these with a 30% burn-in |
| `validation/` | Distances vs astropy and CLASS; published ΛCDM fits reproduced |
| `lean/` | Lean 4 proofs of the closed-form solution and the apparent equation of state |

## Reproduce

Needs Rust, Python ≥ 3.9 with numpy, scipy, emcee and matplotlib, and maturin.

```bash
cargo test --release -p cosmo-bg                     # unit tests (limits, closed form vs ODE, conservation)
(cd rust/cosmo-bg-py && maturin develop --release)   # builds the cosmo_bg module
(cd python && python3 -m p1.data)                    # downloads the data (pinned versions)

python3 scripts/make_numbers.py           # out/numbers.tex and out/sn_table.tex
python3 scripts/flipped_summary.py        # out/numbers_flipped.tex and out/figures/fig_flipped.pdf
python3 scripts/flipped_switch_figure.py  # out/numbers_switches.tex and out/figures/fig_flipped_switches.pdf
python3 scripts/figures.py                # the other paper figures
```

Together the first three write every number in the paper. These four read only `results/`. To recompute `results/` (hours on one core):

| Script | Output |
|---|---|
| `fit_real.py` | `real_fits.json`, `chains/`: best fits and MCMC for ΛCDM, w0wa, RVM |
| `mock_test.py` | `mock_test.json`: w0wa fits to noiseless RVM mocks |
| `noisy_mocks.py n=98 seed=11 tag=a` | `noisy_mocks_a.jsonl`; the paper's 1,000 Pantheon+ realisations are tags a–d: `n=98 seed=22 tag=b`, `n=402 seed=3 tag=c`, `n=402 seed=4 tag=d`. DES-Y5: `sn=desy5 n=45` with `seed=33 tag=a`, `seed=44 tag=b` |
| `refit_check.py` | checks that noisy-mock minima are global |
| `compressed_profile.py` | `compressed_profile.json`: RVM profile in ν |
| `sn_robustness.py sn=desy5` | `sn_robustness_<sn>.json` (also `union3`, `dovekie`, `pantheon+`) |
| `flipped_fit.py` | `flipped_fit_pantheon+.json`: all four models, flipped RVM by grid screen + polish |
| `flipped_switch_map.py` | `flipped_switch_map_pantheon+.json`: the flipped RVM refitted at 117 fixed pairs of switch redshifts |
| `flipped_mocks.py truth=lcdm n=200` | `flipped_mocks_<truth>_a.jsonl` (also `truth=cpl`, `truth=flipped`, `noise=0`) |
| `boltzmann/full_cmb.py` | `full_cmb_pr4.jsonl`, `full_cmb_flipped.jsonl`: full-Planck profiles with PR4 lensing; `full_cmb_actplanck.jsonl`, `full_cmb_flipped_actplanck.jsonl`: the same with ACT DR6 + PR4 lensing; `full_cmb.jsonl`: 2018-lensing cross-check (see `boltzmann/README.md`) |
| `validation/*.py` | consistency checks; exit non-zero on failure |

`scripts/make_cmb_prior.py` writes the compressed CMB prior from DESI DR2 App. A; every real-data result in `results/`
uses it. Noisy-mock tags c and d were drawn about the current best fit. The earlier ensembles (`noisy_mocks_a`, `_b`,
`noisy_mocks_desy5_*`, `flipped_mocks_*_a`) were drawn about best fits from an earlier prior that differed in the fifth
significant figure, far below the mock noise, so rerunning them reproduces the draws but not every digit of the fits.
All p-values are computed against the current real-data Δχ².
The seeds of the `flipped_mocks_*_a` ensembles were not recorded; rerunning them gives statistically equivalent draws,
not identical ones.

## License

MIT; see `LICENSE`.

## Citation and licence

MIT licence. To cite, see `CITATION.cff`. The code was written with AI assistance (Anthropic's Claude), as the paper
states; the author checked the results and is responsible for them.
