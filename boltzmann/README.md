# Full-CMB check with a running-vacuum CLASS

`class_v3.4.1_rvm.patch` adds the running vacuum to CLASS v3.4.1 (lesgourg/class_public, tag v3.4.1).

- **Background:** CDM takes the closed form `rho_cdm = H0^2 [A a^{-3(1-nu)} - Om_b a^{-3} - k Om_r a^{-4}]` with
  `k = 4nu/(1+3nu)` times `rvm_rad`. For `rvm_rad = 1` (canonical, identical to `rust/cosmo-bg`) the vacuum
  `rho_vac0 + nu (H^2 - H0^2)` is solved together with the Friedmann equation. For `rvm_rad = 0` it is
  `rho_vac0 + nu/(1-nu) A (a^{-3(1-nu)} - 1)`, with no running in the radiation era (RRVM-like).
- **Perturbations:** synchronous gauge, `delta rho_vac = 0`, geodesic CDM:
  `delta_cdm' + h'/2 - (rho_vac'/rho_cdm) delta_cdm = 0` (arXiv:2512.20616).
- **Flipped RVM** (arXiv:2512.20616 §II.E): with `rvm_flip_z2 > 0`, total matter goes as `a^{-3(1-nu)}` below
  `rvm_flip_z1`, `a^{-3(1+nu)}` between the switches and `a^{-3}` above `rvm_flip_z2`, continuous at both; CDM alone
  exchanges, the vacuum is constant above z2, and the perturbation equation is the same as above.
- **Inputs:** `rvm_nu` (default 0, patch inactive), `rvm_rad` (default 1), `rvm_flip_z1`, `rvm_flip_z2` (default 0:
  minimal RVM). Use `tol_initial_Omega_r = 0.02` for |nu| of order 1e-3.

## Build

```bash
git clone https://github.com/lesgourg/class_public && cd class_public && git checkout v3.4.1
git apply ../boltzmann/class_v3.4.1_rvm.patch
pip install --no-deps --no-build-isolation --target ../boltzmann/classy_rvm .
```

The scripts import classy from `boltzmann/classy_rvm` (or `$CLASSY_RVM`).

- `validate_class_rvm.py`: background vs cosmo_bg; D_M agrees to ~1e-8 and H(z) to ~5e-7 for nu in {-2, -1, 0, +1}e-3
- `s8_at_bestfit.py`: linear σ8, today's Ω_m and S8 = σ8 (Ω_m/0.3)^½ at each full-Planck best fit (one CLASS run per
  row, no fitting) → `results/s8_bestfits.json`
  and for the flipped RVM at nu in {0.03, 0.12}, (z1, z2) in {(1.1, 1.8), (1.22, 2.16)}; nu = 0 reproduces ΛCDM.
- `full_cmb.py`: profile likelihood (`model=lcdm|cpl|rvm|flipped`) with Planck PR4 CamSpec TTTEEE, Planck 2018
  low-ell TT and EE and Planck PR4 lensing (`lens=pr4`, default; needs `planckpr4lensing` from
  github.com/carronj/planck_PR4_lensing), 2018 lensing (`lens=2018`), or the ACT DR6 + Planck PR4 lensing
  combination used by DESI DR2 (`lens=actplanck`: `act_dr6_lenslike` v1.2.1, variant `actplanck_baseline`, about 3 GB
  of RAM per process; `warm=1 warm_from=results/full_cmb_pr4.jsonl warm_lens=pr4` starts from a PR4 fit), via Cobaya (set `COBAYA_PACKAGES_PATH`), plus
  DESI DR2 BAO and SNe from `python/p1`. The flipped RVM takes `nu`, `z1`, `z2`, and `free_z=1` frees the switches.
- `results/full_cmb.jsonl` (2018 lensing) was run as `model=lcdm`, `model=cpl maxfun=220` and
  `model=rvm nu=... rad=1|0`; the `nu=-0.0015` point needed `warm=1` and `extrap=-0.0005,-0.001` continuations.
- `queue_full_cmb.sh`: the PR4-lensing profiles (minimal RVM, RRVM-like variant, w0wa) and the flipped-RVM profiles on
  two workers, appending to `results/full_cmb_pr4.jsonl` and `results/full_cmb_flipped.jsonl`; restartable.
- `results/full_cmb_actplanck.jsonl` and `results/full_cmb_flipped_actplanck.jsonl` (`out=`): every RVM, RRVM-like and
  flipped-RVM point above, rerun with `lens=actplanck`, first warm-started from the PR4 fits, then from their own best
  fits (`warm=1`); ΛCDM and w0wa twice each. The run list is in `profile_runs.txt`.
- `profile_runs.txt`: every run behind the PR4 files, in order, including the warm reruns that settled each point
  (the paper quotes the lowest χ² per point; `scripts/make_numbers.py` fits the parabolas).
