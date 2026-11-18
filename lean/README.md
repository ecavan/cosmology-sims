# Machine-checked derivations (Lean 4 + Mathlib)

The paper's key analytic results, proved in Lean 4: a file that builds is a proof with no gaps.

| Paper result | Lean theorem | File |
|---|---|---|
| Vacuum law + Friedmann + exchange ⇒ linear ODE for ρ_c (Sec. 2) | `cdm_ode_of_model` | `ClosedForm.lean` |
| Closed form `A a^{-3(1-ν)} − B a^{-3} − kR a^{-4}`, `k = 4ν/(1+3ν)`, solves it | `closedForm_solves` | `ClosedForm.lean` |
| …and is the only solution (general solution) | `ode_solution_unique` | `ClosedForm.lean` |
| Exact apparent EoS `1+w = (Ω_m a³ − Ω̃_m) a^{-3}/ρ̃_DE` (Eq. weff_general) | `weff_add_one` | `Crossing.lean` |
| `w = −1` exactly at the anchor `Ω_m(a)a³ = Ω̃_m` | `weff_eq_neg_one_iff` | `Crossing.lean` |
| For the RVM (ν ≠ 0) the anchor is unique: at most one crossing | `rvm_comoving_matter_injective` | `Crossing.lean` |
| First-order formula `1+w ≈ 3ν(Ω_m/Ω_Λ)(1+z)³ ln[(1+z_A)/(1+z)]` (Eq. weff_linear) | `weff_first_order`, `weff_first_order_redshift` | `Linear.lean` |

`Axioms.lean` prints the axioms each theorem uses. All of them use only `propext`, `Classical.choice` and
`Quot.sound`, Lean's standard axioms, so there is no `sorry` and no hidden assumption.

## What this does and does not establish

These files prove the mathematics of the model: given its assumptions, the formulas follow. They do not check:

- whether the model is physically right
- the numerics, fits or data
- that the Lean statements match the paper's equations.

The last point is the human's job: **read the theorem statements**, not the proofs. Simplifications made in the
formal statements:

- `s = ln a` is the variable.
- Units are `8πG/3 = 1`.
- Massive neutrinos are left out of the exchange, an O(ν Ω_ν) effect; the code makes the same approximation.
- In `Linear.lean`, the matter-era form `Ω_m a³ = A a^{3ν}` is used.
- "First order" means `HasDerivAt` at ν = 0, i.e. the error is o(ν).

## Build

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh   # installs elan (Lean toolchain manager)
cd lean
lake exe cache get   # downloads prebuilt Mathlib (~5 GB in .lake/, not committed)
lake build           # checks every proof (a few minutes)
```

Editor: install VS Code with the "Lean 4" extension and open this folder. Put the cursor inside a proof to see
the live goal state.
