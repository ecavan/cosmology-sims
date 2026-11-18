import Mathlib

/-!
# Apparent equation of state of an interacting vacuum (paper Sec. 3, Eq. `weff_general`)

Work in `s = ln a`. Densities are in units of the critical density today (`8πG/3 = 1`).

* `m s` : true total matter density (CDM + baryons), `v s` : true vacuum density (`w = −1`).
* Matter and vacuum exchange energy, but their sum is conserved:
  `v' + m' + 3 m = 0` (vacuum has `w = −1`, matter has `w = 0`).
  Radiation and neutrinos are separately conserved, so they cancel identically from the observer's
  dark-energy density; they are omitted here without loss of generality.
* An observer who assumes self-conserved matter with normalisation `Mt` (`Ω̃_m`) attributes
  `ρ̃(s) = v s + m s − Mt e^{−3s}` to dark energy, and defines its equation of state by
  `ρ̃' = −3 (1 + w) ρ̃`.

Theorem `weff_add_one`: `1 + w = (m e^{3s} − Mt) e^{−3s} / ρ̃` — Solà & Štefančić (2005), in closed form.
Theorem `weff_eq_neg_one_iff`: `w = −1` exactly where the observer's normalisation equals `m e^{3s}`.
-/

open Real

namespace RvmProofs

/-- The observer's dark-energy density. -/
noncomputable def rhoDE (m v : ℝ → ℝ) (Mt s : ℝ) : ℝ := v s + m s - Mt * exp (-3 * s)

/-- The observer's dark-energy equation of state, from `ρ̃' = −3(1+w)ρ̃`. -/
noncomputable def weff (m v : ℝ → ℝ) (Mt s : ℝ) : ℝ :=
  -1 - deriv (fun t => rhoDE m v Mt t) s / (3 * rhoDE m v Mt s)

theorem hasDerivAt_rhoDE {m v : ℝ → ℝ} {m' v' Mt s : ℝ}
    (hm : HasDerivAt m m' s) (hv : HasDerivAt v v' s) :
    HasDerivAt (fun t => rhoDE m v Mt t) (v' + m' + 3 * Mt * exp (-3 * s)) s := by
  have he : HasDerivAt (fun t => exp (-3 * t)) (exp (-3 * s) * (-3)) s := by
    simpa using ((hasDerivAt_id s).const_mul (-3)).exp
  have := (hv.add hm).sub (he.const_mul Mt)
  unfold rhoDE
  convert this using 1
  ring

/-- **Exact apparent equation of state.** -/
theorem weff_add_one {m v : ℝ → ℝ} {m' v' Mt s : ℝ}
    (hm : HasDerivAt m m' s) (hv : HasDerivAt v v' s)
    (hcons : v' + m' + 3 * m s = 0) (hρ : rhoDE m v Mt s ≠ 0) :
    weff m v Mt s + 1 = (m s * exp (3 * s) - Mt) * exp (-3 * s) / rhoDE m v Mt s := by
  have hd := (hasDerivAt_rhoDE (Mt := Mt) hm hv).deriv
  unfold weff
  rw [hd]
  have hv' : v' = -m' - 3 * m s := by linarith
  rw [hv']
  have hρ' := hρ
  unfold rhoDE at hρ' ⊢
  simp only [show (-3 * s) = -(3 * s) by ring, Real.exp_neg] at hρ' ⊢
  have he : exp (3 * s) ≠ 0 := (exp_pos _).ne'
  field_simp
  ring

/-- **Crossing criterion.** The apparent equation of state equals `−1` exactly at the "anchor", where the
observer's matter normalisation equals the true comoving matter density `m e^{3s}`. -/
theorem weff_eq_neg_one_iff {m v : ℝ → ℝ} {m' v' Mt s : ℝ}
    (hm : HasDerivAt m m' s) (hv : HasDerivAt v v' s)
    (hcons : v' + m' + 3 * m s = 0) (hρ : rhoDE m v Mt s ≠ 0) :
    weff m v Mt s = -1 ↔ m s * exp (3 * s) = Mt := by
  have h := weff_add_one hm hv hcons hρ
  constructor
  · intro hw
    rw [hw] at h
    have : (m s * exp (3 * s) - Mt) * exp (-3 * s) = 0 := by
      have h0 : (m s * exp (3 * s) - Mt) * exp (-3 * s) / rhoDE m v Mt s = 0 := by linarith
      rcases div_eq_zero_iff.mp h0 with h1 | h1
      · exact h1
      · exact absurd h1 hρ
    rcases mul_eq_zero.mp this with h1 | h1
    · linarith
    · exact absurd h1 (exp_pos _).ne'
  · intro hA
    have : weff m v Mt s + 1 = 0 := by rw [h, hA]; simp
    linarith

/-- **At most one crossing for the running vacuum.** For the RVM the true comoving matter density is
`A e^{3νs} − k R e^{−s}` with `k = 4ν/(1+3ν)` (closed form, see `ClosedForm.lean`). If `ν ≠ 0`,
`1 + 3ν > 0`, `A > 0` and `R ≥ 0`, this is strictly monotone in `s`, so it equals any given
normalisation `Mt` at most once: the apparent equation of state crosses `−1` at most once. -/
theorem rvm_comoving_matter_injective {ν A R : ℝ} (hν : ν ≠ 0) (h13 : 0 < 1 + 3 * ν) (hA : 0 < A)
    (hR : 0 ≤ R) :
    Function.Injective (fun s : ℝ => A * exp (3 * ν * s) - 4 * ν / (1 + 3 * ν) * R * exp (-s)) := by
  set f := fun s : ℝ => A * exp (3 * ν * s) - 4 * ν / (1 + 3 * ν) * R * exp (-s) with hf
  have hk : 0 ≤ 4 * ν / (1 + 3 * ν) * R * ν := by
    have : 4 * ν / (1 + 3 * ν) * R * ν = 4 * R * ν ^ 2 / (1 + 3 * ν) := by field_simp
    rw [this]; positivity
  have hderiv : ∀ s, HasDerivAt f (3 * ν * A * exp (3 * ν * s) + 4 * ν / (1 + 3 * ν) * R * exp (-s)) s := by
    intro s
    have h1 : HasDerivAt (fun s => exp (3 * ν * s)) (exp (3 * ν * s) * (3 * ν)) s := by
      simpa using ((hasDerivAt_id s).const_mul (3 * ν)).exp
    have h2 : HasDerivAt (fun s => exp (-s)) (exp (-s) * (-1)) s := by
      simpa using (hasDerivAt_neg s).exp
    convert (h1.const_mul A).sub (h2.const_mul (4 * ν / (1 + 3 * ν) * R)) using 1
    ring
  rcases lt_or_gt_of_ne hν with hneg | hpos
  · -- ν < 0 : strictly decreasing
    apply StrictAnti.injective
    apply strictAnti_of_deriv_neg
    intro s
    rw [(hderiv s).deriv]
    have hkneg : 4 * ν / (1 + 3 * ν) * R ≤ 0 := by
      have : 4 * ν / (1 + 3 * ν) ≤ 0 := div_nonpos_of_nonpos_of_nonneg (by linarith) h13.le
      exact mul_nonpos_of_nonpos_of_nonneg this hR
    have : 3 * ν * A * exp (3 * ν * s) < 0 := by
      have := mul_pos hA (exp_pos (3 * ν * s))
      nlinarith
    nlinarith [exp_pos (-s), mul_nonpos_of_nonpos_of_nonneg hkneg (exp_pos (-s)).le]
  · -- ν > 0 : strictly increasing
    apply StrictMono.injective
    apply strictMono_of_deriv_pos
    intro s
    rw [(hderiv s).deriv]
    have hkpos : 0 ≤ 4 * ν / (1 + 3 * ν) * R := by
      have : 0 ≤ 4 * ν / (1 + 3 * ν) := div_nonneg (by linarith) h13.le
      exact mul_nonneg this hR
    have : 0 < 3 * ν * A * exp (3 * ν * s) := by
      have := mul_pos hA (exp_pos (3 * ν * s))
      nlinarith
    nlinarith [exp_pos (-s), mul_nonneg hkpos (exp_pos (-s)).le]

end RvmProofs
