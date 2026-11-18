import Mathlib

/-!
# Closed-form background of the running vacuum (paper Sec. 2, derivations.md §1)

Units `8πG/3 = 1`, flat, variable `s = ln a`. Baryons `b = B e^{−3s}` and radiation `r = R e^{−4s}` are
separately conserved. The vacuum runs as `v = v₀ + ν (H² − H₀²)` and exchanges energy with CDM `c`:
`c' + 3c = −v'`. With the Friedmann equation `H² = c + b + r + v`, these imply the linear ODE

  `c' = −3(1−ν) c + 3ν b + 4ν r`                                        (`cdm_ode_of_model`)

whose solution is

  `c(s) = A e^{−3(1−ν)s} − B e^{−3s} − k R e^{−4s}`,  `k = 4ν/(1+3ν)`      (`closedForm_solves`)

and it is the only one with a given value of `A` (`ode_solution_unique`).
-/

open Real

namespace RvmProofs

/-- **From the model to the ODE** (pointwise, at one `s`). Given the derivatives of all densities, the
vacuum law, the Friedmann equation and the CDM exchange equation imply the linear ODE. -/
theorem cdm_ode_of_model {ν c b r c' b' r' v' H2' : ℝ}
    (hvlaw : v' = ν * H2')                 -- derivative of `v₀ + ν(H² − H₀²)`
    (hfried : H2' = c' + b' + r' + v')     -- derivative of `H² = c + b + r + v`
    (hb : b' = -3 * b) (hr : r' = -4 * r)  -- conserved baryons and radiation
    (hexch : c' + 3 * c = -v') :           -- vacuum → CDM energy exchange
    c' = -3 * (1 - ν) * c + 3 * ν * b + 4 * ν * r := by
  have h1 : (1 - ν) * v' = ν * (c' + b' + r') := by linear_combination hvlaw + ν * hfried
  linear_combination (1 - ν) * hexch - h1 - ν * hb - ν * hr

/-- The closed form. -/
noncomputable def closedForm (ν A B R s : ℝ) : ℝ :=
  A * exp (-3 * (1 - ν) * s) - B * exp (-3 * s) - 4 * ν / (1 + 3 * ν) * R * exp (-4 * s)

/-- **The closed form solves the ODE**, for every `s`, provided `1 + 3ν ≠ 0`. -/
theorem closedForm_solves (ν A B R s : ℝ) (h : 1 + 3 * ν ≠ 0) :
    HasDerivAt (closedForm ν A B R)
      (-3 * (1 - ν) * closedForm ν A B R s + 3 * ν * (B * exp (-3 * s)) + 4 * ν * (R * exp (-4 * s))) s := by
  have hE : ∀ κ : ℝ, HasDerivAt (fun t => exp (κ * t)) (exp (κ * s) * κ) s := fun κ => by
    simpa using ((hasDerivAt_id s).const_mul κ).exp
  have := ((hE (-3 * (1 - ν))).const_mul A).sub ((hE (-3)).const_mul B) |>.sub
    ((hE (-4)).const_mul (4 * ν / (1 + 3 * ν) * R))
  unfold closedForm
  convert this using 1
  field_simp
  ring

/-- **Uniqueness.** Any differentiable solution of the ODE equals the closed form with
`A = c(0) + B + kR`: so the closed form is the general solution. -/
theorem ode_solution_unique {ν B R : ℝ} {c : ℝ → ℝ} (h : 1 + 3 * ν ≠ 0)
    (hc : ∀ s, HasDerivAt c (-3 * (1 - ν) * c s + 3 * ν * (B * exp (-3 * s)) + 4 * ν * (R * exp (-4 * s))) s) :
    ∀ s, c s = closedForm ν (c 0 + B + 4 * ν / (1 + 3 * ν) * R) B R s := by
  set A := c 0 + B + 4 * ν / (1 + 3 * ν) * R
  -- g = (c − closedForm) e^{3(1−ν)s} has zero derivative
  set g := fun s => (c s - closedForm ν A B R s) * exp (3 * (1 - ν) * s) with hg
  have hgd : ∀ s, HasDerivAt g 0 s := by
    intro s
    have hd := (hc s).sub (closedForm_solves ν A B R s h)
    have he : HasDerivAt (fun t => exp (3 * (1 - ν) * t)) (exp (3 * (1 - ν) * s) * (3 * (1 - ν))) s := by
      simpa using ((hasDerivAt_id s).const_mul (3 * (1 - ν))).exp
    convert hd.mul he using 1
    simp only [Pi.sub_apply]
    ring
  have hconst : ∀ s, g s = g 0 := fun s =>
    is_const_of_deriv_eq_zero (fun t => (hgd t).differentiableAt) (fun t => (hgd t).deriv) s 0
  have hg0 : g 0 = 0 := by
    simp only [hg, closedForm, A, mul_zero, exp_zero, mul_one]
    ring
  intro s
  have h1 := hconst s
  rw [hg0] at h1
  simp only [hg] at h1
  rcases mul_eq_zero.mp h1 with h2 | h2
  · linarith
  · exact absurd h2 (exp_pos _).ne'

end RvmProofs
