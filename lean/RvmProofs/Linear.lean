import Mathlib
import RvmProofs.Crossing

/-!
# First-order apparent equation of state (paper Eq. `weff_linear`)

In the matter era the comoving matter density of the RVM is `m e^{3s} = A e^{3νs}` (closed form without the
radiation term), and an observer anchored at `s_A` uses `Mt = A e^{3ν s_A}`. By `weff_add_one`,
`1 + w = N(ν) / D(ν)` with `N(ν) = (A e^{3νs} − A e^{3ν s_A}) e^{−3s}` and `D(ν) = ρ̃_DE`, a quantity that
tends to `Ω_Λ e^{0}` (here: any `D` differentiable at `ν = 0` with `D 0 ≠ 0`).

`weff_first_order`: `d(1+w)/dν |_{ν=0} = 3 A e^{−3s} (s − s_A) / D 0`, i.e. to first order in ν

  `1 + w ≈ 3ν (Ω_m/Ω_Λ) (1+z)³ ln[(1+z_A)/(1+z)]`   (`s = −ln(1+z)`; `weff_first_order_redshift`).

`HasDerivAt` at `ν = 0` is exactly the statement that the error is `o(ν)`.
-/

open Real

namespace RvmProofs

theorem weff_first_order {A s sA : ℝ} {D : ℝ → ℝ} {D' : ℝ} (hD : HasDerivAt D D' 0) (hD0 : D 0 ≠ 0) :
    HasDerivAt (fun ν => (A * exp (3 * ν * s) - A * exp (3 * ν * sA)) * exp (-3 * s) / D ν)
      (3 * A * exp (-3 * s) * (s - sA) / D 0) 0 := by
  have hE : ∀ κ : ℝ, HasDerivAt (fun ν => exp (3 * ν * κ)) (exp (3 * 0 * κ) * (3 * κ)) 0 := fun κ => by
    have := ((hasDerivAt_id (0 : ℝ)).const_mul 3).mul_const κ |>.exp
    simpa [mul_comm, mul_left_comm, mul_assoc] using this
  have hN := (((hE s).const_mul A).sub ((hE sA).const_mul A)).mul_const (exp (-3 * s))
  convert hN.div hD hD0 using 1
  simp
  rw [pow_two]
  field_simp

/-- The same statement in the paper's variables: `s = −ln(1+z)`, `s_A = −ln(1+z_A)`, `A = Ω_m`,
`D 0 = Ω_Λ`. -/
theorem weff_first_order_redshift {Om OL z zA : ℝ} (hz : 0 < 1 + z) (hzA : 0 < 1 + zA) :
    3 * Om * exp (-3 * (-log (1 + z))) * (-log (1 + z) - -log (1 + zA)) / OL
      = 3 * (Om / OL) * (1 + z) ^ 3 * log ((1 + zA) / (1 + z)) := by
  have h1 : exp (-3 * (-log (1 + z))) = (1 + z) ^ 3 := by
    rw [show -3 * (-log (1 + z)) = (3 : ℕ) * log (1 + z) by push_cast; ring, exp_nat_mul, exp_log hz]
  rw [h1, log_div hzA.ne' hz.ne']
  ring

end RvmProofs
