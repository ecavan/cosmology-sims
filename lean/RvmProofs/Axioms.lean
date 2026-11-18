import RvmProofs.Crossing
import RvmProofs.ClosedForm
import RvmProofs.Linear

/-! Audit: every main theorem should depend only on Lean's standard axioms
(`propext`, `Classical.choice`, `Quot.sound`) — no `sorry`, no custom axioms. -/

#print axioms RvmProofs.weff_add_one
#print axioms RvmProofs.weff_eq_neg_one_iff
#print axioms RvmProofs.rvm_comoving_matter_injective
#print axioms RvmProofs.cdm_ode_of_model
#print axioms RvmProofs.closedForm_solves
#print axioms RvmProofs.ode_solution_unique
#print axioms RvmProofs.weff_first_order
#print axioms RvmProofs.weff_first_order_redshift
