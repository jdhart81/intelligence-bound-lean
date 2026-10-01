import IntelligenceBound

/-!
# Axiom audit

Every headline result must depend only on Lean's standard axioms (`propext`,
`Classical.choice`, `Quot.sound`). CI runs this file after `lake build` and fails
on anything else, including `sorryAx`. Run it locally with:

    lake env lean scripts/AxiomAudit.lean
-/

#print axioms intelligence_bound
#print axioms data_bound_lemma_conditional
#print axioms thermodynamic_bound_lemma
#print axioms finite_memory_dissipation
#print axioms learning_dissipation_link
#print axioms conditional_conservation
#print axioms prediction1_rho_dependence
#print axioms phase_transition_regimes
#print axioms data_wall
