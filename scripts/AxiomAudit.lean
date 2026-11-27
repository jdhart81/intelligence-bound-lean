import IntelligenceBound

/-!
# Axiom audit

Every theorem and lemma in `IntelligenceBound` must depend only on Lean's
standard axioms (`propext`, `Classical.choice`, `Quot.sound`). CI runs this
file after `lake build` and fails on anything else, including `sorryAx`. Run it
locally with:

    lake env lean scripts/AxiomAudit.lean | python3 scripts/check_axioms.py /dev/stdin
-/

-- Headline results (9)
#print axioms intelligence_bound
#print axioms data_bound_lemma_conditional
#print axioms thermodynamic_bound_lemma
#print axioms finite_memory_dissipation
#print axioms learning_dissipation_link
#print axioms conditional_conservation
#print axioms prediction1_rho_dependence
#print axioms phase_transition_regimes
#print axioms data_wall

-- Supporting lemmas (13)
#print axioms indep_self_implies_indep_any
#print axioms mutualInformation_eq_zero_iff_indep
#print axioms entropy_eq_zero_iff_indep_self
#print axioms entropy_eq_zero_implies_mutualInformation_eq_zero
#print axioms mutualInformation_eq_zero_of_entropy_eq_zero
#print axioms eventually_finite_entropy
#print axioms ennreal_div_eq_div_mul_div
#print axioms decomposition_lemma_pointwise
#print axioms intelligence_rate_eq_product_eventually
#print axioms limsup_decomposition
#print axioms thermodynamic_factor_pos_finite
#print axioms phase_transition_algebra
#print axioms conditional_conservation_core