import IntelligenceBound

/-!
# Non-vacuity witness for `intelligence_bound` and `data_wall`

Instantiates the hypotheses of `intelligence_bound` and `data_wall` simultaneously
on a concrete probability space (`Measure.dirac ()` on `Unit`) and constant
Boolean stochastic processes, confirming that neither theorem is vacuously true.
-/

open MeasureTheory ProbabilityTheory ENNReal Filter

namespace IntelligenceBound.NonVacuity

noncomputable def μ₀ : Measure Unit := Measure.dirac ()

noncomputable instance : IsProbabilityMeasure μ₀ := by
  unfold μ₀
  infer_instance

def constProc : Process Unit Bool := fun _ _ => true

lemma observationEntropy_constProc (t : NNReal) :
    observationEntropy μ₀ constProc t = 0 := by
  change shannonEntropy μ₀ (trajectory constProc t) = 0
  have h_ae : AEMeasurable (trajectory constProc t) μ₀ := by unfold trajectory constProc; exact aemeasurable_const
  rw [entropy_eq_zero_iff_indep_self μ₀ (trajectory constProc t) h_ae]
  unfold trajectory constProc; exact ProbabilityTheory.indepFun_const_left (μ := μ₀) (fun _ => true) _

lemma cumulativePredictiveInformation_constProc (t τ : NNReal) :
    cumulativePredictiveInformation μ₀ constProc constProc t τ = 0 :=
  mutualInformation_eq_zero_of_entropy_eq_zero μ₀ _ _ (observationEntropy_constProc t)

lemma intelligenceCreationRate_constProc (τ : NNReal) :
    intelligenceCreationRate μ₀ constProc constProc τ = 0 := by
  simp [intelligenceCreationRate, cumulativePredictiveInformation_constProc]

lemma observationBandwidth_constProc :
    observationBandwidth μ₀ constProc = 0 := by
  simp [observationBandwidth, observationEntropy_constProc]

lemma predictiveRichness_constProc (τ : NNReal) :
    predictiveRichness μ₀ constProc constProc τ = 0 := by
  simp [predictiveRichness, cumulativePredictiveInformation_constProc]

example :
    intelligenceCreationRate μ₀ constProc constProc 0 ≤
      min (predictiveRichness μ₀ constProc constProc 0 * observationBandwidth μ₀ constProc)
        (ENNReal.ofReal ((1 : ℝ) / ((1 : ℝ) * (1 : ℝ) * Real.log 2))) := by
  apply intelligence_bound μ₀ constProc constProc 0 1 1 1 zero_lt_one zero_lt_one
  · simp [SatisfiesLandauerLimit, intelligenceCreationRate_constProc]
  · simp [observationBandwidth_constProc]
  · simp [predictiveRichness_constProc]

example :
    intelligenceCreationRate μ₀ constProc constProc 0 ≤
      (0 : ENNReal) * observationBandwidth μ₀ constProc := by
  apply data_wall μ₀ constProc constProc 0 1 1 1 0 zero_lt_one zero_lt_one
  · simp [SatisfiesLandauerLimit, intelligenceCreationRate_constProc]
  · simp [observationBandwidth_constProc]
  · simp [predictiveRichness_constProc]
  · simp [predictiveRichness_constProc]
  · simp [IsDataLimited, criticalPower, predictiveRichness_constProc,
      observationBandwidth_constProc]

end IntelligenceBound.NonVacuity
