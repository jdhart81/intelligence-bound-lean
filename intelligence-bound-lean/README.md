# The Intelligence Bound — Formal Verification in Lean 4

Formal verification of the mathematical results in
**"The Intelligence Bound: Thermodynamic Limits on Learning Rate and Implications
for Biosphere Information"** (Hart 2025).

## Building

```bash
# Install elan (Lean version manager) if you haven't already
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Clone and build
git clone https://github.com/viridis-llc/intelligence-bound-lean.git
cd intelligence-bound-lean
lake exe cache get   # download prebuilt Mathlib .olean files
lake build           # compile the project
```

Requires Lean 4.24.0 and Mathlib at commit `f897ebcf`.

## Theorem Map

| Paper Result | Lean Name | Description |
|---|---|---|
| **Theorem 1** | `intelligence_bound` | İ(τ) ≤ min(ρB, P/(k_B T ln 2)) |
| **Lemma 1** | `data_bound_lemma_conditional` | İ ≤ ρ · B (data-processing bound) |
| **Lemma 2** | `thermodynamic_bound_lemma` | İ ≤ P/(k_B T ln 2) (Landauer bound) |
| **Lemma 3** | `finite_memory_dissipation` | Bounded memory ⟹ power dissipation ≥ İ · k_B T ln 2 |
| **Proposition 2** | `learning_dissipation_link` | Learning–dissipation link (Landauer limit predicate) |
| **Proposition 7** | `conditional_conservation` | Long-horizon rational agents preserve biosphere |
| **Prediction 1** | `prediction1_rho_dependence` | Data-limited regime: bound = ρ · B |
| **Prediction 2** | `phase_transition_regimes` | Phase transition at critical power P* = ρBK |
| **Data Wall** | `data_wall` | Low ρ caps İ at ρ_max · B regardless of power |

## Key Definitions

| Paper Concept | Lean Name | Type |
|---|---|---|
| Mutual information I(X; Y) | `mutualInformation` | `ENNReal` via `klDiv` |
| Shannon entropy H(X) | `shannonEntropy` | `ENNReal` |
| Intelligence creation rate İ(τ) | `intelligenceCreationRate` | `ENNReal` via `limsup` |
| Observation bandwidth B | `observationBandwidth` | `ENNReal` via `limsup` |
| Predictive richness ρ(τ) | `predictiveRichness` | `ENNReal` via `limsup` |
| Landauer limit predicate | `SatisfiesLandauerLimit` | `Prop` |
| Bounded memory | `BoundedMemory` | `Prop` |
| Thermodynamic factor K | `thermodynamicFactor` | `ENNReal` |
| Critical power P* | `criticalPower` | `ENNReal` |
| Data-limited regime | `IsDataLimited` | `Prop` |
| Power-limited regime | `IsPowerLimited` | `Prop` |

## Empirical Hypotheses

These are stated as predicates (not theorems) since they require experimental
validation:

| Paper Result | Lean Name |
|---|---|
| Proposition 5 | `BiosphereRichnessHypothesis` |
| Prediction 3 | `BiosphereIntegrityHypothesis` |
| Gaia-Intelligence | `GaiaIntelligenceProposition` |

## Architecture

The proof relies on Mathlib's:
- `InformationTheory.klDiv` — KL divergence and `klDiv_eq_zero_iff` (Gibbs' inequality)
- `ProbabilityTheory.IndepFun` — independence and `indepFun_iff_map_prod_eq_prod_map_map`
- `ENNReal` — extended non-negative reals for handling ∞
- `Filter.limsup` — asymptotic definitions of rates

## Authors

- **Justin Hart** (Viridis LLC) — paper and formalization design
- **Aristotle** (Harmonic) — automated proof generation ([aristotle.harmonic.fun](https://aristotle.harmonic.fun))

## License

Apache 2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{hart2025intelligencebound,
  author = {Hart, Justin},
  title  = {The Intelligence Bound: Thermodynamic Limits on Learning Rate
            and Implications for Biosphere Information},
  year   = {2025},
  note   = {Formal verification available at
            https://github.com/viridis-llc/intelligence-bound-lean}
}
```
