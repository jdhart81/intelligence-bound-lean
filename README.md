# The Intelligence Bound — Lean 4 Formalization of the Conditional Mathematics

[![Lean proofs](https://github.com/jdhart81/intelligence-bound-lean/actions/workflows/lean.yml/badge.svg)](https://github.com/jdhart81/intelligence-bound-lean/actions/workflows/lean.yml)

Lean 4 formalization of the mathematical results in
**"The Intelligence Bound: Thermodynamic Limits on Learning Rate and Implications
for Biosphere Information"** (Hart 2025).

> **Status: the physical Intelligence Bound is a conjecture. Lean checks the
> mathematics from stated premises.**
>
> - **Machine-checked** (Lean 4.24.0, Mathlib `f897ebcf`, axioms limited to
>   `propext`, `Classical.choice`, `Quot.sound`, no `sorry`): the nine results in the
>   table below, as implications from their hypotheses. The one substantive
>   information-theoretic step is Lemma 1, İ ≤ ρ·B, for the defined limsup
>   quantities.
> - **Assumed (named premise):** the thermodynamic arm is the hypothesis
>   `SatisfiesLandauerLimit` (P ≥ İ · k_B T ln 2). It is passed into Lemma 2,
>   Theorem 1 and the Data Wall as an argument; it is not derived. Lemma 2 is that
>   premise divided by k_B T ln 2.
> - **Not established:** that any physical learner satisfies the premise.
>   Landauer's principle prices erasing a bit, not acquiring one (Bennett; Wolpert),
>   so the thermodynamic arm applies only to a learner that must erase at least as
>   fast as it learns. The bound İ ≤ min(ρB, P/(k_B T ln 2)) as a law of physics is
>   **the Intelligence Bound conjecture**.
> - **Pipeline:** this repository is rebuilt and axiom-audited by its own CI. It is
>   not re-checked by the current Viridis Comparator pipeline and has no comparator
>   certificate. Status labels follow the Viridis Canon
>   [`THEOREM_STATUS_TAXONOMY.md`](https://github.com/jdhart81/viridis-canon/blob/main/THEOREM_STATUS_TAXONOMY.md)
>   and [`CLAIMS_MATRIX.md`](https://github.com/jdhart81/viridis-canon/blob/main/CLAIMS_MATRIX.md).

> Active development of the Intelligence Bound formalization continues in [Viridis Canon](https://github.com/jdhart81/viridis-canon), the Lean 4 research spine of the Viridis program. This repository keeps the original 2025 formalization reproducible.

## Building

```bash
# Install elan (Lean version manager) if you haven't already
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Clone and build
git clone https://github.com/jdhart81/intelligence-bound-lean.git
cd intelligence-bound-lean
lake exe cache get   # download prebuilt Mathlib .olean files
lake build           # compile the project
```

Requires Lean 4.24.0 and Mathlib at commit `f897ebcf`.

## Verify it yourself

Every push to `main` and every pull request rebuilds every proof in this repository against the pinned Mathlib and audits the axioms of the nine headline results (CI badge above). To run the same audit locally after `lake build`:

```bash
lake env lean scripts/AxiomAudit.lean   # each result should list only propext, Classical.choice, Quot.sound
```

## Theorem Map

Read the "What Lean checks" column, not the name. "Elementary" marks results that
follow by unfolding definitions or one line of order algebra.

| Paper result | Lean name | What Lean checks | Load-bearing assumptions | Status |
|---|---|---|---|---|
| **Theorem 1** | `intelligence_bound` | İ(τ) ≤ min(ρB, P/(k_B T ln 2)) | `SatisfiesLandauerLimit` (assumed premise), B < ∞, ρ ≤ 1 | Conditional |
| **Lemma 1** | `data_bound_lemma_conditional` | İ ≤ ρ · B for the defined limsup rates | probability measure, B < ∞, ρ < ∞ | Derived (mathematics of the defined quantities) |
| **Lemma 2** | `thermodynamic_bound_lemma` | İ ≤ P/(k_B T ln 2) | `SatisfiesLandauerLimit`, which is this inequality multiplied out | Definition-expansion (elementary) |
| **Lemma 3** | `finite_memory_dissipation` | P ≥ İ · k_B T ln 2 | `P ≥ erasureRate · k_B T ln 2` and `erasureRate ≥ İ`, both assumed; the `BoundedMemory` hypothesis is present but unused | Definition-expansion (elementary; transitivity of ≤) |
| **Proposition 2** | `learning_dissipation_link` | `SatisfiesLandauerLimit` holds | the same two assumed inequalities as Lemma 3; `BoundedMemory` unused | Definition-expansion (elementary) |
| **Proposition 7** | `conditional_conservation` | for every large horizon T, r_P·Φ₀·T > r_E·Φ₀·(1 − e^(−λT))/λ | positive rates with r_E > r_P; the two utility formulas are model choices | Conditional (a linear function eventually exceeds a bounded one; no agents or rationality are modeled) |
| **Prediction 1** | `prediction1_rho_dependence` | if P ≥ ρBK then min(ρB, P/K) = ρB, with K = k_B T ln 2 | none beyond positivity | Definition-expansion (elementary) |
| **Prediction 2** | `phase_transition_regimes` | min(ρB, P/K) switches branch at P* = ρBK | none beyond positivity | Definition-expansion (elementary) |
| **Data Wall** | `data_wall` | ρ ≤ ρ_max ⟹ İ ≤ ρ_max · B | as Theorem 1; only the data arm is used, and the data-limited hypothesis is unused | Conditional (corollary of Lemma 1) |

Predictions 1 and 2 are statements about the algebra of the bound's formula. Testing
them physically needs measured ρ, B, P and T, which this repository does not supply.
The Viridis Canon replaced the unused bounded-memory hypothesis with a version in
which bounded memory is load-bearing (`BoundedMemoryDissipation`); this repository
keeps the original 2025 statements unchanged so the record stays reproducible.

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

These are stated as predicates (not theorems). No theorem here assumes or proves
them, and none has been tested against data:

| Paper Result | Lean Name |
|---|---|
| Proposition 5 | `BiosphereRichnessHypothesis` |
| Prediction 3 | `BiosphereIntegrityHypothesis` |
| Gaia-Intelligence | `GaiaIntelligenceProposition` |

## Architecture

The proofs rely on Mathlib's:
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
  note   = {The physical bound is a conjecture. Lean 4 formalization of
            the conditional mathematics:
            https://github.com/jdhart81/intelligence-bound-lean}
}
```
