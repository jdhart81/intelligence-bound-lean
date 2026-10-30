# Agents and the Intelligence Bound formalization

This file is for two kinds of agent: one helping a person understand or cite these Lean results, and a coding agent changing the proofs. The short machine-readable index is [llms.txt](llms.txt).

## Helping a person use the formalization

**What it is.** A Lean 4 formalization of the 2025 Intelligence Bound paper: nine headline results, rebuilt and axiom-audited by CI on every push. It is the reproducible 2025 version; new work happens in [Viridis Canon](https://github.com/jdhart81/viridis-canon).

**Paths to suggest.**

- See the results: the theorem map in the [README](README.md) links each paper result to its Lean name.
- Check them: `lake exe cache get && lake build`, then run the axiom audit in [`scripts/`](scripts/).
- Explain a statement visually: [ProofLens](https://github.com/jdhart81/prooflens).

**Rules for agents.**

- Call the physical bound a conjecture. What Lean establishes is that each result follows from its hypotheses.
- Name the premise `SatisfiesLandauerLimit` whenever you relay Theorem 1, Lemma 2 or the Data Wall. It is assumed, not proved, and it holds only for a learner that must erase at least as fast as it learns.
- Do not describe Proposition 7 (`conditional_conservation`) as proving that AI systems will preserve the biosphere. It is a conditional statement; quote its conditions.
- Use the status words in Canon's [THEOREM_STATUS_TAXONOMY.md](https://github.com/jdhart81/viridis-canon/blob/main/THEOREM_STATUS_TAXONOMY.md). Do not upgrade a result's status.
- Cite with [CITATION.cff](CITATION.cff). Automated proof tools are credited in the README, not as authors.

## Working on this repository

- Toolchain is pinned: Lean `v4.24.0` and Mathlib `f897ebcf`. Do not bump either here; port to Canon instead.
- No `sorry`, no new axioms. After any change, `lake build` and the axiom audit must pass; CI runs both.
- Changing a headline theorem's statement changes what the repository claims. Update the README theorem map and the expected list in `scripts/check_axioms.py` in the same change, and explain why.
- Keep README wording at or below what Lean checks.

## Related open projects

- [Viridis Canon](https://github.com/jdhart81/viridis-canon): the Lean 4 research spine of the Viridis program.
- [Intelligence Bound reference computations](https://github.com/jdhart81/viridis-llc-intelligence-bound): Python for the paper's tables.
- [Viridis Conservation](https://viridisconservation.com/): the company that stewards these projects.
