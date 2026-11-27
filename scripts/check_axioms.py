#!/usr/bin/env python3
"""Fail unless every theorem and lemma in IntelligenceBound is audited exactly once
and uses only Lean's standard axioms.

usage: lake env lean scripts/AxiomAudit.lean > axioms.txt && python3 scripts/check_axioms.py axioms.txt
"""
from pathlib import Path
import re
import sys

ALLOWED = {"propext", "Classical.choice", "Quot.sound"}

DECL_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*(?:private\s+|protected\s+)?(?:theorem|lemma)\s+([^\s(:{\[]+)",
    re.MULTILINE,
)

def collect_expected() -> set[str]:
    root = Path(__file__).resolve().parent.parent / "IntelligenceBound"
    expected: set[str] = set()
    for lean_file in sorted(root.rglob("*.lean")):
        content = lean_file.read_text(encoding="utf-8")
        expected.update(DECL_RE.findall(content))
    return expected

EXPECTED = collect_expected()

text = open(sys.argv[1], encoding="utf-8").read()
results = re.findall(r"'([^']+)' (depends on axioms: \[([^\]]*)\]|does not depend on any axioms)", text)
names = [name for name, _, _ in results]
problems = []
missing = sorted(EXPECTED - set(names))
unexpected = sorted(set(names) - EXPECTED)
duplicated = sorted({name for name in names if names.count(name) > 1})
if missing:
    problems.append("not audited: " + ", ".join(missing))
if unexpected:
    problems.append("not in the expected list: " + ", ".join(unexpected))
if duplicated:
    problems.append("audited more than once: " + ", ".join(duplicated))
for name, _, axioms in results:
    for axiom in filter(None, (a.strip() for a in axioms.replace("\n", " ").split(","))):
        if axiom not in ALLOWED:
            problems.append(f"{name} uses non-standard axiom {axiom}")
if problems:
    sys.exit("axiom audit: FAIL\n" + "\n".join(problems))
print(f"axiom audit: PASS ({len(EXPECTED)} theorems, standard axioms only)")