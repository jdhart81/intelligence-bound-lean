#!/usr/bin/env python3
"""Fail unless every audited theorem uses only Lean's standard axioms.

usage: lake env lean scripts/AxiomAudit.lean > axioms.txt && python3 scripts/check_axioms.py axioms.txt
"""
import re
import sys

ALLOWED = {"propext", "Classical.choice", "Quot.sound"}
EXPECTED = 9

text = open(sys.argv[1], encoding="utf-8").read()
results = re.findall(r"'([^']+)' (depends on axioms: \[([^\]]*)\]|does not depend on any axioms)", text)
if len(results) != EXPECTED:
    sys.exit(f"axiom audit: expected {EXPECTED} results, found {len(results)}")
bad = []
for name, _, axioms in results:
    for axiom in filter(None, (a.strip() for a in axioms.replace("\n", " ").split(","))):
        if axiom not in ALLOWED:
            bad.append(f"{name}: {axiom}")
if bad:
    sys.exit("axiom audit: non-standard axioms\n" + "\n".join(bad))
print(f"axiom audit: PASS ({EXPECTED} theorems, standard axioms only)")
