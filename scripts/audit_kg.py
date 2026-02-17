#!/usr/bin/env python3
"""Audit KG for directional issues."""
import json

kg = json.load(open("data/de_knowledge.json"))

print("=== EWIGKEIT / AUGENBLICK ===")
for t in kg:
    s, r, o = t["subject"], t["relation"], t["object"]
    if s in ("Ewigkeit", "Augenblick") or o in ("Ewigkeit", "Augenblick"):
        print(f"  {s} --{r}--> {o}")

print("\n=== SPOT CHECK: CAUSES direction ===")
# Flag CAUSES relations that seem reversed
questionable_causes = []
for t in kg:
    s, r, o = t["subject"], t["relation"], t["object"]
    if r == "CAUSES":
        # Abstract concepts that rarely "cause" concrete things
        if o in ("Leben", "Mensch", "Natur"):
            questionable_causes.append(f"  ⚠ {s} CAUSES {o}")
        # Death doesn't cause things much
        if s in ("Tod",) and o not in ("Trauer", "Angst", "Schmerz", "Stille"):
            questionable_causes.append(f"  ⚠ {s} CAUSES {o}")
for q in questionable_causes:
    print(q)

print("\n=== OPPOSES symmetry check ===")
opposes = [(t["subject"], t["object"]) for t in kg if t["relation"] == "OPPOSES"]
opp_set = set(opposes)
for a, b in opposes:
    if (b, a) not in opp_set:
        print(f"  One-way: {a} OPPOSES {b} (no reverse)")

print("\n=== NATUR domain ===")
natur = ["Wald", "Meer", "Sturm", "Berg", "Fluss", "Sonne", "Nacht", "Wind", "Regen", "Erde"]
for c in natur:
    rels = [t for t in kg if t["subject"] == c]
    for t in rels:
        print(f"  {c} --{t['relation']}--> {t['object']}")

print("\n=== Counts by relation ===")
counts = {}
for t in kg:
    r = t["relation"]
    counts[r] = counts.get(r, 0) + 1
for r, c in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {r}: {c}")
