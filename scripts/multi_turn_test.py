#!/usr/bin/env python3
"""Multi-turn variety test: 5+ sequential queries, check structural repetition."""
from cce.engine import CognitiveCrystallizationEngine as E

e = E()

print("=" * 60)
print("MULTI-TURN VARIETY TEST")
print("=" * 60)

queries = [
    "Was ist Liebe?",
    "Warum Krieg?",
    "Wie entsteht Mut?",
    "Kennst du Angst?",
    "Was bedeutet Freiheit?",
    "Warum braucht man Hoffnung?",
    "Kennst du Stille?",
]

# Track structural patterns
structures = []
openings = []

for i, q in enumerate(queries, 1):
    print(f"\n--- Turn {i}: {q} ---")
    out = e.generate_from_text(q)
    print(f"  {out}")

    # Analyze structure
    sentences = [s.strip() for s in out.replace("—", ".").replace(".", ".").split(".") if s.strip()]
    opening = out.split()[0] if out else ""
    openings.append(opening)

    # Check for connectors used
    connectors = []
    for conn in ["Denn", "Und", "Aber", "Obwohl", "Da", "Auch wenn", "Weil", "Und so",
                  "Und gerade", "Und genau", "Und dennoch"]:
        if conn in out:
            connectors.append(conn)
    structures.append(connectors)

print("\n" + "=" * 60)
print("PATTERN ANALYSIS")
print("=" * 60)

print(f"\nOpenings: {openings}")
unique_openings = len(set(openings))
print(f"  Unique: {unique_openings}/{len(openings)}")

print(f"\nConnector patterns:")
for i, (q, conns) in enumerate(zip(queries, structures), 1):
    print(f"  Turn {i}: {conns}")

# Check for exact connector pattern repeats
pattern_strs = [str(c) for c in structures]
unique_patterns = len(set(pattern_strs))
print(f"\nUnique connector patterns: {unique_patterns}/{len(structures)}")

if unique_patterns == len(structures):
    print("✓ No structural repetition detected")
elif unique_patterns >= len(structures) * 0.7:
    print("~ Acceptable structural variety")
else:
    print("⚠ Significant structural repetition detected")
