#!/usr/bin/env python3
"""Explore potential metaphors in the KG by finding concepts with shared properties."""
import json
from collections import defaultdict

kg = json.load(open("data/de_knowledge.json"))

# Build Indices
# forward: subject -> list of (relation, object)
# reverse: object -> list of (subject, relation)
forward = defaultdict(list)
reverse = defaultdict(list)

for t in kg:
    s, r, o = t["subject"], t["relation"], t["object"]
    forward[s].append((r, o))
    reverse[o].append((s, r))

def find_metaphors(target: str):
    print(f"\nSearching metaphors for: {target.upper()}")
    
    # 1. Get target properties
    properties = forward[target]
    if not properties:
        print("  No properties found.")
        return

    # 2. Find candidates sharing these properties
    candidates = defaultdict(int) 
    shared_props = defaultdict(list)

    for rel, obj in properties:
        # Who else relates to this object?
        # We generally look for subjects that CAUSES, HAS, or ISA this object
        others = reverse[obj]
        for other_subj, other_rel in others:
            if other_subj == target: continue
            
            # Simple heuristic: meaningful overlap
            # If Target CAUSES Pain, and Knife CAUSES Pain -> Metaphor?
            candidates[other_subj] += 1
            shared_props[other_subj].append(f"{obj} ({rel}/{other_rel})")

    # 3. Sort by overlap strength
    sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])

    # 4. Display top results
    for cand, score in sorted_candidates[:5]:
        props = ", ".join(shared_props[cand])
        print(f"  Candidate: {cand} (Score: {score})")
        print(f"    Shared: {props}")
        
        # Check for divergent properties (Target has X, Candidate has Y)
        # to construct complex metaphors: "Love is Fire; it warms (shared), but it also burns (candidate prop)"
        cand_props = forward[cand]
        unique_cand_props = [f"{r} {o}" for r, o in cand_props if o not in [p[0].split()[0] for p in shared_props[cand]]] # simplified
        if unique_cand_props:
            print(f"    Possible extension: {cand} also {unique_cand_props[0]}")

# Test with abstract concepts
targets = ["Liebe", "Wut", "Freiheit", "Zeit", "Gedanken", "Tod", "Stille"]

for t in targets:
    find_metaphors(t)
