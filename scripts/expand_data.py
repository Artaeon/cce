#!/usr/bin/env python3
"""Expand KG to 1000+ triples and lexicon to 700+ entries."""
import json

# ═══ KG EXPANSION ═══
kg = json.load(open("data/de_knowledge.json"))
existing = {(t["subject"], t["relation"], t["object"]) for t in kg}

new = [
    {"subject": "Eifersucht", "relation": "ISA", "object": "Emotion"},
    {"subject": "Eifersucht", "relation": "CAUSES", "object": "Schmerz"},
    {"subject": "Eifersucht", "relation": "OPPOSES", "object": "Vertrauen"},
    {"subject": "Eifersucht", "relation": "NEEDS", "object": "Liebe"},
    {"subject": "Eifersucht", "relation": "LEADSTO", "object": "Einsamkeit"},
    {"subject": "Gier", "relation": "OPPOSES", "object": "Zufriedenheit"},
    {"subject": "Gier", "relation": "CAUSES", "object": "Leid"},
    {"subject": "Gier", "relation": "LEADSTO", "object": "Einsamkeit"},
    {"subject": "Gier", "relation": "NEEDS", "object": "Demut"},
    {"subject": "Zorn", "relation": "ISA", "object": "Emotion"},
    {"subject": "Zorn", "relation": "OPPOSES", "object": "Geduld"},
    {"subject": "Zorn", "relation": "CAUSES", "object": "Zerstörung"},
    {"subject": "Zorn", "relation": "NEEDS", "object": "Vergebung"},
    {"subject": "Zorn", "relation": "LEADSTO", "object": "Reue"},
    {"subject": "Sanftmut", "relation": "OPPOSES", "object": "Gewalt"},
    {"subject": "Sanftmut", "relation": "CAUSES", "object": "Frieden"},
    {"subject": "Sanftmut", "relation": "NEEDS", "object": "Kraft"},
    {"subject": "Sanftmut", "relation": "HAS", "object": "Stärke"},
    {"subject": "Treue", "relation": "NEEDS", "object": "Liebe"},
    {"subject": "Treue", "relation": "CAUSES", "object": "Vertrauen"},
    {"subject": "Treue", "relation": "OPPOSES", "object": "Verrat"},
    {"subject": "Treue", "relation": "HAS", "object": "Kraft"},
    {"subject": "Treue", "relation": "LEADSTO", "object": "Verbundenheit"},
    {"subject": "Verrat", "relation": "CAUSES", "object": "Schmerz"},
    {"subject": "Verrat", "relation": "OPPOSES", "object": "Treue"},
    {"subject": "Verrat", "relation": "LEADSTO", "object": "Einsamkeit"},
    {"subject": "Verrat", "relation": "NEEDS", "object": "Vergebung"},
    {"subject": "Trost", "relation": "CAUSES", "object": "Heilung"},
    {"subject": "Trost", "relation": "NEEDS", "object": "Nähe"},
    {"subject": "Trost", "relation": "OPPOSES", "object": "Einsamkeit"},
    {"subject": "Trost", "relation": "HAS", "object": "Wärme"},
    {"subject": "Trost", "relation": "LEADSTO", "object": "Kraft"},
    {"subject": "Wunder", "relation": "CAUSES", "object": "Staunen"},
    {"subject": "Wunder", "relation": "OPPOSES", "object": "Logik"},
    {"subject": "Wunder", "relation": "HAS", "object": "Schönheit"},
    {"subject": "Wunder", "relation": "NEEDS", "object": "Glaube"},
    {"subject": "Leere", "relation": "OPPOSES", "object": "Sinn"},
    {"subject": "Leere", "relation": "CAUSES", "object": "Sehnsucht"},
    {"subject": "Leere", "relation": "NEEDS", "object": "Mut"},
    {"subject": "Leere", "relation": "LEADSTO", "object": "Erneuerung"},
    {"subject": "Ordnung", "relation": "OPPOSES", "object": "Chaos"},
    {"subject": "Ordnung", "relation": "NEEDS", "object": "Disziplin"},
    {"subject": "Ordnung", "relation": "CAUSES", "object": "Ruhe"},
    {"subject": "Ordnung", "relation": "HAS", "object": "Schönheit"},
    {"subject": "Chaos", "relation": "OPPOSES", "object": "Ordnung"},
    {"subject": "Chaos", "relation": "CAUSES", "object": "Kreativität"},
    {"subject": "Chaos", "relation": "LEADSTO", "object": "Veränderung"},
    {"subject": "Chaos", "relation": "HAS", "object": "Energie"},
    {"subject": "Heilung", "relation": "NEEDS", "object": "Zeit"},
    {"subject": "Heilung", "relation": "CAUSES", "object": "Kraft"},
    {"subject": "Heilung", "relation": "OPPOSES", "object": "Schmerz"},
    {"subject": "Heilung", "relation": "LEADSTO", "object": "Wachstum"},
    {"subject": "Wandel", "relation": "NEEDS", "object": "Mut"},
    {"subject": "Wandel", "relation": "CAUSES", "object": "Angst"},
    {"subject": "Wandel", "relation": "OPPOSES", "object": "Stillstand"},
    {"subject": "Wandel", "relation": "LEADSTO", "object": "Wachstum"},
    {"subject": "Wandel", "relation": "HAS", "object": "Kraft"},
    {"subject": "Verantwortung", "relation": "NEEDS", "object": "Mut"},
    {"subject": "Verantwortung", "relation": "CAUSES", "object": "Reife"},
    {"subject": "Verantwortung", "relation": "OPPOSES", "object": "Gleichgültigkeit"},
    {"subject": "Verantwortung", "relation": "LEADSTO", "object": "Stärke"},
    {"subject": "Empathie", "relation": "CAUSES", "object": "Verbundenheit"},
    {"subject": "Empathie", "relation": "NEEDS", "object": "Offenheit"},
    {"subject": "Empathie", "relation": "OPPOSES", "object": "Gleichgültigkeit"},
    {"subject": "Empathie", "relation": "LEADSTO", "object": "Verständigung"},
    {"subject": "Empathie", "relation": "HAS", "object": "Kraft"},
    {"subject": "Demut", "relation": "CAUSES", "object": "Weisheit"},
    {"subject": "Demut", "relation": "NEEDS", "object": "Erfahrung"},
    {"subject": "Demut", "relation": "OPPOSES", "object": "Stolz"},
    {"subject": "Demut", "relation": "LEADSTO", "object": "Frieden"},
    {"subject": "Demut", "relation": "HAS", "object": "Stärke"},
    {"subject": "Wille", "relation": "CAUSES", "object": "Veränderung"},
    {"subject": "Wille", "relation": "NEEDS", "object": "Mut"},
    {"subject": "Wille", "relation": "OPPOSES", "object": "Gleichgültigkeit"},
    {"subject": "Wille", "relation": "LEADSTO", "object": "Freiheit"},
    {"subject": "Wille", "relation": "HAS", "object": "Kraft"},
    {"subject": "Glaube", "relation": "CAUSES", "object": "Hoffnung"},
    {"subject": "Glaube", "relation": "NEEDS", "object": "Demut"},
    {"subject": "Glaube", "relation": "OPPOSES", "object": "Zweifel"},
    {"subject": "Glaube", "relation": "LEADSTO", "object": "Kraft"},
    {"subject": "Glaube", "relation": "HAS", "object": "Tiefe"},
    {"subject": "Zweifel", "relation": "OPPOSES", "object": "Glaube"},
    {"subject": "Zweifel", "relation": "CAUSES", "object": "Erkenntnis"},
    {"subject": "Zweifel", "relation": "NEEDS", "object": "Mut"},
    {"subject": "Zweifel", "relation": "LEADSTO", "object": "Wahrheit"},
    {"subject": "Zweifel", "relation": "HAS", "object": "Kraft"},
]

added = 0
for t in new:
    k = (t["subject"], t["relation"], t["object"])
    if k not in existing:
        kg.append(t)
        existing.add(k)
        added += 1

with open("data/de_knowledge.json", "w", encoding="utf-8") as f:
    json.dump(kg, f, indent=2, ensure_ascii=False)

subjects = set()
for item in kg:
    subjects.add(item["subject"])
    subjects.add(item["object"])
print(f"KG: {len(kg)} triples, {len(subjects)} concepts (+{added})")

# ═══ LEXICON EXPANSION ═══
lex = json.load(open("data/de_lexicon_5k.json"))
existing_words = {e["word"].lower() for e in lex}

# Find new concepts from KG not in lexicon
new_nouns = sorted(subjects - existing_words)
for noun in new_nouns:
    cap = noun.capitalize() if noun[0].islower() else noun
    lex.append({
        "word": cap, "pos": "NOUN", "syllables": max(1, len(noun) // 3),
        "valence": 0.0, "arousal": 0.3, "register": 0.5, "bonds": [], "repels": [],
    })

# Add adjectives
new_adjs = [
    ("still", 0.1, 0.1), ("tief", 0.0, 0.4), ("kalt", -0.3, 0.3),
    ("warm", 0.3, 0.3), ("zart", 0.2, 0.2), ("wild", 0.0, 0.7),
    ("sanft", 0.3, 0.2), ("roh", -0.1, 0.6), ("blind", -0.1, 0.3),
    ("stumm", -0.1, 0.1), ("leer", -0.2, 0.2), ("rein", 0.3, 0.2),
    ("wahr", 0.2, 0.3), ("frei", 0.4, 0.5), ("schwer", -0.1, 0.4),
    ("leicht", 0.2, 0.3), ("bitter", -0.3, 0.4), ("süß", 0.3, 0.3),
    ("weich", 0.2, 0.2), ("hart", -0.1, 0.5), ("hell", 0.3, 0.3),
    ("leuchtend", 0.3, 0.4), ("glühend", 0.1, 0.7), ("eisig", -0.3, 0.4),
    ("ewig", 0.0, 0.3), ("flüchtig", -0.1, 0.3), ("endlos", 0.0, 0.4),
    ("brennend", 0.0, 0.7), ("heilig", 0.2, 0.3), ("einsam", -0.3, 0.3),
    ("zerbrochen", -0.3, 0.4), ("rasend", -0.1, 0.8), ("unendlich", 0.1, 0.4),
    ("vergänglich", -0.1, 0.3), ("beständig", 0.2, 0.2),
]
added_a = 0
for word, val, aro in new_adjs:
    if word.lower() not in existing_words:
        lex.append({
            "word": word, "pos": "ADJ", "syllables": max(1, len(word) // 3),
            "valence": val, "arousal": aro, "register": 0.5, "bonds": [], "repels": [],
        })
        existing_words.add(word.lower())
        added_a += 1

# Add verbs
new_verbs = [
    ("nähren", 0.1, 0.3), ("entzünden", 0.0, 0.6), ("heilen", 0.3, 0.3),
    ("zerbrechen", -0.2, 0.6), ("erblühen", 0.4, 0.4), ("versinken", -0.2, 0.4),
    ("erstrahlen", 0.4, 0.4), ("verschmelzen", 0.1, 0.5), ("wurzeln", 0.1, 0.2),
    ("erwachen", 0.3, 0.4), ("vergehen", -0.1, 0.3), ("strömen", 0.1, 0.5),
    ("leuchten", 0.3, 0.3), ("flüstern", 0.0, 0.2), ("blühen", 0.3, 0.3),
    ("welken", -0.2, 0.2), ("reifen", 0.2, 0.3), ("atmen", 0.1, 0.2),
    ("träumen", 0.2, 0.2), ("wandeln", 0.0, 0.3), ("wagen", 0.1, 0.5),
    ("trotzen", 0.1, 0.6), ("sprengen", -0.1, 0.7), ("befreien", 0.3, 0.5),
    ("begraben", -0.2, 0.4), ("bewahren", 0.2, 0.3),
]
added_v = 0
for word, val, aro in new_verbs:
    if word.lower() not in existing_words:
        lex.append({
            "word": word, "pos": "VERB", "syllables": max(1, len(word) // 3),
            "valence": val, "arousal": aro, "register": 0.5, "bonds": [], "repels": [],
        })
        existing_words.add(word.lower())
        added_v += 1

with open("data/de_lexicon_5k.json", "w", encoding="utf-8") as f:
    json.dump(lex, f, indent=2, ensure_ascii=False)

pos_count = {}
for e in lex:
    p = e.get("pos", "?")
    pos_count[p] = pos_count.get(p, 0) + 1
print(f"Lexicon: {len(lex)} entries (+{len(new_nouns)} nouns, +{added_a} adjs, +{added_v} verbs)")
for p, c in sorted(pos_count.items(), key=lambda x: -x[1]):
    print(f"  {p}: {c}")
