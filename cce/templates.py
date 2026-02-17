"""
German Sentence Templates
===========================

Rich template library for generating coherent German text.
Each template category maps to a discourse intent and specifies
slot requirements (S=subject, V=verb, O=object, A/B=concepts, Pred=predicate).

Templates use {slot} syntax. Slots:
  {S}    — Subject (noun, capitalized)
  {V}    — Verb (conjugated 3s)
  {O}    — Object (noun, capitalized)
  {A},{B} — Concept words (for contrast/parallel)
  {Pred} — Predicate adjective
  {Art}  — Article (der/die/das)
  {Adv}  — Adverb or adjective modifier
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Template:
    """A sentence template with metadata."""
    pattern: str
    requires: list[str]        # Required slots: S, V, O, A, B, Pred, Adv
    mood: str = "neutral"      # neutral, reflective, assertive, emotional, contrastive
    min_words: int = 1         # Minimum input concepts needed
    weight: float = 1.0        # Selection probability weight


# ─── Assertion templates ─────────────────────────────────────────
ASSERTION = [
    Template("{Art} {S} {V}.", ["S", "V"], "assertive"),
    Template("{Art} {S} {V} {Art2} {O}.", ["S", "V", "O"], "assertive"),
    Template("{S} ist {Pred}.", ["S", "Pred"], "assertive"),
    Template("{Art} {S} ist mehr als {Pred}.", ["S", "Pred"], "assertive"),
    Template("Es gibt {Art} {S}, {Art2} {V}.", ["S", "V"], "assertive", 2),
    Template("{S} bedeutet {O}.", ["S", "O"], "assertive"),
    Template("{Art} {S} braucht {O}.", ["S", "O"], "assertive"),
    Template("{S} und {O} gehören zusammen.", ["S", "O"], "assertive", 2),
    Template("Ohne {S} gibt es kein {O}.", ["S", "O"], "assertive", 2),
    Template("{Art} {S} {V} — immer.", ["S", "V"], "assertive"),
]

# ─── Reflection templates ────────────────────────────────────────
REFLECTION = [
    Template("Vielleicht ist {S} {Pred}.", ["S", "Pred"], "reflective"),
    Template("Man fragt sich, was {S} wirklich bedeutet.", ["S"], "reflective"),
    Template("Was wäre, wenn {S} nicht {Pred} wäre?", ["S", "Pred"], "reflective"),
    Template("Es gibt Momente, in denen {S} alles verändert.", ["S"], "reflective"),
    Template("Wer {S} versteht, versteht auch {O}.", ["S", "O"], "reflective", 2),
    Template("Am Ende bleibt nur {S}.", ["S"], "reflective"),
    Template("{S} — das ist die eigentliche Frage.", ["S"], "reflective"),
    Template("Manchmal {V} {Art} {S} leise.", ["S", "V"], "reflective"),
]

# ─── Contrast templates ──────────────────────────────────────────
CONTRAST = [
    Template("Nicht {A}, sondern {B}.", ["A", "B"], "contrastive", 2),
    Template("{A}, aber auch {B}.", ["A", "B"], "contrastive", 2),
    Template("Wo {A} endet, beginnt {B}.", ["A", "B"], "contrastive", 2),
    Template("{A} und {B} — zwei Seiten derselben Wahrheit.", ["A", "B"], "contrastive", 2),
    Template("Ohne {A} kein {B}.", ["A", "B"], "contrastive", 2),
    Template("{A} ist nicht das Gegenteil von {B} — es ist der Weg dorthin.", ["A", "B"], "contrastive", 2),
    Template("Wer {A} kennt, kennt auch {B}.", ["A", "B"], "contrastive", 2),
    Template("{A} schließt {B} nicht aus.", ["A", "B"], "contrastive", 2),
]

# ─── Causation templates ─────────────────────────────────────────
CAUSATION = [
    Template("Weil es {S} gibt, gibt es auch {O}.", ["S", "O"], "assertive", 2),
    Template("{S} führt zu {O}.", ["S", "O"], "assertive", 2),
    Template("Aus {S} entsteht {O}.", ["S", "O"], "assertive", 2),
    Template("Ohne {S} wäre {O} nicht möglich.", ["S", "O"], "assertive", 2),
    Template("{S} macht {O} erst {Pred}.", ["S", "O", "Pred"], "assertive", 2),
    Template("{Art} {S} {V}, weil {O} es verlangt.", ["S", "V", "O"], "assertive", 2),
]

# ─── Emotional templates ─────────────────────────────────────────
EMOTIONAL = [
    Template("Die Sehnsucht nach {S} {V} tief.", ["S", "V"], "emotional"),
    Template("Es schmerzt, wenn {S} fehlt.", ["S"], "emotional"),
    Template("{S} — ein Wort, das alles sagt.", ["S"], "emotional"),
    Template("{Art} {Pred} {S} lässt niemanden kalt.", ["S", "Pred"], "emotional"),
    Template("In {S} liegt eine stille Kraft.", ["S"], "emotional"),
    Template("Was bleibt, wenn {S} geht? {O}.", ["S", "O"], "emotional", 2),
    Template("{S} ist das, was uns {Pred} macht.", ["S", "Pred"], "emotional"),
]

# ─── Narrative templates ─────────────────────────────────────────
NARRATIVE = [
    Template("Es beginnt mit {S}.", ["S"], "neutral"),
    Template("Am Ende steht {S}.", ["S"], "neutral"),
    Template("Erst kommt {A}, dann {B}.", ["A", "B"], "neutral", 2),
    Template("Der Weg von {A} zu {B} ist lang.", ["A", "B"], "neutral", 2),
    Template("{S} ist der Anfang. {O} ist das Ziel.", ["S", "O"], "neutral", 2),
    Template("Alles beginnt bei {S} — und führt zu {O}.", ["S", "O"], "neutral", 2),
]

# ─── Question / Reflective-Response templates ────────────────────
QUESTION_RESPONSE = [
    Template("Die Frage ist nicht ob, sondern wie {S} {V}.", ["S", "V"], "reflective"),
    Template("Was bedeutet {S} wirklich?", ["S"], "reflective"),
    Template("{S} — darauf gibt es keine einfache Antwort.", ["S"], "reflective"),
    Template("Vielleicht liegt die Antwort in {S} selbst.", ["S"], "reflective"),
    Template("Die wahre Frage ist: Was macht {S} {Pred}?", ["S", "Pred"], "reflective"),
    Template("Wer nach {S} fragt, sucht eigentlich {O}.", ["S", "O"], "reflective", 2),
]

# ─── Connector templates (for joining clauses) ───────────────────
CONNECTORS = [
    "Und doch: ",
    "Dennoch — ",
    "Gleichzeitig ",
    "Aber: ",
    "Denn ",
    "Deshalb ",
    "Trotzdem ",
    "Gerade deshalb ",
    "Und genau das macht den Unterschied. ",
]

# ─── All templates by category ───────────────────────────────────
ALL_TEMPLATES = {
    "assertion": ASSERTION,
    "reflection": REFLECTION,
    "contrast": CONTRAST,
    "causation": CAUSATION,
    "emotional": EMOTIONAL,
    "narrative": NARRATIVE,
    "question": QUESTION_RESPONSE,
}


def select_templates(
    n_nouns: int,
    n_verbs: int,
    n_adjs: int,
    mood: str = "neutral",
    is_question: bool = False,
    n_results: int = 3,
    seed: int | None = None,
) -> list[Template]:
    """Select appropriate templates based on available content.

    Parameters
    ----------
    n_nouns, n_verbs, n_adjs : int
        Available concept counts by POS
    mood : str
        Desired mood (neutral, reflective, assertive, emotional, contrastive)
    is_question : bool
        Whether input was a question
    n_results : int
        Number of templates to return
    seed : int, optional
        Random seed for reproducibility
    """
    rng = random.Random(seed)

    # Determine which slots we can fill
    available_slots = set()
    if n_nouns >= 1:
        available_slots.update({"S", "A"})
    if n_nouns >= 2:
        available_slots.update({"O", "B"})
    if n_verbs >= 1:
        available_slots.add("V")
    if n_adjs >= 1:
        available_slots.update({"Pred", "Adv"})

    # Score categories by mood match
    if is_question:
        preferred = ["question", "reflection"]
    elif mood == "emotional":
        preferred = ["emotional", "reflection", "narrative"]
    elif mood == "contrastive":
        preferred = ["contrast", "causation"]
    elif mood == "reflective":
        preferred = ["reflection", "question", "narrative"]
    else:
        preferred = ["assertion", "narrative", "reflection"]

    # Collect eligible templates
    candidates: list[tuple[Template, float]] = []
    for cat_name, templates in ALL_TEMPLATES.items():
        cat_bonus = 2.0 if cat_name in preferred else 0.5
        for tmpl in templates:
            # Check if we can fill all required slots
            if all(s in available_slots for s in tmpl.requires):
                score = tmpl.weight * cat_bonus
                # Bonus for mood match
                if tmpl.mood == mood:
                    score *= 1.5
                candidates.append((tmpl, score))

    if not candidates:
        # Absolute fallback
        return [Template("{S}.", ["S"], "neutral")]

    # Weighted random selection without replacement
    selected: list[Template] = []
    remaining = list(candidates)
    for _ in range(min(n_results, len(remaining))):
        weights = [s for _, s in remaining]
        total = sum(weights)
        if total == 0:
            break
        pick = rng.random() * total
        cumulative = 0.0
        for idx, (tmpl, w) in enumerate(remaining):
            cumulative += w
            if cumulative >= pick:
                selected.append(tmpl)
                remaining.pop(idx)
                break

    return selected


def fill_template(
    template: Template,
    nouns: list[str],
    verbs: list[str],
    adjs: list[str],
    articles: dict[str, str] | None = None,
) -> str:
    """Fill a template with actual words.

    Parameters
    ----------
    template : Template
        The template to fill
    nouns : list[str]
        Available nouns (capitalized)
    verbs : list[str]
        Available verbs (conjugated)
    adjs : list[str]
        Available adjectives
    articles : dict
        Pre-computed articles by noun: {"Mut": "der", "Kraft": "die"}
    """
    articles = articles or {}
    pattern = template.pattern

    # Map slots to words
    s = nouns[0] if nouns else "es"
    o = nouns[1] if len(nouns) > 1 else (nouns[0] if nouns else "es")
    a = nouns[0] if nouns else "es"
    b = nouns[1] if len(nouns) > 1 else (nouns[0] if nouns else "es")
    v = verbs[0] if verbs else "ist"
    pred = adjs[0] if adjs else "stark"
    adv = adjs[0] if adjs else ""

    art = articles.get(s, "der") if s != "es" else ""
    art2 = articles.get(o, "die") if o != "es" else ""

    result = pattern.format(
        S=s, V=v, O=o, A=a, B=b,
        Pred=pred, Adv=adv,
        Art=art, Art2=art2,
    )

    # Clean up double spaces and empty articles
    result = result.replace("  ", " ").strip()
    # Remove leading space after empty article
    while result.startswith(" "):
        result = result[1:]

    return result
