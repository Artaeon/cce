"""
Realization: Crystal → Text
============================

Translates crystallized thought-chunks into natural German text.

Architecture (v7 — GRAMMAR-CORRECT COMPOSITION):
1. Extract content words from crystal particle labels
2. Find KG relations between extracted words
3. Detect CHAINS: shared pivot words between relations
4. Build CLAUSES from verbs (not templates) using grammar engine
5. FUSE chained clauses via RELATIVE CLAUSES (not demonstrative repetition)
6. Vary sentence types: main, causal (weil), concessive (obwohl)
7. V2 inversion after adverbial connectors
8. Track used verbs to avoid repetition within one output
"""

from __future__ import annotations

import re
import random
from collections import Counter
from typing import Optional

import numpy as np

from cce.codebook import Codebook
from cce.crystallization import Crystal
from cce.grammar import (
    Case,
    Clause,
    Gender,
    NounPhrase,
    conjugate,
    decline_adjective,
    get_article,
    get_demonstrative,
    get_relative_pronoun,
    guess_gender,
)
from cce.lexicon import ResonantLexicon
from cce.metaphor import MetaphorMatcher
from cce.particle import ParticleCategory


# ═══════════════════════════════════════════════════════════════════
# Verb Banks
# ═══════════════════════════════════════════════════════════════════
# Each relation maps to VERBS, not sentences.
# Format: (infinitive, structure)

VERB_BANK: dict[str, dict[str, list[tuple[str, str]]]] = {
    "CAUSES": {
        "neutral": [
            ("erzeugen", "svo"),
            ("wecken", "svo"),
            ("hervorbringen", "svo"),
            ("bewirken", "svo"),
            ("säen", "svo"),
            ("entstehen", "prep_aus"),
            ("nähren", "svo"),
            ("auslösen", "svo"),
        ],
        "dark": [
            ("gebären", "svo"),
            ("entfesseln", "svo"),
            ("beschwören", "svo"),
            ("hervorbringen", "svo"),
            ("entstehen", "prep_aus"),
            ("vergiften", "svo"),
            ("verschlucken", "svo"),
        ],
        "hopeful": [
            ("schenken", "svo"),
            ("ermöglichen", "svo"),
            ("hervorbringen", "svo"),
            ("säen", "svo"),
            ("wachsen", "prep_aus"),
            ("nähren", "svo"),
            ("entzünden", "svo"),
        ],
        "intense": [
            ("entfesseln", "svo"),
            ("aufbrechen", "svo"),
            ("hervorbringen", "svo"),
            ("explodieren", "prep_in"),
            ("reißen", "prep_in"),
            ("entzünden", "svo"),
        ],
    },
    "OPPOSES": {
        "neutral": [
            ("widersprechen", "sdat"),
            ("gegenüberstehen", "sdat"),
            ("ausschließen", "svo"),
            ("verdrängen", "svo"),
            ("bekämpfen", "svo"),
            ("begrenzen", "svo"),
            ("untergraben", "svo"),
            ("hemmen", "svo"),
        ],
        "dark": [
            ("verschlingen", "svo"),
            ("vernichten", "svo"),
            ("auslöschen", "svo"),
            ("vertreiben", "svo"),
            ("ersticken", "svo"),
            ("verschlucken", "svo"),
        ],
        "hopeful": [
            ("trotzen", "sdat"),
            ("überwinden", "svo"),
            ("besiegen", "svo"),
            ("standhalten", "sdat"),
            ("heilen", "svo"),
            ("befreien", "prep_von"),
        ],
        "intense": [
            ("zerreißen", "svo"),
            ("zerschlagen", "svo"),
            ("vernichten", "svo"),
            ("zerbrechen", "svo"),
            ("sprengen", "svo"),
            ("zermalmen", "svo"),
        ],
    },
    "LEADSTO": {
        "neutral": [
            ("werden", "zu"),
            ("führen", "prep_zu"),
            ("münden", "prep_in"),
            ("sich verwandeln", "prep_in"),
            ("wachsen", "prep_zu"),
            ("reifen", "prep_zu"),
        ],
        "dark": [
            ("enden", "prep_in"),
            ("werden", "zu"),
            ("verfallen", "prep_in"),
            ("versinken", "prep_in"),
            ("zerfallen", "prep_in"),
            ("stürzen", "prep_in"),
        ],
        "hopeful": [
            ("erblühen", "prep_zu"),
            ("reifen", "prep_zu"),
            ("werden", "zu"),
            ("sich entfalten", "prep_zu"),
            ("erstrahlen", "prep_in"),
            ("erwachsen", "prep_zu"),
        ],
        "intense": [
            ("stürzen", "prep_in"),
            ("werden", "zu"),
            ("explodieren", "prep_in"),
            ("umschlagen", "prep_in"),
            ("sich wandeln", "prep_in"),
            ("aufbrechen", "prep_in"),
        ],
    },
    "NEEDS": {
        "neutral": [
            ("brauchen", "svo"),
            ("erfordern", "svo"),
            ("verlangen", "svo"),
            ("ruhen", "prep_auf"),
            ("bedürfen", "svo"),
            ("suchen", "svo"),
        ],
        "dark": [
            ("hungern", "prep_nach"),
            ("verlangen", "svo"),
            ("erfordern", "svo"),
            ("gieren", "prep_nach"),
            ("flehen", "prep_nach"),
            ("lechzen", "prep_nach"),
        ],
        "hopeful": [
            ("finden", "prep_durch"),
            ("brauchen", "svo"),
            ("erfordern", "svo"),
            ("gedeihen", "prep_durch"),
            ("nähren", "prep_durch"),
            ("blühen", "prep_durch"),
        ],
        "intense": [
            ("schreien", "prep_nach"),
            ("verlangen", "svo"),
            ("erfordern", "svo"),
            ("lechzen", "prep_nach"),
            ("verzehren", "prep_nach"),
            ("brennen", "prep_nach"),
        ],
    },
    "ISA": {
        "neutral": [
            ("sein", "pred"),
            ("darstellen", "svo"),
            ("verkörpern", "svo"),
        ],
        "dark": [
            ("sein", "pred_nichts_als"),
            ("sein", "pred"),
            ("verkörpern", "svo"),
            ("darstellen", "svo"),
        ],
        "hopeful": [
            ("sein", "pred_mehr"),
            ("sein", "pred"),
            ("verkörpern", "svo"),
            ("darstellen", "svo"),
        ],
        "intense": [
            ("sein", "pred"),
            ("verkörpern", "svo"),
            ("sein", "pred_nichts_als"),
            ("darstellen", "svo"),
        ],
    },
    "HAS": {
        "neutral": [
            ("tragen", "prep_in_sich"),
            ("bergen", "svo"),
            ("enthalten", "svo"),
            ("umfassen", "svo"),
            ("einschließen", "svo"),
        ],
        "dark": [
            ("verbergen", "svo"),
            ("tragen", "prep_in_sich"),
            ("verschleiern", "svo"),
            ("begraben", "svo"),
            ("umklammern", "svo"),
        ],
        "hopeful": [
            ("bergen", "svo"),
            ("tragen", "prep_in_sich"),
            ("hüten", "svo"),
            ("bewahren", "svo"),
            ("nähren", "svo"),
        ],
        "intense": [
            ("durchdringen", "pass"),
            ("bergen", "svo"),
            ("tragen", "prep_in_sich"),
            ("umfassen", "svo"),
            ("durchströmen", "pass"),
        ],
    },
    "CONTEXT": {
        "neutral": [
            ("gehören", "prep_zu"),
            ("leben", "prep_in"),
            ("wurzeln", "prep_in"),
            ("bestehen", "prep_in"),
        ],
        "dark": [
            ("verbergen", "prep_in"),
            ("gefangen sein", "prep_in"),
            ("versinken", "prep_in"),
            ("ersticken", "prep_in"),
        ],
        "hopeful": [
            ("gehören", "prep_zu"),
            ("erblühen", "prep_in"),
            ("wurzeln", "prep_in"),
            ("gedeihen", "prep_in"),
        ],
        "intense": [
            ("verweben", "prep_mit"),
            ("verschmelzen", "prep_mit"),
            ("leben", "prep_in"),
            ("brennen", "prep_in"),
        ],
    },
    "PARTOF": {
        "neutral": [
            ("gehören", "prep_zu"),
            ("sein", "pred_teil"),
            ("bilden", "svo"),
            ("formen", "svo"),
        ],
        "dark": [
            ("gefangen sein", "prep_in"),
            ("sein", "pred_teil"),
            ("versinken", "prep_in"),
            ("zerfallen", "prep_in"),
        ],
        "hopeful": [
            ("gehören", "prep_zu"),
            ("sein", "pred_teil"),
            ("wachsen", "prep_in"),
            ("aufgehen", "prep_in"),
        ],
        "intense": [
            ("sein", "pred_teil"),
            ("verschmelzen", "prep_mit"),
            ("aufgehen", "prep_in"),
            ("zerfließen", "prep_in"),
        ],
    },
}



# ═══════════════════════════════════════════════════════════════════
# Connector categories
# ═══════════════════════════════════════════════════════════════════
# Coordinating conjunctions: NO V2 inversion (und, aber, doch, denn)
# Adverbial connectors: REQUIRE V2 inversion (zugleich, dennoch, deshalb)

COORD_CONJUNCTIONS = {
    "neutral": ["und", "denn"],
    "dark":    ["aber", "doch"],
    "hopeful": ["und", "denn"],
    "intense": ["und", "doch"],
}

ADVERB_CONNECTORS = {
    "neutral": ["zugleich", "ebenso", "dabei"],
    "dark":    ["dennoch", "trotzdem", "gleichwohl"],
    "hopeful": ["darüber hinaus", "außerdem", "obendrein"],
    "intense": ["zugleich", "gleichzeitig", "dabei"],
}

# Chain connectors: fuse two chained clauses at the pivot
CHAIN_CONNECTORS: dict[str, dict[tuple[str, str], list[str]]] = {
    "neutral": {
        ("CAUSES", "LEADSTO"):  ["und so", "woraufhin"],
        ("CAUSES", "CAUSES"):   ["und zugleich"],
        ("CAUSES", "NEEDS"):    ["und doch"],
        ("OPPOSES", "LEADSTO"): ["und dennoch", "und trotzdem"],
        ("OPPOSES", "CAUSES"):  ["und dadurch"],
        ("NEEDS", "CAUSES"):    ["denn"],
        ("NEEDS", "LEADSTO"):   ["und erst dann"],
        ("LEADSTO", "CAUSES"):  ["und daraus"],
        ("LEADSTO", "LEADSTO"): ["und am Ende"],
        ("LEADSTO", "NEEDS"):   ["doch dafür"],
    },
    "dark": {
        ("CAUSES", "LEADSTO"):  ["doch", "aber"],
        ("CAUSES", "CAUSES"):   ["und schlimmer noch"],
        ("OPPOSES", "LEADSTO"): ["und dennoch"],
        ("NEEDS", "CAUSES"):    ["denn"],
        ("LEADSTO", "CAUSES"):  ["und daraus"],
    },
    "hopeful": {
        ("CAUSES", "LEADSTO"):  ["doch", "aber gerade deshalb"],
        ("CAUSES", "CAUSES"):   ["und daraus"],
        ("OPPOSES", "LEADSTO"): ["und gerade deshalb"],
        ("NEEDS", "CAUSES"):    ["denn"],
        ("LEADSTO", "CAUSES"):  ["und daraus erwächst"],
    },
    "intense": {
        ("CAUSES", "LEADSTO"):  ["doch", "aber"],
        ("CAUSES", "CAUSES"):   ["und"],
        ("OPPOSES", "LEADSTO"): ["und trotzdem"],
        ("NEEDS", "CAUSES"):    ["denn"],
    },
}

DEFAULT_CONNECTORS = {
    "neutral": ["und", "denn"],
    "dark":    ["doch", "aber"],
    "hopeful": ["und", "denn"],
    "intense": ["und", "doch"],
}

# Subordinate clause connectors mapped by semantic fit
SUBORDINATE_CONNECTORS = {
    "causal":     ["weil", "da"],
    "concessive": ["obwohl", "auch wenn"],
    "temporal":   ["während"],
}

# Which relation types prefer which subordinate type
RELATION_CLAUSE_BIAS: dict[str, str] = {
    "CAUSES": "causal",
    "NEEDS":  "causal",
    "OPPOSES": "concessive",
    "LEADSTO": "causal",
    "ISA": "causal",
    "HAS": "causal",
    "CONTEXT": "temporal",
    "PARTOF": "causal",
}

# Conclusive connectors for the last sentence (synthesis feel)
# coord_concl: coordinating (no V2 inversion) — "denn"
# adverb_concl: adverbial (requires V2 inversion) — "und so"
CONCLUSIVE_COORD = {
    "neutral":  ["denn"],
    "dark":     ["denn"],
    "hopeful":  ["denn"],
    "intense":  ["denn"],
}
CONCLUSIVE_ADVERB = {
    "neutral":  ["und so", "und genau darin"],
    "dark":     ["und gerade deshalb", "und so"],
    "hopeful":  ["und genau darin", "und so"],
    "intense":  ["und genau das", "und so"],
}


# ═══════════════════════════════════════════════════════════════════
# SKIP_WORDS and extraction config
# ═══════════════════════════════════════════════════════════════════

SKIP_WORDS = {
    "und", "aber", "doch", "oder", "nicht", "kein", "keine",
    "zu", "sehr", "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einen", "einem", "einer", "eines",
    "es", "ist", "sind", "hat", "haben", "wird", "werden",
    "also", "denn", "weil", "sondern", "daher", "darum",
    "deshalb", "allerdings", "jedoch", "auch", "dazu",
    "außerdem", "nämlich", "besonders", "ich", "du",
    "er", "sie", "wir", "ihr", "mein", "dein", "sein",
    "so", "in", "an", "auf", "von", "mit", "für", "über",
    "nach", "bei", "durch", "vor", "bis", "um", "aus",
    "noch", "schon", "nur", "dann", "wenn", "ob", "als",
    "wie", "was", "wer", "wo", "am", "im", "zum", "zur",
    "beim", "vom", "ans", "ins", "ja", "nein", "mal",
    "halt", "eben", "nun", "da", "hier", "dort", "jetzt",
    "sich", "man", "mir", "dir", "uns", "etwas", "nichts",
    "alles", "bin", "bist", "seid", "habe", "kann", "muss",
    "soll", "will", "darf", "keinen", "keinem",
    "braucht", "macht", "geht", "gibt", "nimmt", "kommt",
    "steht", "liegt", "bleibt", "heißt", "heißen",
    "stellt", "stell", "fühlt", "fühlen", "kennt", "kennst",
    "lässt", "lassen", "könnte", "möchte", "müssen",
    "sagen", "gesagt", "finden", "glauben", "findet",
    "arbeite", "arbeitet", "arbeiten", "heiße",
    "weißt", "weiß", "wissen", "denke", "denkst", "denkt",
    "mache", "machst", "gehe", "gehst", "bleibe", "bleibst",
    "bedeutet", "bedeuten", "sagt", "sagst", "findest",
    "fühlst", "stehe", "stehst", "lebe", "lebst", "lebt",
    "spricht", "sprichst", "sprechen", "erzähle", "erzählt",
    "mutige", "mutiger", "mutiges", "mutigem", "mutigen",
    "starke", "starker", "starkes", "starkem", "starken",
    "stärker", "schwächer", "manchmal",
    "dunklen", "dunkler", "dunkle",
    "ehrlich", "eigentlich", "wirklich", "wahrscheinlich",
    "vielleicht", "genau", "bestimmt", "natürlich",
}

MAX_CONTENT = 5
MAX_CRYSTALS = 3

# Mood detection word sets
DARK_WORDS = {
    "traurig", "dunkel", "einsam", "verloren", "kalt",
    "negativ", "frustriert", "wütend", "bitter", "hoffnungslos",
    "melancholisch", "düster", "verzweifelt", "leer", "schwer",
    "ängstlich", "besorgt", "zornig", "enttäuscht", "müde",
}
HOPEFUL_WORDS = {
    "hoffnungsvoll", "positiv", "warm", "hell", "mutig",
    "stark", "freudig", "optimistisch", "zuversichtlich",
    "dankbar", "liebevoll", "sanft", "ruhig", "friedlich",
    "begeistert", "inspiriert", "lebendig", "frei",
}
INTENSE_WORDS = {
    "leidenschaftlich", "intensiv", "brennend", "wild",
    "stürmisch", "rasend", "explosiv", "unbändig",
    "sehnend", "tief", "roh", "unkontrolliert",
}

# Gender overrides for common nouns where guess_gender fails
GENDER_OVERRIDES: dict[str, Gender] = {
    "angst": Gender.F, "liebe": Gender.F, "kraft": Gender.F,
    "freude": Gender.F, "trauer": Gender.F, "sehnsucht": Gender.F,
    "hoffnung": Gender.F, "kunst": Gender.F, "macht": Gender.F,
    "natur": Gender.F, "sprache": Gender.F, "stille": Gender.F,
    "ruhe": Gender.F, "würde": Gender.F, "chance": Gender.F,
    "krise": Gender.F, "reise": Gender.F, "schule": Gender.F,
    "hilfe": Gender.F, "wärme": Gender.F, "nähe": Gender.F,
    "stärke": Gender.F, "schwäche": Gender.F, "schuld": Gender.F,
    "scham": Gender.F, "arbeit": Gender.F, "musik": Gender.F,
    "familie": Gender.F, "gesellschaft": Gender.F,
    "demokratie": Gender.F, "philosophie": Gender.F,
    "religion": Gender.F, "tradition": Gender.F, "kultur": Gender.F,
    "vertrauen": Gender.N, "kind": Gender.N, "leid": Gender.N,
    "glück": Gender.N, "recht": Gender.N, "gesetz": Gender.N,
    "geld": Gender.N, "feuer": Gender.N, "wasser": Gender.N,
    "leben": Gender.N, "wissen": Gender.N, "schweigen": Gender.N,
    "licht": Gender.N,
}

# ═══════════════════════════════════════════════════════════════════
# Adjective Bank — mood-driven adjective injection
# ═══════════════════════════════════════════════════════════════════
# Keys: mood → lowercase_noun → list of adjective stems (pre-declension)
# "_generic" is fallback when noun not in bank

ADJECTIVE_BANK: dict[str, dict[str, list[str]]] = {
    "neutral": {
        "_generic":     ["tief", "still", "wahr", "rein"],
        "schmerz":      ["tief", "still", "leise"],
        "freude":       ["rein", "still", "schlicht"],
        "liebe":        ["tief", "wahr", "bedingungslos"],
        "angst":        ["leise", "tief", "dumpf"],
        "mut":          ["still", "ruhig", "fest"],
        "kraft":        ["ruhig", "tief", "stetig"],
        "hass":         ["blind", "stumm", "kalt"],
        "vertrauen":    ["tief", "still", "fest"],
        "hoffnung":     ["leise", "fern", "zart"],
        "einsamkeit":   ["tief", "still", "schwer"],
        "freiheit":     ["wahr", "rein", "weit"],
        "wahrheit":     ["nackt", "klar", "hart"],
        "leid":         ["tief", "stumm", "schwer"],
        "krieg":        ["blind", "endlos", "kalt"],
        "frieden":      ["zart", "still", "brüchig"],
        "tod":          ["still", "kalt", "gewiss"],
        "leben":        ["kurz", "rein", "flüchtig"],
        "gerechtigkeit": ["blind", "streng", "rein"],
        "schuld":       ["tief", "schwer", "stumm"],
        "erkenntnis":   ["tief", "klar", "still"],
        "wachstum":     ["stetig", "still", "langsam"],
        "zerstörung":   ["blind", "kalt", "lautlos"],
        "veränderung":  ["tief", "still", "stetig"],
    },
    "dark": {
        "_generic":     ["bitter", "kalt", "dunkel", "schwer"],
        "schmerz":      ["bitter", "bodenlos", "eisig"],
        "freude":       ["trügerisch", "flüchtig", "hohl"],
        "liebe":        ["verzweifelt", "blind", "brennend"],
        "angst":        ["lähmend", "eisig", "bodenlos"],
        "mut":          ["verzweifelt", "trotzend", "bitter"],
        "kraft":        ["zerstörerisch", "dunkel", "roh"],
        "hass":         ["gnadenlos", "brennend", "tödlich"],
        "vertrauen":    ["zerbrechlich", "blind", "brüchig"],
        "hoffnung":     ["trügerisch", "hohl", "ersterbend"],
        "einsamkeit":   ["eisig", "endlos", "erdrückend"],
        "freiheit":     ["illusorisch", "hohl", "unerreichbar"],
        "wahrheit":     ["grausam", "nackt", "unbarmherzig"],
        "leid":         ["endlos", "bodenlos", "stumm"],
        "krieg":        ["gnadenlos", "endlos", "sinnlos"],
        "frieden":      ["brüchig", "trügerisch", "flüchtig"],
        "tod":          ["lautlos", "kalt", "unerbittlich"],
        "leben":        ["flüchtig", "grausam", "vergeblich"],
        "gerechtigkeit": ["blind", "kalt", "unerreichbar"],
        "schuld":       ["erdrückend", "endlos", "unauslöschlich"],
        "erkenntnis":   ["bitter", "grausam", "spät"],
        "wachstum":     ["qualvoll", "erzwungen", "schmerzhaft"],
        "zerstörung":   ["total", "gnadenlos", "endgültig"],
    },
    "hopeful": {
        "_generic":     ["sanft", "warm", "leise", "wachsend"],
        "schmerz":      ["leise", "heilend", "vorübergehend"],
        "freude":       ["wachsend", "warm", "strahlend"],
        "liebe":        ["warm", "beständig", "sanft"],
        "angst":        ["schwindend", "leise", "überwindbar"],
        "mut":          ["wachsend", "leuchtend", "unerschütterlich"],
        "kraft":        ["wachsend", "sanft", "lebendig"],
        "hass":         ["schwindend", "überwindbar", "verblassend"],
        "vertrauen":    ["wachsend", "tief", "beständig"],
        "hoffnung":     ["leuchtend", "unzerstörbar", "warm"],
        "einsamkeit":   ["vorübergehend", "lehrend", "schwindend"],
        "freiheit":     ["wahr", "errungen", "leuchtend"],
        "wahrheit":     ["befreiend", "heilend", "tröstend"],
        "leid":         ["vorübergehend", "lehrend", "heilend"],
        "krieg":        ["endend", "überwindbar", "letzt"],
        "frieden":      ["wachsend", "beständig", "errungen"],
        "leben":        ["blühend", "reich", "kostbar"],
        "erkenntnis":   ["befreiend", "leuchtend", "wachsend"],
        "wachstum":     ["blühend", "stetig", "natürlich"],
    },
    "intense": {
        "_generic":     ["rasend", "glühend", "unbändig", "roh"],
        "schmerz":      ["rasend", "zerreißend", "glühend"],
        "freude":       ["ekstatisch", "rasend", "überwältigend"],
        "liebe":        ["verzehrend", "glühend", "unbedingt"],
        "angst":        ["rasend", "panisch", "überwältigend"],
        "mut":          ["unbezwingbar", "rasend", "unbedingt"],
        "kraft":        ["entfesselt", "explosiv", "unbändig"],
        "hass":         ["rasend", "verzehrend", "glühend"],
        "vertrauen":    ["blind", "absolut", "unerschütterlich"],
        "hoffnung":     ["brennend", "unzerstörbar", "rasend"],
        "einsamkeit":   ["zerreißend", "unerträglich", "rasend"],
        "freiheit":     ["absolut", "grenzenlos", "unbedingt"],
        "wahrheit":     ["überwältigend", "erschütternd", "unausweichlich"],
        "leid":         ["unerträglich", "zerreißend", "maßlos"],
        "krieg":        ["total", "rasend", "entfesselt"],
        "leben":        ["rasend", "überbordend", "ekstatisch"],
        "erkenntnis":   ["erschütternd", "überwältigend", "blitzartig"],
        "zerstörung":   ["total", "entfesselt", "absolut"],
    },
}

def _pronominalize_metaphor_subject(text: str, subject: str, lexicon: ResonantLexicon) -> str:
    """Replaces the subject at the beginning of a sentence with a pronoun if it matches."""
    if not text or not subject:
        return text

    # Check if the sentence starts with the subject (case-insensitive)
    # and if the subject is a single word (for simplicity)
    if text.lower().startswith(subject.lower()) and ' ' not in subject:
        # Get gender of the subject
        gender = guess_gender(subject)
        if gender == Gender.M:
            pronoun = "er"
        elif gender == Gender.F:
            pronoun = "sie"
        elif gender == Gender.N:
            pronoun = "es"
        else:
            return text # Cannot determine gender, don't pronominalize

        # Replace the subject with the pronoun, preserving case of the rest of the sentence
        # and ensuring the pronoun is lowercase as it's mid-sentence
        return pronoun + text[len(subject):]
    return text


def _select_metaphor_verb(
    relation: str,
    mood: str,
    rng: random.Random
) -> tuple[str, str]:
    """Select a suitable verb and structure for a metaphor component."""
    options = VERB_BANK.get(relation, {}).get(mood) or VERB_BANK.get(relation, {}).get("neutral")
    if not options:
        # Fallback defaults
        if relation == "HAS": return ("bergen", "svo")
        if relation == "ISA": return ("sein", "pred")
        if relation == "CAUSES": return ("erzeugen", "svo")
        if relation == "NEEDS": return ("brauchen", "svo")
        if relation == "LEADSTO": return ("führen", "prep_zu")
        if relation == "OPPOSES": return ("widersprechen", "sdat")
        return ("haben", "svo")
    return rng.choice(options)


# Mood-aware transition phrases from image-world back to abstract content
_METAPHOR_BRIDGES: dict[str, list[str]] = {
    "neutral": [
        "Und darin zeigt sich:",
        "Denn dahinter steht:",
        "Was das bedeutet:",
    ],
    "dark":    [
        "Und gerade deshalb:",
        "Was das heißt:",
        "Dahinter verbirgt sich:",
    ],
    "hopeful": [
        "Und darin liegt:",
        "Denn im Kern gilt:",
        "Was daraus wächst:",
    ],
    "intense": [
        "Und das bedeutet:",
        "Denn dahinter brennt:",
        "Was das heißt:",
    ],
}


def _metaphor_bridge(mood: str, rng: random.Random) -> str:
    """Select a mood-appropriate bridging phrase from image-world to abstract."""
    bridges = _METAPHOR_BRIDGES.get(mood, _METAPHOR_BRIDGES["neutral"])
    return rng.choice(bridges)


# ═══════════════════════════════════════════════════════════════════
# Main realize function
# ═══════════════════════════════════════════════════════════════════

def realize(
    crystals: list[Crystal],
    lexicon: ResonantLexicon,
    codebook: Codebook,
    avoid_words: set[str] | None = None,
    is_question: bool = False,
    knowledge: object | None = None,
    emotion: str = "",
    question_type: str = "",
    metaphor_matcher: MetaphorMatcher | None = None,
) -> str:
    """Turn crystallized thought-chunks into German text.

    v7: Correct V2 word order, verb dedup, sentence type variation.
    """
    if not crystals:
        return ""

    # Initialize RNG
    seed = int(sum(c.emotional_charge for c in crystals) * 1000)
    rng = random.Random(seed)

    # 1. Metaphor Injection
    metaphor_text = ""
    if metaphor_matcher and rng.random() < 0.4:  # 40% chance of metaphor
        # Count main topics
        c_particles = []
        for c in crystals:
            for p in c.particles:
                # Extract base concept from label "love#123" -> "love"
                c_particles.append(p.label.split("#")[0].lower())
        
        if c_particles:
            main_topic = Counter(c_particles).most_common(1)
            topic = main_topic[0][0]
            metaphor = metaphor_matcher.find_metaphor(topic, rng)
            if metaphor:
                metaphor_text = metaphor.render_as_denkraum(rng)

    all_words: list[tuple[str, str]] = []
    crystal_emotions: list[float] = []

    for crystal in crystals[:MAX_CRYSTALS]:
        nouns, verbs, adjs = _extract_words(crystal, lexicon)
        for n in nouns:
            all_words.append((n, "NOUN"))
        for v in verbs:
            all_words.append((v, "VERB"))
        for a in adjs:
            all_words.append((a, "ADJ"))
        crystal_emotions.append(crystal.emotional_charge)

    if not all_words:
        return ""

    # Deduplicate preserving order
    seen_w: set[str] = set()
    unique: list[tuple[str, str]] = []
    for w, p in all_words:
        key = w.lower()
        if key not in seen_w:
            seen_w.add(key)
            unique.append((w, p))

    nouns = [w for w, p in unique if p == "NOUN"]
    mood = _determine_mood(crystal_emotions, emotion, is_question)

    # Find KG relations
    relations: list[tuple[str, str, str]] = []
    if knowledge is not None and len(nouns) >= 1:
        relations = _find_relations(nouns, knowledge)

    rng = random.Random(hash(tuple(nouns)) & 0xFFFFFFFF)

    if not relations:
        return _fallback_clause(nouns, mood, is_question, rng, lexicon)

    # Detect chains (shared pivots between relations)
    chains, standalone = _detect_chains(relations)

    # ── Build output with verb dedup ──
    used_verbs: set[str] = set()
    parts: list[str] = []

    # Fused chain sentences (with relative clauses)
    for chain in chains:
        if len(parts) >= 2:
            break
        sentence = _fuse_chain(chain, mood, rng, lexicon, used_verbs)
        if sentence:
            parts.append(sentence)

    # Standalone relation clauses — structure: bound pair (middle) + main (closing)
    # Reserve the last standalone for a plain main clause (gets conclusive connector)
    if len(standalone) >= 2 and len(parts) >= 1:
        # Build a bound pair from the first two standalones
        s1, r1, o1 = standalone[0]
        s2, r2, o2 = standalone[1]
        combined = _build_bound_pair(
            s1, r1, o1, s2, r2, o2,
            mood, rng, lexicon, used_verbs,
        )
        if combined:
            parts.append(combined)
        remaining = standalone[2:]
    else:
        remaining = list(standalone)

    # Remaining standalones as plain main clauses (last gets conclusive)
    for subj, rel, obj in remaining:
        if len(parts) >= 3:
            break
        clause = _build_clause(subj, rel, obj, mood, rng, lexicon, used_verbs)
        if clause:
            parts.append(clause)

    assembled = _assemble(parts, mood, rng)
    
    result = assembled
    if metaphor_text:
        # ── Apoptosis: metaphor carries the weight, keep only 1 coda clause ──
        # The image-world + bridge is already 3-4 sentences.
        # Adding more than 1 abstract clause dilutes the impact.
        if len(parts) > 1:
            coda = _assemble(parts[:1], mood, rng)
        else:
            coda = assembled
        bridge = _metaphor_bridge(mood, rng)
        result = f"{metaphor_text} {bridge} {coda}".strip()

    # Apply pronominalization globally over the entire text
    result = _pronominalize(result, lexicon)

    if question_type:
        result = _frame_for_question(result, question_type, rng)
    return result


# ═══════════════════════════════════════════════════════════════════
# Clause Building — the core compositional engine
# ═══════════════════════════════════════════════════════════════════

def _build_clause(
    subject: str,
    relation: str,
    obj: str,
    mood: str,
    rng: random.Random,
    lexicon: ResonantLexicon,
    used_verbs: set[str],
) -> str:
    """Build a main clause from a relation using the verb bank.

    Tracks used_verbs to avoid repetition within a single output.
    """
    verb_inf, structure = _pick_verb(relation, mood, rng, used_verbs)
    if not verb_inf:
        return ""

    return _render_structure(subject, verb_inf, obj, structure, lexicon, mood, rng)


def _build_subordinate(
    subject: str,
    relation: str,
    obj: str,
    mood: str,
    rng: random.Random,
    lexicon: ResonantLexicon,
    used_verbs: set[str],
) -> str:
    """Build a subordinate clause (weil/obwohl) from a relation.

    Produces e.g.: "weil Liebe Schmerz erzeugt" (V-final)
    NEVER used standalone — always paired with a main clause via _build_bound_pair.
    """
    verb_inf, structure = _pick_verb(relation, mood, rng, used_verbs)
    if not verb_inf:
        return ""

    sub_type = RELATION_CLAUSE_BIAS.get(relation, "causal")
    connectors = SUBORDINATE_CONNECTORS.get(sub_type, SUBORDINATE_CONNECTORS["causal"])
    connector = rng.choice(connectors)

    return _render_subordinate(subject, verb_inf, obj, structure, connector, lexicon)


def _build_bound_pair(
    subj1: str, rel1: str, obj1: str,
    subj2: str, rel2: str, obj2: str,
    mood: str,
    rng: random.Random,
    lexicon: ResonantLexicon,
    used_verbs: set[str],
) -> str:
    """Build a subordinate+main pair as one sentence.

    'Obwohl Liebe Hass widerspricht, erfordert sie Vertrauen.'

    The subordinate clause occupies position 1, so the main clause
    starts with verb at position 2 (V2 inversion).
    """
    # Decide which relation is better as subordinate
    # OPPOSES → concessive (obwohl), CAUSES/NEEDS → causal (weil)
    sub_type1 = RELATION_CLAUSE_BIAS.get(rel1, "causal")
    sub_type2 = RELATION_CLAUSE_BIAS.get(rel2, "causal")

    # Prefer the relation with a concessive bias as the subordinate
    if sub_type2 == "concessive" and sub_type1 != "concessive":
        # Swap: rel2 becomes the subordinate
        sub_clause = _build_subordinate(subj2, rel2, obj2, mood, rng, lexicon, used_verbs)
        main_clause = _build_clause(subj1, rel1, obj1, mood, rng, lexicon, used_verbs)
    else:
        sub_clause = _build_subordinate(subj1, rel1, obj1, mood, rng, lexicon, used_verbs)
        main_clause = _build_clause(subj2, rel2, obj2, mood, rng, lexicon, used_verbs)

    if not sub_clause or not main_clause:
        return main_clause or sub_clause or ""

    # Apply V2 inversion to main clause (verb before subject after subordinate)
    inverted_main = _invert_v2(main_clause)

    # Lowercase the inverted main (it's mid-sentence after comma)
    if inverted_main:
        inverted_main = inverted_main[0].lower() + inverted_main[1:]

    # Strip trailing punctuation from both before combining
    sub_clean = sub_clause.rstrip(".!?")
    main_clean = inverted_main.rstrip(".!?")

    return f"{_ensure_cap(sub_clean)}, {main_clean}"


def _pick_verb(
    relation: str,
    mood: str,
    rng: random.Random,
    used_verbs: set[str],
) -> tuple[str, str]:
    """Pick a verb from the bank, avoiding already-used verbs.

    Returns (infinitive, structure) or ("", "") if nothing available.
    """
    bank = VERB_BANK.get(relation, VERB_BANK.get("CAUSES", {}))
    mood_verbs = bank.get(mood, bank.get("neutral", []))
    if not mood_verbs:
        return "", ""

    # Prefer unused verbs
    unused = [(v, s) for v, s in mood_verbs if v not in used_verbs]
    if unused:
        verb_inf, structure = rng.choice(unused)
    else:
        # All used — pick any
        verb_inf, structure = rng.choice(mood_verbs)

    used_verbs.add(verb_inf)
    return verb_inf, structure


# ═══════════════════════════════════════════════════════════════════
# Structure Rendering
# ═══════════════════════════════════════════════════════════════════

def _ensure_cap(text: str) -> str:
    """Capitalize first letter without lowering the rest.

    'dieser Schmerz' → 'Dieser Schmerz' (not 'Dieser schmerz').
    """
    if not text:
        return text
    return text[0].upper() + text[1:]


def _conjugate_and_split(verb_inf: str) -> tuple[str, str, str]:
    """Conjugate verb and split into main verb, reflexive, and particle.

    Returns (main_verb, reflexive, particle):
      "hervorbringen" → ("bringt", "", "hervor")
      "sich entfalten" → ("entfaltet", "sich", "")
      "sich verwandeln" → ("verwandelt", "sich", "")
      "erzeugen" → ("erzeugt", "", "")
      "gegenüberstehen" → ("steht", "", "gegenüber")
    """
    v = conjugate(verb_inf, "3s")
    parts = v.split()

    if len(parts) == 2:
        # "sich verwandelt" → sich at position 0
        if parts[0] == "sich":
            return parts[1], "sich", ""
        # "entfaltet sich" → sich at position 1
        if parts[1] == "sich":
            return parts[0], "sich", ""
        # Separable verb: "bringt hervor"
        return parts[0], "", parts[1]
    elif len(parts) == 3:
        return parts[0], "", " ".join(parts[1:])

    return v, "", ""


def _inject_adjective(
    noun: str,
    mood: str,
    rng: random.Random | None,
    lexicon: ResonantLexicon,
    case: Case = Case.ACC,
) -> str:
    """Optionally inject a mood-appropriate adjective before a noun.

    Returns "adj Noun" ~50% of the time, "Noun" otherwise.
    Uses strong declension (no article context).
    """
    noun = _ensure_cap(noun)
    if not rng or rng.random() > 0.5:
        return noun

    lower = noun.lower()
    mood_adjs = ADJECTIVE_BANK.get(mood, ADJECTIVE_BANK.get("neutral", {}))
    candidates = mood_adjs.get(lower, mood_adjs.get("_generic", []))
    if not candidates:
        return noun

    adj_stem = rng.choice(candidates)
    gender = _get_gender(noun, lexicon)
    adj_declined = decline_adjective(adj_stem, gender, case, "none")

    return f"{adj_declined} {noun}"


def _render_structure(
    subject: str,
    verb_inf: str,
    obj: str,
    structure: str,
    lexicon: ResonantLexicon,
    mood: str = "neutral",
    rng: random.Random | None = None,
) -> str:
    """Render a main clause from subject, verb, object, and structure type."""
    s = _ensure_cap(subject)
    # Adjective injection: apply per-structure with correct case
    o_acc = _inject_adjective(obj, mood, rng, lexicon, Case.ACC) if rng else _ensure_cap(obj)
    o_dat = _inject_adjective(obj, mood, rng, lexicon, Case.DAT) if rng else _ensure_cap(obj)
    main_v, refl, particle = _conjugate_and_split(verb_inf)

    # Build verb string: "erzeugt" or "entfaltet sich"
    verb_str = f"{main_v} {refl}".strip() if refl else main_v

    if structure == "svo":
        if particle:
            return Clause(subject=s, verb=verb_str, object=o_acc, adverbial=particle).render()
        return Clause(subject=s, verb=verb_str, object=o_acc).render()

    elif structure == "sdat":
        if particle:
            return Clause(subject=s, verb=verb_str, object=o_dat, adverbial=particle).render()
        return Clause(subject=s, verb=verb_str, object=o_dat).render()

    elif structure == "prep_aus":
        return f"aus {s} {main_v} {o_acc}".strip()

    elif structure == "prep_in":
        return Clause(subject=s, verb=verb_str, object=f"in {o_dat}").render()

    elif structure == "prep_zu":
        return Clause(subject=s, verb=verb_str, object=f"zu {o_dat}").render()

    elif structure == "prep_nach":
        return Clause(subject=s, verb=verb_str, object=f"nach {o_dat}").render()

    elif structure == "prep_auf":
        return Clause(subject=s, verb=verb_str, object=f"auf {o_dat}").render()

    elif structure == "prep_durch":
        return Clause(subject=s, verb="findet sich", object=f"durch {o_acc}").render()

    elif structure == "prep_mit":
        return Clause(subject=s, verb="ist verwoben", object=f"mit {o_dat}").render()

    elif structure == "prep_von":
        return Clause(subject=s, verb=verb_str, object=f"von {o_dat}").render()

    elif structure == "prep_in_sich":
        return Clause(subject=s, verb=verb_str, object=o_acc, adverbial="in sich").render()

    elif structure == "zu":
        return Clause(subject=s, verb=verb_str, object=f"zu {o_dat}").render()

    elif structure == "pred":
        return Clause(subject=s, verb=main_v, predicate=o_acc).render()

    elif structure == "pred_nichts_als":
        return Clause(subject=s, verb="ist", predicate=f"nichts als {o_acc}").render()

    elif structure == "pred_mehr":
        return Clause(subject=s, verb="ist", predicate=f"{o_acc} — und vielleicht mehr").render()

    elif structure == "pred_teil":
        return Clause(subject=s, verb="ist", predicate=f"Teil von {o_dat}").render()

    elif structure == "pass":
        return Clause(subject=s, verb=f"ist {main_v}", object=f"von {o_dat}").render()

    else:
        if particle:
            return Clause(subject=s, verb=verb_str, object=o_acc, adverbial=particle).render()
        return Clause(subject=s, verb=verb_str, object=o_acc).render()


def _render_subordinate(
    subject: str,
    verb_inf: str,
    obj: str,
    structure: str,
    connector: str,
    lexicon: ResonantLexicon,
) -> str:
    """Render a subordinate clause with V-final word order.

    "weil Liebe Schmerz erzeugt" (verb at end, not position 2).
    """
    s = _ensure_cap(subject)
    o = _ensure_cap(obj)
    v_full = conjugate(verb_inf, "3s")

    # For subordinate clauses, reassemble the verb as one unit at the end
    # Separable verbs rejoin: "bringt hervor" → "hervorbringt"
    # This is a simplification — use the infinitive-like recombined form
    main_v, refl, particle = _conjugate_and_split(verb_inf)
    if particle:
        # Separable verbs rejoin in subordinate: "hervor" + "bringt" → "hervorbringt"
        verb_final = particle + main_v
    elif refl:
        verb_final = f"{refl} {main_v}"  # "sich verwandelt"
    else:
        verb_final = main_v

    # Build the object/complement part based on structure
    if structure in ("svo", "sdat"):
        complement = o
    elif structure == "prep_aus":
        complement = f"aus {o}"
    elif structure == "prep_in":
        complement = f"in {o}"
    elif structure == "prep_zu":
        complement = f"zu {o}"
    elif structure == "prep_nach":
        complement = f"nach {o}"
    elif structure == "prep_auf":
        complement = f"auf {o}"
    elif structure == "zu":
        complement = f"zu {o}"
    elif structure.startswith("pred"):
        complement = o
        verb_final = "ist"
    else:
        complement = o

    # Subordinate clause: connector + subject + complement + verb (V-final)
    return Clause(
        subject=s,
        verb=verb_final,
        object=complement,
        clause_type="subordinate",
        connector=connector,
    ).render()


# ═══════════════════════════════════════════════════════════════════
# Chain Detection & Fusion
# ═══════════════════════════════════════════════════════════════════

def _detect_chains(
    relations: list[tuple[str, str, str]],
) -> tuple[list[tuple[str, str, str, str, str]], list[tuple[str, str, str]]]:
    """Find chains where the object of one relation is the subject of another."""
    chains: list[tuple[str, str, str, str, str]] = []
    used_indices: set[int] = set()

    for i, (s1, r1, o1) in enumerate(relations):
        if i in used_indices:
            continue
        for j, (s2, r2, o2) in enumerate(relations):
            if j in used_indices or i == j:
                continue
            if o1.lower() == s2.lower():
                chains.append((s1, r1, o1, r2, o2))
                used_indices.add(i)
                used_indices.add(j)
                break

    standalone = [
        (s, r, o) for idx, (s, r, o) in enumerate(relations)
        if idx not in used_indices
    ]

    return chains, standalone


def _fuse_chain(
    chain: tuple[str, str, str, str, str],
    mood: str,
    rng: random.Random,
    lexicon: ResonantLexicon,
    used_verbs: set[str],
) -> str:
    """Fuse two chained clauses through their shared pivot word.

    Uses RELATIVE CLAUSES instead of demonstrative repetition:
      "Liebe weckt Schmerz, der zu Kraft wird"
    instead of:
      "Liebe weckt Schmerz, doch dieser Schmerz wird zu Kraft"
    """
    subj1, rel1, pivot, rel2, obj2 = chain

    # Build first clause (main)
    verb1_inf, struct1 = _pick_verb(rel1, mood, rng, used_verbs)
    if not verb1_inf:
        return ""
    clause1 = _render_structure(subj1, verb1_inf, pivot, struct1, lexicon)
    if not clause1:
        return ""

    # Build second clause as RELATIVE clause
    verb2_inf, struct2 = _pick_verb(rel2, mood, rng, used_verbs)
    if not verb2_inf:
        return clause1

    gender = _get_gender(pivot, lexicon)
    rel_pronoun = get_relative_pronoun(gender, Case.NOM)

    # Render the relative clause with V-final word order
    rel_clause = _render_relative(
        rel_pronoun, verb2_inf, obj2, struct2, lexicon,
    )

    if not rel_clause:
        return clause1

    return f"{clause1}, {rel_clause}"


def _render_relative(
    rel_pronoun: str,
    verb_inf: str,
    obj: str,
    structure: str,
    lexicon: ResonantLexicon,
) -> str:
    """Render a relative clause with V-final word order.

    "der zu Kraft wird" / "die Freude vertreibt"
    """
    o = _ensure_cap(obj)
    main_v, refl, particle = _conjugate_and_split(verb_inf)

    # Reassemble verb for V-final position
    if particle:
        verb_final = particle + main_v  # "hervor" + "bringt" = "hervorbringt"
    elif refl:
        verb_final = f"{refl} {main_v}"  # "sich verwandelt" (not "verwandeltsich")
    else:
        verb_final = main_v

    # Build complement based on structure
    if structure in ("svo", "sdat"):
        # "der Freude vertreibt" — object then verb
        return f"{rel_pronoun} {o} {verb_final}"
    elif structure == "prep_in":
        return f"{rel_pronoun} in {o} {verb_final}"
    elif structure == "prep_zu" or structure == "zu":
        return f"{rel_pronoun} zu {o} {verb_final}"
    elif structure == "prep_aus":
        return f"{rel_pronoun} aus {o} {verb_final}"
    elif structure == "prep_nach":
        return f"{rel_pronoun} nach {o} {verb_final}"
    elif structure == "prep_auf":
        return f"{rel_pronoun} auf {o} {verb_final}"
    elif structure == "prep_durch":
        return f"{rel_pronoun} {refl} durch {o} {verb_final}".strip()
    elif structure.startswith("pred"):
        return f"{rel_pronoun} {o} ist"
    else:
        return f"{rel_pronoun} {o} {verb_final}"


# ═══════════════════════════════════════════════════════════════════
# V2 Inversion
# ═══════════════════════════════════════════════════════════════════

def _invert_v2(sentence: str) -> str:
    """Apply V2 inversion: swap subject and verb.

    "Liebe weckt Freude." → "weckt Liebe Freude."

    After an adverbial connector occupies position 1,
    the finite verb must move to position 2 (before the subject).
    """
    # Strip trailing punctuation for processing
    punct = ""
    text = sentence.strip()
    if text and text[-1] in ".!?":
        punct = text[-1]
        text = text[:-1].strip()

    words = text.split()
    if len(words) < 2:
        return sentence

    # words[0] = subject, words[1] = verb → swap
    # But handle multi-word verbs: "entfaltet sich"
    subject = words[0]
    verb = words[1]

    # Check if word[2] is "sich" (reflexive) — keep it with verb
    if len(words) > 2 and words[2] == "sich":
        # Reflexive V2 inversion: "verwandelt sie sich" (not "verwandelt sich sie")
        rest = words[3:]
        result = f"{verb} {subject} sich"
        if rest:
            result += " " + " ".join(rest)
        if punct:
            result += punct
        return result
    else:
        rest = words[2:]

    # Inverted: verb + subject + rest
    result = f"{verb} {subject}"
    if rest:
        result += " " + " ".join(rest)
    if punct:
        result += punct

    return result


# ═══════════════════════════════════════════════════════════════════
# Assembly
# ═══════════════════════════════════════════════════════════════════

def _assemble(
    sentences: list[str],
    mood: str,
    rng: random.Random,
) -> str:
    """Join sentences into coherent output.

    Flow: first sentence stands alone, middle gets conjunction or dash,
    last sentence gets conclusive connector for synthesis.
    Bound pairs (starting with subordinate connector) get dash prefix.
    """
    if not sentences:
        return ""

    # Filter out empty
    clean: list[str] = []
    for s in sentences:
        s = s.strip()
        if s and len(s) > 3:
            clean.append(s)
    if not clean:
        return ""

    SUB_STARTERS = {"weil", "da", "obwohl", "auch", "während"}
    coord = COORD_CONJUNCTIONS.get(mood, COORD_CONJUNCTIONS["neutral"])
    concl_coord = CONCLUSIVE_COORD.get(mood, CONCLUSIVE_COORD["neutral"])
    concl_adverb = CONCLUSIVE_ADVERB.get(mood, CONCLUSIVE_ADVERB["neutral"])

    parts: list[str] = []
    for i, sentence in enumerate(clean):
        # Capitalize and punctuate
        sentence = _ensure_cap(sentence)
        if sentence[-1] not in ".!?—":
            sentence += "."

        if i == 0:
            # First sentence: no connector
            pass
        else:
            # Check if this sentence is a bound pair
            first_word = sentence.split()[0].lower().rstrip(",")
            is_bound = first_word in SUB_STARTERS

            is_last = (i == len(clean) - 1) and len(clean) > 1

            if is_bound:
                # Bound pair: dash prefix only (already complete)
                sentence = f"— {sentence}"
            elif is_last:
                # Last non-bound sentence: conclusive connector
                if rng.random() < 0.5:
                    # Coordinating conclusive (no inversion): "Denn X."
                    connector = rng.choice(concl_coord)
                    sentence = f"{_ensure_cap(connector)} {sentence}"
                else:
                    # Adverbial conclusive (V2 inversion): "Und so V S..."
                    connector = rng.choice(concl_adverb)
                    inverted = _invert_v2(sentence)
                    sentence = f"{_ensure_cap(connector)} {inverted}"
            else:
                # Middle sentence: coordinating conjunction
                conj = rng.choice(coord)
                sentence = f"{conj.capitalize()} {sentence}"

        parts.append(sentence)

    result = " ".join(parts).strip()
    return result


# ═══════════════════════════════════════════════════════════════════
# Question-Type Response Framing
# ═══════════════════════════════════════════════════════════════════

# Framing prefixes per question type
# NOTE: "warum" uses only "Denn" — "Weil" and "Da" require verb-final
# word order (Nebensatz) which the engine doesn't produce.
# "Denn" is a coordinating conjunction that preserves V2 order.
_QUESTION_FRAMES: dict[str, list[str]] = {
    "warum":     ["Denn"],
    "kennst_du": ["Vielleicht.", "Ja —", "Gewiss:"],
}


def _frame_for_question(text: str, question_type: str, rng: random.Random) -> str:
    """Add question-type-appropriate framing to the response.

    - warum → prepend 'Denn' (coordinating, keeps V2 order)
    - kennst_du → add reflective opener
    - was/wer/wie/general → keep as-is (the statement IS the answer)
    """
    # was/wer/wie/general/wo/wann → the statement IS the answer, no meta-frame
    if not text or question_type in {"was", "wer", "wie", "general", "wo", "wann", ""}:
        return text

    frames = _QUESTION_FRAMES.get(question_type)
    if not frames:
        return text

    prefix = rng.choice(frames)
    return f"{prefix} {text}"


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _determine_mood(
    crystal_emotions: list[float],
    emotion_text: str,
    is_question: bool,
) -> str:
    """Determine emotional mood from crystal charges and emotion text."""
    words = set(emotion_text.lower().split())

    if words & INTENSE_WORDS:
        return "intense"
    if words & DARK_WORDS:
        return "dark"
    if words & HOPEFUL_WORDS:
        return "hopeful"

    if crystal_emotions:
        avg = sum(crystal_emotions) / len(crystal_emotions)
        if avg < -0.3:
            return "dark"
        if avg > 0.3:
            return "hopeful"

    return "neutral"

# ═══════════════════════════════════════════════════════════════════
# Pronominalization
# ═══════════════════════════════════════════════════════════════════

_PRONOUN_MAP = {
    Gender.M: "er",
    Gender.F: "sie",
    Gender.N: "es",
}

# Connectors/articles that should NOT be pronominalized
_PRON_SKIP = {
    "und", "aber", "doch", "denn", "oder", "auch", "so",
    "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einem", "einen", "einer",
    "weil", "da", "obwohl", "wenn", "während",
    "—", "\u2014",
}


def _pronominalize(text: str, lexicon: ResonantLexicon) -> str:
    """Replace repeated noun subjects with pronouns (er/sie/es).

    Rules:
    - 1st mention of a noun → keep as-is
    - 2nd+ mention in SUBJECT position only → pronoun
    - Subject = first capitalized content word in a clause
    - Only one replacement per clause (the subject)
    - Never replace after prepositions or in object position
    """
    if not text:
        return text

    # Track which nouns have been mentioned
    mentioned: dict[str, str] = {}  # lowercase_noun → pronoun

    import re
    # Split on sentence boundaries: ". " followed by capital letter or dash
    segments = re.split(r'(?<=\.) (?=[A-ZÄÖÜ—])|(?= — )', text)

    result_segments: list[str] = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        # Within this segment, find clauses separated by ", "
        clauses = seg.split(", ")
        new_clauses: list[str] = []

        for clause in clauses:
            words = clause.split()
            new_words: list[str] = []
            replaced_in_clause = False  # Only replace subject (first noun)

            for j, w in enumerate(words):
                clean = w.rstrip(".,!?;:")
                lower = clean.lower()
                punct = w[len(clean):]

                is_capitalized = clean and clean[0].isupper() and len(clean) > 1
                is_skip = lower in _PRON_SKIP

                if is_capitalized and not is_skip and not replaced_in_clause:
                    # Check if this is in subject position:
                    # - First content word (j==0 or j==1 after dash/connector)
                    # - Not after a preposition
                    prev = new_words[-1].lower().rstrip(".,!?;:") if new_words else ""
                    preps = {"in", "aus", "zu", "nach", "auf", "von", "für",
                             "durch", "über", "unter", "zwischen", "vor", "bei"}

                    if prev in preps:
                        # After preposition — object position, skip
                        new_words.append(w)
                        continue

                    # Check if this is a verb (verbs are lowercase in German)
                    # Only nouns are capitalized mid-sentence
                    if lower in mentioned:
                        pronoun = mentioned[lower]
                        # Only capitalize at true sentence start (j==0, no preceding connector)
                        if j == 0 and not new_words:
                            pronoun = pronoun.capitalize()
                        new_words.append(pronoun + punct)
                        replaced_in_clause = True
                    else:
                        # First mention — register
                        gender = _get_gender(clean, lexicon)
                        mentioned[lower] = _PRONOUN_MAP.get(gender, "es")
                        new_words.append(w)
                        replaced_in_clause = True  # subject found, don't replace more
                else:
                    new_words.append(w)

            new_clauses.append(" ".join(new_words))

        result_segments.append(", ".join(new_clauses))

    return " ".join(result_segments)


def _get_gender(word: str, lexicon: ResonantLexicon) -> Gender:
    """Get grammatical gender for a noun."""
    lower = word.lower()

    if lower in GENDER_OVERRIDES:
        return GENDER_OVERRIDES[lower]

    profile = lexicon.get(lower)
    if profile and profile.gender:
        return {"M": Gender.M, "F": Gender.F, "N": Gender.N}.get(
            profile.gender, Gender.M
        )

    return guess_gender(word)


def _find_relations(
    nouns: list[str],
    knowledge: object,
) -> list[tuple[str, str, str]]:
    """Find KG relations between nouns AND from nouns to KG targets."""
    results: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    noun_set = {n.lower() for n in nouns}

    # Priority 1: Direct relations between input nouns
    for noun in nouns:
        rels = knowledge.get_related(noun.lower())
        for rel_type, target, sim in rels:
            if target.lower() in noun_set:
                key = (noun.lower(), target.lower())
                reverse = (target.lower(), noun.lower())
                if key not in seen and reverse not in seen:
                    results.append((noun, rel_type, target.capitalize()))
                    seen.add(key)

    # Priority 2: Expand with KG targets to enable chain formation
    if len(results) < 4:
        for noun in nouns:
            if len(results) >= 5:
                break
            rels = knowledge.get_related(noun.lower())
            for rel_type, target, sim in rels:
                if target.lower() not in noun_set:
                    key = (noun.lower(), target.lower())
                    if key not in seen:
                        results.append((noun, rel_type, target.capitalize()))
                        seen.add(key)
                        if len(results) >= 5:
                            break

    return results


def _extract_words(
    crystal: Crystal,
    lexicon: ResonantLexicon,
) -> tuple[list[str], list[str], list[str]]:
    """Extract original words from crystal particle labels."""
    concept_counts: Counter[str] = Counter()
    for p in crystal.particles:
        raw = p.label.split("#")[0]
        raw = re.sub(r'\[.*?\]', '', raw)
        for part in raw.split("+"):
            for subpart in part.split("~"):
                base = subpart.strip()
                if not base:
                    continue
                clean = base.strip(".,!?;:\"'()[]{}").lower()
                if clean and len(clean) >= 2 and clean not in SKIP_WORDS:
                    concept_counts[clean] += 1

    top = concept_counts.most_common(MAX_CONTENT * 2)

    nouns: list[str] = []
    verbs: list[str] = []
    adjs: list[str] = []
    seen: set[str] = set()

    for word, count in top:
        if len(nouns) + len(verbs) + len(adjs) >= MAX_CONTENT:
            break
        if word in seen:
            continue

        profile = lexicon.get(word)
        pos = None
        base_word = word

        if profile:
            pos = profile.pos
        else:
            base_word, pos = _find_base_form(word, lexicon)

        if pos is None:
            pos = _guess_pos(word)

        if pos == "NOUN":
            nouns.append(base_word.capitalize())
        elif pos == "VERB":
            verbs.append(base_word)
        elif pos == "ADJ":
            adjs.append(base_word)
        else:
            nouns.append(base_word.capitalize())

        seen.add(word)

    return nouns, verbs, adjs


def _find_base_form(
    word: str,
    lexicon: ResonantLexicon,
) -> tuple[str, str | None]:
    """Try to find the base form of a word in the lexicon."""
    w = word.lower()

    verb_endings = [("t", "en"), ("st", "en"), ("e", "en")]
    for suffix, replacement in verb_endings:
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            candidate = w[:-len(suffix)] + replacement
            profile = lexicon.get(candidate)
            if profile and profile.pos == "VERB":
                return candidate, "VERB"

    adj_endings = ["e", "er", "es", "em", "en"]
    for ending in adj_endings:
        if w.endswith(ending) and len(w) > len(ending) + 2:
            candidate = w[:-len(ending)]
            profile = lexicon.get(candidate)
            if profile and profile.pos == "ADJ":
                return candidate, "ADJ"

    return word, None


def _guess_pos(word: str) -> str:
    """Heuristic POS guessing for unknown words."""
    w = word.lower()
    if w.endswith(("lich", "isch", "ig", "haft", "sam", "los", "voll", "bar")):
        return "ADJ"
    if w.endswith(("ige", "iger", "iges", "igem", "igen")):
        return "ADJ"
    if w.endswith(("en", "eln", "ern")) and len(w) >= 4:
        return "VERB"
    if w.endswith("t") and len(w) >= 4 and any(
        w.endswith(x) for x in ("cht", "nnt", "llt", "mmt", "sst")
    ):
        return "VERB"
    if w.endswith(("ung", "heit", "keit", "schaft", "nis", "tum", "ment")):
        return "NOUN"
    if len(w) < 4:
        return "VERB"
    return "NOUN"


def _fallback_clause(
    nouns: list[str],
    mood: str,
    is_question: bool,
    rng: random.Random,
    lexicon: ResonantLexicon,
) -> str:
    """Build a clause when no KG relations are available."""
    if not nouns:
        return ""

    s = nouns[0]

    if is_question and len(nouns) >= 2:
        return f"Was verbindet {s} mit {nouns[1]}?"
    if is_question:
        return f"Was bedeutet {s}?"

    if len(nouns) >= 2:
        structures = [
            f"Wer {s} versteht, versteht auch {nouns[1]}.",
            f"Zwischen {s} und {nouns[1]} entsteht Bedeutung.",
            f"{s} und {nouns[1]} — darin liegt etwas Ungesagtes.",
        ]
        return rng.choice(structures)

    structures = [
        f"Was {s} wirklich bedeutet, zeigt sich erst im Handeln.",
        f"Die Frage nach {s} ist größer als jede Antwort.",
        f"An {s} entscheidet sich vieles.",
    ]
    return rng.choice(structures)
