"""
German Grammar Engine
======================

Rule-based morphology for German text generation:
- Grammatical gender for nouns (M/F/N)
- Article declension (der/die/das × NOM/ACC/DAT/GEN)
- Adjective declension (strong/weak/mixed)
- Verb conjugation (present tense, common irregulars)
- V2 word order (main clauses) and V-final (subordinate clauses)

This module transforms semantic slot-fillings into grammatically
correct German surface forms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Case(Enum):
    NOM = "NOM"
    ACC = "ACC"
    DAT = "DAT"
    GEN = "GEN"


class Gender(Enum):
    M = "M"
    F = "F"
    N = "N"


# ── Article declension tables ─────────────────────────────────
# Definite articles: der/die/das
DEF_ARTICLES: dict[tuple[Gender, Case], str] = {
    (Gender.M, Case.NOM): "der",
    (Gender.M, Case.ACC): "den",
    (Gender.M, Case.DAT): "dem",
    (Gender.M, Case.GEN): "des",
    (Gender.F, Case.NOM): "die",
    (Gender.F, Case.ACC): "die",
    (Gender.F, Case.DAT): "der",
    (Gender.F, Case.GEN): "der",
    (Gender.N, Case.NOM): "das",
    (Gender.N, Case.ACC): "das",
    (Gender.N, Case.DAT): "dem",
    (Gender.N, Case.GEN): "des",
}

# Indefinite articles: ein/eine
INDEF_ARTICLES: dict[tuple[Gender, Case], str] = {
    (Gender.M, Case.NOM): "ein",
    (Gender.M, Case.ACC): "einen",
    (Gender.M, Case.DAT): "einem",
    (Gender.M, Case.GEN): "eines",
    (Gender.F, Case.NOM): "eine",
    (Gender.F, Case.ACC): "eine",
    (Gender.F, Case.DAT): "einer",
    (Gender.F, Case.GEN): "einer",
    (Gender.N, Case.NOM): "ein",
    (Gender.N, Case.ACC): "ein",
    (Gender.N, Case.DAT): "einem",
    (Gender.N, Case.GEN): "eines",
}

# Negative articles: kein/keine
NEG_ARTICLES: dict[tuple[Gender, Case], str] = {
    (Gender.M, Case.NOM): "kein",
    (Gender.M, Case.ACC): "keinen",
    (Gender.M, Case.DAT): "keinem",
    (Gender.M, Case.GEN): "keines",
    (Gender.F, Case.NOM): "keine",
    (Gender.F, Case.ACC): "keine",
    (Gender.F, Case.DAT): "keiner",
    (Gender.F, Case.GEN): "keiner",
    (Gender.N, Case.NOM): "kein",
    (Gender.N, Case.ACC): "kein",
    (Gender.N, Case.DAT): "keinem",
    (Gender.N, Case.GEN): "keines",
}

# Demonstrative articles: dieser/diese/dieses
DEM_ARTICLES: dict[tuple[Gender, Case], str] = {
    (Gender.M, Case.NOM): "dieser",
    (Gender.M, Case.ACC): "diesen",
    (Gender.M, Case.DAT): "diesem",
    (Gender.M, Case.GEN): "dieses",
    (Gender.F, Case.NOM): "diese",
    (Gender.F, Case.ACC): "diese",
    (Gender.F, Case.DAT): "dieser",
    (Gender.F, Case.GEN): "dieser",
    (Gender.N, Case.NOM): "dieses",
    (Gender.N, Case.ACC): "dieses",
    (Gender.N, Case.DAT): "diesem",
    (Gender.N, Case.GEN): "dieses",
}

# ── Adjective declension endings ───────────────────────────────
# Strong declension (no article)
ADJ_STRONG: dict[tuple[Gender, Case], str] = {
    (Gender.M, Case.NOM): "er", (Gender.M, Case.ACC): "en",
    (Gender.M, Case.DAT): "em", (Gender.M, Case.GEN): "en",
    (Gender.F, Case.NOM): "e",  (Gender.F, Case.ACC): "e",
    (Gender.F, Case.DAT): "er", (Gender.F, Case.GEN): "er",
    (Gender.N, Case.NOM): "es", (Gender.N, Case.ACC): "es",
    (Gender.N, Case.DAT): "em", (Gender.N, Case.GEN): "en",
}

# Weak declension (after definite article)
ADJ_WEAK: dict[tuple[Gender, Case], str] = {
    (Gender.M, Case.NOM): "e",  (Gender.M, Case.ACC): "en",
    (Gender.M, Case.DAT): "en", (Gender.M, Case.GEN): "en",
    (Gender.F, Case.NOM): "e",  (Gender.F, Case.ACC): "e",
    (Gender.F, Case.DAT): "en", (Gender.F, Case.GEN): "en",
    (Gender.N, Case.NOM): "e",  (Gender.N, Case.ACC): "e",
    (Gender.N, Case.DAT): "en", (Gender.N, Case.GEN): "en",
}

# Mixed declension (after indefinite article)
ADJ_MIXED: dict[tuple[Gender, Case], str] = {
    (Gender.M, Case.NOM): "er", (Gender.M, Case.ACC): "en",
    (Gender.M, Case.DAT): "en", (Gender.M, Case.GEN): "en",
    (Gender.F, Case.NOM): "e",  (Gender.F, Case.ACC): "e",
    (Gender.F, Case.DAT): "en", (Gender.F, Case.GEN): "en",
    (Gender.N, Case.NOM): "es", (Gender.N, Case.ACC): "es",
    (Gender.N, Case.DAT): "en", (Gender.N, Case.GEN): "en",
}

# ── Verb conjugation ──────────────────────────────────────────
# Regular present tense endings
VERB_ENDINGS = {
    "1s": "e",   # ich
    "2s": "st",  # du
    "3s": "t",   # er/sie/es
    "1p": "en",  # wir
    "2p": "t",   # ihr
    "3p": "en",  # sie/Sie
}

# Common irregular verbs (3rd person singular present)
IRREGULAR_3S: dict[str, str] = {
    "sein": "ist",
    "haben": "hat",
    "werden": "wird",
    "können": "kann",
    "müssen": "muss",
    "sollen": "soll",
    "wollen": "will",
    "dürfen": "darf",
    "wissen": "weiß",
    "geben": "gibt",
    "nehmen": "nimmt",
    "sehen": "sieht",
    "lesen": "liest",
    "sprechen": "spricht",
    "treffen": "trifft",
    "helfen": "hilft",
    "sterben": "stirbt",
    "werfen": "wirft",
    "brechen": "bricht",
    "essen": "isst",
    "vergessen": "vergisst",
    "fallen": "fällt",
    "halten": "hält",
    "laufen": "läuft",
    "schlafen": "schläft",
    "tragen": "trägt",
    "fahren": "fährt",
    "wachsen": "wächst",
    "lassen": "lässt",
    "stoßen": "stößt",
    # Compound/prefix verbs used in realization verb banks
    "widersprechen": "widerspricht",
    "gegenüberstehen": "steht gegenüber",
    "ausschließen": "schließt aus",
    "hervorbringen": "bringt hervor",
    "aufbrechen": "bricht auf",
    "verschlingen": "verschlingt",
    "vernichten": "vernichtet",
    "auslöschen": "löscht aus",
    "zerreißen": "zerreißt",
    "zerschlagen": "zerschlägt",
    "entfesseln": "entfesselt",
    "beschwören": "beschwört",
    "ermöglichen": "ermöglicht",
    "bewirken": "bewirkt",
    "gebären": "gebiert",
    "schenken": "schenkt",
    "erfordern": "erfordert",
    "verlangen": "verlangt",
    "überwinden": "überwindet",
    "besiegen": "besiegt",
    "standhalten": "hält stand",
    "umschlagen": "schlägt um",
    "zerbrechen": "zerbricht",
    "durchdringen": "durchdringt",
    "enthalten": "enthält",
    "verbergen": "verbirgt",
    "bergen": "birgt",
    "trotzen": "trotzt",
    "erzeugen": "erzeugt",
    "wecken": "weckt",
    "säen": "sät",
    "entstehen": "entsteht",
    "führen": "führt",
    "münden": "mündet",
    "sich entfalten": "entfaltet sich",
    "enden": "endet",
    "verfallen": "verfällt",
    "erblühen": "erblüht",
    "reifen": "reift",
    "stürzen": "stürzt",
    "explodieren": "explodiert",
    "brauchen": "braucht",
    "ruhen": "ruht",
    "hungern": "hungert",
    "finden": "findet",
    "schreien": "schreit",
    "gehören": "gehört",
    "bergen": "birgt",
    "enthalten": "enthält",
    "verbergen": "verbirgt",
    "durchdringen": "durchdringt",
    "verweben": "verwebt",
    "leben": "lebt",
    "gefangen sein": "ist gefangen",
    # New verbs from expanded verb bank
    "nähren": "nährt",
    "auslösen": "löst aus",
    "vergiften": "vergiftet",
    "verschlucken": "verschluckt",
    "entzünden": "entzündet",
    "begrenzen": "begrenzt",
    "ersticken": "erstickt",
    "heilen": "heilt",
    "befreien": "befreit",
    "sprengen": "sprengt",
    "zermalmen": "zermalmt",
    "erstrahlen": "erstrahlt",
    "erwachsen": "erwächst",
    "sich wandeln": "wandelt sich",
    "bedürfen": "bedarf",
    "flehen": "fleht",
    "blühen": "blüht",
    "verzehren": "verzehrt",
    "darstellen": "stellt dar",
    "verkörpern": "verkörpert",
    "umfassen": "umfasst",
    "einschließen": "schließt ein",
    "verschleiern": "verschleiert",
    "begraben": "begräbt",
    "umklammern": "umklammert",
    "hüten": "hütet",
    "bewahren": "bewahrt",
    "durchströmen": "durchströmt",
    "wurzeln": "wurzelt",
    "bestehen": "besteht",
    "verschmelzen": "verschmilzt",
    "bilden": "bildet",
    "formen": "formt",
    "aufgehen": "geht auf",
    "zerfließen": "zerfließt",
    "untergraben": "untergräbt",
    "bekämpfen": "bekämpft",
    "zerfallen": "zerfällt",
    "versinken": "versinkt",
    "gedeihen": "gedeiht",
    "reißen": "reißt",
    "vertreiben": "vertreibt",
    "suchen": "sucht",
    "brennen": "brennt",
    "gieren": "giert",
    "lechzen": "lechzt",
    "verdrängen": "verdrängt",
}


# ── Gender guessing heuristics ─────────────────────────────────
# These cover ~75% of German nouns correctly

# Feminine suffixes
F_SUFFIXES = (
    "ung", "heit", "keit", "schaft", "tion", "sion", "tät", "enz",
    "anz", "ie", "ik", "ur", "ei",
)

# Neuter suffixes
N_SUFFIXES = (
    "ment", "tum", "nis", "chen", "lein", "um", "ma",
)

# Masculine suffixes
M_SUFFIXES = (
    "ling", "ismus", "ist", "or", "ant", "ent",
)


def guess_gender(word: str) -> Gender:
    """Estimate grammatical gender from suffix patterns.

    Accuracy ~75%. Can be overridden by lexicon data.
    """
    lower = word.lower()

    # Check suffixes (longest first for specificity)
    for suffix in sorted(F_SUFFIXES, key=len, reverse=True):
        if lower.endswith(suffix):
            return Gender.F

    for suffix in sorted(N_SUFFIXES, key=len, reverse=True):
        if lower.endswith(suffix):
            return Gender.N

    for suffix in sorted(M_SUFFIXES, key=len, reverse=True):
        if lower.endswith(suffix):
            return Gender.M

    # Default heuristic: most German nouns are masculine
    return Gender.M


def get_article(
    gender: Gender,
    case: Case = Case.NOM,
    article_type: str = "def",
) -> str:
    """Get the correct article for a given gender and case."""
    table = {
        "def": DEF_ARTICLES,
        "indef": INDEF_ARTICLES,
        "neg": NEG_ARTICLES,
        "dem": DEM_ARTICLES,
    }.get(article_type, DEF_ARTICLES)

    return table.get((gender, case), "die")


def get_demonstrative(
    gender: Gender,
    case: Case = Case.NOM,
) -> str:
    """Get demonstrative article: dieser/diese/dieses."""
    return DEM_ARTICLES.get((gender, case), "dieser")


def get_relative_pronoun(
    gender: Gender,
    case: Case = Case.NOM,
) -> str:
    """Get relative pronoun: der/die/das (nominative matches definite articles)."""
    # In German, relative pronouns share forms with definite articles
    # (except genitive: dessen/deren, but we use NOM/ACC here)
    return DEF_ARTICLES.get((gender, case), "der")


def decline_adjective(
    adj: str,
    gender: Gender,
    case: Case = Case.NOM,
    article_type: str = "def",
) -> str:
    """Decline a German adjective based on gender, case, and article context."""
    if article_type == "def":
        table = ADJ_WEAK
    elif article_type == "indef":
        table = ADJ_MIXED
    else:
        table = ADJ_STRONG

    ending = table.get((gender, case), "e")

    # Extract stem by removing known adjectival endings.
    # Only strip actual suffixes, not root consonants.
    # e.g. "wahr" stays "wahr", "leise" → "leis", "dunkel" → "dunkl"
    stem = adj
    if adj.endswith("er") and len(adj) > 3:
        # "bitter" → "bitt", "sicher" → "sich" — but keep short roots
        stem = adj[:-2]
    elif adj.endswith("el") and len(adj) > 3:
        # "dunkel" → "dunkl"
        stem = adj[:-2] + adj[-1]
    elif adj.endswith("e") and len(adj) > 2:
        # "leise" → "leis", "stille" → "still"
        stem = adj[:-1]

    if not stem:
        stem = adj

    return stem + ending


def conjugate(
    verb: str,
    person: str = "3s",
) -> str:
    """Conjugate a German verb in present tense.

    Parameters
    ----------
    verb : str
        Infinitive form
    person : str
        Person+number: "1s", "2s", "3s", "1p", "2p", "3p"
    """
    lower = verb.lower()

    # Check irregulars for 3rd person singular
    if person == "3s" and lower in IRREGULAR_3S:
        return IRREGULAR_3S[lower]

    # Regular conjugation: strip -en/-n, add ending
    if lower.endswith("en"):
        stem = lower[:-2]
    elif lower.endswith("n"):
        stem = lower[:-1]
    else:
        stem = lower

    ending = VERB_ENDINGS.get(person, "t")

    # Handle stems ending in t/d (need extra 'e' before st/t)
    if stem.endswith(("t", "d")) and ending in ("st", "t"):
        ending = "e" + ending

    return stem + ending


@dataclass
class NounPhrase:
    """A declined noun phrase."""
    noun: str
    gender: Gender
    case: Case = Case.NOM
    adjective: str = ""
    article_type: str = "def"  # "def", "indef", "neg", "none"

    def render(self) -> str:
        """Render the complete noun phrase."""
        parts = []

        # Article
        if self.article_type != "none":
            art = get_article(self.gender, self.case, self.article_type)
            parts.append(art)

        # Adjective
        if self.adjective:
            declined = decline_adjective(
                self.adjective, self.gender, self.case, self.article_type,
            )
            parts.append(declined)

        # Noun (always capitalized in German)
        parts.append(self.noun.capitalize())

        return " ".join(parts)


@dataclass
class Clause:
    """A German clause with grammar-aware rendering.

    Supports main clause (V2) and subordinate clause (V-final) word order.
    """
    subject: str = ""
    verb: str = ""
    object: str = ""
    predicate: str = ""  # Predicate adjective/noun
    adverbial: str = ""
    negation: bool = False
    clause_type: str = "main"  # "main" or "subordinate"
    connector: str = ""  # "weil", "obwohl", etc.

    def render(self) -> str:
        """Render with correct German word order."""
        parts = []

        if self.clause_type == "subordinate" and self.connector:
            # Subordinate: connector + subject + ... + verb (V-final)
            parts.append(self.connector)
            if self.subject:
                parts.append(self.subject)
            if self.negation:
                parts.append("nicht")
            if self.object:
                parts.append(self.object)
            if self.adverbial:
                parts.append(self.adverbial)
            if self.predicate:
                parts.append(self.predicate)
            if self.verb:
                parts.append(self.verb)
        else:
            # Main clause: V2 word order
            # Subject + Verb + rest
            if self.subject:
                parts.append(self.subject)
            if self.verb:
                parts.append(self.verb)
            if self.negation:
                parts.append("nicht")
            if self.object:
                parts.append(self.object)
            if self.adverbial:
                parts.append(self.adverbial)
            if self.predicate:
                parts.append(self.predicate)

        return " ".join(p for p in parts if p)
