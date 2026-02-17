"""Metaphor Engine: Finds poetic comparisons in the Knowledge Graph.

Conceptual Blending via Semantic Overlap:
1. Limit Source Domain to ELEMENTAL_SOURCES (Nature, Elements) for poetic quality.
2. Find Shared Properties: Target -> Relation -> X <- Relation <- Source.
3. Find Distinct Properties: Source -> Relation -> Y.
"""

from typing import Any
import random
from collections import defaultdict
from dataclasses import dataclass

# Avoid circular import - Use Any or simplistic typing
# from cce.knowledge import KnowledgeGraph 

# Elemental concepts that serve as high-quality metaphor sources
ELEMENTAL_SOURCES = {
    "feuer", "wasser", "meer", "sturm", "nacht", "licht", "stein", 
    "wind", "eis", "wald", "wüste", "sonne", "schatten", "asche",
    "fluss", "berg", "himmel", "stern"
}

# ═══════════════════════════════════════════════════════════════════
# Image Lexicons — domain-specific vocabulary per elemental source
# ═══════════════════════════════════════════════════════════════════
# Each source provides:
#   verbs      — actions native to that domain
#   nouns      — concrete objects from that world
#   adjectives — qualities evoking that domain
#   connectors — sentence templates for the "shared property" sentence
#                {verb} = domain verb (conjugated externally)
#                {obj}  = the shared/distinct property object
#   contrast   — templates for the "distinct property" sentence
#   gender     — grammatical gender of the source noun (m/f/n)

IMAGE_LEXICONS: dict[str, dict] = {
    "meer": {
        "gender": "n",
        "verbs":    ["tragen", "verschlingen", "umspülen", "fluten", "wiegen"],
        "nouns":    ["Welle", "Tiefe", "Strömung", "Ufer", "Brandung", "Gischt"],
        "adjectives": ["unergründlich", "salzig", "tosend", "still", "endlos"],
        "connectors": [
            "Seine Wellen tragen {obj}.",
            "In seiner Tiefe ruht {obj}.",
            "Seine Strömung trägt {obj}.",
            "An seinem Ufer wartet {obj}.",
        ],
        "contrast": [
            "Doch unter der Oberfläche lauert {obj}.",
            "Doch seine Tiefe birgt auch {obj}.",
            "Allein seine Brandung kennt {obj}.",
        ],
    },
    "feuer": {
        "gender": "n",
        "verbs":    ["verbrennen", "wärmen", "lodern", "verzehren", "glühen", "entzünden"],
        "nouns":    ["Flamme", "Glut", "Asche", "Hitze", "Funke", "Brand"],
        "adjectives": ["glühend", "verzehrend", "lodernd", "schwelend", "hell"],
        "connectors": [
            "Seine Flammen nähren {obj}.",
            "In seiner Glut lodert {obj}.",
            "Seine Hitze gebiert {obj}.",
        ],
        "contrast": [
            "Doch am Ende bleibt nur {obj}.",
            "Doch seine Asche trägt {obj}.",
            "Und dennoch schwelt in ihm {obj}.",
        ],
    },
    "sturm": {
        "gender": "m",
        "verbs":    ["peitschen", "toben", "aufwirbeln", "fegen", "tosen"],
        "nouns":    ["Donner", "Böe", "Blitz", "Orkan", "Wolkenbruch"],
        "adjectives": ["tosend", "rasend", "entfesselt", "wild", "grollend"],
        "connectors": [
            "Seine Böen tragen {obj}.",
            "In seinem Toben verbirgt sich {obj}.",
            "Sein Donner verkündet {obj}.",
        ],
        "contrast": [
            "Doch nach dem Sturm bleibt {obj}.",
            "Allein in seiner Stille danach wächst {obj}.",
            "Und dennoch reißt er {obj} mit sich.",
        ],
    },
    "wald": {
        "gender": "m",
        "verbs":    ["wurzeln", "wachsen", "verbergen", "umschließen", "atmen"],
        "nouns":    ["Wurzel", "Blatt", "Dickicht", "Schatten", "Lichtung", "Moos"],
        "adjectives": ["dicht", "dunkel", "still", "moosig", "alt"],
        "connectors": [
            "Seine Wurzeln nähren {obj}.",
            "In seinem Dickicht verbirgt sich {obj}.",
            "Sein Blätterdach schützt {obj}.",
        ],
        "contrast": [
            "Doch in seinem Schatten lauert {obj}.",
            "Allein sein Dickicht kennt auch {obj}.",
            "Und dennoch birgt er {obj} in sich.",
        ],
    },
    "nacht": {
        "gender": "f",
        "verbs":    ["verhüllen", "umhüllen", "verschlucken", "bergen", "stillen"],
        "nouns":    ["Dunkelheit", "Stille", "Mond", "Schatten", "Dämmerung"],
        "adjectives": ["dunkel", "still", "endlos", "samtweich", "kühl"],
        "connectors": [
            "Ihre Dunkelheit birgt {obj}.",
            "In ihrer Stille wächst {obj}.",
            "Ihr Schatten umhüllt {obj}.",
        ],
        "contrast": [
            "Doch in ihrer Tiefe wartet {obj}.",
            "Allein ihre Stille kennt {obj}.",
            "Und dennoch bricht in ihr {obj} hervor.",
        ],
    },
    "eis": {
        "gender": "n",
        "verbs":    ["erstarren", "bewahren", "versiegeln", "brechen", "schmelzen"],
        "nouns":    ["Kälte", "Riss", "Kristall", "Frost", "Schmelze"],
        "adjectives": ["kalt", "klar", "spröde", "gläsern", "schneidend"],
        "connectors": [
            "Sein Kristall bewahrt {obj}.",
            "In seiner Kälte ruht {obj}.",
            "Seine Oberfläche spiegelt {obj}.",
        ],
        "contrast": [
            "Doch unter seiner Oberfläche lauert {obj}.",
            "Allein sein Schmelzen offenbart {obj}.",
            "Und dennoch bricht in ihm {obj} hervor.",
        ],
    },
    "licht": {
        "gender": "n",
        "verbs":    ["erhellen", "durchdringen", "blenden", "enthüllen", "leuchten"],
        "nouns":    ["Strahl", "Glanz", "Schein", "Brechung", "Schimmer"],
        "adjectives": ["hell", "strahlend", "blendend", "warm", "flüchtig"],
        "connectors": [
            "Sein Strahl enthüllt {obj}.",
            "In seinem Glanz lebt {obj}.",
            "Sein Schein wärmt {obj}.",
        ],
        "contrast": [
            "Doch sein Schatten verbirgt {obj}.",
            "Allein seine Brechung zeigt {obj}.",
            "Und dennoch blendet es vor {obj}.",
        ],
    },
    "stein": {
        "gender": "m",
        "verbs":    ["tragen", "überdauern", "widerstehen", "schweigen", "ruhen"],
        "nouns":    ["Fels", "Riss", "Kante", "Grund", "Geröll"],
        "adjectives": ["hart", "stumm", "kalt", "schwer", "unerschütterlich"],
        "connectors": [
            "Sein Fels trägt {obj}.",
            "In seiner Härte ruht {obj}.",
            "Sein Gewicht birgt {obj}.",
        ],
        "contrast": [
            "Doch selbst Stein kennt {obj}.",
            "Allein seine Risse verraten {obj}.",
            "Und dennoch verbirgt er {obj}.",
        ],
    },
    "wind": {
        "gender": "m",
        "verbs":    ["tragen", "flüstern", "wehen", "aufwirbeln", "treiben"],
        "nouns":    ["Brise", "Hauch", "Böe", "Atem", "Zug"],
        "adjectives": ["unsichtbar", "leise", "schneidend", "warm", "ruhelos"],
        "connectors": [
            "Sein Hauch trägt {obj}.",
            "In seiner Brise flüstert {obj}.",
            "Seine Böen tragen {obj} davon.",
        ],
        "contrast": [
            "Doch seine Stille verschweigt {obj}.",
            "Allein sein Schweigen birgt {obj}.",
            "Und dennoch reißt er {obj} mit sich.",
        ],
    },
    "wüste": {
        "gender": "f",
        "verbs":    ["verdorren", "blenden", "verschlucken", "prüfen", "brennen"],
        "nouns":    ["Sand", "Düne", "Hitze", "Oase", "Horizont"],
        "adjectives": ["endlos", "karg", "gnadenlos", "still", "glühend"],
        "connectors": [
            "Ihr Sand verschluckt {obj}.",
            "In ihrer Weite wartet {obj}.",
            "Ihre Hitze brennt {obj} ein.",
        ],
        "contrast": [
            "Doch in ihr verbirgt sich {obj}.",
            "Allein ihre Oase kennt {obj}.",
            "Und dennoch birgt sie {obj}.",
        ],
    },
    "sonne": {
        "gender": "f",
        "verbs":    ["wärmen", "blenden", "nähren", "verbrennen", "aufgehen"],
        "nouns":    ["Strahl", "Glut", "Wärme", "Licht", "Aufgang"],
        "adjectives": ["strahlend", "warm", "gnadenlos", "golden", "leuchtend"],
        "connectors": [
            "Ihre Strahlen nähren {obj}.",
            "In ihrem Licht wächst {obj}.",
            "Ihre Wärme gebiert {obj}.",
        ],
        "contrast": [
            "Doch ihre Glut verbrennt auch {obj}.",
            "Allein ihr Untergang kennt {obj}.",
            "Und dennoch blendet sie vor {obj}.",
        ],
    },
    "schatten": {
        "gender": "m",
        "verbs":    ["verbergen", "folgen", "umhüllen", "verschleiern", "warten"],
        "nouns":    ["Dunkelheit", "Kontur", "Schleier", "Stille", "Rand"],
        "adjectives": ["lautlos", "still", "kühl", "tief", "unsichtbar"],
        "connectors": [
            "Seine Stille birgt {obj}.",
            "In seiner Dunkelheit wächst {obj}.",
            "Sein Schleier verhüllt {obj}.",
        ],
        "contrast": [
            "Doch hinter ihm wartet {obj}.",
            "Allein sein Verschwinden zeigt {obj}.",
            "Und dennoch verbirgt er {obj}.",
        ],
    },
    "asche": {
        "gender": "f",
        "verbs":    ["bedecken", "bewahren", "ersticken", "nähren", "verwehen"],
        "nouns":    ["Staub", "Rest", "Glut", "Erinnerung", "Boden"],
        "adjectives": ["grau", "still", "kalt", "leicht", "vergänglich"],
        "connectors": [
            "Ihr Staub bewahrt {obj}.",
            "In ihrer Stille ruht {obj}.",
            "Unter ihr schwelt {obj}.",
        ],
        "contrast": [
            "Doch in ihr glimmt noch {obj}.",
            "Allein ihr Boden nährt {obj}.",
            "Und dennoch verweht sie {obj}.",
        ],
    },
    "fluss": {
        "gender": "m",
        "verbs":    ["fließen", "tragen", "spiegeln", "umspülen", "nähren"],
        "nouns":    ["Strömung", "Ufer", "Quelle", "Mündung", "Strudel"],
        "adjectives": ["ruhig", "tief", "klar", "unaufhaltsam", "still"],
        "connectors": [
            "Seine Strömung trägt {obj}.",
            "An seinem Ufer wartet {obj}.",
            "Sein Wasser spiegelt {obj}.",
        ],
        "contrast": [
            "Doch sein Strudel verschlingt {obj}.",
            "Allein seine Tiefe birgt {obj}.",
            "Und dennoch reißt er {obj} mit sich.",
        ],
    },
    "berg": {
        "gender": "m",
        "verbs":    ["überragen", "tragen", "schweigen", "trotzen", "ruhen"],
        "nouns":    ["Gipfel", "Fels", "Hang", "Abgrund", "Kluft"],
        "adjectives": ["erhaben", "still", "unerschütterlich", "steil", "einsam"],
        "connectors": [
            "Sein Gipfel trägt {obj}.",
            "An seinem Hang wächst {obj}.",
            "Sein Fels birgt {obj}.",
        ],
        "contrast": [
            "Doch sein Abgrund kennt auch {obj}.",
            "Allein seine Kluft verbirgt {obj}.",
            "Und dennoch schweigt er über {obj}.",
        ],
    },
    "himmel": {
        "gender": "m",
        "verbs":    ["umfassen", "tragen", "leuchten", "schweigen", "öffnen"],
        "nouns":    ["Weite", "Wolke", "Horizont", "Dämmerung", "Blau"],
        "adjectives": ["endlos", "weit", "still", "leuchtend", "grenzenlos"],
        "connectors": [
            "Seine Weite umfasst {obj}.",
            "Am Horizont leuchtet {obj}.",
            "Unter seinem Blau ruht {obj}.",
        ],
        "contrast": [
            "Doch seine Wolken verbergen {obj}.",
            "Allein seine Dämmerung kennt {obj}.",
            "Und dennoch schweigt er über {obj}.",
        ],
    },
    "stern": {
        "gender": "m",
        "verbs":    ["leuchten", "führen", "brennen", "verglühen", "strahlen"],
        "nouns":    ["Licht", "Glanz", "Ferne", "Bahn", "Funkeln"],
        "adjectives": ["fern", "kalt", "leuchtend", "einsam", "ewig"],
        "connectors": [
            "Sein Licht führt zu {obj}.",
            "In seiner Ferne leuchtet {obj}.",
            "Sein Glanz erhellt {obj}.",
        ],
        "contrast": [
            "Doch seine Ferne verbirgt {obj}.",
            "Allein sein Verglühen kennt {obj}.",
            "Und dennoch brennt er für {obj}.",
        ],
    },
    "wasser": {
        "gender": "n",
        "verbs":    ["fließen", "tragen", "spiegeln", "lösen", "reinigen"],
        "nouns":    ["Tropfen", "Strom", "Oberfläche", "Tiefe", "Quelle"],
        "adjectives": ["klar", "tief", "still", "kühl", "rein"],
        "connectors": [
            "Sein Strom trägt {obj}.",
            "In seiner Tiefe ruht {obj}.",
            "Seine Oberfläche spiegelt {obj}.",
        ],
        "contrast": [
            "Doch unter seiner Oberfläche lauert {obj}.",
            "Allein seine Tiefe kennt {obj}.",
            "Und dennoch löst es {obj} auf.",
        ],
    },
}


# ── Domain-aware translation: abstract KG objects → image-world nouns ──
# Prevents semantic breaks like "Seine Flammen kennen Tiefe" (Feuer+Tiefe).
# Each source maps common abstract concepts to domain-appropriate equivalents.
# Unmapped objects pass through as-is.

_DOMAIN_MAP: dict[str, dict[str, str]] = {
    "meer":     {"Kraft": "Strömung", "Schmerz": "Salzlast", "Wärme": "Woge",
                 "Angst": "Abgrund", "Freude": "Glitzern", "Leid": "Brandung",
                 "Mut": "Fahrt", "Erkenntnis": "Klarheit", "Zerstörung": "Sturmflut",
                 "Hoffnung": "Horizont", "Vertrauen": "Ankergrund", "Zweifel": "Nebel"},
    "feuer":    {"Tiefe": "Glut", "Kraft": "Flamme", "Schmerz": "Brand",
                 "Angst": "Funkenflug", "Freude": "Licht", "Leid": "Asche",
                 "Mut": "Entfachen", "Erkenntnis": "Helligkeit", "Zerstörung": "Feuersturm",
                 "Hoffnung": "Glimmen", "Vertrauen": "Herdfeuer", "Zweifel": "Rauch"},
    "wald":     {"Tiefe": "Wurzelwerk", "Kraft": "Stamm", "Wärme": "Lichtung",
                 "Schmerz": "Dornen", "Angst": "Dickicht", "Freude": "Vogelruf",
                 "Leid": "Windbruch", "Mut": "Wachstum", "Erkenntnis": "Waldlicht",
                 "Zerstörung": "Kahlschlag", "Hoffnung": "Trieb", "Zweifel": "Nebel"},
    "sturm":    {"Tiefe": "Donner", "Kraft": "Böe", "Wärme": "Windstille",
                 "Schmerz": "Peitsche", "Angst": "Blitz", "Freude": "Aufklaren",
                 "Leid": "Verwüstung", "Mut": "Gegenwind", "Erkenntnis": "Stille",
                 "Zerstörung": "Hagel", "Hoffnung": "Lücke", "Zweifel": "Wirbel"},
    "eis":      {"Tiefe": "Kristall", "Kraft": "Härte", "Wärme": "Schmelze",
                 "Schmerz": "Splitter", "Angst": "Kälte", "Freude": "Glanz",
                 "Leid": "Erstarrung", "Mut": "Brechen", "Erkenntnis": "Klarheit"},
    "nacht":    {"Tiefe": "Dunkelheit", "Kraft": "Stille", "Wärme": "Mondlicht",
                 "Schmerz": "Schlaflosigkeit", "Angst": "Schatten", "Freude": "Stern",
                 "Leid": "Finsternis", "Mut": "Wachen", "Erkenntnis": "Dämmerung"},
    "sonne":    {"Tiefe": "Kern", "Kraft": "Strahl", "Schmerz": "Hitze",
                 "Angst": "Finsternis", "Freude": "Wärme", "Leid": "Dürre",
                 "Mut": "Aufgang", "Erkenntnis": "Licht", "Zerstörung": "Glut"},
    "wind":     {"Tiefe": "Stille", "Kraft": "Böe", "Wärme": "Föhn",
                 "Schmerz": "Schnitt", "Angst": "Heulen", "Freude": "Brise",
                 "Leid": "Kälte", "Mut": "Sturm", "Erkenntnis": "Klarheit"},
    "berg":     {"Tiefe": "Abgrund", "Kraft": "Fels", "Wärme": "Sonnenseite",
                 "Schmerz": "Steinschlag", "Angst": "Kluft", "Freude": "Gipfel",
                 "Leid": "Lawine", "Mut": "Aufstieg", "Erkenntnis": "Weitblick"},
    "wüste":    {"Tiefe": "Hitze", "Kraft": "Weite", "Wärme": "Glut",
                 "Schmerz": "Durst", "Angst": "Leere", "Freude": "Oase",
                 "Leid": "Verdorren", "Mut": "Wanderung", "Erkenntnis": "Fata Morgana"},
    "wasser":   {"Tiefe": "Grund", "Kraft": "Strömung", "Schmerz": "Strudel",
                 "Angst": "Trübe", "Freude": "Quelle", "Leid": "Flut",
                 "Mut": "Durchbruch", "Erkenntnis": "Klarheit"},
    "fluss":    {"Tiefe": "Grund", "Kraft": "Strömung", "Wärme": "Quellwärme",
                 "Schmerz": "Strudel", "Angst": "Trübe", "Freude": "Mündung",
                 "Leid": "Hochwasser", "Mut": "Durchbruch", "Erkenntnis": "Klarheit"},
    "schatten":  {"Tiefe": "Dunkelheit", "Kraft": "Kühle", "Wärme": "Dämmerung",
                 "Schmerz": "Kälte", "Angst": "Verschwinden", "Freude": "Schutz",
                 "Leid": "Finsternis", "Mut": "Hervortreten", "Erkenntnis": "Kontur"},
    "licht":    {"Tiefe": "Brechung", "Kraft": "Strahl", "Schmerz": "Schatten",
                 "Angst": "Erlöschen", "Freude": "Glanz", "Leid": "Finsternis",
                 "Mut": "Leuchten", "Erkenntnis": "Erhellung"},
    "stein":    {"Tiefe": "Kern", "Kraft": "Härte", "Wärme": "Sonnenwärme",
                 "Schmerz": "Riss", "Angst": "Abtragung", "Freude": "Schliff",
                 "Leid": "Erosion", "Mut": "Standhaftigkeit", "Erkenntnis": "Ader"},
    "stern":    {"Tiefe": "Ferne", "Kraft": "Leuchten", "Wärme": "Glühen",
                 "Schmerz": "Kälte", "Angst": "Erlöschen", "Freude": "Funkeln",
                 "Leid": "Verglühen", "Mut": "Aufleuchten", "Erkenntnis": "Licht"},
    "asche":    {"Tiefe": "Rest", "Kraft": "Wärme", "Schmerz": "Verlust",
                 "Angst": "Leere", "Freude": "Glimmen", "Leid": "Verlust",
                 "Mut": "Neubeginn", "Erkenntnis": "Spur"},
    "himmel":   {"Tiefe": "Weite", "Kraft": "Unendlichkeit", "Schmerz": "Verdunkelung",
                 "Angst": "Leere", "Freude": "Blau", "Leid": "Gewitter",
                 "Mut": "Aufklaren", "Erkenntnis": "Horizont"},
}


def _map_to_domain(obj: str, source: str) -> str:
    """Translate an abstract KG object into domain-appropriate vocabulary.

    Returns the mapped noun if found, otherwise the original object.
    Example: _map_to_domain("Tiefe", "feuer") → "Glut"
    """
    source_map = _DOMAIN_MAP.get(source.lower(), {})
    return source_map.get(obj, obj)


def _cap(s: str) -> str:
    """Capitalize first letter, preserve rest."""
    return s[0].upper() + s[1:] if s else s


@dataclass
class Metaphor:
    target: str
    source: str
    shared_prop: tuple[str, str]  # (rel, obj) e.g. ("HAS", "Wärme")
    distinct_prop: tuple[str, str] # (rel, obj) e.g. ("CAUSES", "Schmerz")

    def render(self, lexicon=None) -> str:
        """Render the metaphor as a short poetic sequence (legacy)."""
        return f"{self.target} ist {self.source}."

    def render_as_denkraum(self, rng: random.Random) -> str:
        """Render the metaphor as an immersive image-world.

        Uses IMAGE_LEXICONS to stay inside the source domain's vocabulary.
        Picks from 5 structural templates for variety.
        """
        source_lower = self.source.lower()
        lex = IMAGE_LEXICONS.get(source_lower)

        # Fallback to simple render if no lexicon available
        if not lex:
            return self._render_simple(rng)

        target = _cap(self.target)
        source = _cap(self.source)
        _, s_obj = self.shared_prop
        _, d_obj = self.distinct_prop
        # Domain-map: translate abstract KG objects into image-world nouns
        s_obj = _map_to_domain(s_obj, source_lower)
        d_obj = _map_to_domain(d_obj, source_lower)
        obj_cap = _cap(s_obj)
        d_obj_cap = _cap(d_obj)

        # Pick a structural template
        templates = [
            self._tmpl_klassisch,
            self._tmpl_apposition,
            self._tmpl_genitiv,
            self._tmpl_wie,
            self._tmpl_inverted,
        ]
        template_fn = rng.choice(templates)
        return template_fn(target, source, obj_cap, d_obj_cap, lex, rng)

    # ── Template 1: Klassisch ──────────────────────────────────────
    # "Freiheit ist ein Meer. Seine Wellen tragen die Weite.
    #  Doch unter der Oberfläche lauert das Unbekannte."

    def _tmpl_klassisch(
        self, target: str, source: str, obj: str, d_obj: str,
        lex: dict, rng: random.Random,
    ) -> str:
        article = _article_for(lex["gender"])
        opening = f"{target} ist {article}{source}"
        shared_sent = rng.choice(lex["connectors"]).format(obj=obj)
        contrast_sent = rng.choice(lex["contrast"]).format(obj=d_obj)
        return f"{opening}. {shared_sent} {contrast_sent}"

    # ── Template 2: Apposition ─────────────────────────────────────
    # "Freiheit — ein Meer — trägt die Weite in ihren Wellen."

    def _tmpl_apposition(
        self, target: str, source: str, obj: str, d_obj: str,
        lex: dict, rng: random.Random,
    ) -> str:
        article = _article_for(lex["gender"])
        adj = rng.choice(lex["adjectives"])
        noun = rng.choice(lex["nouns"])
        noun_dat_pl = _dative_plural(noun)
        adj_declined = _decline_adj(adj)
        opening = f"{target} — {article}{source} — birgt {obj} zwischen {adj_declined}en {noun_dat_pl}"
        contrast_sent = rng.choice(lex["contrast"]).format(obj=d_obj)
        return f"{opening}. {contrast_sent}"

    # ── Template 3: Genitiv ────────────────────────────────────────
    # "Die Wellen der Freiheit tragen die Weite."

    def _tmpl_genitiv(
        self, target: str, source: str, obj: str, d_obj: str,
        lex: dict, rng: random.Random,
    ) -> str:
        noun = rng.choice(lex["nouns"])
        adj = rng.choice(lex["adjectives"])
        adj_declined = _decline_adj(adj)
        # "Die [adjective]en [noun-plural] der [target] [verb] [obj]."
        noun_pl = _pluralize(noun)
        opening = f"Die {adj_declined}en {noun_pl} der {target} tragen {obj}"
        contrast_sent = rng.choice(lex["contrast"]).format(obj=d_obj)
        return f"{opening}. {contrast_sent}"

    # ── Template 4: Wie-Vergleich ──────────────────────────────────
    # "Wie ein Meer trägt Freiheit die Weite in sich."

    def _tmpl_wie(
        self, target: str, source: str, obj: str, d_obj: str,
        lex: dict, rng: random.Random,
    ) -> str:
        article = _article_for(lex["gender"])
        adj = rng.choice(lex["adjectives"])
        adj_stem = _decline_adj(adj)
        opening = f"Wie {article}{adj_stem}es {source} trägt {target} {obj} in sich"
        contrast_sent = rng.choice(lex["contrast"]).format(obj=d_obj)
        return f"{opening}. {contrast_sent}"

    # ── Template 5: Inverted ───────────────────────────────────────
    # "Ein Meer ist sie, die Freiheit. Ihre Wellen..."

    def _tmpl_inverted(
        self, target: str, source: str, obj: str, d_obj: str,
        lex: dict, rng: random.Random,
    ) -> str:
        article = _article_for(lex["gender"]).strip().capitalize()
        pronoun = _pronoun_for(lex["gender"])
        opening = f"{article} {source} ist {pronoun}, die {target}"
        shared_sent = rng.choice(lex["connectors"]).format(obj=obj)
        contrast_sent = rng.choice(lex["contrast"]).format(obj=d_obj)
        return f"{opening}. {shared_sent} {contrast_sent}"

    def _render_simple(self, rng: random.Random) -> str:
        """Fallback when no image lexicon is available."""
        target = _cap(self.target)
        source = _cap(self.source)
        _, s_obj = self.shared_prop
        _, d_obj = self.distinct_prop
        return (
            f"{target} ist {source}. "
            f"Denn darin liegt {_cap(s_obj)}. "
            f"Doch zugleich auch {_cap(d_obj)}."
        )


def _article_for(gender: str) -> str:
    """Return indefinite article for gender."""
    return {"m": "ein ", "f": "eine ", "n": "ein "}[gender]


def _possessive_for(gender: str) -> str:
    """Return possessive pronoun (dative-ish) for gender."""
    return {"m": "seinen ", "f": "ihren ", "n": "seinen "}[gender]


def _pronoun_for(gender: str) -> str:
    """Return personal pronoun for gender."""
    return {"m": "er", "f": "sie", "n": "es"}[gender]


# Explicit plural overrides for irregular German nouns used in IMAGE_LEXICONS
_PLURAL_OVERRIDES: dict[str, str] = {
    "Blatt": "Blätter",
    "Fels": "Felsen",
    "Licht": "Lichter",
    "Wald": "Wälder",
    "Riss": "Risse",
    "Gischt": "Gischten",
    "Blitz": "Blitze",
    "Zug": "Züge",
    "Rand": "Ränder",
    "Frost": "Fröste",
    "Brand": "Brände",
    "Strom": "Ströme",
    "Mond": "Monde",
    "Hang": "Hänge",
    "Glanz": "Glänze",
    "Staub": "Stäube",
    "Rest": "Reste",
    "Sand": "Sande",
    "Blau": "Blaue",
    "Grund": "Gründe",
    "Ufer": "Ufer",
    "Funke": "Funken",
    "Atem": "Atem",
    "Abgrund": "Abgründe",
    "Horizont": "Horizonte",
    "Schein": "Scheine",
    "Aufgang": "Aufgänge",
    "Schleier": "Schleier",
    "Hauch": "Hauche",
}


def _pluralize(noun: str) -> str:
    """German noun pluralization for image-lexicon nouns.

    Uses explicit overrides for irregular forms, then heuristics.
    """
    if noun in _PLURAL_OVERRIDES:
        return _PLURAL_OVERRIDES[noun]
    if noun.endswith("e"):
        return noun + "n"
    if noun.endswith(("el", "er", "en")):
        return noun
    if noun.endswith(("ung", "heit", "keit", "schaft", "ät")):
        return noun + "en"
    # Common masculine: add -e
    return noun + "e"


def _dative_plural(noun: str) -> str:
    """Return dative plural form of a noun.

    German rule: ALL dative plural nouns end in -n.
    Pluralize first, then ensure the -n ending.
    Examples: Blätter → Blättern, Ufer → Ufern, Wellen → Wellen (already -n)
    """
    plural = _pluralize(noun)
    if plural.endswith("n"):
        return plural
    return plural + "n"


def _decline_adj(adj: str) -> str:
    """Return the adjective stem ready for adding endings (-en, -es, -er, etc.).

    Handles German elision rules:
    - Adjectives ending in -el: drop the inner -e- (dunkel → dunkl)
    - Adjectives ending in -er: drop the inner -e- (bitter → bittr)
    - Adjectives ending in -e: drop the -e (leise → leis)
    - Others: return as-is (still → still)
    """
    if adj.endswith("el"):
        # dunkel → dunkl, spröde is handled by -e case
        return adj[:-2] + "l"
    if adj.endswith("er") and len(adj) > 3:
        # bitter → bittr  (but not for short words like "er")
        return adj[:-2] + "r"
    if adj.endswith("e"):
        # leise → leis, spröde → spröd
        return adj[:-1]
    return adj


class MetaphorMatcher:
    def __init__(self, knowledge: Any):
        self.kg = knowledge # KnowledgeGraph object
        self.forward = defaultdict(list) # subject -> list of (relation, object)
        self.reverse = defaultdict(list) # object -> list of (subject, relation)
        self._build_indices()

    def _build_indices(self):
        """Build forward and reverse indices from KnowledgeGraph nodes."""
        # kg.nodes is dict[str, KnowledgeNode]
        # KnowledgeNode.relations is list[tuple[str, str, np.ndarray]] -> (rel, target, vec)
        for concept, node in self.kg.nodes.items():
            for rel_type, target, _ in node.relations:
                self.forward[concept].append((rel_type, target))
                self.reverse[target].append((concept, rel_type))

    def find_metaphor(self, target: str, rng: random.Random) -> Metaphor | None:
        """Find a poetic metaphor for the target concept."""
        # 1. Get properties of the target
        target_lower = target.lower()
        target_props = self.forward.get(target_lower)
        if not target_props:
            return None

        # 2. Find candidate sources from ELEMENTAL_SOURCES that share properties
        candidates = []
        
        # Collect all shared properties per elemental source
        shared_by_source = defaultdict(list)
        
        for rel, obj in target_props:
            # Look for other subjects that relate to this object
            # We focus on properties (HAS, CAUSES, NEEDS, LEADSTO)
            if rel not in {"HAS", "CAUSES", "NEEDS", "LEADSTO", "ISA"}:
                continue
                
            others = self.reverse.get(obj, [])
            for other_subj, other_rel in others:
                if other_subj == target:
                    continue
                if other_subj in ELEMENTAL_SOURCES:
                    # Found an elemental source sharing this object!
                    shared_by_source[other_subj].append((rel, obj))

        # 3. Score and select candidates
        # We prefer sources with at least one meaningful shared property
        valid_sources = [s for s, props in shared_by_source.items() if props]
        if not valid_sources:
            return None
            
        # Pick a random source to avoid repetitiveness, 
        # but weighted by number of shared properties could be an improvement later
        source = rng.choice(valid_sources)
        shared_props = shared_by_source[source]
        
        # Pick one shared property to highlight
        # Prefer HAS > CAUSES > NEEDS > LEADSTO for the "explanation" part
        # "Liebe ist Feuer. Denn sie HAT Wärme." is stronger than "Denn sie BRAUCHT Wärme."
        def score_rel(r): 
            return {"HAS": 4, "CAUSES": 3, "LEADSTO": 2, "NEEDS": 1, "ISA": 1}.get(r, 0)
            
        shared_props.sort(key=lambda x: score_rel(x[0]), reverse=True)
        best_shared = shared_props[0]

        # 4. Find a DISTINCT property of the source for contrast/expansion
        # Something the Source has, but the Target does NOT explicitly have in the KG
        source_props = self.forward.get(source, [])
        target_obj_set = {p[1] for p in target_props}
        
        distinct_candidates = []
        for r, o in source_props:
            if o not in target_obj_set and r in {"CAUSES", "HAS", "LEADSTO", "OPPOSES"}:
                distinct_candidates.append((r, o))
        
        best_distinct = rng.choice(distinct_candidates) if distinct_candidates else None
        
        if not best_distinct:
            # Fallback: just use another property of source even if not strictly unique
            # It's poetic, strict logical distinctness isn't always required
            valid_source_props = [p for p in source_props if p[0] in {"CAUSES", "HAS"}]
            if valid_source_props:
                best_distinct = rng.choice(valid_source_props)
            else:
                return None # Metaphor needs expansion to be good

        return Metaphor(
            target=target,
            source=source,
            shared_prop=best_shared,
            distinct_prop=best_distinct
        )
