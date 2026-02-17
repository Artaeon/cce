<h1 align="center">
  ❄️ CCE — Cognitive Crystallization Engine
</h1>

<p align="center">
  <em>Konzeptbasierte Textkomposition durch Phasenübergänge im Konzeptraum.</em><br>
  <em>Ohne neuronale Netze. Ohne GPU. Ohne Halluzinationen.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/GPU-nicht_benötigt-green?style=flat-square" alt="No GPU">
  <img src="https://img.shields.io/badge/latenz-<10ms-orange?style=flat-square" alt="Latency">
  <img src="https://img.shields.io/badge/relationen-996-purple?style=flat-square" alt="Relations">
</p>

---

```
» Stille — ein Wald — birgt Ruhe zwischen dichten Blättern.
  Doch in seinem Schatten lauert Dunkelheit.
  Und darin zeigt sich: Aus Stille entsteht Erkenntnis. «

  ⏱  5.8ms  |  CPU only  |  0 API calls  |  0 tokens
```

---

## Was ist CCE?

CCE ist eine **deterministische Textkompositions-Engine**, die Konzepte in poetisch-philosophische deutsche Kurzformen verwandelt — nicht durch statistische Token-Vorhersage, sondern durch einen physikalisch inspirierten Kristallisationsprozess.

Das Projekt entstand aus einer einfachen Frage: *Ist bedeutungsvolle Sprachgenerierung ohne statistische Modelle möglich?*

**Der Unterschied zu LLMs:**

| | LLM | CCE |
|---|---|---|
| Weiß, was es nicht weiß | ❌ Halluziniert | ✅ Schweigt ehrlich |
| Weiß, wann es aufhören soll | ❌ Redet weiter | ✅ Apoptose |
| Reproduzierbar | ❌ Stochastisch | ✅ Deterministisch bei gleichem Seed |
| Latenz | 500ms–5s | **< 10ms** |
| Abhängigkeiten | Cloud, GPU, API-Key | **numpy** |

## Architektur

CCE modelliert Sprache als physikalischen Phasenübergang in vier Stufen:

<<<<<<< HEAD
```mermaid
graph LR
    A["🌊 Plasma<br/>Konzepte als Hochenergie-Partikel"] --> B["💎 Keimbildung<br/>Semantische Cluster formen Keime"]
    B --> C["❄️ Kristallisation<br/>Keime wachsen zu Kristallstrukturen"]
    C --> D["📝 Realisierung<br/>Kristalle werden zu deutschem Text"]

    style A fill:#ff6b6b,stroke:#333,color:#fff
    style B fill:#feca57,stroke:#333,color:#333
    style C fill:#48dbfb,stroke:#333,color:#333
    style D fill:#ff9ff3,stroke:#333,color:#333
=======
```
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  🌊 PLASMA  │───▶│ 💎 KEIMBILD.│───▶│ ❄️ KRISTALL.│───▶│ 📝 REALIS.  │
  │             │    │             │    │             │    │             │
  │  Konzepte   │    │  Semantische│    │  Keime      │    │  Kristalle  │
  │  als HDC-   │    │  Cluster    │    │  wachsen zu │    │  werden zu  │
  │  Partikel   │    │  formen     │    │  Kristall-  │    │  deutschem  │
  │  (10.000-d) │    │  Keime      │    │  strukturen │    │  Text       │
  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
>>>>>>> 6d5970a (docs: release README with architecture, examples, and philosophy)
```

**Plasma** — Eingabekonzepte werden in einen 10.000-dimensionalen Vektorraum projiziert (Hyperdimensional Computing). Temperatur bestimmt die Assoziationsweite.

**Keimbildung** — Partikel mit semantischer Nähe bilden Keime. Ein Knowledge Graph mit 996 Relationen liefert die Bindungskräfte (HAS, CAUSES, OPPOSES, NEEDS, LEADSTO).

**Kristallisation** — Keime wachsen zu Kristallen mit definierter Form (parallel, verschränkt, kaskadierend). Die Form bestimmt die spätere Satzstruktur.

**Realisierung** — Kristalle werden in grammatisch korrektes Deutsch übersetzt. 18 Bildwelten (Meer, Feuer, Wald, Sturm…) liefern domänenspezifisches Vokabular für Metaphern als *Denkräume*.

## Installation

```bash
git clone https://github.com/your-org/cce.git
cd cce
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Benutzung

### Interaktive CLI

```bash
python -m cce.cli
```

```
CCE ▸ Was ist Liebe?

  » Liebe ist ein Feuer. Seine Flammen nähren Wärme.
    Und dennoch schwelt in ihm Schmerz.
    Was das bedeutet: Liebe bekämpft kalten Hass. «

  ⏱  5.8ms  |  📊 841 Wörter  |  ✅ 996 Relationen
```

### Natürliche Eingabe

Die CLI versteht natürliches Deutsch. Der Parser extrahiert Intent, Emotion und Kontext automatisch:

```
CCE ▸ Erzähl mir etwas Trauriges über Einsamkeit
CCE ▸ Was bedeutet Freiheit?
CCE ▸ /emotion dunkel
CCE ▸ /intent Krieg
```

### Programmatische API

```python
from cce.engine import CognitiveCrystallizationEngine

engine = CognitiveCrystallizationEngine()

# Neutral
output = engine.generate(intent="Stille", emotion="neutral")
# → Stille — ein Wald — birgt Ruhe zwischen dichten Blättern.
#   Doch in seinem Schatten lauert Dunkelheit.
#   Und darin zeigt sich: Aus Stille entsteht Erkenntnis.

# Stimmungsvariation
output = engine.generate(intent="Liebe Schmerz", emotion="dunkel")
# → Liebe weckt Schmerz. Und gerade deshalb: sie verlangt Mut.
```

## Beispiel-Outputs

```
Stille
  Stille — ein Wald — birgt Ruhe zwischen dichten Blättern.
  Doch in seinem Schatten lauert Dunkelheit.
  Und darin zeigt sich: Aus Stille entsteht Erkenntnis.

Liebe
  Liebe ist ein Feuer. Seine Flammen nähren Wärme.
  Und dennoch schwelt in ihm Schmerz.

Krieg
  Krieg verdrängt zarten Frieden.
  Denn er sät stummes Leid.
  Und so bringt er kalte Zerstörung hervor.
```

### Interaktiver Dialog

```
CCE ▸ Kennst du Angst?
  » Angst sät Zweifel. Denn sie braucht Mut.
    Und genau darin wächst sie zu Erkenntnis. «

CCE ▸ Und was ist das Gegenteil?
  » Mut bekämpft Angst. Denn er ruht auf Vertrauen. «
```

## Technische Daten

| Metrik | Wert |
|--------|------|
| Quelltext | ~5.800 Zeilen Python |
| Module | 16 |
| Knowledge Graph | 996 Relationen |
| Lexikon | 841 Wörter |
| Bildwelten | 18 (Meer, Feuer, Wald, Sturm, Eis, Nacht…) |
| Metapher-Templates | 5 Strukturvarianten |
| HDC-Dimension | 10.000 |
| Median-Latenz | < 10ms (CPU) |
| Abhängigkeiten | numpy |
| GPU | Nicht benötigt |
| API-Calls | 0 |
| Trainingskosten | 0 € |

## Kernideen

**Metapher als Denkraum** — Metaphern sind keine Etiketten ("X ist Y"), sondern Bildwelten in denen die Engine *denkt*. "Meer" aktiviert Wellen, Tiefe, Brandung, Strömung — das gesamte Vokabular bleibt im Bild.

**Apoptose** — Der Text endet wenn der stärkste Punkt gemacht ist, nicht wenn das Material erschöpft ist. Bewusstes Schweigen ist eine Fähigkeit, kein Mangel.

**Ehrliches Nichtwissen** — Die Engine halluziniert nicht. Wenn sie ein Konzept nicht kennt, produziert sie weniger Output statt falschen.

**Evolvierbare Haut** — Die Physikschichten (Plasma, Keimbildung, Kristallisation) sind seit v1 unverändert. Jede Verbesserung geschieht in der Realisierungsschicht. Community-Beiträge — neue Lexikon-Einträge, Bildwelten, Templates — erfordern keine Änderungen am Kern.

## Projektstruktur

```
cce/
├── engine.py          # Orchestrierung der 4 Phasen
├── plasma.py          # HDC-Vektorraum + Temperatur
├── particle.py        # Partikel-Repräsentation
├── nucleation.py      # Keimbildung aus Partikel-Clustern
├── crystallization.py # Kristallwachstum + Formbestimmung
├── realization.py     # Kristall → deutscher Text
├── metaphor.py        # 18 Bildwelten + 5 Templates
├── knowledge.py       # Knowledge Graph (996 Relationen)
├── lexicon.py         # Resonanzlexikon (841 Wörter)
├── grammar.py         # Deutsche Grammatik-Engine
├── memory.py          # Working Memory + Vermeidung
├── parser.py          # NL-Eingabeparser
├── codebook.py        # HDC Codebook-Vektoren
├── templates.py       # Satzstruktur-Templates
├── cli.py             # Interaktive CLI
└── __init__.py
```

## Grenzen

CCE ist kein Allzweck-Sprachmodell. Die Engine deckt ein **philosophisch-poetisches Terrain** ab — abstrakte Konzepte wie Liebe, Freiheit, Stille, Krieg. Sie ersetzt kein LLM für Alltagsfragen, Codegeneration oder Faktenwissen. Sie generiert ausschließlich Deutsch.

Was sie nicht kann, tut sie nicht. Das ist Absicht.

## Beitragen

Neue Bildwelten, Lexikon-Einträge und Knowledge-Graph-Relationen sind willkommen. Die Physikschichten (Plasma, Keimbildung, Kristallisation) müssen dafür nicht verändert werden — alles Sprachliche lebt in der Realisierungsschicht.

## Lizenz

MIT

---

<p align="center">
  <em>Gebaut ohne ein einziges neuronales Netz.<br>
  Jeder Satz ist nachvollziehbar, reproduzierbar, und erklärt sich selbst.</em>
</p>
