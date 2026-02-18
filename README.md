<h1 align="center">
  ❄️ CCE — Cognitive Crystallization Engine
</h1>

<p align="center">
  <strong>Deterministic Text Composition via Phase Transitions in Concept Space</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/GPU-not_required-green?style=flat-square" alt="No GPU">
  <img src="https://img.shields.io/badge/latency-<10ms-orange?style=flat-square" alt="Latency">
  <img src="https://img.shields.io/badge/knowledge_graph-996_relations-purple?style=flat-square" alt="Relations">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen?style=flat-square" alt="MIT License">
</p>

---

## Abstract

CCE is a **deterministic text composition engine** that transforms abstract concepts into poetisch-philosophische German short-form prose — not through statistical token prediction, but through a physically-inspired crystallization process operating in a 10,000-dimensional hypervector space.

The system requires **no neural networks, no GPU, and no API calls**. All outputs are fully reproducible given the same seed, traceable through every processing stage, and generated in under 10 ms on commodity hardware.

> *Ist bedeutungsvolle Sprachgenerierung ohne statistische Modelle möglich?*
>
> — The founding question behind this project.

```
» Stille — ein Wald — birgt Ruhe zwischen dichten Blättern.
  Doch in seinem Schatten lauert Dunkelheit.
  Und darin zeigt sich: Aus Stille entsteht Erkenntnis. «

  ⏱  5.8ms  |  CPU only  |  0 API calls  |  0 tokens
```

---

## 1. Motivation

Large Language Models (LLMs) achieve impressive fluency but rely on stochastic token sampling, external infrastructure, and opaque internal representations. CCE explores the opposite end of the design spectrum:

| Property | LLM | CCE |
|---|---|---|
| Epistemic honesty | ❌ Halluziniert | ✅ Schweigt ehrlich |
| Termination criterion | ❌ Redet weiter | ✅ Apoptose |
| Reproducibility | ❌ Stochastisch | ✅ Deterministisch (same seed) |
| Latency | 500 ms – 5 s | **< 10 ms** |
| Dependencies | Cloud · GPU · API key | **numpy** |

---

## 2. Architecture

CCE models language generation as a **physical phase transition** in four successive stages. Each stage maps to a well-defined computational module:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   🌊 PLASMA      │────▷│  💎 NUCLEATION   │────▷│ ❄️ CRYSTALLIZE   │────▷│ 📝 REALIZATION   │
│                  │     │                  │     │                  │     │                  │
│  Concepts as     │     │  Semantic        │     │  Seeds grow into │     │  Crystals are    │
│  high-energy     │     │  clusters form   │     │  crystal         │     │  translated into │
│  particles in    │     │  nucleation      │     │  structures with │     │  grammatically   │
│  10,000-d space  │     │  seeds           │     │  defined shape   │     │  correct German  │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
        plasma.py              nucleation.py          crystallization.py        realization.py
```

### 2.1 Phase I — Plasma

Input concepts are projected into a **10,000-dimensional vector space** using Hyperdimensional Computing (HDC). A temperature parameter governs the breadth of semantic association: high temperature activates distant analogies, low temperature constrains output to closely related concepts.

### 2.2 Phase II — Nucleation

Particles with high semantic proximity aggregate into **nucleation seeds**. Binding forces are supplied by a curated knowledge graph containing **996 relations** across five primary link types: `HAS`, `CAUSES`, `OPPOSES`, `NEEDS`, and `LEADSTO`.

### 2.3 Phase III — Crystallization

Seeds grow into crystal structures with a **defined morphology** — parallel, entangled, or cascading. Crystal geometry directly determines the syntactic template that will be used in the final text.

### 2.4 Phase IV — Realization

Crystals are translated into grammatically correct German text. A library of **18 Bildwelten** (image domains: ocean, fire, forest, storm, ice, night, …) provides domain-specific vocabulary. Metaphors operate as *Denkräume* — coherent conceptual spaces rather than surface-level decorations.

---

## 3. Key Concepts

**Metaphor as Denkraum.** Metaphors are not labels ("X is Y") but entire image-worlds in which the engine *thinks*. Selecting "Meer" (ocean) activates waves, depth, surf, currents — all output vocabulary remains within that conceptual domain.

**Apoptosis.** Text generation terminates when the strongest rhetorical point has been made — not when source material is exhausted. Deliberate silence is a feature, not a deficiency.

**Epistemic Honesty.** The engine does not hallucinate. When it lacks knowledge of a concept, it produces less output rather than fabricated content.

**Evolvable Skin.** The physics layers (Plasma, Nucleation, Crystallization) have remained unchanged since v1. All improvements are applied in the Realization layer. Community contributions — new lexicon entries, image domains, templates — require no changes to the core pipeline.

---

## 4. Getting Started

### 4.1 Installation

```bash
git clone https://github.com/Artaeon/cce.git
cd cce
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 4.2 Interactive CLI

```bash
python -m cce.cli
```

The CLI accepts natural German input. The parser automatically extracts intent, emotion, and context:

```
CCE ▸ Was ist Liebe?

  » Liebe ist ein Feuer. Seine Flammen nähren Wärme.
    Und dennoch schwelt in ihm Schmerz.
    Was das bedeutet: Liebe bekämpft kalten Hass. «

  ⏱  5.8ms  |  📊 841 Wörter  |  ✅ 996 Relationen
```

```
CCE ▸ Erzähl mir etwas Trauriges über Einsamkeit
CCE ▸ Was bedeutet Freiheit?
CCE ▸ /emotion dunkel
CCE ▸ /intent Krieg
```

### 4.3 Programmatic API

```python
from cce.engine import CognitiveCrystallizationEngine

engine = CognitiveCrystallizationEngine()

output = engine.generate(intent="Stille", emotion="neutral")
# → Stille — ein Wald — birgt Ruhe zwischen dichten Blättern.
#   Doch in seinem Schatten lauert Dunkelheit.
#   Und darin zeigt sich: Aus Stille entsteht Erkenntnis.

output = engine.generate(intent="Liebe Schmerz", emotion="dunkel")
# → Liebe weckt Schmerz. Und gerade deshalb: sie verlangt Mut.
```

---

## 5. Sample Outputs

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

### Multi-Turn Dialogue

```
CCE ▸ Kennst du Angst?
  » Angst sät Zweifel. Denn sie braucht Mut.
    Und genau darin wächst sie zu Erkenntnis. «

CCE ▸ Und was ist das Gegenteil?
  » Mut bekämpft Angst. Denn er ruht auf Vertrauen. «
```

---

## 6. Technical Specifications

| Metric | Value |
|---|---|
| Source code | ~5,800 lines Python |
| Modules | 16 |
| Knowledge graph | 996 relations |
| Lexicon | 841 words |
| Image domains (Bildwelten) | 18 |
| Metaphor templates | 5 structural variants |
| HDC dimensionality | 10,000 |
| Median latency | < 10 ms (CPU) |
| Runtime dependencies | numpy |
| GPU required | No |
| External API calls | 0 |
| Training cost | € 0 |

---

## 7. Project Structure

```
cce/
├── engine.py          # Pipeline orchestration (4 phases)
├── plasma.py          # HDC vector space + temperature
├── particle.py        # Particle representation
├── nucleation.py      # Seed formation from particle clusters
├── crystallization.py # Crystal growth + morphology
├── realization.py     # Crystal → German text
├── metaphor.py        # 18 Bildwelten + 5 templates
├── knowledge.py       # Knowledge graph (996 relations)
├── lexicon.py         # Resonance lexicon (841 words)
├── grammar.py         # German grammar engine
├── memory.py          # Working memory + avoidance
├── parser.py          # Natural-language input parser
├── codebook.py        # HDC codebook vectors
├── templates.py       # Sentence structure templates
├── cli.py             # Interactive CLI
└── __init__.py
```

---

## 8. Scope & Limitations

CCE is not a general-purpose language model. The engine covers a **philosophical-poetic domain** — abstract concepts such as love, freedom, silence, and war. It does not replace LLMs for everyday questions, code generation, or factual retrieval. All output is generated exclusively in German.

> What it cannot do, it does not attempt. This is by design.

---

## 9. Contributing

New Bildwelten, lexicon entries, and knowledge-graph relations are welcome. The physics layers (Plasma, Nucleation, Crystallization) require no modification — all linguistic evolution happens in the Realization layer.

## License

MIT

---

<p align="center">
  <em>Built without a single neural network.<br>
  Every sentence is traceable, reproducible, and self-explanatory.</em>
</p>
