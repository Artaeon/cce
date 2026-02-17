"""
CCE Interactive CLI
====================

Interactive command-line interface for the Cognitive Crystallization Engine.
Provides phase-by-phase visualization of the crystallization process.
"""

from __future__ import annotations

import sys
from pathlib import Path


# ANSI color codes
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    WHITE = "\033[37m"
    
    BG_DARK = "\033[48;5;233m"


BANNER = f"""
{C.CYAN}{C.BOLD}
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║    ██████╗ ██████╗███████╗                                   ║
  ║   ██╔════╝██╔════╝██╔════╝                                   ║
  ║   ██║     ██║     █████╗   Cognitive Crystallization Engine  ║
  ║   ██║     ██║     ██╔══╝   ─────────────────────────────     ║
  ║   ╚██████╗╚██████╗███████╗ Phase transitions in concept      ║
  ║    ╚═════╝ ╚═════╝╚══════╝ space, not neural networks.      ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
{C.RESET}"""

HELP = f"""
{C.DIM}Befehle:{C.RESET}
  {C.YELLOW}/intent{C.RESET} <text>    — Konzepte direkt setzen (Stichwörter)
  {C.YELLOW}/emotion{C.RESET} <text>   — Emotionale Färbung
  {C.YELLOW}/context{C.RESET} <text>   — Situativer Kontext
  {C.YELLOW}/persona{C.RESET} <text>   — Sprechercharakter
  {C.YELLOW}/generate{C.RESET}        — Kristallisation aus gesetzten Slots starten
  {C.YELLOW}/demo{C.RESET}            — Vorgefertigte Demo laufen lassen
  {C.YELLOW}/reset{C.RESET}           — Gedächtnis zurücksetzen
  {C.YELLOW}/help{C.RESET}            — Diese Hilfe anzeigen
  {C.YELLOW}/quit{C.RESET}            — Beenden

{C.DIM}Oder einfach natürlichen deutschen Text eingeben —
der Parser extrahiert Intent, Emotion und Kontext automatisch.{C.RESET}
"""

DEMOS = [
    {
        "name": "Quartalsbericht",
        "intent": "Zahlen schlecht Trend positiv Wachstum",
        "emotion": "frustriert hoffnungsvoll",
        "context": "Quartalsbericht Vorstandssitzung",
        "persona": "direkt ehrlich analytisch",
    },
    {
        "name": "Unternehmer-Einsamkeit",
        "intent": "Einsamkeit Unternehmer Leere Fülle Idee Mensch verstehen Mangel",
        "emotion": "melancholisch kämpferisch sehnend",
        "context": "Reflexion Nacht",
        "persona": "tief ehrlich mutig",
    },
    {
        "name": "Mut und Angst",
        "intent": "Mut Angst Kraft Zweifel Weg Ziel",
        "emotion": "Angst Hoffnung Kraft",
        "context": "Entscheidung Risiko Chance",
        "persona": "ehrlich mutig",
    },
]


def _find_lexicon_path() -> Path:
    """Find the lexicon data file (prefer expanded lexicon)."""
    candidates = [
        Path("data/de_lexicon_5k.json"),
        Path(__file__).parent.parent / "data" / "de_lexicon_5k.json",
        Path("data/de_lexicon.json"),
        Path(__file__).parent.parent / "data" / "de_lexicon.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    print(f"{C.RED}Fehler: Lexikon nicht gefunden!{C.RESET}")
    print(f"Gesucht in: {[str(c) for c in candidates]}")
    sys.exit(1)


def _visualize_plasma(engine) -> None:
    """Visualize the plasma state."""
    field = engine._last_field
    if not field:
        return

    print(f"\n  {C.MAGENTA}{C.BOLD}🌊 PHASE 1: PLASMA{C.RESET}")
    print(f"  {C.DIM}{'─' * 50}{C.RESET}")

    # Show particle distribution
    from collections import Counter
    from cce.particle import ParticleCategory
    cats = Counter(p.category for p in field.particles)
    for cat in ParticleCategory:
        count = cats.get(cat, 0)
        bar = "█" * count + "░" * (20 - min(count, 20))
        color = {
            ParticleCategory.INTENT: C.CYAN,
            ParticleCategory.EMOTION: C.MAGENTA,
            ParticleCategory.CONTEXT: C.YELLOW,
            ParticleCategory.PERSONA: C.GREEN,
        }.get(cat, C.WHITE)
        print(f"    {color}{cat.name:10s}{C.RESET} [{bar}] {count}")

    # Show plasma visualization
    print(f"\n    {C.DIM}Plasma-Feld:{C.RESET}")
    import random
    random.seed(42)
    symbols = "~≈∿∾≋"
    for row in range(4):
        line = "    "
        for col in range(50):
            if random.random() < 0.15:
                line += f"{C.CYAN}◆{C.RESET}"
            else:
                line += f"{C.DIM}{random.choice(symbols)}{C.RESET}"
        print(line)

    print(f"\n    {C.DIM}Temperatur: {field.temperature:.3f}{C.RESET}")
    print(f"    {C.DIM}Partikel: {len(field)}{C.RESET}")


def _visualize_nucleation(engine) -> None:
    """Visualize the nucleation results."""
    nuclei = engine._last_nuclei
    if not nuclei:
        return

    print(f"\n  {C.YELLOW}{C.BOLD}💎 PHASE 2: KEIMBILDUNG{C.RESET}")
    print(f"  {C.DIM}{'─' * 50}{C.RESET}")

    for i, nucleus in enumerate(nuclei):
        strength_bar = "▓" * int(nucleus.strength * 2) + "░" * max(0, 20 - int(nucleus.strength * 2))
        emo_str = f"emo={nucleus.emotional_intensity:.2f}" if nucleus.emotional_intensity > 0 else "neutral"
        print(f"    {C.YELLOW}Keim {i+1}{C.RESET}: [{strength_bar}] {nucleus.label}")
        print(f"           Stärke: {nucleus.strength:.2f}  |  {emo_str}  |  Partikel: {nucleus.size}")


def _visualize_crystallization(engine) -> None:
    """Visualize the crystallization results."""
    crystals = engine._last_crystals
    if not crystals:
        return

    print(f"\n  {C.BLUE}{C.BOLD}❄️  PHASE 3: KRISTALLISATION{C.RESET}")
    print(f"  {C.DIM}{'─' * 50}{C.RESET}")

    shape_icons = {
        "simple": "□",
        "contrast": "◇",
        "parallel": "▣",
        "fragment": "▪",
    }

    for i, crystal in enumerate(crystals):
        icon = shape_icons.get(crystal.shape, "?")
        emo_color = C.GREEN if crystal.emotional_charge > 0 else (C.RED if crystal.emotional_charge < 0 else C.DIM)
        temp_bar = "🔥" * max(1, int(crystal.residual_temperature * 5))

        print(f"    {C.BLUE}Kristall {i+1}{C.RESET} {icon} [{crystal.shape}]")
        print(f"           Label: {crystal.label}")
        print(f"           Form: {crystal.shape} | Elemente: {crystal.element_count}")
        print(f"           Emotion: {emo_color}{crystal.emotional_charge:+.2f}{C.RESET}")
        print(f"           Energie: {temp_bar} ({crystal.residual_temperature:.2f})")


def _run_generation(engine, intent: str, emotion: str, context: str, persona: str) -> None:
    """Run generation with full visualization."""
    print(f"\n  {C.BOLD}{'═' * 56}{C.RESET}")
    print(f"  {C.BOLD}  Input{C.RESET}")
    print(f"  {C.BOLD}{'═' * 56}{C.RESET}")
    print(f"    Intent:  {C.CYAN}{intent}{C.RESET}")
    if emotion:
        print(f"    Emotion: {C.MAGENTA}{emotion}{C.RESET}")
    if context:
        print(f"    Kontext: {C.YELLOW}{context}{C.RESET}")
    if persona:
        print(f"    Persona: {C.GREEN}{persona}{C.RESET}")

    # Generate
    text = engine.generate(
        intent=intent,
        emotion=emotion,
        context=context,
        persona=persona,
    )

    # Visualize each phase
    _visualize_plasma(engine)
    _visualize_nucleation(engine)
    _visualize_crystallization(engine)

    # Final output
    print(f"\n  {C.GREEN}{C.BOLD}📝 PHASE 4: REALISIERUNG{C.RESET}")
    print(f"  {C.DIM}{'─' * 50}{C.RESET}")
    print()
    print(f"    {C.BOLD}{C.WHITE}» {text} «{C.RESET}")
    print()

    # Memory status
    ctx = engine.memory.get_context()
    print(f"  {C.DIM}🧠 Turn {ctx.turn_number} | Vermiedene Wörter: {len(ctx.avoid_words)}{C.RESET}")
    print()

    # Timing
    engine.print_logs()


def _run_nl_generation(engine, text: str) -> None:
    """Run generation from natural language text using the parser."""
    parsed = engine.parser.parse(text)

    print(f"\n  {C.DIM}{'─' * 50}{C.RESET}")
    print(f"  {C.BOLD}  🔍 Parsed:{C.RESET}")
    print(f"    Intent:  {C.CYAN}{parsed.intent or '(leer)'}{C.RESET}")
    if parsed.emotion:
        print(f"    Emotion: {C.MAGENTA}{parsed.emotion}{C.RESET}")
    if parsed.context:
        print(f"    Kontext: {C.YELLOW}{parsed.context}{C.RESET}")
    if parsed.persona:
        print(f"    Persona: {C.GREEN}{parsed.persona}{C.RESET}")
    if parsed.is_question:
        print(f"    Typ:     {C.YELLOW}Frage{C.RESET}")

    if not parsed.intent:
        print(f"    {C.RED}Kein Intent erkannt. Bitte deutlicher formulieren.{C.RESET}")
        return

    _run_generation(
        engine,
        intent=parsed.intent,
        emotion=parsed.emotion,
        context=parsed.context,
        persona=parsed.persona,
    )


def main() -> None:
    """Main CLI entry point."""
    print(BANNER)

    from cce.engine import CognitiveCrystallizationEngine

    lexicon_path = _find_lexicon_path()
    print(f"  {C.DIM}Lade Lexikon: {lexicon_path}{C.RESET}")
    
    engine = CognitiveCrystallizationEngine(lexicon_path=lexicon_path)
    print(f"  {C.GREEN}✓ Engine bereit ({engine.lexicon.size} Wörter, {engine.dim:,}D){C.RESET}")
    print(f"  {C.DIM}  Knowledge: {engine.knowledge.size} Relationen | Memory: aktiv{C.RESET}")
    print(HELP)

    # State
    intent = ""
    emotion = ""
    context = ""
    persona = ""

    while True:
        try:
            raw = input(f"  {C.CYAN}CCE ▸{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {C.DIM}Auf Wiedersehen.{C.RESET}")
            break

        if not raw:
            continue

        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit" or cmd == "/exit":
                print(f"  {C.DIM}Auf Wiedersehen.{C.RESET}")
                break
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/intent":
                intent = arg
                print(f"    {C.CYAN}Intent ← {arg}{C.RESET}")
            elif cmd == "/emotion":
                emotion = arg
                print(f"    {C.MAGENTA}Emotion ← {arg}{C.RESET}")
            elif cmd == "/context":
                context = arg
                print(f"    {C.YELLOW}Kontext ← {arg}{C.RESET}")
            elif cmd == "/persona":
                persona = arg
                print(f"    {C.GREEN}Persona ← {arg}{C.RESET}")
            elif cmd == "/generate":
                if not intent:
                    print(f"    {C.RED}Kein Intent gesetzt! Nutze /intent <text>{C.RESET}")
                else:
                    _run_generation(engine, intent, emotion, context, persona)
            elif cmd == "/demo":
                for i, demo in enumerate(DEMOS):
                    print(f"\n  {C.BOLD}{'━' * 56}{C.RESET}")
                    print(f"  {C.BOLD}  Demo {i+1}: {demo['name']}{C.RESET}")
                    print(f"  {C.BOLD}{'━' * 56}{C.RESET}")
                    _run_generation(
                        engine,
                        demo["intent"],
                        demo["emotion"],
                        demo["context"],
                        demo["persona"],
                    )
            elif cmd == "/reset":
                engine.memory.reset()
                intent = emotion = context = persona = ""
                print(f"    {C.GREEN}✓ Gedächtnis und Slots zurückgesetzt{C.RESET}")
            else:
                print(f"    {C.RED}Unbekannter Befehl: {cmd}{C.RESET}")
                print(f"    {C.DIM}Nutze /help für Hilfe{C.RESET}")
        else:
            # Natural language input → parse and generate
            _run_nl_generation(engine, raw)


if __name__ == "__main__":
    main()
