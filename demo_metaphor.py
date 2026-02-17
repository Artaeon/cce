"""Demo: Metaphor als Denkraum — Image Lexicons + Structural Variety."""

import random
from cce.engine import CognitiveCrystallizationEngine
from cce.metaphor import Metaphor, IMAGE_LEXICONS


def test_render_templates():
    """Test all 5 structural templates directly."""
    print("=" * 60)
    print("  TEMPLATE VARIETY TEST (direct render_as_denkraum)")
    print("=" * 60)

    # Synthetic metaphor: Freiheit → Meer
    m = Metaphor(
        target="freiheit",
        source="meer",
        shared_prop=("HAS", "Weite"),
        distinct_prop=("HAS", "Tiefe"),
    )

    for i in range(10):
        rng = random.Random(i)
        text = m.render_as_denkraum(rng)
        print(f"\n  [{i}] {text}")


def test_all_sources():
    """Test that every elemental source with a lexicon produces output."""
    print("\n" + "=" * 60)
    print("  IMAGE LEXICON COVERAGE (one per source)")
    print("=" * 60)

    for source in sorted(IMAGE_LEXICONS.keys()):
        m = Metaphor(
            target="liebe",
            source=source,
            shared_prop=("HAS", "Wärme"),
            distinct_prop=("CAUSES", "Schmerz"),
        )
        rng = random.Random(42)
        text = m.render_as_denkraum(rng)
        print(f"\n  [{source:>10}] {text}")


def test_full_engine():
    """End-to-end: full CCE pipeline with KG-seeded metaphors."""
    print("\n" + "=" * 60)
    print("  FULL ENGINE TEST (KG-driven metaphors)")
    print("=" * 60)

    engine = CognitiveCrystallizationEngine()

    # Seed KG
    engine.knowledge.add_relation("Liebe", "HAS", "Wärme")
    engine.knowledge.add_relation("Feuer", "HAS", "Wärme")
    engine.knowledge.add_relation("Feuer", "CAUSES", "Asche")

    engine.knowledge.add_relation("Freiheit", "HAS", "Weite")
    engine.knowledge.add_relation("Meer", "HAS", "Weite")
    engine.knowledge.add_relation("Meer", "HAS", "Tiefe")

    engine.knowledge.add_relation("Krieg", "CAUSES", "Schmerz")
    engine.knowledge.add_relation("Sturm", "CAUSES", "Schmerz")
    engine.knowledge.add_relation("Sturm", "CAUSES", "Zerstörung")

    engine.knowledge.add_relation("Stille", "HAS", "Ruhe")
    engine.knowledge.add_relation("Wald", "HAS", "Ruhe")
    engine.knowledge.add_relation("Wald", "HAS", "Schatten")

    # Rebuild indices
    engine.metaphor_matcher._build_indices()

    topics = ["Liebe", "Freiheit", "Krieg", "Stille"]
    for t in topics:
        found = False
        for attempt in range(10):
            res = engine.generate(intent=t, emotion="neutral")
            # Check if it contains image-world vocabulary
            if any(kw in res for kw in [
                "Welle", "Flamme", "Böe", "Wurzel", "Meer", "Feuer",
                "Sturm", "Wald", "Tiefe", "Glut", "Donner", "Dickicht",
                "ist ein", "ist eine", "Wie ein", "—",
            ]):
                print(f"\n  [{t}] DENKRAUM: {res}")
                found = True
                break
        if not found:
            print(f"\n  [{t}] Standard: {res}")


if __name__ == "__main__":
    test_render_templates()
    test_all_sources()
    test_full_engine()
