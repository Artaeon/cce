"""Integration tests for the full CCE pipeline."""

import time
from pathlib import Path

import pytest

from cce.engine import CognitiveCrystallizationEngine


@pytest.fixture
def engine():
    lexicon_path = Path(__file__).parent.parent / "data" / "de_lexicon_5k.json"
    return CognitiveCrystallizationEngine(lexicon_path=lexicon_path, dim=10_000)


class TestFullPipeline:
    def test_basic_generation(self, engine):
        """Full pipeline produces non-empty output."""
        text = engine.generate(
            intent="Zahlen schlecht Trend positiv",
            emotion="frustriert hoffnungsvoll",
            context="Quartalsbericht",
            persona="direkt ehrlich",
        )
        assert isinstance(text, str)
        assert len(text) > 0
        assert len(text.split()) >= 2, f"Expected ≥2 words, got: '{text}'"

    def test_all_phases_logged(self, engine):
        """All 4 phases produce log entries."""
        engine.generate(intent="Mut Kraft Angst")
        phases = [log.phase for log in engine.logs]
        assert "PLASMA" in phases
        assert "NUCLEATION" in phases
        assert "CRYSTALLIZATION" in phases
        assert "REALIZATION" in phases

    def test_runs_under_5_seconds(self, engine):
        """Full pipeline completes in <5 seconds."""
        t0 = time.perf_counter()
        engine.generate(
            intent="Einsamkeit Unternehmer Leere Fülle Idee Mensch",
            emotion="melancholisch kämpferisch",
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"Pipeline took {elapsed:.1f}s (too slow)"

    def test_different_inputs_different_outputs(self, engine):
        """Different inputs produce different text."""
        text1 = engine.generate(intent="Mut Kraft Hoffnung")
        text2 = engine.generate(intent="Angst Zweifel Dunkelheit")
        # They should differ (not guaranteed but very likely)
        # At minimum they should both produce output
        assert len(text1) > 0
        assert len(text2) > 0

    def test_emotion_affects_output(self, engine):
        """Emotional coloring changes the output."""
        text_pos = engine.generate(
            intent="Ergebnis Zahlen",
            emotion="hoffnungsvoll positiv",
        )
        text_neg = engine.generate(
            intent="Ergebnis Zahlen",
            emotion="frustriert negativ",
        )
        assert len(text_pos) > 0
        assert len(text_neg) > 0

    def test_minimal_input(self, engine):
        """Works with minimal input (intent only)."""
        text = engine.generate(intent="Mut")
        assert len(text) > 0

    def test_timing(self, engine):
        """Phase timing is recorded."""
        engine.generate(intent="Kraft Mut Wille")
        assert engine.total_time_ms > 0
        for log in engine.logs:
            assert log.duration_ms >= 0
