"""Tests for the Natural Language Input Parser (parser.py)."""

import pytest

from cce.codebook import Codebook
from cce.lexicon import ResonantLexicon
from cce.parser import InputParser


@pytest.fixture
def lexicon():
    from pathlib import Path
    cb = Codebook(dim=10_000)
    lex = ResonantLexicon(cb)
    path = Path(__file__).parent.parent / "data" / "de_lexicon_5k.json"
    lex.load(path)
    return lex


@pytest.fixture
def parser(lexicon):
    return InputParser(lexicon)


class TestInputParser:
    """Tests for natural language parsing."""

    def test_question_detection(self, parser):
        p = parser.parse("Was macht Mut so stark?")
        assert p.is_question

    def test_statement_not_question(self, parser):
        p = parser.parse("Mut ist Kraft.")
        assert not p.is_question

    def test_intent_extraction(self, parser):
        p = parser.parse("Die Kraft des Mutes überwindet die Angst")
        intent_lower = p.intent.lower()
        assert "kraft" in intent_lower
        assert "angst" in intent_lower

    def test_emotion_extraction(self, parser):
        p = parser.parse("Es ist frustrierend und hoffnungslos")
        # Adjectives with strong valence should be extracted as emotion
        assert len(p.emotion) > 0

    def test_persona_extraction(self, parser):
        p = parser.parse("Ehrlich gesagt, das ist schwierig")
        assert "ehrlich" in p.persona or "direkt" in p.persona

    def test_stop_words_excluded(self, parser):
        p = parser.parse("Ich bin nicht in der Lage zu verstehen")
        intent_words = p.intent.lower().split()
        assert "ich" not in intent_words
        assert "bin" not in intent_words
        assert "nicht" not in intent_words

    def test_empty_input(self, parser):
        p = parser.parse("")
        assert p.intent == ""
        assert p.emotion == ""

    def test_parsed_repr(self, parser):
        p = parser.parse("Mut und Angst")
        result = repr(p)
        assert "intent=" in result

    def test_full_roundtrip(self, parser):
        """Full parse → generate would need engine, so just verify
        the parse produces usable output."""
        p = parser.parse("Warum fällt es so schwer, mutig zu sein in Zeiten der Angst?")
        assert p.is_question
        assert len(p.intent) > 0

    def test_parser_without_lexicon(self):
        """Parser should work without a lexicon, using fallback heuristics."""
        parser = InputParser(lexicon=None)
        p = parser.parse("Die Wirtschaft wächst trotz Krise")
        assert len(p.intent) > 0
