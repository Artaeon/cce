"""
Natural Language Input Parser
==============================

Extracts semantic slots (intent, emotion, context, persona) from
natural German text input.

This enables the CCE to accept free-text queries instead of
structured keyword inputs. It uses:
- Lexicon-aware POS tagging (nouns/verbs → intent, adjectives → emotion)
- Sentiment analysis via word valence profiles
- Question detection and discourse markers
- Named entity patterns (for context extraction)

No external NLP dependencies — pure rule-based with lexicon support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from cce.lexicon import ResonantLexicon


# Question markers
QUESTION_WORDS = {
    "was", "wer", "wie", "wo", "warum", "wann", "woher", "wohin",
    "welche", "welcher", "welches", "wessen", "wem", "wen",
    "wieso", "weshalb",
}

# Discourse markers that indicate speaker attitude (persona)
PERSONA_MARKERS = {
    "ich denke": "nachdenklich",
    "ich glaube": "überzeugend",
    "ich finde": "direkt",
    "meiner meinung nach": "direkt ehrlich",
    "ehrlich gesagt": "direkt ehrlich",
    "offen gesagt": "direkt ehrlich",
    "ich frage mich": "nachdenklich zweifelnd",
    "man muss": "bestimmt direkt",
    "wir sollten": "kooperativ",
    "wir müssen": "dringend direkt",
    "stell dir vor": "visionär kreativ",
}

# Context indicator phrases
CONTEXT_PATTERNS = [
    (r"(?:im|beim|zum)\s+(meeting|gespräch|bericht|quartal|projekt)", "business"),
    (r"(?:in der|bei der)\s+(arbeit|firma|schule|universität)", "beruf"),
    (r"(?:zu hause|zuhause|privat|persönlich)", "privat"),
    (r"(?:nachts|abends|morgens|am morgen)", "reflexion"),
]

# Emotion intensifiers
INTENSIFIERS = {"sehr", "extrem", "unglaublich", "total", "absolut",
                "zutiefst", "wahnsinnig", "furchtbar", "unheimlich"}
DIMINISHERS = {"etwas", "leicht", "ein bisschen", "vielleicht",
               "kaum", "wenig", "eher"}

# Stop words to exclude from intent extraction
STOP_WORDS = {
    "ich", "du", "er", "sie", "es", "wir", "ihr", "mein", "dein",
    "sein", "unser", "euer", "sich", "man", "mir", "dir", "uns",
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen",
    "einem", "einer", "eines", "kein", "keine", "keinen", "keinem",
    "und", "oder", "aber", "sondern", "denn", "weil", "dass", "wenn",
    "ob", "als", "wie", "so", "zu", "von", "mit", "für", "auf",
    "an", "in", "aus", "um", "über", "nach", "bei", "durch", "vor",
    "bis", "ohne", "gegen", "zwischen", "unter", "neben", "hinter",
    "am", "im", "zum", "zur", "vom", "beim", "ans", "ins",
    "nicht", "noch", "schon", "auch", "nur", "erst", "mal", "doch",
    "ja", "nein", "halt", "eben", "nun", "dann", "da", "hier",
    "dort", "jetzt", "heute", "morgen", "gestern", "immer", "nie",
    "bin", "bist", "ist", "sind", "seid", "hat", "habe", "haben",
    "wird", "werden", "kann", "muss", "soll", "will", "darf",
    "wurde", "hatte", "konnte", "musste", "sollte", "wollte",
    "etwas", "alles", "nichts", "was", "wer", "wie", "wo",
}


@dataclass
class ParsedInput:
    """Structured extraction from natural language input."""
    intent: str           # Core concepts (nouns, verbs)
    emotion: str          # Emotional coloring (adjectives, sentiment words)
    context: str          # Situational context
    persona: str          # Speaker character traits
    is_question: bool     # Whether the input is a question
    question_type: str    # Type: warum/was/wie/wer/kennst_du/general/""
    raw_text: str         # Original input

    def __repr__(self) -> str:
        parts = [f"intent='{self.intent}'"]
        if self.emotion:
            parts.append(f"emotion='{self.emotion}'")
        if self.context:
            parts.append(f"context='{self.context}'")
        if self.persona:
            parts.append(f"persona='{self.persona}'")
        if self.is_question:
            qt = self.question_type or 'general'
            parts.append(f"question='{qt}'")
        return f"ParsedInput({', '.join(parts)})"


class InputParser:
    """Extracts CCE semantic slots from free German text.

    Usage::

        parser = InputParser(lexicon)
        parsed = parser.parse("Was macht den Mut so stark in Zeiten der Angst?")
        engine.generate(
            intent=parsed.intent,
            emotion=parsed.emotion,
            context=parsed.context,
            persona=parsed.persona,
        )
    """

    def __init__(self, lexicon: Optional[ResonantLexicon] = None) -> None:
        self.lexicon = lexicon

    def parse(self, text: str) -> ParsedInput:
        """Parse natural German text into CCE input slots."""
        text = text.strip()
        lower = text.lower()

        is_question = self._detect_question(text, lower)
        question_type = self._detect_question_type(lower) if is_question else ""
        persona = self._extract_persona(lower)
        context = self._extract_context(lower)
        emotion_words = self._extract_emotion(text, lower)
        intent_words = self._extract_intent(text, lower, emotion_words)

        return ParsedInput(
            intent=" ".join(intent_words),
            emotion=" ".join(emotion_words),
            context=context,
            persona=persona,
            is_question=is_question,
            question_type=question_type,
            raw_text=text,
        )

    def _detect_question(self, text: str, lower: str) -> bool:
        """Detect whether input is a question."""
        if text.endswith("?"):
            return True
        first_word = lower.split()[0] if lower.split() else ""
        return first_word in QUESTION_WORDS

    def _detect_question_type(self, lower: str) -> str:
        """Classify question type for response framing.

        Returns one of: warum, was, wie, wer, kennst_du, general
        """
        # Check for "kennst du" pattern first (familiarity questions)
        if "kennst du" in lower or "kennen sie" in lower:
            return "kennst_du"

        words = lower.split()
        if not words:
            return "general"

        first = words[0].rstrip("?")
        if first in {"warum", "wieso", "weshalb"}:
            return "warum"
        elif first in {"was", "welche", "welcher", "welches"}:
            return "was"
        elif first in {"wie"}:
            return "wie"
        elif first in {"wer", "wem", "wen", "wessen"}:
            return "wer"
        elif first in {"wo", "woher", "wohin"}:
            return "wo"
        elif first in {"wann"}:
            return "wann"
        return "general"

    def _extract_persona(self, lower: str) -> str:
        """Extract speaker persona from discourse markers."""
        personas = []
        for marker, persona in PERSONA_MARKERS.items():
            if marker in lower:
                personas.extend(persona.split())
        return " ".join(sorted(set(personas))) if personas else ""

    def _extract_context(self, lower: str) -> str:
        """Extract situational context from patterns."""
        contexts = []
        for pattern, ctx in CONTEXT_PATTERNS:
            if re.search(pattern, lower):
                contexts.append(ctx)
        return " ".join(contexts)

    def _extract_emotion(self, text: str, lower: str) -> list[str]:
        """Extract emotional words from the text.

        Uses lexicon valence when available, otherwise relies on
        adjective suffix patterns.
        """
        words = text.split()
        emotions: list[str] = []
        intensified = False

        for i, word in enumerate(words):
            clean = word.strip(".,!?;:\"'()[]{}").lower()
            if not clean or clean in STOP_WORDS:
                # Check if it's an intensifier
                if clean in INTENSIFIERS:
                    intensified = True
                continue

            # Check lexicon for emotional words
            is_emotional = False
            if self.lexicon:
                profile = self.lexicon.get(clean)
                if profile:
                    # High arousal or strong valence = emotional
                    if abs(profile.valence) > 0.35 and profile.pos == "ADJ":
                        is_emotional = True
                    elif profile.arousal > 0.6 and profile.pos == "ADJ":
                        is_emotional = True

            # Fallback: adjective suffix patterns
            if not is_emotional:
                adj_suffixes = ("lich", "isch", "ig", "haft", "sam", "los",
                                "voll", "bar", "los")
                if clean.endswith(adj_suffixes) and clean not in STOP_WORDS:
                    is_emotional = True

            if is_emotional:
                emotions.append(clean)
                if intensified:
                    emotions.append("intensiv")
                    intensified = False

        return emotions

    def _extract_intent(
        self,
        text: str,
        lower: str,
        emotion_words: list[str],
    ) -> list[str]:
        """Extract core intent concepts (nouns, action verbs).

        Everything that isn't a stop word, emotion word, or question word.
        """
        words = text.split()
        intents: list[str] = []
        emotion_set = set(emotion_words)
        seen: set[str] = set()

        for word in words:
            clean = word.strip(".,!?;:\"'()[]{}").lower()
            if not clean or len(clean) < 2:
                continue
            if clean in STOP_WORDS or clean in QUESTION_WORDS:
                continue
            if clean in emotion_set:
                continue
            if clean in seen:
                continue

            # Check lexicon for relevance
            if self.lexicon:
                profile = self.lexicon.get(clean)
                if profile:
                    # Nouns and significant verbs are intent
                    if profile.pos in ("NOUN", "VERB"):
                        # Capitalize nouns for the intent
                        display = clean.capitalize() if profile.pos == "NOUN" else clean
                        intents.append(display)
                        seen.add(clean)
                        continue

            # Fallback: include if it looks like a content word
            if len(clean) >= 3 and clean not in STOP_WORDS:
                intents.append(clean.capitalize())
                seen.add(clean)

        return intents
