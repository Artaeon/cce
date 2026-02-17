"""
Resonant Lexicon
=================

Every word is stored not as a flat definition but as a rich profile:
semantic vector, rhythm, phonetics, emotion, register, bonds.

Word selection works by *resonance*: the crystal slot emits a
"vibration" and the word that vibrates most harmoniously wins.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from cce.codebook import Codebook


@dataclass
class WordProfile:
    """The multi-dimensional profile of a single word."""

    word: str
    vec: np.ndarray                  # Semantic HDC vector
    syllables: int = 1              # Rhythmic weight
    valence: float = 0.0            # Emotional valence: -1 (dark) to +1 (bright)
    arousal: float = 0.5            # Activation: 0 (calm) to 1 (intense)
    register: float = 0.5           # 0 = very informal, 0.5 = neutral, 1 = very formal
    pos: str = "NOUN"              # Part of speech
    gender: str = ""               # Grammatical gender: M, F, N (nouns only)
    bonds: list[str] = field(default_factory=list)    # Words that go well together
    repels: list[str] = field(default_factory=list)   # Words that clash

    @property
    def rhythm_weight(self) -> float:
        """Rhythmic weight — longer words have more weight."""
        return math.log1p(self.syllables)


class ResonantLexicon:
    """The word-store with resonance-based retrieval.

    Words are retrieved by multi-dimensional matching:
    semantics, rhythm, emotion, register, bonds.
    """

    def __init__(self, codebook: Codebook) -> None:
        self.codebook = codebook
        self.words: dict[str, WordProfile] = {}
        self._vec_cache: Optional[np.ndarray] = None
        self._word_list: list[str] = []

    def load(self, path: str | Path) -> None:
        """Load word profiles from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            word = entry["word"]
            vec = self.codebook.encode(word.lower())

            profile = WordProfile(
                word=word,
                vec=vec,
                syllables=entry.get("syllables", 1),
                valence=entry.get("valence", 0.0),
                arousal=entry.get("arousal", 0.5),
                register=entry.get("register", 0.5),
                pos=entry.get("pos", "NOUN"),
                gender=entry.get("gender", ""),
                bonds=entry.get("bonds", []),
                repels=entry.get("repels", []),
            )
            self.words[word.lower()] = profile

        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        """Build the vectorized lookup cache for fast nearest-neighbor."""
        self._word_list = list(self.words.keys())
        if self._word_list:
            vecs = np.stack([self.words[w].vec for w in self._word_list])
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            self._vec_cache = vecs / norms

    def semantic_neighbors(self, vec: np.ndarray, n: int = 50) -> list[str]:
        """Find the n closest words by cosine similarity."""
        if self._vec_cache is None or len(self._word_list) == 0:
            return []

        norm = np.linalg.norm(vec)
        if norm == 0:
            return self._word_list[:n]

        query = vec / norm
        sims = self._vec_cache @ query
        top_idx = np.argsort(-sims)[:n]
        return [self._word_list[i] for i in top_idx]

    def find_word(
        self,
        meaning_vec: np.ndarray,
        target_syllables: int = 2,
        target_emotion: float = 0.0,
        target_register: float = 0.5,
        target_arousal: float = 0.5,
        neighbor_words: Optional[list[str]] = None,
        pos_filter: Optional[str] = None,
        n_candidates: int = 30,
    ) -> tuple[str, float]:
        """Find the word that best resonates with the crystal slot.

        Returns (word, score).
        """
        candidates = self.semantic_neighbors(meaning_vec, n=n_candidates)

        if pos_filter:
            candidates = [
                w for w in candidates
                if self.words[w].pos == pos_filter
            ] or candidates[:5]

        if not candidates:
            return ("", 0.0)

        best_word = ""
        best_score = -1.0

        for w in candidates:
            profile = self.words[w]
            score = self._score_resonance(
                profile, meaning_vec, target_syllables,
                target_emotion, target_register, target_arousal,
                neighbor_words,
            )
            if score > best_score:
                best_score = score
                best_word = w

        return (best_word, best_score)

    def find_word_by_pos(
        self,
        meaning_vec: np.ndarray,
        pos: str,
        target_emotion: float = 0.0,
        target_register: float = 0.5,
    ) -> tuple[str, float]:
        """Convenience: find best word with a specific part of speech."""
        return self.find_word(
            meaning_vec=meaning_vec,
            pos_filter=pos,
            target_emotion=target_emotion,
            target_register=target_register,
        )

    def _score_resonance(
        self,
        profile: WordProfile,
        meaning_vec: np.ndarray,
        target_syllables: int,
        target_emotion: float,
        target_register: float,
        target_arousal: float,
        neighbor_words: Optional[list[str]],
    ) -> float:
        """Multi-dimensional resonance score.

        Weighted combination of:
        - Semantic similarity (0.30)
        - Rhythm match      (0.15)
        - Emotion match     (0.20)
        - Register match    (0.10)
        - Arousal match     (0.10)
        - Bond score        (0.15)
        """
        # Semantic similarity
        sem_sim = self.codebook.similarity(profile.vec, meaning_vec)
        sem_score = max(0, sem_sim)

        # Rhythm: prefer words close to target syllable count
        rhythm_diff = abs(profile.syllables - target_syllables)
        rhythm_score = 1.0 / (1.0 + rhythm_diff)

        # Emotion match
        emotion_diff = abs(profile.valence - target_emotion)
        emotion_score = 1.0 - min(emotion_diff, 1.0)

        # Register match
        register_diff = abs(profile.register - target_register)
        register_score = 1.0 - min(register_diff, 1.0)

        # Arousal match
        arousal_diff = abs(profile.arousal - target_arousal)
        arousal_score = 1.0 - min(arousal_diff, 1.0)

        # Bond score: bonus if this word bonds well with neighbors
        bond_score = 0.5  # Neutral default
        if neighbor_words:
            bond_count = sum(1 for n in neighbor_words if n in profile.bonds)
            repel_count = sum(1 for n in neighbor_words if n in profile.repels)
            bond_score = 0.5 + 0.25 * bond_count - 0.25 * repel_count
            bond_score = max(0, min(1, bond_score))

        return (
            0.30 * sem_score
            + 0.15 * rhythm_score
            + 0.20 * emotion_score
            + 0.10 * register_score
            + 0.10 * arousal_score
            + 0.15 * bond_score
        )

    def add_word(
        self,
        word: str,
        pos: str = "NOUN",
        valence: float = 0.0,
        arousal: float = 0.5,
        register: float = 0.5,
        syllables: int | None = None,
        bonds: list[str] | None = None,
    ) -> None:
        """Dynamically add a word to the lexicon."""
        key = word.lower()
        if key in self.words:
            return  # Already known

        vec = self.codebook.encode(key)
        if syllables is None:
            # Estimate syllables via vowel counting
            vowels = set("aeiouyäöü")
            count = 0
            in_vowel = False
            for ch in key:
                if ch in vowels:
                    if not in_vowel:
                        count += 1
                        in_vowel = True
                else:
                    in_vowel = False
            syllables = max(1, count)

        profile = WordProfile(
            word=word,
            vec=vec,
            syllables=syllables,
            valence=valence,
            arousal=arousal,
            register=register,
            pos=pos,
            bonds=bonds or [],
        )
        self.words[key] = profile
        self._rebuild_cache()

    def add_from_text(self, text: str) -> int:
        """Auto-register unknown words from input text.

        Uses suffix heuristics for POS detection. Returns count of new words.
        """
        added = 0
        for word in text.strip().split():
            key = word.lower().strip()
            if not key or key in self.words:
                continue

            # POS heuristics from German suffix patterns
            pos = "NOUN"  # Default
            if key.endswith(("lich", "isch", "bar", "sam", "haft", "los", "voll", "ig")):
                pos = "ADJ"
            elif key.endswith(("en", "ern", "eln", "ieren")):
                pos = "VERB"

            self.add_word(word, pos=pos)
            added += 1
        return added

    @property
    def size(self) -> int:
        return len(self.words)

    def get(self, word: str) -> Optional[WordProfile]:
        return self.words.get(word.lower())
