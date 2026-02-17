"""
Working Memory
===============

HDC-based dialogue context that persists across multiple generation calls.

Working memory stores:
- Recent topic vectors (what was discussed)
- Emotional trajectory (how emotion evolves)
- Used vocabulary (to avoid repetition)
- Discourse state (what structures were used)

The memory decays over time — older context fades, recent context
dominates — mimicking how human working memory functions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cce.codebook import Codebook


@dataclass
class MemoryFrame:
    """A single snapshot of context from one generation turn."""
    topic_vec: np.ndarray            # Bundled topic vector for this turn
    emotion: float                   # Emotional valence of this turn
    words_used: set[str]             # Words produced in this turn
    crystal_shapes: list[str]        # Shapes used (for structure variety)
    turn_id: int = 0                 # Sequential turn number
    weight: float = 1.0              # Decay weight (fades over time)


class WorkingMemory:
    """HDC-encoded dialogue context with natural decay.

    Maintains a sliding window of recent context and provides:
    - Topic continuity vectors (what to stay near)
    - Vocabulary exclusion sets (what not to repeat)
    - Emotional trajectory (where the mood is heading)
    - Structural variety hints (avoid same shapes)

    Usage::

        memory = WorkingMemory(codebook, capacity=8)
        # After each engine.generate() call:
        memory.commit(topic_vec, emotion, words_used, shapes)
        # Before next generation:
        context = memory.get_context()
    """

    def __init__(
        self,
        codebook: Codebook,
        capacity: int = 8,
        decay_rate: float = 0.75,
    ) -> None:
        self.codebook = codebook
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.frames: deque[MemoryFrame] = deque(maxlen=capacity)
        self._turn_counter = 0

    def commit(
        self,
        topic_vec: np.ndarray,
        emotion: float,
        words_used: set[str],
        crystal_shapes: list[str],
    ) -> None:
        """Record context from a completed generation turn."""
        self._turn_counter += 1

        frame = MemoryFrame(
            topic_vec=topic_vec.copy(),
            emotion=emotion,
            words_used=set(w.lower() for w in words_used),
            crystal_shapes=list(crystal_shapes),
            turn_id=self._turn_counter,
            weight=1.0,
        )
        self.frames.append(frame)

        # Apply decay to older frames
        for f in self.frames:
            if f.turn_id < self._turn_counter:
                f.weight *= self.decay_rate

    def get_context(self) -> MemoryContext:
        """Build the current dialogue context for the next generation."""
        if not self.frames:
            return MemoryContext.empty(self.codebook.dim)

        # Topic continuity: weighted bundle of recent topic vectors
        weights = np.array([f.weight for f in self.frames])
        total_w = weights.sum()
        if total_w > 0:
            weights /= total_w

        topic_vecs = np.stack([f.topic_vec for f in self.frames])
        # Weighted average, then binarize
        blended = (weights[:, None] * topic_vecs).sum(axis=0)
        topic_continuity = np.sign(blended).astype(np.float32)

        # Emotional trajectory
        emotions = [f.emotion for f in self.frames]
        if len(emotions) >= 2:
            # Trend: difference between recent and older
            recent = np.mean(emotions[-2:])
            older = np.mean(emotions[:-2]) if len(emotions) > 2 else emotions[0]
            emotion_trend = recent - older
        else:
            emotion_trend = 0.0
        emotion_current = emotions[-1] if emotions else 0.0

        # Vocabulary exclusion: union of recently used words (weighted by recency)
        avoid_words: set[str] = set()
        for f in self.frames:
            if f.weight > 0.3:  # Only recent-enough frames
                avoid_words.update(f.words_used)

        # Structural variety: shapes used recently
        recent_shapes: list[str] = []
        for f in list(self.frames)[-3:]:  # Last 3 turns
            recent_shapes.extend(f.crystal_shapes)

        return MemoryContext(
            topic_continuity=topic_continuity,
            emotion_current=emotion_current,
            emotion_trend=emotion_trend,
            avoid_words=avoid_words,
            recent_shapes=recent_shapes,
            turn_number=self._turn_counter,
        )

    def build_topic_vec(
        self,
        intent: str,
        emotion: str,
        context: str,
    ) -> np.ndarray:
        """Build a topic vector from input strings.

        Used to create the topic_vec for commit().
        """
        all_words = f"{intent} {emotion} {context}".strip().split()
        if not all_words:
            return np.zeros(self.codebook.dim, dtype=np.float32)

        vecs = [self.codebook.encode(w.lower()) for w in all_words]
        bundled = np.mean(vecs, axis=0)
        return np.sign(bundled).astype(np.float32)

    def reset(self) -> None:
        """Clear all memory."""
        self.frames.clear()
        self._turn_counter = 0

    @property
    def turn_count(self) -> int:
        return self._turn_counter

    @property
    def is_empty(self) -> bool:
        return len(self.frames) == 0


@dataclass
class MemoryContext:
    """The extracted dialogue context for the next generation turn."""
    topic_continuity: np.ndarray     # Where the conversation is
    emotion_current: float           # Current emotional valence
    emotion_trend: float             # Emotional direction
    avoid_words: set[str]            # Words to not repeat
    recent_shapes: list[str]         # Crystal shapes used recently
    turn_number: int                 # How many turns so far

    @staticmethod
    def empty(dim: int) -> MemoryContext:
        """Create empty context for first turn."""
        return MemoryContext(
            topic_continuity=np.zeros(dim, dtype=np.float32),
            emotion_current=0.0,
            emotion_trend=0.0,
            avoid_words=set(),
            recent_shapes=[],
            turn_number=0,
        )

    @property
    def has_context(self) -> bool:
        return self.turn_number > 0
