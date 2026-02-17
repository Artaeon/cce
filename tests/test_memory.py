"""Tests for the Working Memory module (memory.py)."""

import pytest
import numpy as np

from cce.codebook import Codebook
from cce.memory import WorkingMemory, MemoryContext


@pytest.fixture
def codebook():
    return Codebook(dim=10_000)


@pytest.fixture
def memory(codebook):
    return WorkingMemory(codebook, capacity=4, decay_rate=0.5)


class TestWorkingMemory:
    """Tests for dialogue memory."""

    def test_starts_empty(self, memory):
        assert memory.is_empty
        assert memory.turn_count == 0

    def test_commit_increments_turn(self, memory, codebook):
        vec = codebook.encode("test")
        memory.commit(vec, 0.5, {"wort"}, ["simple"])
        assert memory.turn_count == 1
        assert not memory.is_empty

    def test_context_empty_before_commit(self, memory):
        ctx = memory.get_context()
        assert not ctx.has_context
        assert ctx.turn_number == 0
        assert len(ctx.avoid_words) == 0

    def test_context_after_commit(self, memory, codebook):
        vec = codebook.encode("mut")
        memory.commit(vec, 0.6, {"mut", "kraft"}, ["contrast"])
        ctx = memory.get_context()
        assert ctx.has_context
        assert ctx.turn_number == 1
        assert "mut" in ctx.avoid_words
        assert "kraft" in ctx.avoid_words
        assert ctx.emotion_current == 0.6

    def test_decay_reduces_weight(self, memory, codebook):
        vec1 = codebook.encode("alt")
        memory.commit(vec1, 0.3, {"alt"}, ["simple"])
        vec2 = codebook.encode("neu")
        memory.commit(vec2, 0.8, {"neu"}, ["contrast"])

        # After second commit, first frame should be decayed
        assert memory.frames[0].weight < 1.0
        assert memory.frames[1].weight == 1.0

    def test_capacity_limit(self, memory, codebook):
        for i in range(10):
            vec = codebook.encode(f"word{i}")
            memory.commit(vec, 0.0, {f"w{i}"}, ["simple"])
        assert len(memory.frames) == 4  # capacity

    def test_emotional_trajectory(self, memory, codebook):
        # Start negative, go positive
        vec = codebook.encode("a")
        memory.commit(vec, -0.5, set(), ["simple"])
        memory.commit(vec, -0.3, set(), ["simple"])
        memory.commit(vec, 0.3, set(), ["simple"])
        memory.commit(vec, 0.7, set(), ["simple"])

        ctx = memory.get_context()
        assert ctx.emotion_trend > 0  # Trending positive

    def test_reset_clears(self, memory, codebook):
        vec = codebook.encode("test")
        memory.commit(vec, 0.5, {"test"}, ["simple"])
        memory.reset()
        assert memory.is_empty
        assert memory.turn_count == 0

    def test_build_topic_vec(self, memory):
        vec = memory.build_topic_vec("Mut Kraft", "stark", "Kampf")
        assert vec.shape == (10_000,)
        assert set(np.unique(vec)).issubset({-1.0, 0.0, 1.0})

    def test_recent_shapes(self, memory, codebook):
        vec = codebook.encode("x")
        memory.commit(vec, 0.0, set(), ["contrast", "simple"])
        memory.commit(vec, 0.0, set(), ["parallel"])
        ctx = memory.get_context()
        assert "contrast" in ctx.recent_shapes
        assert "parallel" in ctx.recent_shapes
