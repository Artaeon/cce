"""Tests for the HDC Codebook."""

import numpy as np
import pytest

from cce.codebook import Codebook


@pytest.fixture
def cb():
    return Codebook(dim=10_000, seed=42)


class TestEncode:
    def test_deterministic(self, cb):
        """Same symbol always produces same vector."""
        v1 = cb.encode("Mut")
        v2 = cb.encode("Mut")
        assert np.array_equal(v1, v2)

    def test_bipolar(self, cb):
        """Vectors are bipolar {-1, +1}."""
        v = cb.encode("Kraft")
        unique = set(v.tolist())
        assert unique == {-1.0, 1.0}

    def test_near_orthogonal(self, cb):
        """Different symbols produce nearly orthogonal vectors."""
        v1 = cb.encode("Mut")
        v2 = cb.encode("Angst")
        sim = cb.similarity(v1, v2)
        assert abs(sim) < 0.05, f"Expected near-orthogonal, got {sim}"

    def test_self_similarity(self, cb):
        """A vector is maximally similar to itself."""
        v = cb.encode("Hoffnung")
        sim = cb.similarity(v, v)
        assert abs(sim - 1.0) < 1e-6


class TestBind:
    def test_orthogonal_to_inputs(self, cb):
        """bind(a, b) is nearly orthogonal to both a and b."""
        a = cb.encode("Mut")
        b = cb.encode("Kraft")
        bound = cb.bind(a, b)
        assert abs(cb.similarity(bound, a)) < 0.06
        assert abs(cb.similarity(bound, b)) < 0.06

    def test_self_inverse(self, cb):
        """bind(bind(a, b), b) ≈ a."""
        a = cb.encode("Mut")
        b = cb.encode("Kraft")
        recovered = cb.bind(cb.bind(a, b), b)
        sim = cb.similarity(recovered, a)
        assert sim > 0.99


class TestBundle:
    def test_closer_to_majority(self, cb):
        """bundle(a, a, b) is closer to a than to b."""
        a = cb.encode("Mut")
        b = cb.encode("Angst")
        bundled = cb.bundle(a, a, b)
        sim_a = cb.similarity(bundled, a)
        sim_b = cb.similarity(bundled, b)
        assert sim_a > sim_b

    def test_bipolar_result(self, cb):
        """Bundle result is bipolar."""
        a = cb.encode("X")
        b = cb.encode("Y")
        c = cb.bundle(a, b)
        unique = set(c.tolist())
        assert unique == {-1.0, 1.0}


class TestPermute:
    def test_nearly_orthogonal(self, cb):
        """Permuted vector is nearly orthogonal to original."""
        v = cb.encode("Sequenz")
        pv = cb.permute(v, shift=1)
        sim = cb.similarity(v, pv)
        assert abs(sim) < 0.05


class TestDecompose:
    def test_particles_similar_to_source(self, cb):
        """Decomposed particles are similar to the source vector."""
        base = cb.encode("Wahrheit")
        particles = cb.decompose("Wahrheit", n=5)
        assert len(particles) == 5
        for p in particles:
            sim = cb.similarity(p, base)
            assert sim > 0.3, f"Particle too far from source: {sim}"


class TestCleanup:
    def test_finds_exact(self, cb):
        """Cleanup finds exact match."""
        cb.encode("Mut")
        cb.encode("Angst")
        cb.encode("Kraft")
        query = cb.encode("Mut")
        found, sim = cb.cleanup(query)
        assert found == "Mut"
        assert sim > 0.99
