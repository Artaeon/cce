"""Tests for Particle and ParticleField."""

import numpy as np
import pytest

from cce.codebook import Codebook
from cce.particle import Particle, ParticleCategory, ParticleField


@pytest.fixture
def cb():
    return Codebook(dim=10_000, seed=42)


class TestParticle:
    def test_creation(self, cb):
        p = Particle(
            position=cb.encode("Mut"),
            charge=1.0,
            temperature=0.5,
            label="Mut#0",
            category=ParticleCategory.INTENT,
        )
        assert p.charge == 1.0
        assert p.temperature == 0.5
        assert not p.frozen

    def test_similarity(self, cb):
        p1 = Particle(position=cb.encode("Mut"))
        p2 = Particle(position=cb.encode("Mut"))
        p3 = Particle(position=cb.encode("Angst"))
        assert p1.similarity_to(p2) > 0.99
        assert abs(p1.similarity_to(p3)) < 0.05


class TestParticleField:
    def test_add_and_count(self, cb):
        field = ParticleField()
        for label in ["Mut", "Kraft", "Angst"]:
            field.add(Particle(position=cb.encode(label), label=label))
        assert len(field) == 3
        assert len(field.free_particles()) == 3

    def test_freeze(self, cb):
        field = ParticleField()
        p = Particle(position=cb.encode("Mut"), label="Mut")
        field.add(p)
        p.frozen = True
        assert len(field.free_particles()) == 0
        assert len(field.frozen_particles()) == 1

    def test_cool(self, cb):
        field = ParticleField()
        p = Particle(position=cb.encode("Mut"), temperature=0.5)
        field.add(p)
        field.cool(rate=0.1)
        assert p.temperature == pytest.approx(0.4)

    def test_compute_density(self, cb):
        """Clustered particles produce density peaks."""
        field = ParticleField()
        # Add 5 particles from similar vectors (will cluster)
        for part in cb.decompose("Mut", n=5):
            field.add(Particle(position=part, charge=1.0, label="Mut"))
        # Add 5 from different concept
        for part in cb.decompose("Angst", n=5):
            field.add(Particle(position=part, charge=1.0, label="Angst"))
        
        peaks = field.compute_density(min_distance=0.3)
        assert len(peaks) >= 2, f"Expected ≥2 peaks, got {len(peaks)}"
