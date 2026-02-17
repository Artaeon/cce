"""
Particle Physics for the Meaning-Plasma
========================================

Particles are the atoms of the pre-linguistic state. Each carries:
- A position in meaning-space (HDC vector)
- A charge (how strongly it attracts others)
- A temperature (how freely it moves)
- A label (what concept it came from)
- A category (intent / emotion / context / persona)

The ParticleField manages collections of particles and computes
density maps, forces, and cooling — the physics that enables
crystallization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


class ParticleCategory(Enum):
    INTENT = auto()
    EMOTION = auto()
    CONTEXT = auto()
    PERSONA = auto()


@dataclass
class Particle:
    """A single meaning-particle in the plasma."""

    position: np.ndarray          # HDC vector — location in meaning space
    charge: float = 1.0           # Attraction strength
    temperature: float = 0.5      # Volatility / freedom of movement
    label: str = ""               # Human-readable origin
    category: ParticleCategory = ParticleCategory.INTENT
    mass: float = 1.0             # Resistance to movement
    frozen: bool = False          # Once frozen, particle is locked in crystal

    def similarity_to(self, other: "Particle") -> float:
        """Cosine similarity to another particle."""
        na = np.linalg.norm(self.position)
        nb = np.linalg.norm(other.position)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(self.position, other.position) / (na * nb))


@dataclass
class DensityPeak:
    """A point of high semantic density in the field."""

    position: np.ndarray
    density: float
    contributing_particles: list[Particle] = field(default_factory=list)

    def dominant_emotion(self) -> Optional[np.ndarray]:
        """Return the average position of emotional particles near this peak."""
        emo_particles = [
            p for p in self.contributing_particles
            if p.category == ParticleCategory.EMOTION
        ]
        if not emo_particles:
            return None
        return np.mean([p.position for p in emo_particles], axis=0).astype(np.float32)

    def dominant_category(self) -> ParticleCategory:
        """Which category dominates this peak?"""
        from collections import Counter
        cats = Counter(p.category for p in self.contributing_particles)
        return cats.most_common(1)[0][0]


class ParticleField:
    """A collection of meaning-particles — the plasma.

    Provides physics operations: density estimation, force application,
    and cooling.
    """

    def __init__(self) -> None:
        self._particles: list[Particle] = []

    def add(self, particle: Particle) -> None:
        self._particles.append(particle)

    def add_many(self, particles: list[Particle]) -> None:
        self._particles.extend(particles)

    @property
    def particles(self) -> list[Particle]:
        return self._particles

    def free_particles(self) -> list[Particle]:
        """Return particles not yet frozen into a crystal."""
        return [p for p in self._particles if not p.frozen]

    def frozen_particles(self) -> list[Particle]:
        return [p for p in self._particles if p.frozen]

    @property
    def temperature(self) -> float:
        """Average temperature of all free particles."""
        free = self.free_particles()
        if not free:
            return 0.0
        return float(np.mean([p.temperature for p in free]))

    def cool(self, rate: float = 0.05) -> None:
        """Lower the temperature of all free particles."""
        for p in self.free_particles():
            p.temperature = max(0.0, p.temperature - rate)

    # ------------------------------------------------------------------
    # Density estimation — finding where meaning clusters
    # ------------------------------------------------------------------

    def compute_density(self, min_distance: float = 0.3) -> list[DensityPeak]:
        """Find peaks of semantic density in the field.

        Uses a greedy clustering approach:
        1. For each particle, sum similarity-weighted charges of neighbors
        2. Take the highest-density particle as a peak
        3. Remove particles within min_distance, repeat

        This is O(n²) in particle count but n is small (~55).
        """
        free = self.free_particles()
        if not free:
            return []

        n = len(free)

        # Precompute similarity matrix
        positions = np.stack([p.position for p in free])
        norms = np.linalg.norm(positions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = positions / norms
        sim_matrix = normalized @ normalized.T  # (n, n)

        # Compute density for each particle
        charges = np.array([p.charge for p in free])
        densities = np.zeros(n)
        for i in range(n):
            # Density = sum of (similarity × charge) for all neighbors
            # Only count positive similarities (same "direction")
            sims = np.maximum(sim_matrix[i], 0.0)
            densities[i] = float(np.dot(sims, charges))

        # Greedy peak extraction
        peaks: list[DensityPeak] = []
        used = np.zeros(n, dtype=bool)

        while True:
            # Find highest unused density
            masked = np.where(used, -np.inf, densities)
            idx = int(np.argmax(masked))
            if masked[idx] <= 0:
                break

            # Collect contributing particles (those within min_distance)
            contributors = []
            for j in range(n):
                if not used[j] and sim_matrix[idx, j] >= min_distance:
                    contributors.append(free[j])
                    used[j] = True

            if not contributors:
                used[idx] = True
                continue

            peak = DensityPeak(
                position=free[idx].position.copy(),
                density=float(densities[idx]),
                contributing_particles=contributors,
            )
            peaks.append(peak)

            if np.all(used):
                break

        # Collect any remaining orphan particles into their own peaks
        for j in range(n):
            if not used[j]:
                peaks.append(DensityPeak(
                    position=free[j].position.copy(),
                    density=float(densities[j]),
                    contributing_particles=[free[j]],
                ))
                used[j] = True

        return peaks

    def apply_forces(self, dt: float = 0.1) -> None:
        """Move free particles toward high-charge neighbors.

        The force on particle i from particle j is proportional to
        similarity(i,j) × charge(j) / mass(i), scaled by temperature
        (hot particles jitter more).
        """
        free = self.free_particles()
        if len(free) < 2:
            return

        for p in free:
            if p.frozen:
                continue

            # Compute net force
            force = np.zeros_like(p.position)
            for other in free:
                if other is p:
                    continue
                sim = float(np.dot(p.position, other.position)) / (
                    np.linalg.norm(p.position) * np.linalg.norm(other.position) + 1e-8
                )
                if sim > 0:
                    # Attract toward other
                    direction = other.position - p.position
                    force += sim * other.charge * direction / (p.mass + 1e-8)

            # Thermal noise — high temperature means more random wandering
            noise = np.random.randn(len(p.position)).astype(np.float32)
            force += p.temperature * noise * 0.1

            # Apply force
            p.position = p.position + force * dt
            # Re-binarize to stay in HDC space
            p.position = np.sign(p.position).astype(np.float32)
            # Replace zeros
            zeros = p.position == 0
            if np.any(zeros):
                p.position[zeros] = np.random.choice([-1, 1], size=int(zeros.sum()))

    def __len__(self) -> int:
        return len(self._particles)

    def __repr__(self) -> str:
        n_free = len(self.free_particles())
        n_frozen = len(self.frozen_particles())
        return (
            f"ParticleField(total={len(self)}, free={n_free}, "
            f"frozen={n_frozen}, temp={self.temperature:.3f})"
        )
