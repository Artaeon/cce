"""
Phase 2: Nucleation
====================

Finds crystallization seeds — the points of highest semantic density
in the plasma. Each nucleus is a proto-thought: not yet a word, but
already with shape and direction.

Like snowflake nucleation: at certain points in the supercooled cloud,
the first ice crystals begin to form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cce.codebook import Codebook
from cce.particle import DensityPeak, Particle, ParticleCategory, ParticleField


@dataclass
class CrystalNucleus:
    """A crystallization seed — a proto-thought.

    Not yet language, but already more than plasma. It has a center
    (meaning), strength (importance), and emotional charge (coloring).
    """

    center: np.ndarray                # Position in meaning space
    strength: float                   # How much density it captured
    emotional_charge: Optional[np.ndarray] = None  # Dominant emotion vector
    absorbed_particles: list[Particle] = field(default_factory=list)
    label: str = ""                   # Human-readable summary

    @property
    def total_charge(self) -> float:
        """Sum of charges of all absorbed particles."""
        return sum(p.charge for p in self.absorbed_particles)

    @property
    def residual_temperature(self) -> float:
        """Average temperature of absorbed particles."""
        if not self.absorbed_particles:
            return 0.0
        return float(np.mean([p.temperature for p in self.absorbed_particles]))

    @property
    def emotional_intensity(self) -> float:
        """How emotionally charged is this nucleus? (0-1)"""
        if not self.absorbed_particles:
            return 0.0
        emo_count = sum(
            1 for p in self.absorbed_particles
            if p.category == ParticleCategory.EMOTION
        )
        return emo_count / len(self.absorbed_particles)

    @property
    def size(self) -> int:
        return len(self.absorbed_particles)


def nucleate(
    field: ParticleField,
    codebook: Codebook,
    min_density_distance: float = 0.25,
) -> list[CrystalNucleus]:
    """Find crystallization seeds in the plasma.

    Parameters
    ----------
    field : ParticleField
        The plasma field to analyze
    codebook : Codebook
        For similarity computations
    min_density_distance : float
        Minimum similarity distance between density peaks

    Returns
    -------
    list[CrystalNucleus]
        Nuclei ordered by strength (strongest first = main thought)
    """
    # Find density peaks
    peaks = field.compute_density(min_distance=min_density_distance)

    if not peaks:
        return []

    nuclei = []
    for peak in peaks:
        # Build a label from the most common particle labels
        labels = [p.label.split("#")[0] for p in peak.contributing_particles]
        from collections import Counter
        label_counts = Counter(labels)
        top_labels = [l for l, _ in label_counts.most_common(3)]
        label = "+".join(top_labels)

        nucleus = CrystalNucleus(
            center=peak.position.copy(),
            strength=peak.density,
            emotional_charge=peak.dominant_emotion(),
            absorbed_particles=list(peak.contributing_particles),
            label=label,
        )
        nuclei.append(nucleus)

    # Sort by strength — the main thought comes first
    nuclei.sort(key=lambda n: -n.strength)

    # Mark absorbed particles as frozen
    for nucleus in nuclei:
        for p in nucleus.absorbed_particles:
            p.frozen = True

    return nuclei
