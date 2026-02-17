"""
Phase 3: Crystallization
=========================

Simulated annealing with a warmup phase. The plasma cools and nuclei
grow by absorbing nearby particles. As temperature drops, particles
lock into place.

KEY IMPROVEMENTS:
- Lower merge threshold (0.15) so related concepts fuse
- Warmup phase: particles interact before freezing
- Max crystals capped at 5 for coherent output
- Shape detection considers crystal relationships
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cce.codebook import Codebook
from cce.nucleation import CrystalNucleus
from cce.particle import Particle, ParticleCategory, ParticleField


@dataclass
class Crystal:
    """A fully crystallized thought-chunk.

    Each crystal will become one clause or phrase in the output.
    """

    nucleus: CrystalNucleus
    shape: str = "simple"
    element_count: int = 0
    emotional_charge: float = 0.0
    residual_temperature: float = 0.0
    contrast_partner: Optional["Crystal"] = None
    children: list["Crystal"] = field(default_factory=list)  # Sub-crystals for hierarchy

    @property
    def particles(self) -> list[Particle]:
        return self.nucleus.absorbed_particles

    @property
    def center(self) -> np.ndarray:
        return self.nucleus.center

    @property
    def label(self) -> str:
        return self.nucleus.label

    @property
    def strength(self) -> float:
        return self.nucleus.strength

    @property
    def unique_concepts(self) -> list[str]:
        """Extract unique concept names from absorbed particles."""
        labels = set()
        for p in self.particles:
            base = p.label.split("#")[0].split("~")[0].split("+")[0]
            if base:
                labels.add(base)
        return sorted(labels)


def _compute_attraction(
    nucleus: CrystalNucleus,
    particle: Particle,
) -> float:
    """How strongly does this nucleus attract this particle?"""
    na = np.linalg.norm(nucleus.center)
    nb = np.linalg.norm(particle.position)
    if na == 0 or nb == 0:
        return 0.0
    sim = float(np.dot(nucleus.center, particle.position) / (na * nb))

    if sim <= 0:
        return 0.0

    charge_factor = nucleus.total_charge * particle.charge * 0.01
    return sim * (1 + charge_factor)


def _force_merge_small(nuclei: list[CrystalNucleus], codebook: Codebook, max_crystals: int = 5) -> None:
    """Force-merge smallest nuclei until we have at most max_crystals.

    This ensures coherent output — too many crystals produce choppy text.
    """
    while len(nuclei) > max_crystals:
        # Find the smallest nucleus
        smallest_idx = min(range(len(nuclei)), key=lambda i: nuclei[i].strength)
        smallest = nuclei[smallest_idx]

        # Find the most compatible neighbor to merge into
        best_target = -1
        best_score = -1.0
        for j, other in enumerate(nuclei):
            if j == smallest_idx:
                continue
            sim = codebook.similarity(smallest.center, other.center)
            # Score: prefer high similarity + large target
            score = sim + 0.1 * other.strength
            if score > best_score:
                best_score = score
                best_target = j

        if best_target >= 0:
            target = nuclei[best_target]
            target.absorbed_particles.extend(smallest.absorbed_particles)
            target.strength += smallest.strength * 0.5
            # Weighted center update
            n_t = len(target.absorbed_particles) - len(smallest.absorbed_particles)
            n_s = len(smallest.absorbed_particles)
            if n_t + n_s > 0:
                target.center = (
                    (target.center * n_t + smallest.center * n_s) / (n_t + n_s)
                ).astype(np.float32)
            target.label += "+" + smallest.label
            nuclei.pop(smallest_idx)
        else:
            break


def _merge_nearby_nuclei(
    nuclei: list[CrystalNucleus],
    codebook: Codebook,
    merge_threshold: float = 0.15,
) -> None:
    """Merge nuclei with similarity above threshold."""
    i = 0
    while i < len(nuclei):
        j = i + 1
        while j < len(nuclei):
            sim = codebook.similarity(nuclei[i].center, nuclei[j].center)
            if sim >= merge_threshold:
                n_i = max(len(nuclei[i].absorbed_particles), 1)
                n_j = len(nuclei[j].absorbed_particles)
                nuclei[i].absorbed_particles.extend(nuclei[j].absorbed_particles)
                nuclei[i].strength += nuclei[j].strength
                nuclei[i].center = (
                    (nuclei[i].center * n_i + nuclei[j].center * n_j) / (n_i + n_j)
                ).astype(np.float32)
                nuclei[i].label += "+" + nuclei[j].label
                nuclei.pop(j)
            else:
                j += 1
        i += 1


def _compute_emotional_valence(nucleus: CrystalNucleus) -> float:
    """Compute overall emotional orientation (-1 to +1)."""
    if nucleus.emotional_charge is None:
        return 0.0
    return float(np.clip(np.mean(nucleus.emotional_charge), -1, 1))


def _detect_shapes(crystals: list[Crystal], codebook: Codebook) -> None:
    """Detect shapes based on inter-crystal relationships.

    Shapes:
    - "contrast" : two crystals with opposite emotional charge or low similarity
    - "parallel" : crystal with 4+ elements showing internal symmetry
    - "fragment" : 1-2 unique concepts, punchy
    - "simple"   : default, SVO-like
    """
    for crystal in crystals:
        n_concepts = crystal.element_count
        if n_concepts <= 1:
            crystal.shape = "fragment"
            continue

        # Check for contrast with another crystal
        found_contrast = False
        for other in crystals:
            if other is crystal:
                continue
            sim = codebook.similarity(crystal.center, other.center)
            emo_diff = abs(crystal.emotional_charge - other.emotional_charge)
            if sim < 0.05 or emo_diff > 0.3:
                crystal.contrast_partner = other
                crystal.shape = "contrast"
                found_contrast = True
                break

        if found_contrast:
            continue

        if n_concepts >= 4:
            crystal.shape = "parallel"
        elif n_concepts >= 2:
            crystal.shape = "simple"
        else:
            crystal.shape = "fragment"


def crystallize(
    field: ParticleField,
    nuclei: list[CrystalNucleus],
    codebook: Codebook,
    cooling_rate: float = 0.05,
    min_temperature: float = 0.01,
    max_crystals: int = 4,
) -> list[Crystal]:
    """The phase transition: nuclei grow as the plasma cools.

    Parameters
    ----------
    max_crystals : int
        Maximum number of output crystals (= output clauses).
        Fewer crystals = more coherent, focused output.
    """
    if not nuclei:
        return []

    temperature = 1.0
    rng = np.random.RandomState(42)

    # ── Merge pass 1: aggressive similarity-based merging ────
    _merge_nearby_nuclei(nuclei, codebook, merge_threshold=0.15)

    # ── Absorption loop ──────────────────────────────────────
    while temperature > min_temperature:
        free = field.free_particles()
        if not free:
            break

        for nucleus in nuclei:
            for particle in free:
                if particle.frozen:
                    continue

                attraction = _compute_attraction(nucleus, particle)
                threshold = temperature * rng.random()
                if attraction > threshold:
                    nucleus.absorbed_particles.append(particle)
                    particle.frozen = True
                    n = len(nucleus.absorbed_particles)
                    nucleus.center = (
                        (nucleus.center * (n - 1) + particle.position) / n
                    ).astype(np.float32)

        # Merge pass during cooling
        _merge_nearby_nuclei(nuclei, codebook, merge_threshold=0.12)
        temperature -= cooling_rate

    # Absorb remaining free particles
    for particle in field.free_particles():
        if not particle.frozen and nuclei:
            best_nucleus = max(
                nuclei, key=lambda n: _compute_attraction(n, particle)
            )
            best_nucleus.absorbed_particles.append(particle)
            particle.frozen = True

    # ── Force-merge to max_crystals ──────────────────────────
    _force_merge_small(nuclei, codebook, max_crystals=max_crystals)

    # ── Build Crystal objects ────────────────────────────────
    crystals = []
    for nucleus in nuclei:
        if not nucleus.absorbed_particles:
            continue

        unique_concepts = set()
        for p in nucleus.absorbed_particles:
            base = p.label.split("#")[0].split("~")[0].split("+")[0]
            if base:
                unique_concepts.add(base)

        crystal = Crystal(
            nucleus=nucleus,
            element_count=len(unique_concepts),
            emotional_charge=_compute_emotional_valence(nucleus),
            residual_temperature=nucleus.residual_temperature,
        )
        crystals.append(crystal)

    # Detect shapes
    _detect_shapes(crystals, codebook)

    # Recursive nesting: small crystals become children of larger ones
    _recursive_nest(crystals, codebook)

    # Order by strength
    crystals.sort(key=lambda c: -c.strength)

    return crystals


def _recursive_nest(
    crystals: list[Crystal],
    codebook: Codebook,
    sim_threshold: float = 0.25,
    min_parent_elements: int = 3,
) -> None:
    """Nest small crystals as children of compatible larger ones.

    A small crystal becomes a child (subclause) of a larger one when:
    1. The parent has enough content (min_parent_elements)
    2. The similarity is above sim_threshold (topically related)
    3. The child is small (1-2 unique concepts)

    This enables realization to produce complex sentences like:
    'Die Kraft — die aus dem Zweifel wächst — ist unerschütterlich.'
    """
    if len(crystals) < 2:
        return

    to_remove = []
    for i, child in enumerate(crystals):
        if child.element_count > 2:  # Only small crystals can nest
            continue

        best_parent = None
        best_sim = sim_threshold

        for j, parent in enumerate(crystals):
            if i == j or parent.element_count < min_parent_elements:
                continue
            sim = codebook.similarity(child.center, parent.center)
            if sim > best_sim:
                best_sim = sim
                best_parent = parent

        if best_parent is not None:
            best_parent.children.append(child)
            to_remove.append(i)

    # Remove nested crystals from the top level (reverse order)
    for idx in sorted(to_remove, reverse=True):
        crystals.pop(idx)
