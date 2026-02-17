"""
Phase 1: Plasma Creation
=========================

Creates the pre-linguistic state — a warm, chaotic field of
meaning-particles derived from intent, emotion, context, and persona.

This is the "warm ocean" where everything is dissolved, nothing has
form yet. All particles interact with everything.

KEY IMPROVEMENT: Cross-concept binding particles create bridges
between concepts, enabling merging during crystallization.
"""

from __future__ import annotations

import itertools

import numpy as np

from cce.codebook import Codebook
from cce.particle import Particle, ParticleCategory, ParticleField


# ------------------------------------------------------------------
# Charge & temperature profiles per category
# ------------------------------------------------------------------
PROFILES = {
    ParticleCategory.INTENT: {
        "n_particles": 6,    # Per word (fewer but richer)
        "charge": 1.0,
        "temperature": 0.5,
        "mass": 1.0,
    },
    ParticleCategory.EMOTION: {
        "n_particles": 4,
        "charge": 0.8,
        "temperature": 0.9,
        "mass": 0.5,
    },
    ParticleCategory.CONTEXT: {
        "n_particles": 3,
        "charge": 0.5,
        "temperature": 0.3,
        "mass": 1.5,
    },
    ParticleCategory.PERSONA: {
        "n_particles": 2,
        "charge": 0.3,
        "temperature": 0.1,
        "mass": 2.0,
    },
}


def create_plasma(
    intent: str,
    emotion: str,
    context: str,
    persona: str,
    codebook: Codebook,
    knowledge: object | None = None,
) -> ParticleField:
    """Create the pre-linguistic plasma from semantic inputs.

    Creates four kinds of particles:
    1. Core particles — directly from each concept word
    2. Binding particles — blend of pairs of related words
    3. Emotional coloring — emotion vectors blended into intent particles
    4. Knowledge inference — particles from known relations (NEW)
    """
    field = ParticleField()

    inputs = {
        ParticleCategory.INTENT: intent,
        ParticleCategory.EMOTION: emotion,
        ParticleCategory.CONTEXT: context,
        ParticleCategory.PERSONA: persona,
    }

    # Collect all words per category for cross-binding
    words_by_cat: dict[ParticleCategory, list[str]] = {}

    for category, text in inputs.items():
        if not text.strip():
            continue

        profile = PROFILES[category]
        words = text.strip().split()
        words_by_cat[category] = words

        for word in words:
            # Core particles: high signal (65%), moderate noise (35%)
            sub_vecs = codebook.decompose(word.lower(), n=profile["n_particles"])
            for i, vec in enumerate(sub_vecs):
                field.add(Particle(
                    position=vec,
                    charge=profile["charge"],
                    temperature=profile["temperature"],
                    label=f"{word}#{i}",
                    category=category,
                    mass=profile["mass"],
                ))

    # ── Cross-concept binding particles ──────────────────────
    intent_words = words_by_cat.get(ParticleCategory.INTENT, [])
    if len(intent_words) >= 2:
        rng = np.random.RandomState(42)
        for w1, w2 in itertools.combinations(intent_words, 2):
            v1 = codebook.encode(w1.lower())
            v2 = codebook.encode(w2.lower())
            blended = 0.5 * v1 + 0.5 * v2
            noise = rng.choice([-1, 1], size=codebook.dim).astype(np.float32)
            blended = np.sign(0.6 * blended + 0.4 * noise).astype(np.float32)
            field.add(Particle(
                position=blended,
                charge=0.6,
                temperature=0.6,
                label=f"{w1}+{w2}",
                category=ParticleCategory.INTENT,
                mass=0.8,
            ))

    # ── Emotional coloring of intent ─────────────────────────
    emotion_words = words_by_cat.get(ParticleCategory.EMOTION, [])
    if emotion_words and intent_words:
        rng = np.random.RandomState(99)
        for ew in emotion_words:
            ev = codebook.encode(ew.lower())
            for iw in intent_words[:3]:
                iv = codebook.encode(iw.lower())
                colored = 0.4 * iv + 0.6 * ev
                noise = rng.choice([-1, 1], size=codebook.dim).astype(np.float32)
                colored = np.sign(0.65 * colored + 0.35 * noise).astype(np.float32)
                field.add(Particle(
                    position=colored,
                    charge=0.5,
                    temperature=0.7,
                    label=f"{iw}~{ew}",
                    category=ParticleCategory.EMOTION,
                    mass=0.7,
                ))

    # ── Knowledge inference particles (NEW) ──────────────────
    # If a knowledge graph is provided, expand intent concepts
    # with inferred related concepts from known relations.
    if knowledge is not None:
        all_concepts = intent_words + words_by_cat.get(ParticleCategory.CONTEXT, [])
        expansions = knowledge.expand_concepts(all_concepts, max_expansions=6)
        rng = np.random.RandomState(77)
        for source, relation, inferred in expansions:
            inf_vec = codebook.encode(inferred.lower())
            src_vec = codebook.encode(source.lower())
            # Blend inferred concept with source (keeps it connected)
            blended = 0.3 * src_vec + 0.7 * inf_vec
            noise = rng.choice([-1, 1], size=codebook.dim).astype(np.float32)
            blended = np.sign(0.7 * blended + 0.3 * noise).astype(np.float32)
            field.add(Particle(
                position=blended,
                charge=0.4,  # Lower charge: inferred, not stated
                temperature=0.5,
                label=f"{inferred}[{relation}:{source}]",
                category=ParticleCategory.CONTEXT,
                mass=1.2,  # Heavier: contextual knowledge
            ))

    return field
