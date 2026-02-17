"""
HDC Knowledge Graph
====================

World knowledge encoded as hyperdimensional vector bindings.

Each relation is stored as: bind(role_vec, filler_vec) on the subject.
Example: "Unternehmer ISA Person" → the Unternehmer node stores
         bind(encode("ISA"), encode("Person"))

This enables:
- Inference: query("Unternehmer", "ISA") → "Person"
- Expansion: given "Quartalsbericht", infer Zahlen, Ergebnis, Zeit
- Contrast: look up OPPOSES to find natural antonyms
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from cce.codebook import Codebook


# Relation types
RELATION_TYPES = [
    "ISA",       # Unternehmer ISA Person
    "HAS",       # Quartal HAS Zahlen
    "CAUSES",    # Einsamkeit CAUSES Schmerz
    "OPPOSES",   # Hoffnung OPPOSES Verzweiflung
    "PARTOF",    # Monat PARTOF Quartal
    "NEEDS",     # Erfolg NEEDS Arbeit
    "LEADSTO",   # Mut LEADSTO Freiheit
    "CONTEXT",   # Bilanz CONTEXT Geschäft
]


@dataclass
class KnowledgeNode:
    """A concept node with relational bindings."""
    concept: str
    vec: np.ndarray
    relations: list[tuple[str, str, np.ndarray]] = field(default_factory=list)
    # Each relation is (relation_type, target_concept, binding_vec)


class KnowledgeGraph:
    """World knowledge encoded as HDC vector bindings.

    Relations are stored as bind(role_vec, filler_vec) on each subject node.
    Querying unbinds to recover the target concept.
    """

    def __init__(self, codebook: Codebook) -> None:
        self.codebook = codebook
        self.nodes: dict[str, KnowledgeNode] = {}
        self._role_vecs: dict[str, np.ndarray] = {}

        # Pre-encode relation role vectors
        for rel in RELATION_TYPES:
            self._role_vecs[rel] = codebook.encode(f"__REL_{rel}__")

    def _get_or_create_node(self, concept: str) -> KnowledgeNode:
        """Get or create a knowledge node for a concept."""
        key = concept.lower()
        if key not in self.nodes:
            self.nodes[key] = KnowledgeNode(
                concept=concept,
                vec=self.codebook.encode(key),
            )
        return self.nodes[key]

    def add_relation(
        self,
        subject: str,
        relation: str,
        obj: str,
    ) -> None:
        """Add a typed relation between two concepts.

        The relation is stored as a binding on the subject node:
        binding = bind(role_vec, target_vec)
        """
        node = self._get_or_create_node(subject)
        # Ensure target node also exists
        self._get_or_create_node(obj)

        role_vec = self._role_vecs.get(relation)
        if role_vec is None:
            # Unknown relation type — encode it
            role_vec = self.codebook.encode(f"__REL_{relation}__")
            self._role_vecs[relation] = role_vec

        target_vec = self.codebook.encode(obj.lower())
        binding = Codebook.bind(role_vec, target_vec)

        node.relations.append((relation, obj, binding))

    def query(
        self,
        subject: str,
        relation: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Query: what does subject have via this relation?

        Unbinds the relation from the subject's stored bindings
        and cleanup-matches against all known concepts.
        """
        node = self.nodes.get(subject.lower())
        if not node:
            return []

        role_vec = self._role_vecs.get(relation)
        if role_vec is None:
            return []

        results = []
        for rel_type, target, binding_vec in node.relations:
            if rel_type != relation:
                continue
            # Unbind to recover target
            recovered = Codebook.bind(role_vec, binding_vec)
            # Since bind is self-inverse: bind(role, bind(role, target)) ≈ target
            sim = Codebook.similarity(recovered, self.codebook.encode(target.lower()))
            results.append((target, sim))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def get_related(
        self,
        concept: str,
        max_results: int = 10,
    ) -> list[tuple[str, str, float]]:
        """Get all relations of a concept.

        Returns (relation_type, target, strength) tuples.
        """
        node = self.nodes.get(concept.lower())
        if not node:
            return []

        results = []
        for rel_type, target, binding in node.relations:
            # Strength is the cosine of the binding to its expected value
            expected = Codebook.bind(
                self._role_vecs[rel_type],
                self.codebook.encode(target.lower()),
            )
            sim = Codebook.similarity(binding, expected)
            results.append((rel_type, target, sim))

        return results[:max_results]

    def expand_concepts(
        self,
        concepts: list[str],
        max_expansions: int = 5,
    ) -> list[tuple[str, str, str]]:
        """Expand a set of concepts using knowledge relations.

        Returns (source_concept, relation, inferred_concept) tuples.
        Used during plasma creation to add inference particles.
        """
        expansions: list[tuple[str, str, str]] = []
        seen = {c.lower() for c in concepts}

        for concept in concepts:
            node = self.nodes.get(concept.lower())
            if not node:
                continue

            for rel_type, target, _ in node.relations:
                if target.lower() not in seen and len(expansions) < max_expansions:
                    expansions.append((concept, rel_type, target))
                    seen.add(target.lower())

        return expansions

    def find_opposites(self, concept: str) -> list[str]:
        """Find antonyms/opposites of a concept."""
        results = self.query(concept, "OPPOSES")
        return [target for target, _ in results]

    def load(self, path: str | Path) -> None:
        """Load knowledge from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            self.add_relation(
                subject=entry["subject"],
                relation=entry["relation"],
                obj=entry["object"],
            )

    @property
    def size(self) -> int:
        return sum(len(n.relations) for n in self.nodes.values())

    def stats(self) -> dict:
        """Get graph statistics."""
        rel_counts: dict[str, int] = {}
        for node in self.nodes.values():
            for rel_type, _, _ in node.relations:
                rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
        return {
            "nodes": len(self.nodes),
            "relations": self.size,
            "by_type": rel_counts,
        }
