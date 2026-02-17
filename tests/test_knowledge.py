"""Tests for the HDC Knowledge Graph (knowledge.py)."""

import pytest
import numpy as np

from cce.codebook import Codebook
from cce.knowledge import KnowledgeGraph


@pytest.fixture
def codebook():
    return Codebook(dim=10_000)


@pytest.fixture
def graph(codebook):
    g = KnowledgeGraph(codebook)
    g.add_relation("Mut", "OPPOSES", "Angst")
    g.add_relation("Mut", "LEADSTO", "Freiheit")
    g.add_relation("Einsamkeit", "CAUSES", "Schmerz")
    g.add_relation("Hoffnung", "OPPOSES", "Verzweiflung")
    g.add_relation("Unternehmer", "ISA", "Mensch")
    g.add_relation("Unternehmer", "NEEDS", "Mut")
    return g


class TestKnowledgeGraph:
    """Tests for the knowledge graph."""

    def test_add_creates_nodes(self, graph):
        assert "mut" in graph.nodes
        assert "angst" in graph.nodes
        assert "freiheit" in graph.nodes

    def test_relation_count(self, graph):
        assert graph.size == 6

    def test_query_opposes(self, graph):
        results = graph.query("Mut", "OPPOSES")
        targets = [t for t, s in results]
        assert "Angst" in targets

    def test_query_causes(self, graph):
        results = graph.query("Einsamkeit", "CAUSES")
        targets = [t for t, s in results]
        assert "Schmerz" in targets

    def test_query_empty(self, graph):
        results = graph.query("Liebe", "OPPOSES")
        assert results == []

    def test_query_unknown_concept(self, graph):
        results = graph.query("Katze", "ISA")
        assert results == []

    def test_get_related(self, graph):
        relations = graph.get_related("Mut")
        rel_types = [r for r, t, s in relations]
        assert "OPPOSES" in rel_types
        assert "LEADSTO" in rel_types

    def test_expand_concepts(self, graph):
        expansions = graph.expand_concepts(["Mut"], max_expansions=5)
        inferred_concepts = [c for _, _, c in expansions]
        assert "Angst" in inferred_concepts
        assert "Freiheit" in inferred_concepts

    def test_expand_avoids_duplicates(self, graph):
        expansions = graph.expand_concepts(["Mut", "Angst"], max_expansions=5)
        inferred = [c for _, _, c in expansions]
        # Should not include "Angst" since it's already in input
        assert "Angst" not in inferred

    def test_find_opposites(self, graph):
        opposites = graph.find_opposites("Mut")
        assert "Angst" in opposites

    def test_stats(self, graph):
        stats = graph.stats()
        assert stats["nodes"] > 0
        assert stats["relations"] == 6
        assert "OPPOSES" in stats["by_type"]

    def test_load_from_file(self, codebook, tmp_path):
        import json
        data = [
            {"subject": "A", "relation": "ISA", "object": "B"},
            {"subject": "C", "relation": "OPPOSES", "object": "D"},
        ]
        path = tmp_path / "test_kg.json"
        with open(path, "w") as f:
            json.dump(data, f)

        g = KnowledgeGraph(codebook)
        g.load(path)
        assert g.size == 2
        assert "a" in g.nodes
        assert "c" in g.nodes
