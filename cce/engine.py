"""
Cognitive Crystallization Engine — Orchestrator
=================================================

The top-level pipeline that ties all phases together:
  Input → Plasma → Nucleation → Crystallization → Realization → Text

Each phase is logged for introspection and debugging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cce.codebook import Codebook
from cce.crystallization import Crystal, crystallize
from cce.knowledge import KnowledgeGraph
from cce.lexicon import ResonantLexicon
from cce.memory import WorkingMemory
from cce.nucleation import CrystalNucleus, nucleate
from cce.parser import InputParser
from cce.particle import ParticleField
from cce.plasma import create_plasma
from cce.realization import realize
from cce.metaphor import MetaphorMatcher


@dataclass
class PhaseLog:
    """Log entry for one engine phase."""
    phase: str
    duration_ms: float
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"[{self.phase}] {self.duration_ms:.1f}ms — {self.details}"


class CognitiveCrystallizationEngine:
    """The complete CCE pipeline.

    Usage::

        engine = CognitiveCrystallizationEngine("data/de_lexicon.json")
        text = engine.generate(
            intent="Zahlen schlecht Trend positiv",
            emotion="frustriert hoffnungsvoll",
            context="Quartalsbericht",
            persona="direkt ehrlich",
        )
        print(text)
    """

    def __init__(
        self,
        lexicon_path: str | Path = "data/de_lexicon_5k.json",
        knowledge_path: str | Path = "data/de_knowledge.json",
        dim: int = 10_000,
    ) -> None:
        self.codebook = Codebook(dim=dim)
        self.lexicon = ResonantLexicon(self.codebook)
        self.lexicon.load(lexicon_path)
        self.knowledge = KnowledgeGraph(self.codebook)
        knowledge_file = Path(knowledge_path)
        if knowledge_file.exists():
            self.knowledge.load(knowledge_file)
        self.memory = WorkingMemory(self.codebook)
        self.parser = InputParser(self.lexicon)
        self.metaphor_matcher = MetaphorMatcher(self.knowledge)
        self.logs: list[PhaseLog] = []
        self.dim = dim

        # State for introspection
        self._last_field: Optional[ParticleField] = None
        self._last_nuclei: Optional[list[CrystalNucleus]] = None
        self._last_crystals: Optional[list[Crystal]] = None

    def generate(
        self,
        intent: str,
        emotion: str = "",
        context: str = "",
        persona: str = "",
        cooling_rate: float = 0.05,
        is_question: bool = False,
        question_type: str = "",
    ) -> str:
        """Run the full pipeline: intent → text.

        Parameters
        ----------
        intent : str
            What to say (concepts, not sentences)
        emotion : str
            Emotional coloring
        context : str
            Situational context
        persona : str
            Speaker character
        cooling_rate : float
            How fast crystallization proceeds

        Returns
        -------
        str
            Generated text
        """
        self.logs.clear()

        # Auto-register unknown input words so the lexicon always covers them
        all_text = f"{intent} {emotion} {context} {persona}"
        self.lexicon.add_from_text(all_text)

        # ── Phase 1: Plasma ──────────────────────────────────
        t0 = time.perf_counter()
        plasma_field = create_plasma(
            intent=intent,
            emotion=emotion,
            context=context,
            persona=persona,
            codebook=self.codebook,
            knowledge=self.knowledge,
        )
        dt = (time.perf_counter() - t0) * 1000
        self._last_field = plasma_field
        self.logs.append(PhaseLog(
            phase="PLASMA",
            duration_ms=dt,
            details={
                "total_particles": len(plasma_field),
                "temperature": f"{plasma_field.temperature:.3f}",
            },
        ))

        # ── Phase 2: Nucleation ──────────────────────────────
        t0 = time.perf_counter()
        nuclei = nucleate(plasma_field, self.codebook)
        dt = (time.perf_counter() - t0) * 1000
        self._last_nuclei = nuclei
        self.logs.append(PhaseLog(
            phase="NUCLEATION",
            duration_ms=dt,
            details={
                "nuclei_found": len(nuclei),
                "labels": [n.label for n in nuclei],
                "strengths": [f"{n.strength:.2f}" for n in nuclei],
            },
        ))

        # ── Phase 3: Crystallization ─────────────────────────
        t0 = time.perf_counter()
        crystals = crystallize(
            field=plasma_field,
            nuclei=nuclei,
            codebook=self.codebook,
            cooling_rate=cooling_rate,
        )
        dt = (time.perf_counter() - t0) * 1000
        self._last_crystals = crystals
        self.logs.append(PhaseLog(
            phase="CRYSTALLIZATION",
            duration_ms=dt,
            details={
                "crystals": len(crystals),
                "shapes": [c.shape for c in crystals],
                "labels": [c.label for c in crystals],
                "emotions": [f"{c.emotional_charge:.2f}" for c in crystals],
                "temperatures": [f"{c.residual_temperature:.2f}" for c in crystals],
            },
        ))

        # ── Phase 4: Realization ─────────────────────────────
        t0 = time.perf_counter()
        # Pass memory's avoid-words to realization for vocabulary variety
        mem_ctx = self.memory.get_context()
        text = realize(
            crystals, self.lexicon, self.codebook,
            avoid_words=mem_ctx.avoid_words,
            is_question=is_question,
            knowledge=self.knowledge,
            emotion=emotion,
            question_type=question_type,
            metaphor_matcher=self.metaphor_matcher,
        )
        dt = (time.perf_counter() - t0) * 1000
        self.logs.append(PhaseLog(
            phase="REALIZATION",
            duration_ms=dt,
            details={
                "output_length": len(text),
                "word_count": len(text.split()),
                "memory_turn": mem_ctx.turn_number,
            },
        ))

        # ── Commit to working memory ─────────────────────────
        topic_vec = self.memory.build_topic_vec(intent, emotion, context)
        avg_emotion = 0.0
        if crystals:
            avg_emotion = sum(c.emotional_charge for c in crystals) / len(crystals)
        words_used = set(text.lower().split())
        shapes = [c.shape for c in crystals]
        self.memory.commit(topic_vec, avg_emotion, words_used, shapes)

        return text

    @property
    def total_time_ms(self) -> float:
        return sum(log.duration_ms for log in self.logs)

    def print_logs(self) -> None:
        """Pretty-print phase logs."""
        print("\n╔══════════════════════════════════════════════╗")
        print("║   Cognitive Crystallization Engine — Log     ║")
        print("╚══════════════════════════════════════════════╝\n")

        phase_icons = {
            "PLASMA": "🌊",
            "NUCLEATION": "💎",
            "CRYSTALLIZATION": "❄️",
            "REALIZATION": "📝",
        }

        for log in self.logs:
            icon = phase_icons.get(log.phase, "▸")
            print(f"  {icon} {log.phase} ({log.duration_ms:.1f}ms)")
            for k, v in log.details.items():
                print(f"     {k}: {v}")
            print()

        print(f"  ⏱  Total: {self.total_time_ms:.1f}ms")
        print(f"  📊 Lexikon: {self.lexicon.size} Wörter")
        print(f"  📐 Dimension: {self.dim:,}")
        print(f"  ✅ Knowledge: {self.knowledge.size} Relationen")
        print(f"  🧠 Memory: {self.memory.turn_count} Turns")
        print()

    def generate_from_text(
        self,
        text: str,
        persona: str = "",
        cooling_rate: float = 0.05,
    ) -> str:
        """Generate from natural German text input.

        Parses the text to extract intent, emotion, and context,
        then runs the standard pipeline.

        Parameters
        ----------
        text : str
            Free-form German text (question, statement, topic)
        persona : str
            Optional persona override
        cooling_rate : float
            How fast crystallization proceeds

        Returns
        -------
        str
            Generated text
        """
        parsed = self.parser.parse(text)

        return self.generate(
            intent=parsed.intent,
            emotion=parsed.emotion,
            context=parsed.context or "",
            persona=persona or parsed.persona,
            cooling_rate=cooling_rate,
            is_question=parsed.is_question,
            question_type=parsed.question_type,
        )
