"""
Hyperdimensional Computing (HDC) Codebook
=========================================

The mathematical spine of the CCE. All meaning is represented as
high-dimensional bipolar vectors {-1, +1}^D where D=10,000.

Key properties exploited:
- Random high-dimensional vectors are nearly orthogonal (cosine ≈ 0)
- Binding (element-wise multiply) creates new orthogonal vectors
- Bundling (element-wise majority) creates superpositions
- These operations are O(D) — linear, not quadratic
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


class Codebook:
    """Manages symbol → hyperdimensional vector mappings.

    Vectors are bipolar {-1, +1}^dim, generated deterministically
    from symbol hashes so results are reproducible.
    """

    def __init__(self, dim: int = 10_000, seed: int = 42) -> None:
        self.dim = dim
        self.seed = seed
        self._memory: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Core vector operations
    # ------------------------------------------------------------------

    def encode(self, symbol: str) -> np.ndarray:
        """Deterministically encode a symbol as a bipolar vector.

        Uses SHA-256 hash expanded via a seeded RNG so every call with
        the same symbol returns the identical vector.
        """
        if symbol in self._memory:
            return self._memory[symbol]

        # Deterministic seed from symbol content
        h = hashlib.sha256(symbol.encode("utf-8")).digest()
        sym_seed = int.from_bytes(h[:8], "little") ^ self.seed
        rng = np.random.RandomState(sym_seed & 0xFFFF_FFFF)

        vec = rng.choice([-1, 1], size=self.dim).astype(np.float32)
        self._memory[symbol] = vec
        return vec

    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Role-filler binding via element-wise multiplication.

        bind(a, b) is orthogonal to both a and b — it creates a
        *new* concept from the relation of two existing ones.
        """
        return a * b

    @staticmethod
    def bundle(*vecs: np.ndarray) -> np.ndarray:
        """Superposition via element-wise sum + sign.

        The result is similar to each input — it represents the
        *set* of all bundled concepts.
        """
        s = np.sum(np.stack(vecs), axis=0)
        # Replace zeros randomly to keep bipolar
        zeros = s == 0
        if np.any(zeros):
            s[zeros] = np.random.choice([-1, 1], size=int(zeros.sum()))
        return np.sign(s).astype(np.float32)

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity — the universal "how close?" measure."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def permute(v: np.ndarray, shift: int = 1) -> np.ndarray:
        """Cyclic permutation — encodes position / sequence order.

        permute(v, 1) is nearly orthogonal to v, which lets us
        encode ordered sequences without interference.
        """
        return np.roll(v, shift)

    def cleanup(self, v: np.ndarray) -> tuple[str, float]:
        """Find the closest known symbol to vector v.

        Returns (symbol, similarity).
        """
        best_sym = ""
        best_sim = -1.0
        for sym, mem_vec in self._memory.items():
            sim = self.similarity(v, mem_vec)
            if sim > best_sim:
                best_sim = sim
                best_sym = sym
        return best_sym, best_sim

    def cleanup_top_k(self, v: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Find the k closest known symbols to vector v."""
        scored = [
            (sym, self.similarity(v, mem_vec))
            for sym, mem_vec in self._memory.items()
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    # ------------------------------------------------------------------
    # Decomposition — break a concept into sub-particles
    # ------------------------------------------------------------------

    def decompose(self, symbol: str, n: int = 10) -> list[np.ndarray]:
        """Break a symbol's vector into n noisy sub-particles.

        Each particle carries a fraction of the original meaning
        plus individual variation — like dissolving a crystal into
        warm water.
        """
        base = self.encode(symbol)
        particles = []
        rng = np.random.RandomState(hash(symbol) & 0xFFFF_FFFF)
        for i in range(n):
            noise = rng.choice([-1, 1], size=self.dim).astype(np.float32)
            # Mix: 70% signal, 30% noise → particle retains meaning
            # but has its own "personality"
            mixed = 0.7 * base + 0.3 * noise
            particles.append(np.sign(mixed).astype(np.float32))
        return particles

    def encode_phrase(self, phrase: str) -> np.ndarray:
        """Encode a multi-word phrase using permutation-based sequencing.

        Each word's position is encoded via cyclic permutation,
        then all are bundled into a single vector.
        """
        words = phrase.strip().split()
        if not words:
            return np.zeros(self.dim, dtype=np.float32)
        positioned = []
        for i, w in enumerate(words):
            vec = self.encode(w.lower())
            positioned.append(self.permute(vec, shift=i))
        return self.bundle(*positioned)

    @property
    def known_symbols(self) -> list[str]:
        """List all symbols currently in memory."""
        return list(self._memory.keys())
