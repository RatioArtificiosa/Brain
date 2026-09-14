"""Connectome dataset abstraction (plan PH4-WI01, spec §12).

A ``ConnectomeDataset`` is anything the generators and analysis can consume:
neuron IDs, directed weighted edges, and provenance metadata. Three
implementations ship now; FlyWire (PH4-WI02) will be the fourth:

- ``ToyDataset``: hand-designed 12-neuron microcircuit (chain + recurrence +
  hub). Exact structure, for tests — never for science claims.
- ``SyntheticDataset``: seeded planted-partition graph (dense modules, sparse
  bridges). Foreshadows §18 modular expansion; determinism via ``random.Random``.
- ``RandomDataset``: seeded directed G(n, p) with uniform weights. The null
  model every structural claim must beat.

All randomness is stdlib ``random.Random`` (or the keyed stream where the
oracle needs it) — no global ``random`` calls anywhere, so datasets are
reproducible from (kind, seed, params) alone.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["ConnectomeDataset", "RandomDataset", "SyntheticDataset", "ToyDataset"]


@runtime_checkable
class ConnectomeDataset(Protocol):
    """Structural contract every dataset honors (plan §7, spec §12)."""

    @property
    def name(self) -> str: ...
    @property
    def provenance(self) -> dict[str, str]: ...
    def neurons(self) -> list[int]: ...
    def successors(self, neuron_id: int) -> list[tuple[int, float]]: ...
    def edge_count(self) -> int: ...


def _check_weights(edges: dict[int, list[tuple[int, float]]], n: int) -> None:
    seen: set[int] = set()
    for src, outs in edges.items():
        if not isinstance(src, int) or isinstance(src, bool) or not 0 <= src < n:
            raise ValueError(f"source {src} out of range 0..{n}")
        for tgt, w in outs:
            if not isinstance(tgt, int) or isinstance(tgt, bool) or not 0 <= tgt < n:
                raise ValueError(f"target {tgt} of {src} out of range 0..{n}")
            if (
                not isinstance(w, (int, float))
                or isinstance(w, bool)
                or not math.isfinite(w)
            ):
                raise ValueError(f"weight {w!r} on {src}->{tgt} must be finite")
        seen.add(src)


@dataclass
class ToyDataset:
    """Hand-built microcircuit: chain 0→1→…→7, 7→2 recurrence, hub 8 fanning
    to all, 9/10/11 isolated-but-registered (isolation checks need them)."""

    _edges: dict[int, list[tuple[int, float]]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        chain = {i: [(i + 1, 0.6)] for i in range(7)}
        chain[7] = [(2, 0.8)]
        chain[8] = [(i, 0.4) for i in range(8)]
        chain.update({9: [], 10: [], 11: []})
        _check_weights(chain, 12)
        self._edges = chain

    @property
    def name(self) -> str:
        return "toy-v1"

    @property
    def provenance(self) -> dict[str, str]:
        return {
            "source": "hand-designed",
            "version": "1",
            "neurons": "12",
            "seed": "n/a",
        }

    def neurons(self) -> list[int]:
        return list(range(12))

    def successors(self, neuron_id: int) -> list[tuple[int, float]]:
        try:
            return list(self._edges[neuron_id])
        except KeyError:
            raise KeyError(f"neuron {neuron_id} not in toy dataset") from None

    def edge_count(self) -> int:
        return sum(len(v) for v in self._edges.values())


@dataclass
class SyntheticDataset:
    """Planted-partition modules: ``n_modules`` dense blocks (p_in) with sparse
    bridges (p_out). Seeded; same params always give the same graph."""

    n_neurons: int = 300
    n_modules: int = 3
    p_in: float = 0.08
    p_out: float = 0.005
    weight_lo: float = 0.2
    weight_hi: float = 0.8
    seed: int = 0

    _edges: dict[int, list[tuple[int, float]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.n_neurons <= 0 or self.n_modules <= 0:
            raise ValueError("n_neurons and n_modules must be positive")
        if self.n_modules > self.n_neurons:
            raise ValueError("n_modules cannot exceed n_neurons")
        for p in (self.p_in, self.p_out):
            if not 0.0 <= p <= 1.0:
                raise ValueError("connection probabilities must be in [0, 1]")
        if self.weight_hi < self.weight_lo:
            raise ValueError("weight_hi must be >= weight_lo")
        rng = random.Random(self.seed)
        edges: dict[int, list[tuple[int, float]]] = {
            i: [] for i in range(self.n_neurons)
        }
        for src in range(self.n_neurons):
            for tgt in range(self.n_neurons):
                if src == tgt:
                    continue
                same = (src % self.n_modules) == (tgt % self.n_modules)
                if rng.random() < (self.p_in if same else self.p_out):
                    w = self.weight_lo + rng.random() * (
                        self.weight_hi - self.weight_lo
                    )
                    edges[src].append((tgt, w))
        _check_weights(edges, self.n_neurons)
        self._edges = edges

    @property
    def name(self) -> str:
        return f"synthetic-m{self.n_modules}-n{self.n_neurons}-s{self.seed}"

    @property
    def provenance(self) -> dict[str, str]:
        return {
            "source": "planted-partition",
            "version": "1",
            "seed": str(self.seed),
            "params": f"modules={self.n_modules},p_in={self.p_in},p_out={self.p_out}",
        }

    def neurons(self) -> list[int]:
        return list(range(self.n_neurons))

    def successors(self, neuron_id: int) -> list[tuple[int, float]]:
        try:
            return list(self._edges[neuron_id])
        except KeyError:
            raise KeyError(f"neuron {neuron_id} not in synthetic dataset") from None

    def edge_count(self) -> int:
        return sum(len(v) for v in self._edges.values())


@dataclass
class RandomDataset:
    """Directed G(n, p) null model, uniform weights, seeded."""

    n_neurons: int = 300
    p_edge: float = 0.02
    weight_lo: float = 0.2
    weight_hi: float = 0.8
    seed: int = 0

    _edges: dict[int, list[tuple[int, float]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.n_neurons <= 0:
            raise ValueError("n_neurons must be positive")
        if not 0.0 <= self.p_edge <= 1.0:
            raise ValueError("p_edge must be in [0, 1]")
        if self.weight_hi < self.weight_lo:
            raise ValueError("weight_hi must be >= weight_lo")
        rng = random.Random(self.seed)
        edges: dict[int, list[tuple[int, float]]] = {
            i: [] for i in range(self.n_neurons)
        }
        for src in range(self.n_neurons):
            for tgt in range(self.n_neurons):
                if src == tgt:
                    continue
                if rng.random() < self.p_edge:
                    w = self.weight_lo + rng.random() * (
                        self.weight_hi - self.weight_lo
                    )
                    edges[src].append((tgt, w))
        _check_weights(edges, self.n_neurons)
        self._edges = edges

    @property
    def name(self) -> str:
        return f"random-n{self.n_neurons}-p{self.p_edge}-s{self.seed}"

    @property
    def provenance(self) -> dict[str, str]:
        return {"source": "directed-gnp", "version": "1", "seed": str(self.seed)}

    def neurons(self) -> list[int]:
        return list(range(self.n_neurons))

    def successors(self, neuron_id: int) -> list[tuple[int, float]]:
        try:
            return list(self._edges[neuron_id])
        except KeyError:
            raise KeyError(f"neuron {neuron_id} not in random dataset") from None

    def edge_count(self) -> int:
        return sum(len(v) for v in self._edges.values())
