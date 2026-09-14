"""Generator framework (plan PH4-WI04, spec §16).

Every generator is a pure function: (source edges, params, seed) -> (new
edges, metadata). Determinism is structural — same inputs always give the
same graph (``random.Random`` instances, never global random). Metadata
records the full lineage so any generated graph is traceable to its parent
dataset, generator, version, and seed (§16 metadata block).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = [
    "GENERATOR_VERSION",
    "GenerationMetadata",
    "GeneratorParams",
    "mean_weight",
    "rng_for",
]

GENERATOR_VERSION = 1


@dataclass(frozen=True)
class GeneratorParams:
    """Validated common knobs. Individual generators add their own dataclass."""

    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an int")


@dataclass(frozen=True)
class GenerationMetadata:
    """Lineage block (§16): what made this graph, exactly."""

    generator: str
    version: int
    seed: int
    params: dict[str, float | int | str]
    parent: str
    n_in: int
    e_in: int
    n_out: int
    e_out: int
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def as_dict(self) -> dict:
        return {
            "generator": self.generator,
            "version": self.version,
            "seed": self.seed,
            "params": self.params,
            "parent": self.parent,
            "n_in": self.n_in,
            "e_in": self.e_in,
            "n_out": self.n_out,
            "e_out": self.e_out,
            "timestamp": self.timestamp,
        }


Edges = dict[int, list[tuple[int, float]]]


def rng_for(seed: int, stream: str) -> random.Random:
    """Independent deterministic stream per purpose (no cross-talk).

    random.Random takes str seeds (tuples are rejected), so the stream key
    is a string — same (seed, stream) always replays identically.
    """
    return random.Random(f"vnr:{seed}:{stream}")


def mean_weight(edges: Edges) -> float:
    total, count = 0.0, 0
    for outs in edges.values():
        for _, w in outs:
            total += w
            count += 1
    return (total / count) if count else 0.5


def check_edges(edges: Edges, n: int, what: str) -> None:
    """Validate generated edges (ids in range, weights finite, no self-loops
    unless the generator documents them)."""
    for src, outs in edges.items():
        if not 0 <= src < n:
            raise ValueError(f"{what}: source {src} out of range 0..{n}")
        for tgt, w in outs:
            if not 0 <= tgt < n:
                raise ValueError(f"{what}: target {tgt} out of range 0..{n}")
            if (
                not isinstance(w, (int, float))
                or isinstance(w, bool)
                or not math.isfinite(w)
            ):
                raise ValueError(f"{what}: bad weight {w!r} on {src}->{tgt}")


@runtime_checkable
class Generator(Protocol):
    """Structural contract (plan §7): generate + describe."""

    @property
    def name(self) -> str: ...
    def generate(
        self, source: Edges, n_source: int
    ) -> tuple[Edges, GenerationMetadata]: ...
