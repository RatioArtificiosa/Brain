"""Procedural connectivity (plan PH1-WI06, spec §22, decision D2.3).

Outgoing edges are generated on demand, never stored: for a source neuron the
pipeline samples ``out_degree`` target IDs, assigns weights, and assigns
delivery ticks. Every stage is a pure function of the D2.3 key
``(global_seed, source_id, target_population_id, connectivity_version)`` —
no global randomness anywhere, so identical calls return identical events
(§68) and a seed change yields controlled divergence (same pipeline, new
targets).

Reference-oracle RNG is stdlib ``hashlib`` (blake2b) over the key, giving
stateless counter-style determinism. The PH3 CUDA backend will use torch
Philox with the identical key layout; target sets must match exactly.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field

from vnr.core.events import NeuralEvent

__all__ = [
    "ConnectivityParams",
    "ProceduralConnectivity",
    "keyed_uniform",
]

_MASK64 = (1 << 64) - 1
_PACK_EDGE = struct.Struct("<4Q2I").pack
_MAX_DEGREE = 1_000_000

_DOMAIN_TARGET = 0
_DOMAIN_UNIFORM = 1


def _keyed_int(
    global_seed: int,
    source_id: int,
    population_id: int,
    version: int,
    salt: int,
    domain: int,
) -> int:
    """One blake2b draw from the D2.3 key. ``domain`` keeps the target and
    uniform draw streams disjoint."""
    digest = hashlib.blake2b(
        _PACK_EDGE(
            global_seed & _MASK64,
            source_id & _MASK64,
            population_id & _MASK64,
            version & _MASK64,
            salt & 0xFFFFFFFF,
            domain,
        ),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little")


def keyed_uniform(
    global_seed: int, source_id: int, population_id: int, version: int, salt: int
) -> float:
    """Deterministic uniform in [0, 1) from the D2.3 key plus a stage salt.

    Stateless: no RNG object, no shared state, safe to call in any order.
    """
    if not isinstance(salt, int) or isinstance(salt, bool) or salt < 0:
        raise ValueError("salt must be a non-negative int")
    return (
        _keyed_int(
            global_seed, source_id, population_id, version, salt, _DOMAIN_UNIFORM
        )
        / 2**64
    )


@dataclass(frozen=True)
class ConnectivityParams:
    """Static connectivity rule (structural layer input, plan §26)."""

    global_seed: int = 0
    target_population_id: int = 0
    connectivity_version: int = 1
    out_degree: int = 64
    weight: float = 0.5
    weight_std: float = 0.0
    weight_bits: int | None = None
    delay_ticks: int = 1

    def __post_init__(self) -> None:
        for name in (
            "global_seed",
            "target_population_id",
            "connectivity_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
            if not 0 <= value <= _MASK64:
                raise ValueError(f"{name} must fit in uint64")
        if not isinstance(self.out_degree, int) or isinstance(self.out_degree, bool):
            raise TypeError("out_degree must be an int")
        if not 1 <= self.out_degree <= _MAX_DEGREE:
            raise ValueError(f"out_degree must be in 1..{_MAX_DEGREE}")
        if not isinstance(self.weight, (int, float)) or isinstance(self.weight, bool):
            raise TypeError("weight must be a number")
        if not math.isfinite(self.weight):
            raise ValueError("weight must be finite")
        if not isinstance(self.weight_std, (int, float)) or isinstance(
            self.weight_std, bool
        ):
            raise TypeError("weight_std must be a number")
        if not math.isfinite(self.weight_std) or self.weight_std < 0:
            raise ValueError("weight_std must be a non-negative finite number")
        if self.weight_bits is not None:
            if not isinstance(self.weight_bits, int) or isinstance(
                self.weight_bits, bool
            ):
                raise TypeError("weight_bits must be an int or None")
            if not 1 <= self.weight_bits <= 32:
                raise ValueError("weight_bits must be in 1..32 or None")
        if not isinstance(self.delay_ticks, int) or isinstance(self.delay_ticks, bool):
            raise TypeError("delay_ticks must be an int")
        if self.delay_ticks < 0:
            raise ValueError("delay_ticks must be non-negative")


@dataclass
class ProceduralConnectivity:
    """§22 pipeline bound to one rule set (the plan's ``ctx``)."""

    params: ConnectivityParams = field(default_factory=ConnectivityParams)

    @staticmethod
    def _check_source(source_id: int) -> None:
        if not isinstance(source_id, int) or isinstance(source_id, bool):
            raise TypeError("source_id must be an int")
        if not 0 <= source_id <= _MASK64:
            raise ValueError("source_id must fit in uint64")

    @staticmethod
    def _check_tick(tick: int) -> None:
        if not isinstance(tick, int) or isinstance(tick, bool):
            raise TypeError("tick must be an int")
        if tick < 0:
            raise ValueError("tick must be non-negative")

    def sample_targets(self, source_id: int) -> list[int]:
        """Stage 1: deterministic target IDs. Stable across ticks — the same
        source always projects to the same set under one rule version."""
        self._check_source(source_id)
        p = self.params
        return [
            _keyed_int(
                p.global_seed,
                source_id,
                p.target_population_id,
                p.connectivity_version,
                edge,
                _DOMAIN_TARGET,
            )
            for edge in range(p.out_degree)
        ]

    def assign_weights(self, source_id: int) -> list[float]:
        """Stage 2: per-edge weights.

        Defaults reproduce the uniform rule weight exactly (fast path, and the
        §68 determinism gate pins this behavior). With ``weight_std > 0`` each
        edge draws ``weight + std * (2u - 1)`` from the keyed uniform stream
        (same D2.3 key family, edge index as salt; disjoint from target draws
        by domain tag and from drive draws by population id). With
        ``weight_bits`` set, weights quantize to ``2**bits`` uniform levels —
        the compression knob the §71 equivalence experiment turns.
        """
        self._check_source(source_id)
        p = self.params
        if p.weight_std == 0 and p.weight_bits is None:
            return [float(p.weight)] * p.out_degree
        lo = float(p.weight) - float(p.weight_std)
        hi = float(p.weight) + float(p.weight_std)
        out = []
        for edge in range(p.out_degree):
            u = keyed_uniform(
                p.global_seed,
                source_id,
                p.target_population_id,
                p.connectivity_version,
                edge,
            )
            w = float(p.weight) + float(p.weight_std) * (2.0 * u - 1.0)
            if p.weight_bits is not None and hi > lo:
                levels = 2**p.weight_bits - 1
                step = (hi - lo) / levels
                w = lo + round((w - lo) / (hi - lo) * levels) * step
            out.append(w)
        return out

    def assign_delays(self, source_id: int) -> list[int]:
        """Stage 3: per-edge delivery delays in integer ticks (D2.4)."""
        self._check_source(source_id)
        return [self.params.delay_ticks] * self.params.out_degree

    def generate_outgoing_events(self, source_id: int, tick: int) -> list[NeuralEvent]:
        """Full §22 pipeline: sample → weigh → delay → events.

        Delivery tick is ``tick + delay``; targets never depend on ``tick``,
        so the same source re-spiking later replays the same fan-out.
        """
        self._check_source(source_id)
        self._check_tick(tick)
        targets = self.sample_targets(source_id)
        weights = self.assign_weights(source_id)
        delays = self.assign_delays(source_id)
        return [
            NeuralEvent(
                tick=tick + delay,
                source_id=source_id,
                target_id=target,
                weight=weight,
            )
            for target, weight, delay in zip(targets, weights, delays)
        ]
