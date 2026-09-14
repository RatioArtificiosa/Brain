"""Materialization and eviction (plan PH1-WI05, spec §26, decision D2.6).

Three state layers, separated from day one:

- structural: frozen ``LIFParams`` per neuron, fixed at build time;
- learned: persistent ``dict[str, float]`` per neuron (empty for static nets,
  but the layer and its write path exist now, not later);
- transient: evictable ``LIFNeuron`` runtime state, resident only while the
  neuron is materialized.

``Materializer`` owns the layers. ``materialize_neuron`` builds transient
state from the structural record plus any eviction snapshot; ``evict_neuron``
persists the learned layer FIRST, then snapshots transient state and frees
it. Nothing transient may ever silently carry learned information (D2.6):
learned values live only in the learned layer and are copied by value, never
aliased, across every boundary.

``NeuronRuntimeState`` is a frozen snapshot copy: mutating it cannot affect
resident state. Evict → rematerialize is bit-exact for static nets (§67):
floats are copied exactly, never quantized or reconstructed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from vnr.core.neuron import LIFNeuron, LIFParams

__all__ = [
    "Materializer",
    "NeuronRuntimeState",
    "TransientSnapshot",
]

_MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class NeuronRuntimeState:
    """Plan §7 contract: snapshot view of one materialized neuron."""

    neuron_id: int
    v: float
    refractory_until_tick: int
    tick: int


@dataclass(frozen=True)
class TransientSnapshot:
    """Exact transient image saved at eviction, restored at rematerialize."""

    v: float
    refractory_until_tick: int


@dataclass
class Materializer:
    """Three-layer store with explicit materialize/evict lifecycle."""

    _structural: dict[int, LIFParams] = field(default_factory=dict, init=False)
    _learned: dict[int, dict[str, float]] = field(default_factory=dict, init=False)
    _snapshots: dict[int, TransientSnapshot] = field(default_factory=dict, init=False)
    _resident: dict[int, LIFNeuron] = field(default_factory=dict, init=False)
    writes: list[tuple[str, int]] = field(default_factory=list, init=False)
    materializations: int = field(default=0, init=False)
    cache_hits: int = field(default=0, init=False)
    evictions: int = field(default=0, init=False)
    learned_writebacks: int = field(default=0, init=False)

    @staticmethod
    def _check_id(neuron_id: int) -> None:
        if not isinstance(neuron_id, int) or isinstance(neuron_id, bool):
            raise TypeError("neuron_id must be an int")
        if not 0 <= neuron_id <= _MASK64:
            raise ValueError("neuron_id must fit in uint64")

    @staticmethod
    def _check_tick(tick: int) -> None:
        if not isinstance(tick, int) or isinstance(tick, bool):
            raise TypeError("tick must be an int")
        if tick < 0:
            raise ValueError("tick must be non-negative")

    def __len__(self) -> int:
        return len(self._structural)

    def register(self, neuron_id: int, params: LIFParams | None = None) -> None:
        """Fix the structural layer for one neuron. Duplicate → ValueError."""
        self._check_id(neuron_id)
        if params is not None and not isinstance(params, LIFParams):
            raise TypeError("params must be a LIFParams or None")
        if neuron_id in self._structural:
            raise ValueError(f"neuron {neuron_id} already registered")
        self._structural[neuron_id] = params or LIFParams()
        self._learned[neuron_id] = {}

    def set_learned(self, neuron_id: int, key: str, value: float) -> None:
        """Write one persistent learned value. Unregistered → KeyError."""
        self._check_id(neuron_id)
        if not isinstance(key, str) or not key:
            raise ValueError("learned key must be a non-empty str")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("learned value must be a number")
        if not math.isfinite(value):
            raise ValueError("learned value must be finite")
        try:
            learned = self._learned[neuron_id]
        except KeyError:
            raise KeyError(f"neuron {neuron_id} not registered") from None
        learned[key] = float(value)

    def learned_of(self, neuron_id: int) -> dict[str, float]:
        """Copy of the persistent learned layer (never an alias)."""
        self._check_id(neuron_id)
        try:
            return dict(self._learned[neuron_id])
        except KeyError:
            raise KeyError(f"neuron {neuron_id} not registered") from None

    def is_resident(self, neuron_id: int) -> bool:
        """True while transient state is materialized."""
        self._check_id(neuron_id)
        return neuron_id in self._resident

    def resident_ids(self) -> list[int]:
        """Sorted resident IDs (deterministic order for sweeps)."""
        return sorted(self._resident)

    def materialize_neuron(self, neuron_id: int, tick: int = 0) -> NeuronRuntimeState:
        """Build (or return) transient state. Unregistered → KeyError."""
        self._check_id(neuron_id)
        self._check_tick(tick)
        try:
            params = self._structural[neuron_id]
        except KeyError:
            raise KeyError(f"neuron {neuron_id} not registered") from None
        resident = self._resident.get(neuron_id)
        if resident is not None:
            self.cache_hits += 1
        else:
            resident = LIFNeuron(params=params)
            snapshot = self._snapshots.get(neuron_id)
            if snapshot is not None:
                resident.state.v = snapshot.v
                resident.state.refractory_until_tick = snapshot.refractory_until_tick
            self._resident[neuron_id] = resident
            self.materializations += 1
        return NeuronRuntimeState(
            neuron_id=neuron_id,
            v=resident.state.v,
            refractory_until_tick=resident.state.refractory_until_tick,
            tick=tick,
        )

    def step_neuron(self, neuron_id: int, tick: int, i_syn: float = 0.0) -> bool:
        """Advance one resident neuron. Non-resident → KeyError (no silent
        auto-materialization: the caller owns the lifecycle)."""
        self._check_id(neuron_id)
        self._check_tick(tick)
        try:
            resident = self._resident[neuron_id]
        except KeyError:
            raise KeyError(f"neuron {neuron_id} not resident") from None
        return resident.step(tick, i_syn)

    def evict_neuron(self, neuron_id: int) -> TransientSnapshot:
        """Persist learned layer FIRST, snapshot transient, free resident.

        Non-resident → KeyError. The ``writes`` journal records
        ``("learned", id)`` before ``("transient", id)`` on every eviction.
        """
        self._check_id(neuron_id)
        try:
            resident = self._resident[neuron_id]
        except KeyError:
            raise KeyError(f"neuron {neuron_id} not resident") from None
        self.writes.append(("learned", neuron_id))
        self.learned_writebacks += 1
        self.writes.append(("transient", neuron_id))
        snapshot = TransientSnapshot(
            v=resident.state.v,
            refractory_until_tick=resident.state.refractory_until_tick,
        )
        self._snapshots[neuron_id] = snapshot
        del self._resident[neuron_id]
        self.evictions += 1
        return snapshot

    def stats(self) -> dict[str, int]:
        """Cumulative lifecycle counters plus current residency."""
        return {
            "registered": len(self._structural),
            "resident": len(self._resident),
            "materializations": self.materializations,
            "cache_hits": self.cache_hits,
            "evictions": self.evictions,
            "learned_writebacks": self.learned_writebacks,
        }
