"""Active frontier state machine (plan PH1-WI04, spec §25, decisions D2.4/D2.6).

Tracks which virtual neurons hold resident transient state (plan §26) without
owning that state: ``ActiveFrontier`` maps uint64 virtual IDs to one of six
residency states::

    DORMANT -> CANDIDATE -> ACTIVE -> QUIESCENT -> EVICTING -> EVICTED
                   ^                     |  ^          |
                   |                     |  |          v
                   +------ re-entry -----+  +-- rescue-+

Hysteresis (no instant create/delete churn): a single observation never
materializes — ``CANDIDATE`` promotes to ``ACTIVE`` only after
``activate_count`` (>= 2) observations; a single idle tick never evicts —
``ACTIVE`` cools through ``QUIESCENT`` (after ``quiet_ticks`` idle) and only
``QUIESCENT`` past ``evict_ticks`` idle (strictly greater than
``quiet_ticks``) becomes ``EVICTING``. Abandoned candidates fall back to
``DORMANT`` after ``candidate_timeout_ticks`` without enough evidence.

Time model: event-driven ``observe()`` plus a periodic sweep ``update(tick)``.
All ticks are ints (D2.4); ``update`` ticks must be non-decreasing so replays
are deterministic. ``update`` iterates IDs in sorted order; transition lists
are therefore deterministic for identical call sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "RESIDENT_STATES",
    "ActiveFrontier",
    "FrontierEntry",
    "FrontierParams",
    "FrontierState",
]

_MASK64 = (1 << 64) - 1


class FrontierState(Enum):
    """Residency lifecycle of one virtual neuron."""

    DORMANT = "dormant"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    QUIESCENT = "quiescent"
    EVICTING = "evicting"
    EVICTED = "evicted"


RESIDENT_STATES = frozenset(
    {FrontierState.ACTIVE, FrontierState.QUIESCENT, FrontierState.EVICTING}
)
"""States that still hold resident transient state (count toward budgets)."""


@dataclass(frozen=True)
class FrontierParams:
    """Hysteresis thresholds. All tick counts are integer ticks (D2.4)."""

    activate_count: int = 3
    quiet_ticks: int = 100
    evict_ticks: int = 1000
    candidate_timeout_ticks: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.activate_count, int) or isinstance(
            self.activate_count, bool
        ):
            raise TypeError("activate_count must be an int")
        if self.activate_count < 2:
            raise ValueError("activate_count must be >= 2 (no single-event churn)")
        for name in ("quiet_ticks", "evict_ticks", "candidate_timeout_ticks"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.evict_ticks <= self.quiet_ticks:
            raise ValueError("evict_ticks must exceed quiet_ticks (hysteresis band)")


@dataclass
class FrontierEntry:
    """Per-neuron frontier record."""

    neuron_id: int
    state: FrontierState = FrontierState.CANDIDATE
    activity_count: int = 0
    first_seen_tick: int = 0
    last_activity_tick: int = 0
    resident_ticks: int = 0


@dataclass
class ActiveFrontier:
    """Six-state residency tracker over uint64 virtual IDs."""

    params: FrontierParams = field(default_factory=FrontierParams)
    _entries: dict[int, FrontierEntry] = field(default_factory=dict, init=False)
    _tick: int = field(default=0, init=False)
    activations: int = field(default=0, init=False)
    reactivations: int = field(default=0, init=False)
    quiet_demotions: int = field(default=0, init=False)
    eviction_requests: int = field(default=0, init=False)
    evictions: int = field(default=0, init=False)
    candidate_abandons: int = field(default=0, init=False)

    def __len__(self) -> int:
        return len(self._entries)

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

    def state_of(self, neuron_id: int) -> FrontierState:
        """Residency state; untracked IDs are DORMANT (query creates nothing)."""
        self._check_id(neuron_id)
        entry = self._entries.get(neuron_id)
        return entry.state if entry is not None else FrontierState.DORMANT

    def observe(self, neuron_id: int, tick: int) -> FrontierState:
        """Register activity at ``tick``; returns the resulting state."""
        self._check_id(neuron_id)
        self._check_tick(tick)
        entry = self._entries.get(neuron_id)
        if entry is None:
            self._entries[neuron_id] = FrontierEntry(
                neuron_id=neuron_id,
                state=FrontierState.CANDIDATE,
                activity_count=1,
                first_seen_tick=tick,
                last_activity_tick=tick,
            )
            return FrontierState.CANDIDATE
        entry.last_activity_tick = max(entry.last_activity_tick, tick)
        if entry.state is FrontierState.DORMANT:
            entry.state = FrontierState.CANDIDATE
            entry.activity_count = 1
            entry.first_seen_tick = tick
        elif entry.state is FrontierState.CANDIDATE:
            entry.activity_count += 1
            if entry.activity_count >= self.params.activate_count:
                entry.state = FrontierState.ACTIVE
                self.activations += 1
        elif entry.state is FrontierState.EVICTED:
            entry.state = FrontierState.CANDIDATE
            entry.activity_count = 1
            entry.first_seen_tick = tick
        elif entry.state in (FrontierState.QUIESCENT, FrontierState.EVICTING):
            entry.state = FrontierState.ACTIVE
            self.reactivations += 1
        return entry.state

    def update(self, tick: int) -> list[tuple[int, FrontierState, FrontierState]]:
        """Advance time-driven demotions; returns deterministic transitions."""
        self._check_tick(tick)
        if tick < self._tick:
            raise ValueError("update tick must be non-decreasing")
        self._tick = tick
        p = self.params
        transitions: list[tuple[int, FrontierState, FrontierState]] = []
        for neuron_id in sorted(self._entries):
            entry = self._entries[neuron_id]
            old = entry.state
            if entry.state in RESIDENT_STATES:
                entry.resident_ticks += 1
            if entry.state is FrontierState.CANDIDATE:
                if tick - entry.first_seen_tick >= p.candidate_timeout_ticks:
                    entry.state = FrontierState.DORMANT
                    self.candidate_abandons += 1
            elif entry.state is FrontierState.ACTIVE:
                if tick - entry.last_activity_tick >= p.quiet_ticks:
                    entry.state = FrontierState.QUIESCENT
                    self.quiet_demotions += 1
            elif entry.state is FrontierState.QUIESCENT:
                if tick - entry.last_activity_tick >= p.evict_ticks:
                    entry.state = FrontierState.EVICTING
                    self.eviction_requests += 1
            elif entry.state is FrontierState.EVICTING:
                entry.state = FrontierState.EVICTED
                self.evictions += 1
            if entry.state is not old:
                transitions.append((neuron_id, old, entry.state))
        return transitions

    def counts(self) -> dict[FrontierState, int]:
        """Tracked-record count per state (untracked virtual IDs excluded)."""
        out: dict[FrontierState, int] = {state: 0 for state in FrontierState}
        for entry in self._entries.values():
            out[entry.state] += 1
        return out

    def resident_ids(self) -> list[int]:
        """Sorted IDs still holding transient state (budget-relevant set)."""
        return sorted(
            nid for nid, e in self._entries.items() if e.state in RESIDENT_STATES
        )

    def stats(self) -> dict[str, int]:
        """Cumulative counters plus current residency snapshot."""
        counts = self.counts()
        resident = sum(counts[s] for s in RESIDENT_STATES)
        return {
            "tracked": len(self._entries),
            "resident": resident,
            "activations": self.activations,
            "reactivations": self.reactivations,
            "quiet_demotions": self.quiet_demotions,
            "eviction_requests": self.eviction_requests,
            "evictions": self.evictions,
            "candidate_abandons": self.candidate_abandons,
        }
