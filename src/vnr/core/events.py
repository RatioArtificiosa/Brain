"""Event engine primitives (plan PH1-WI03, spec §24, decision D2.4).

Events are the runtime's unit of causal work: a spike (or delivery) scheduled
on an integer tick. ``NeuralEvent`` is an immutable record; ``EventQueue`` is
the reference priority queue all later backends (NumPy/CUDA/GeNN) drain from.

Ordering is ``(tick, seq)``: earliest tick first, FIFO within a tick via an
assign-on-push sequence number. The queue accepts out-of-order pushes and
always pops in order. Batch drains (``drain_tick`` / ``drain_until``) exist
for the future kernel boundary (§5: orchestrate in batches, never per-event
Python in production).
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

__all__ = ["EventQueue", "NeuralEvent"]

_MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class NeuralEvent:
    """One scheduled spike/delivery on an integer tick (D2.4).

    ``tick`` is the delivery tick (non-negative int). ``source_id`` /
    ``target_id`` are uint64 virtual IDs (plan §20). ``weight`` is the
    synaptic weight in mV applied on delivery. ``causal_parent`` is the
    ``seq`` of the spike event that caused this delivery (None for
    externally injected spikes). ``trace_id`` groups causally related
    events for sampled traces (PH5-WI05). ``seq`` is assigned by
    ``EventQueue.push`` for deterministic FIFO order within a tick; it
    must not be set by hand except in tests comparing explicit order.
    """

    tick: int
    source_id: int
    target_id: int
    weight: float = 1.0
    causal_parent: int | None = None
    trace_id: int = 0
    seq: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.tick, int) or isinstance(self.tick, bool):
            raise TypeError("tick must be an int")
        if self.tick < 0:
            raise ValueError("tick must be non-negative")
        for name in ("source_id", "target_id"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
            if not 0 <= value <= _MASK64:
                raise ValueError(f"{name} must fit in uint64")
        if not isinstance(self.weight, (int, float)) or isinstance(self.weight, bool):
            raise TypeError("weight must be a number")
        if not math.isfinite(float(self.weight)):
            raise ValueError("weight must be finite")
        if self.causal_parent is not None:
            if not isinstance(self.causal_parent, int) or isinstance(
                self.causal_parent, bool
            ):
                raise TypeError("causal_parent must be a non-negative int or None")
            if self.causal_parent < 0:
                raise ValueError("causal_parent must be a non-negative int or None")
        if not isinstance(self.trace_id, int) or isinstance(self.trace_id, bool):
            raise TypeError("trace_id must be an int")
        if not 0 <= self.trace_id <= _MASK64:
            raise ValueError("trace_id must fit in uint64")
        if not isinstance(self.seq, int) or isinstance(self.seq, bool):
            raise TypeError("seq must be a non-negative int")
        if self.seq < 0:
            raise ValueError("seq must be a non-negative int")


@dataclass
class EventQueue:
    """Min-heap priority queue on ``(tick, seq)`` with batch drains."""

    _heap: list[tuple[int, int, NeuralEvent]] = field(default_factory=list, init=False)
    _counter: itertools.count = field(default_factory=itertools.count, init=False)

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        """True when no events are scheduled."""
        return not self._heap

    @property
    def next_tick(self) -> int | None:
        """Earliest scheduled tick, or None when empty."""
        return self._heap[0][0] if self._heap else None

    def push(self, event: NeuralEvent) -> NeuralEvent:
        """Schedule ``event``, assigning a FIFO ``seq``. Returns stored event."""
        stored = NeuralEvent(
            tick=event.tick,
            source_id=event.source_id,
            target_id=event.target_id,
            weight=event.weight,
            causal_parent=event.causal_parent,
            trace_id=event.trace_id,
            seq=next(self._counter),
        )
        heapq.heappush(self._heap, (stored.tick, stored.seq, stored))
        return stored

    def peek(self) -> NeuralEvent | None:
        """Earliest event without removing it, or None when empty."""
        return self._heap[0][2] if self._heap else None

    def pop(self) -> NeuralEvent:
        """Remove and return the earliest event. Raises IndexError if empty."""
        if not self._heap:
            raise IndexError("pop from empty EventQueue")
        return heapq.heappop(self._heap)[2]

    def drain_tick(self, tick: int) -> list[NeuralEvent]:
        """Remove and return all events at exactly ``tick`` in FIFO order."""
        out: list[NeuralEvent] = []
        while self._heap and self._heap[0][0] == tick:
            out.append(heapq.heappop(self._heap)[2])
        return out

    def drain_until(self, max_tick: int) -> list[NeuralEvent]:
        """Remove and return all events with ``tick <= max_tick`` in order."""
        if not isinstance(max_tick, int) or isinstance(max_tick, bool):
            raise TypeError("max_tick must be an int")
        out: list[NeuralEvent] = []
        while self._heap and self._heap[0][0] <= max_tick:
            out.append(heapq.heappop(self._heap)[2])
        return out

    def drain_all(self) -> list[NeuralEvent]:
        """Remove and return every scheduled event in ``(tick, seq)`` order."""
        out = [entry[2] for entry in sorted(self._heap)]
        self._heap.clear()
        return out
