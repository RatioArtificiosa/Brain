"""PH1-WI03 tests: NeuralEvent validation + EventQueue ordering/drains."""

import itertools
import random
import time

import pytest

from vnr.core.events import EventQueue, NeuralEvent


def _ev(tick: int, source: int = 1, target: int = 2) -> NeuralEvent:
    return NeuralEvent(tick=tick, source_id=source, target_id=target)


def test_push_assigns_fifo_seq_within_tick():
    q = EventQueue()
    a = q.push(_ev(5, source=1))
    b = q.push(_ev(5, source=2))
    c = q.push(_ev(5, source=3))
    assert (a.seq, b.seq, c.seq) == (0, 1, 2)
    assert [e.source_id for e in q.drain_tick(5)] == [1, 2, 3]


def test_pop_orders_by_tick_despite_out_of_order_push():
    q = EventQueue()
    for tick in (10, 0, 7, 3, 3, 100, 1):
        q.push(_ev(tick))
    assert [e.tick for e in q.drain_all()] == [0, 1, 3, 3, 7, 10, 100]


def test_100k_ordering_and_fifo():
    rng = random.Random(1234)
    ticks = [rng.randrange(0, 5000) for _ in range(100_000)]
    q = EventQueue()
    start = time.perf_counter()
    for i, tick in enumerate(ticks):
        q.push(_ev(tick, source=i % 1000, target=(i * 7) % 1000))
    push_s = time.perf_counter() - start
    start = time.perf_counter()
    out = q.drain_all()
    drain_s = time.perf_counter() - start
    assert len(out) == 100_000
    keys = [(e.tick, e.seq) for e in out]
    assert keys == sorted(keys)
    assert all(b >= a for a, b in itertools.pairwise(keys))
    print(f"\n100K push: {push_s:.2f}s drain: {drain_s:.2f}s")


def test_drain_until_and_next_tick_peek():
    q = EventQueue()
    for tick in (0, 2, 2, 5, 9):
        q.push(_ev(tick))
    assert q.next_tick == 0
    assert q.peek() is not None and q.peek().tick == 0
    batch = q.drain_until(2)
    assert [e.tick for e in batch] == [0, 2, 2]
    assert q.next_tick == 5
    assert len(q) == 2
    assert q.drain_until(1) == []
    assert [e.tick for e in q.drain_all()] == [5, 9]
    assert q.is_empty()
    assert q.next_tick is None
    assert q.peek() is None


def test_pop_empty_raises_and_drain_tick_missing_is_empty_list():
    q = EventQueue()
    with pytest.raises(IndexError):
        q.pop()
    assert q.drain_tick(42) == []


def test_causal_parent_and_trace_ids_preserved():
    q = EventQueue()
    parent = q.push(NeuralEvent(tick=1, source_id=9, target_id=10, trace_id=77))
    child = q.push(
        NeuralEvent(
            tick=2,
            source_id=10,
            target_id=11,
            weight=0.5,
            causal_parent=parent.seq,
            trace_id=77,
        )
    )
    assert child.causal_parent == parent.seq == 0
    assert child.trace_id == 77
    first, second = q.pop(), q.pop()
    assert (first.seq, second.seq) == (0, 1)
    assert second.causal_parent == first.seq


def test_replay_identical_for_same_push_sequence():
    def run() -> list[tuple[int, int]]:
        q = EventQueue()
        rng = random.Random(99)
        for _ in range(5000):
            q.push(_ev(rng.randrange(100)))
        return [(e.tick, e.seq) for e in q.drain_all()]

    assert run() == run()


def test_invalid_events_rejected():
    with pytest.raises(ValueError):
        NeuralEvent(tick=-1, source_id=1, target_id=2)
    with pytest.raises(TypeError):
        NeuralEvent(tick=1.5, source_id=1, target_id=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NeuralEvent(tick=0, source_id=-1, target_id=2)
    with pytest.raises(ValueError):
        NeuralEvent(tick=0, source_id=1, target_id=2**64)
    with pytest.raises(ValueError):
        NeuralEvent(tick=0, source_id=1, target_id=2, weight=float("inf"))
    with pytest.raises(TypeError):
        NeuralEvent(tick=0, source_id=1, target_id=2, weight="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NeuralEvent(tick=0, source_id=1, target_id=2, causal_parent=-1)
    with pytest.raises(TypeError):
        NeuralEvent(tick=0, source_id=1, target_id=2, causal_parent=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NeuralEvent(tick=0, source_id=1, target_id=2, seq=-1)
    with pytest.raises(TypeError):
        EventQueue().drain_until(1.5)  # type: ignore[arg-type]
