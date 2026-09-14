"""PH1-WI06 tests: §22 pipeline + §68 determinism (100 calls identical)."""

import time

import pytest

from vnr.core.events import EventQueue
from vnr.core.procedural import (
    ConnectivityParams,
    ProceduralConnectivity,
    keyed_uniform,
)


def _conn(**kwargs) -> ProceduralConnectivity:
    return ProceduralConnectivity(params=ConnectivityParams(**kwargs))


def _sig(events) -> list[tuple[int, int, int, float]]:
    return [(e.tick, e.source_id, e.target_id, e.weight) for e in events]


def test_pipeline_shape_and_contract():
    conn = _conn(out_degree=8, weight=0.5, delay_ticks=2)
    events = conn.generate_outgoing_events(42, 10)
    assert len(events) == 8
    assert all(e.tick == 12 and e.source_id == 42 for e in events)
    assert all(e.weight == 0.5 for e in events)
    assert all(0 <= e.target_id < 2**64 for e in events)


def test_section68_100_calls_identical():
    conn = _conn()
    first = _sig(conn.generate_outgoing_events(7, 3))
    for _ in range(100):
        assert _sig(conn.generate_outgoing_events(7, 3)) == first


def test_targets_stable_across_ticks_deliveries_shift():
    conn = _conn()
    early = conn.generate_outgoing_events(7, 0)
    late = conn.generate_outgoing_events(7, 500)
    assert [e.target_id for e in early] == [e.target_id for e in late]
    assert [e.tick for e in late] == [t + 500 for t in [e.tick for e in early]]


def test_seed_change_gives_controlled_divergence():
    base = _sig(_conn(global_seed=1).generate_outgoing_events(7, 0))
    same = _sig(_conn(global_seed=1).generate_outgoing_events(7, 0))
    other = _sig(_conn(global_seed=2).generate_outgoing_events(7, 0))
    assert same == base
    assert other != base
    assert len({e[2] for e in other}) > 1  # still a real fan-out, not collapse


def test_source_and_version_change_targets():
    conn = _conn()
    assert conn.sample_targets(1) != conn.sample_targets(2)
    v1 = _sig(_conn(connectivity_version=1).generate_outgoing_events(7, 0))
    v2 = _sig(_conn(connectivity_version=2).generate_outgoing_events(7, 0))
    assert v1 != v2


def test_stages_agree_with_pipeline():
    conn = _conn(out_degree=16)
    events = conn.generate_outgoing_events(9, 4)
    assert [e.target_id for e in events] == conn.sample_targets(9)
    assert [e.weight for e in events] == conn.assign_weights(9)
    assert [e.tick for e in events] == [4 + d for d in conn.assign_delays(9)]


def test_events_feed_queue_in_order():
    conn = _conn(out_degree=4, delay_ticks=1)
    queue = EventQueue()
    for event in conn.generate_outgoing_events(3, 10):
        queue.push(event)
    assert [e.tick for e in queue.drain_all()] == [11] * 4


def test_fanout_spreads_no_collapse():
    conn = _conn(out_degree=1000)
    targets = conn.sample_targets(12345)
    assert len(set(targets)) > 990  # 64-bit space: collisions ~impossible


def test_keyed_uniform_range_and_stability():
    draws = [keyed_uniform(5, 6, 7, 1, s) for s in range(1000)]
    assert all(0.0 <= d < 1.0 for d in draws)
    assert len(set(draws)) == 1000
    assert keyed_uniform(5, 6, 7, 1, 0) == keyed_uniform(5, 6, 7, 1, 0)
    assert keyed_uniform(5, 6, 7, 1, 0) != keyed_uniform(5, 6, 7, 2, 0)
    with pytest.raises(ValueError):
        keyed_uniform(5, 6, 7, 1, -1)


def test_10k_sources_benchmark():
    conn = _conn(out_degree=64)
    start = time.perf_counter()
    total = sum(len(conn.generate_outgoing_events(s, 0)) for s in range(10_000))
    elapsed = time.perf_counter() - start
    assert total == 640_000
    print(f"\n10K sources x64 edges: {elapsed:.2f}s ({total / elapsed:,.0f} edges/s)")


def test_invalid_inputs_rejected():
    conn = _conn()
    with pytest.raises(TypeError):
        conn.generate_outgoing_events(True, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        conn.generate_outgoing_events(-1, 0)
    with pytest.raises(ValueError):
        conn.generate_outgoing_events(2**64, 0)
    with pytest.raises(TypeError):
        conn.generate_outgoing_events(1, 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        conn.generate_outgoing_events(1, -1)
    with pytest.raises(ValueError):
        ConnectivityParams(out_degree=0)
    with pytest.raises(ValueError):
        ConnectivityParams(out_degree=2_000_000)
    with pytest.raises(TypeError):
        ConnectivityParams(out_degree=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ConnectivityParams(weight="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ConnectivityParams(weight=float("nan"))
    with pytest.raises(ValueError):
        ConnectivityParams(delay_ticks=-1)
    with pytest.raises(ValueError):
        ConnectivityParams(global_seed=2**64)
    with pytest.raises(TypeError):
        ConnectivityParams(connectivity_version=1.5)  # type: ignore[arg-type]


def test_weight_distribution_deterministic_and_bounded():
    conn = _conn(out_degree=64, weight=0.5, weight_std=0.2)
    first = conn.assign_weights(7)
    assert first == conn.assign_weights(7)
    assert len(first) == 64 and len(set(first)) > 1
    assert all(0.3 <= w <= 0.7 for w in first)


def test_weight_quantization_levels_and_exact_default():
    assert _conn(out_degree=8).assign_weights(3) == [0.5] * 8
    q = _conn(out_degree=200, weight=0.5, weight_std=0.2, weight_bits=2).assign_weights(
        3
    )
    assert len(set(q)) <= 4
    assert all(0.3 - 1e-9 <= w <= 0.7 + 1e-9 for w in q)
    q8 = _conn(
        out_degree=200, weight=0.5, weight_std=0.2, weight_bits=8
    ).assign_weights(3)
    assert len(set(q8)) <= 256


def test_weight_params_rejected():
    with pytest.raises(ValueError, match="weight_std must be a non-negative"):
        _conn(weight_std=-0.1)
    with pytest.raises(ValueError, match="weight_bits must be in"):
        _conn(weight_bits=0)
    with pytest.raises(TypeError):
        _conn(weight_bits="8")  # type: ignore[arg-type]
