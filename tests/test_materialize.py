"""PH1-WI05 tests: 3-layer separation + §67 exact roundtrip (static net)."""

import random
import time

import pytest

from vnr.core.materialize import Materializer
from vnr.core.neuron import LIFNeuron, LIFParams


def _net(n: int) -> Materializer:
    mat = Materializer()
    for nid in range(n):
        mat.register(nid)
    return mat


def test_fresh_materialize_matches_init():
    mat = _net(1)
    assert not mat.is_resident(0)
    snap = mat.materialize_neuron(0, tick=0)
    assert (snap.v, snap.refractory_until_tick) == (0.0, 0)
    assert mat.is_resident(0)
    assert mat.resident_ids() == [0]


def test_snapshot_is_a_copy_not_an_alias():
    mat = _net(1)
    snap = mat.materialize_neuron(0)
    mat.step_neuron(0, 0, 3.0)
    live = mat.materialize_neuron(0)
    assert live.v != snap.v  # one driven tick lifts V above init
    assert (snap.v, snap.refractory_until_tick) == (0.0, 0)  # capture intact
    assert mat.learned_of(0) == {}
    leaked = mat.learned_of(0)
    leaked["x"] = 1.0
    assert mat.learned_of(0) == {}


def test_step_matches_plain_lif_oracle():
    params = LIFParams()
    drive = [3.0] * 60 + [0.0] * 60
    ref = LIFNeuron(params=params).run(0, drive)
    mat = Materializer()
    mat.register(5, params)
    mat.materialize_neuron(5)
    got = [t for t, i in enumerate(drive) if mat.step_neuron(5, t, i)]
    assert got == ref


def test_evict_persists_learned_first_then_frees():
    mat = _net(1)
    mat.set_learned(0, "bias", 0.5)
    mat.materialize_neuron(0)
    mat.step_neuron(0, 0, 3.0)
    before = mat.materialize_neuron(0)
    mat.evict_neuron(0)
    assert not mat.is_resident(0)
    assert mat.writes[-2:] == [("learned", 0), ("transient", 0)]
    assert mat.learned_of(0) == {"bias": 0.5}
    again = mat.materialize_neuron(0)
    assert (again.v, again.refractory_until_tick) == (
        before.v,
        before.refractory_until_tick,
    )
    stats = mat.stats()
    assert stats["evictions"] == 1
    assert stats["learned_writebacks"] == 1
    assert stats["materializations"] == 2


def test_section67_roundtrip_exact_static_net():
    rng = random.Random(20260914)
    n, ticks, mid = 200, 2000, 1000
    drives = [[rng.choice([0.0, 0.5, 3.0]) for _ in range(ticks)] for _ in range(n)]
    ref_spikes = [LIFNeuron().run(0, drive) for drive in drives]

    mat = _net(n)
    for nid in range(n):
        mat.materialize_neuron(nid)
    got_spikes: list[list[int]] = [[] for _ in range(n)]
    for t in range(ticks):
        if t == mid:
            for nid in range(n):  # full eviction sweep mid-run
                mat.evict_neuron(nid)
            assert mat.resident_ids() == []
            for nid in range(n):
                mat.materialize_neuron(nid)
        for nid in range(n):
            if mat.step_neuron(nid, t, drives[nid][t]):
                got_spikes[nid].append(t)
    assert got_spikes == ref_spikes
    refs = [LIFNeuron() for _ in range(n)]
    for nid in range(n):
        refs[nid].run(0, drives[nid])
    for nid in range(n):  # bit-exact final state: same op order, exact restore
        live = mat.materialize_neuron(nid, tick=ticks)
        assert live.v == refs[nid].state.v
        assert live.refractory_until_tick == refs[nid].state.refractory_until_tick
    assert mat.stats()["evictions"] == n


def test_repeated_evict_cycles_stay_exact():
    mat = _net(4)
    for nid in range(4):
        mat.materialize_neuron(nid)
    ref = [LIFNeuron() for _ in range(4)]
    for t in range(300):
        for nid in range(4):
            assert mat.step_neuron(nid, t, 3.0) == ref[nid].step(t, 3.0)
        if t % 50 == 49:
            for nid in range(4):
                mat.evict_neuron(nid)
            for nid in range(4):
                mat.materialize_neuron(nid)


def test_learned_survives_eviction_cycles():
    mat = _net(2)
    mat.set_learned(1, "gain", 2.5)
    mat.materialize_neuron(1)
    mat.evict_neuron(1)
    mat.materialize_neuron(1)
    mat.evict_neuron(1)
    assert mat.learned_of(1) == {"gain": 2.5}
    assert mat.writes == [("learned", 1), ("transient", 1)] * 2


def test_lifecycle_failures_are_loud():
    mat = Materializer()
    with pytest.raises(KeyError):
        mat.materialize_neuron(9)
    with pytest.raises(KeyError):
        mat.step_neuron(9, 0)
    with pytest.raises(KeyError):
        mat.evict_neuron(9)
    with pytest.raises(KeyError):
        mat.set_learned(9, "k", 1.0)
    with pytest.raises(KeyError):
        mat.learned_of(9)
    mat.register(1)
    with pytest.raises(ValueError):
        mat.register(1)
    with pytest.raises(KeyError):
        mat.step_neuron(1, 0)  # registered but never materialized
    with pytest.raises(KeyError):
        mat.evict_neuron(1)
    with pytest.raises(TypeError):
        mat.register(True)
    with pytest.raises(ValueError):
        mat.register(-1)
    with pytest.raises(ValueError):
        mat.register(2**64)
    with pytest.raises(TypeError):
        mat.materialize_neuron(1, tick=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        mat.materialize_neuron(1, tick=-1)
    with pytest.raises(TypeError):
        mat.register(3, params="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        mat.set_learned(1, "", 1.0)
    with pytest.raises(TypeError):
        mat.set_learned(1, "k", "x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        mat.set_learned(1, "k", float("inf"))


def test_stats_and_determinism():
    def run() -> tuple[list[int], dict[str, int], list[tuple[str, int]]]:
        mat = _net(3)
        for nid in (2, 0, 1):
            mat.materialize_neuron(nid)
        assert mat.resident_ids() == [0, 1, 2]
        for nid in (2, 0, 1):
            mat.evict_neuron(nid)
        return mat.resident_ids(), mat.stats(), list(mat.writes)

    assert run() == run()


def test_20k_materialize_evict_cycle_fast():
    mat = Materializer()
    for nid in range(20_000):
        mat.register(nid)
    start = time.perf_counter()
    for nid in range(20_000):
        mat.materialize_neuron(nid)
    for nid in range(20_000):
        mat.evict_neuron(nid)
    cycle_s = time.perf_counter() - start
    assert mat.stats()["evictions"] == 20_000
    assert mat.resident_ids() == []
    print(f"\n20K materialize+evict: {cycle_s:.2f}s")
