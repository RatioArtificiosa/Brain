"""PH1-WI04 tests: six-state frontier, hysteresis, residency counters."""

import time

import pytest

from vnr.core.frontier import (
    RESIDENT_STATES,
    ActiveFrontier,
    FrontierParams,
    FrontierState,
)


def _params() -> FrontierParams:
    return FrontierParams(
        activate_count=3, quiet_ticks=10, evict_ticks=30, candidate_timeout_ticks=10
    )


def _activate(frontier: ActiveFrontier, nid: int, tick: int = 0) -> None:
    for _ in range(frontier.params.activate_count):
        assert frontier.observe(nid, tick) in (
            FrontierState.CANDIDATE,
            FrontierState.ACTIVE,
        )
    assert frontier.state_of(nid) is FrontierState.ACTIVE


def test_unknown_is_dormant_without_tracking():
    frontier = ActiveFrontier(params=_params())
    assert frontier.state_of(42) is FrontierState.DORMANT
    assert len(frontier) == 0


def test_single_observe_never_activates():
    frontier = ActiveFrontier(params=_params())
    assert frontier.observe(7, 0) is FrontierState.CANDIDATE
    assert frontier.state_of(7) is FrontierState.CANDIDATE
    assert frontier.stats()["activations"] == 0


def test_promotion_needs_full_evidence_count():
    frontier = ActiveFrontier(params=_params())
    frontier.observe(7, 0)
    frontier.observe(7, 1)
    assert frontier.state_of(7) is FrontierState.CANDIDATE
    assert frontier.observe(7, 2) is FrontierState.ACTIVE
    assert frontier.stats()["activations"] == 1


def test_abandoned_candidate_returns_to_dormant():
    frontier = ActiveFrontier(params=_params())
    frontier.observe(7, 0)
    frontier.observe(7, 1)  # still below threshold of 3
    moved = frontier.update(10)  # first_seen=0, timeout=10
    assert frontier.state_of(7) is FrontierState.DORMANT
    assert (7, FrontierState.CANDIDATE, FrontierState.DORMANT) in moved
    assert frontier.stats()["candidate_abandons"] == 1


def test_active_ignores_single_idle_tick():
    frontier = ActiveFrontier(params=_params())
    _activate(frontier, 7, tick=0)
    frontier.update(1)
    assert frontier.state_of(7) is FrontierState.ACTIVE


def test_active_cools_to_quiescent_after_quiet_ticks():
    frontier = ActiveFrontier(params=_params())
    _activate(frontier, 7, tick=0)
    frontier.update(9)
    assert frontier.state_of(7) is FrontierState.ACTIVE
    moved = frontier.update(10)
    assert frontier.state_of(7) is FrontierState.QUIESCENT
    assert (7, FrontierState.ACTIVE, FrontierState.QUIESCENT) in moved
    assert frontier.stats()["quiet_demotions"] == 1


def test_quiescent_reactivates_on_activity():
    frontier = ActiveFrontier(params=_params())
    _activate(frontier, 7, tick=0)
    frontier.update(10)
    assert frontier.observe(7, 11) is FrontierState.ACTIVE
    assert frontier.stats()["reactivations"] == 1


def test_quiescent_needs_full_evict_band():
    frontier = ActiveFrontier(params=_params())
    _activate(frontier, 7, tick=0)
    frontier.update(10)  # -> QUIESCENT (last activity 0)
    frontier.update(29)
    assert frontier.state_of(7) is FrontierState.QUIESCENT
    frontier.update(30)
    assert frontier.state_of(7) is FrontierState.EVICTING
    assert frontier.stats()["eviction_requests"] == 1
    frontier.update(31)
    assert frontier.state_of(7) is FrontierState.EVICTED
    assert frontier.stats()["evictions"] == 1


def test_evicting_rescued_by_activity():
    frontier = ActiveFrontier(params=_params())
    _activate(frontier, 7, tick=0)
    frontier.update(10)
    frontier.update(30)
    assert frontier.state_of(7) is FrontierState.EVICTING
    assert frontier.observe(7, 31) is FrontierState.ACTIVE
    assert frontier.stats()["evictions"] == 0


def test_evicted_reenters_as_candidate():
    frontier = ActiveFrontier(params=_params())
    _activate(frontier, 7, tick=0)
    frontier.update(10)
    frontier.update(30)
    frontier.update(31)
    assert frontier.state_of(7) is FrontierState.EVICTED
    assert frontier.observe(7, 32) is FrontierState.CANDIDATE
    # Full evidence required again — no instant re-materialization.
    assert frontier.state_of(7) is FrontierState.CANDIDATE
    frontier.observe(7, 32)
    assert frontier.observe(7, 32) is FrontierState.ACTIVE
    assert frontier.stats()["activations"] == 2


def test_resident_set_and_counts_consistent():
    frontier = ActiveFrontier(params=_params())
    _activate(frontier, 1, tick=0)
    _activate(frontier, 2, tick=0)
    frontier.observe(3, 0)  # CANDIDATE: not resident
    assert frontier.resident_ids() == [1, 2]
    counts = frontier.counts()
    assert counts[FrontierState.ACTIVE] == 2
    assert counts[FrontierState.CANDIDATE] == 1
    assert sum(counts.values()) == len(frontier) == 3
    assert FrontierState.ACTIVE in RESIDENT_STATES
    assert FrontierState.CANDIDATE not in RESIDENT_STATES
    assert FrontierState.EVICTED not in RESIDENT_STATES


def test_resident_ticks_accumulate_only_while_resident():
    frontier = ActiveFrontier(params=_params())
    frontier.observe(7, 0)  # CANDIDATE
    frontier.update(0)
    assert frontier._entries[7].resident_ticks == 0
    _activate(frontier, 8, tick=0)
    frontier.update(1)
    assert frontier._entries[8].resident_ticks == 1


def test_transitions_deterministic_sorted_order():
    def run() -> list[tuple[int, FrontierState, FrontierState]]:
        frontier = ActiveFrontier(params=_params())
        for nid in (9, 3, 6):
            _activate(frontier, nid, tick=0)
        return frontier.update(10)

    first, second = run(), run()
    assert first == second
    assert [nid for nid, _, _ in first] == sorted(nid for nid, _, _ in first)


def test_100k_tracked_fast_and_consistent():
    frontier = ActiveFrontier(params=_params())
    start = time.perf_counter()
    for nid in range(100_000):
        frontier.observe(nid, 0)
    observe_s = time.perf_counter() - start
    start = time.perf_counter()
    frontier.update(0)
    update_s = time.perf_counter() - start
    assert len(frontier) == 100_000
    assert sum(frontier.counts().values()) == 100_000
    print(f"\n100K observe: {observe_s:.2f}s update: {update_s:.2f}s")


def test_invalid_ids_ticks_params_rejected():
    frontier = ActiveFrontier(params=_params())
    with pytest.raises(TypeError):
        frontier.observe(True, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        frontier.observe(-1, 0)
    with pytest.raises(ValueError):
        frontier.observe(2**64, 0)
    with pytest.raises(TypeError):
        frontier.observe(1, 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        frontier.observe(1, -1)
    with pytest.raises(TypeError):
        frontier.state_of("x")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        frontier.update(1.5)  # type: ignore[arg-type]
    frontier.update(5)
    with pytest.raises(ValueError):
        frontier.update(4)  # backward time breaks replay
    with pytest.raises(ValueError):
        FrontierParams(activate_count=1)
    with pytest.raises(ValueError):
        FrontierParams(quiet_ticks=30, evict_ticks=30)
    with pytest.raises(ValueError):
        FrontierParams(quiet_ticks=30, evict_ticks=10)
    with pytest.raises(ValueError):
        FrontierParams(quiet_ticks=0)
    with pytest.raises(TypeError):
        FrontierParams(activate_count=True)  # type: ignore[arg-type]
