"""PH2-WI05 tests: graph-health gate fails loud on all five modes (spec §73)."""

import pytest

from vnr.backend.graph_health import (
    GraphHealthError,
    check_graph_health,
)
from vnr.backend.reference import StaticNetSpec, run_explicit


def _trains(n: int, spikes: dict[int, list[int]]) -> dict[int, list[int]]:
    return {i: list(spikes.get(i, [])) for i in range(n)}


def test_healthy_run_passes_all():
    # 10 neurons, 1 spread spike each, 200 ticks (20 ms): mean 50 Hz, sync 0.1.
    trains = {i: [5 + i * 15] for i in range(10)}
    report = check_graph_health(_trains(10, trains), ticks=200)
    assert report.healthy is True
    assert report.failures == []


def test_dead_network_fails_loud():
    with pytest.raises(GraphHealthError, match="completely silent"):
        check_graph_health(_trains(10, {}), ticks=100)


def test_exploding_network_fails():
    trains = {i: list(range(100)) for i in range(10)}  # every neuron, every tick
    with pytest.raises(GraphHealthError, match="mean rate"):
        check_graph_health(trains, ticks=100)


def test_lockstep_synchrony_fails():
    trains = {i: [50] for i in range(20)}  # all spikes share one tick
    with pytest.raises(GraphHealthError, match="busiest tick"):
        check_graph_health(trains, ticks=100)


def test_hub_dominated_fails():
    trains = {0: list(range(200))}  # one neuron does everything
    full = _trains(50, trains)
    with pytest.raises(GraphHealthError, match="Gini"):
        check_graph_health(full, ticks=300)


def test_isolated_desert_fails():
    trains = {0: [5], 1: [9]}  # 2 of 50 active
    with pytest.raises(GraphHealthError, match="silent"):
        check_graph_health(_trains(50, trains), ticks=100)


def test_each_check_visible_individually():
    trains = {0: list(range(200))}
    report = check_graph_health(_trains(50, trains), ticks=300, raise_on_fail=False)
    by_name = {c.name: c for c in report.checks}
    assert by_name["hub_dominated"].passed is False
    assert by_name["dead"].passed is True
    assert report.healthy is False


def test_reference_run_passes_gate():
    # Thirds must exceed the ~81-tick climb to threshold at amplitude 3.0;
    # a 120-tick run switches halves every 40 ticks and stays silent — the
    # gate caught exactly that when this test first used ticks=120.
    spec = StaticNetSpec(n_neurons=64, ticks=600, out_degree=4, seed=5)
    ref = run_explicit(spec)
    assert ref.total_spikes > 0
    report = check_graph_health(ref.spikes, ticks=spec.ticks)
    assert report.healthy is True, report.failures


def test_bad_inputs_rejected():
    with pytest.raises(ValueError, match="positive int"):
        check_graph_health({0: [1]}, ticks=0)
    with pytest.raises(ValueError, match="at least one neuron"):
        check_graph_health({}, ticks=10)
