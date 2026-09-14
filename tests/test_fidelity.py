"""PH2-WI04 tests: StructuralFidelityScore per-metric distances (spec §72)."""

import pytest

from vnr.backend.fidelity import FidelityThresholds, compare_spike_trains
from vnr.backend.reference import StaticNetSpec, run_explicit, run_virtualized


def _trains(n: int, spikes: dict[int, list[int]]) -> dict[int, list[int]]:
    return {i: list(spikes.get(i, [])) for i in range(n)}


def test_identical_trains_score_perfect():
    ref = _trains(4, {0: [1, 5, 9], 1: [2], 3: [7, 8]})
    score = compare_spike_trains(ref, ref, ticks=10)
    assert score.rate_ratio == 1.0
    assert score.count_correlation == 1.0
    assert score.binned_cosine == 1.0
    assert score.max_count_diff == 0
    assert score.overall_pass is True


def test_timing_shift_fires_only_binned_cosine():
    # Same counts, moved spikes: rate metrics blind, timing metric fires.
    # This is the §72 point — one scalar would hide it.
    ref = _trains(2, {0: [1, 2, 3], 1: [4, 5, 6]})
    shifted = _trains(2, {0: [51, 52, 53], 1: [54, 55, 56]})
    score = compare_spike_trains(ref, shifted, ticks=60, bin_width=10)
    assert score.rate_ratio == 1.0
    assert score.count_correlation == 1.0
    assert score.binned_cosine < 1.0
    assert score.passed["binned_cosine"] is False
    assert score.overall_pass is False


def test_silent_candidate_fails_loud():
    ref = _trains(3, {0: [1], 1: [2], 2: [3]})
    score = compare_spike_trains(ref, _trains(3, {}), ticks=10)
    assert score.rate_ratio == 0.0
    assert score.overall_pass is False


def test_mismatched_ids_and_bad_args_rejected():
    ref = _trains(2, {0: [1]})
    with pytest.raises(ValueError, match="same neuron IDs"):
        compare_spike_trains(ref, _trains(3, {0: [1]}), ticks=10)
    with pytest.raises(ValueError, match="ticks must be a positive int"):
        compare_spike_trains(ref, ref, ticks=0)
    with pytest.raises(ValueError, match="bin_width must be a positive int"):
        compare_spike_trains(ref, ref, ticks=10, bin_width=-1)


def test_explicit_vs_virtualized_passes_gate():
    spec = StaticNetSpec(n_neurons=64, ticks=120, out_degree=4, seed=5)
    ref = run_explicit(spec)
    virt, _stats = run_virtualized(spec)
    score = compare_spike_trains(ref.spikes, virt.spikes, ticks=spec.ticks)
    assert score.overall_pass is True, score.passed


def test_custom_thresholds_travel_with_result():
    ref = _trains(2, {0: [1, 2, 3], 1: [4, 5, 6]})
    shifted = _trains(2, {0: [11, 12, 13], 1: [14, 15, 16]})
    loose = FidelityThresholds(binned_cosine_min=0.0)
    score = compare_spike_trains(ref, shifted, ticks=60, bin_width=10, thresholds=loose)
    assert score.thresholds.binned_cosine_min == 0.0
    assert score.passed["binned_cosine"] is True
