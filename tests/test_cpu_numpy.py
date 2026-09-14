"""PH3-WI01 tests: NumPy backend bit-exact vs oracle + B001/B003 timings."""

import time

import numpy as np
import pytest

from vnr.backend.cpu_numpy import NumpyLIF, run_numpy
from vnr.backend.reference import StaticNetSpec, run_explicit
from vnr.core.neuron import LIFNeuron


def test_single_neuron_matches_oracle_tick_for_tick():
    oracle = LIFNeuron()
    pop = NumpyLIF(1)
    pattern = [(i * 37 % 11) * 0.3 for i in range(200)]
    for t, current in enumerate(pattern):
        expected = oracle.step(t, current)
        (got,) = pop.step(t, np.array([current]))
        assert bool(got) == expected
        assert pop.v[0] == oracle.state.v
        assert pop.refractory_until[0] == oracle.state.refractory_until_tick


def test_numpy_matches_explicit_exactly():
    spec = StaticNetSpec(n_neurons=64, ticks=600, out_degree=4, seed=5)
    expected = run_explicit(spec)
    got = run_numpy(spec)
    assert got.spikes == expected.spikes
    assert got.final_v == expected.final_v
    assert got.final_refractory == expected.final_refractory
    assert got.events_delivered == expected.events_delivered


def test_numpy_matches_explicit_medium_net():
    spec = StaticNetSpec(n_neurons=256, ticks=300, out_degree=8, seed=9)
    expected = run_explicit(spec)
    got = run_numpy(spec)
    assert got.spikes == expected.spikes
    assert got.final_v == expected.final_v


def test_b001_b003_timings_reported():
    # B001 (explicit) vs B003 (numpy): informational — the gate is exactness,
    # but the ratio tells us what vectorization alone buys before kernels.
    spec = StaticNetSpec(n_neurons=256, ticks=300, out_degree=8, seed=9)
    start = time.perf_counter()
    run_explicit(spec)
    t_explicit = time.perf_counter() - start
    start = time.perf_counter()
    run_numpy(spec)
    t_numpy = time.perf_counter() - start
    print(
        f"\nB001 explicit={t_explicit:.2f}s B003 numpy={t_numpy:.2f}s ratio={t_explicit / t_numpy:.2f}x"
    )
    assert t_numpy > 0


def test_numpy_rejects_bad_size():
    with pytest.raises(ValueError, match="positive int"):
        NumpyLIF(0)
    with pytest.raises(ValueError, match="positive int"):
        NumpyLIF(True)  # type: ignore[arg-type]
