"""PH3-WI02 tests: torch backend exactness, keyed RNG, VRAM guard."""

import time

import pytest
import torch

from vnr.backend.reference import StaticNetSpec, run_explicit
from vnr.backend.torch_backend import TorchLIF, keyed_generator, run_torch, vram_status
from vnr.core.neuron import LIFNeuron


def test_single_neuron_matches_oracle():
    oracle = LIFNeuron()
    pop = TorchLIF(1)
    pattern = [(i * 37 % 11) * 0.3 for i in range(200)]
    for t, current in enumerate(pattern):
        expected = oracle.step(t, current)
        (got,) = pop.step(t, torch.tensor([current], dtype=torch.float64)).tolist()
        assert bool(got) == expected
        assert float(pop.v[0].item()) == oracle.state.v
        assert int(pop.refractory_until[0].item()) == oracle.state.refractory_until_tick


def test_torch_matches_explicit_exactly():
    spec = StaticNetSpec(n_neurons=64, ticks=600, out_degree=4, seed=5)
    expected = run_explicit(spec)
    got = run_torch(spec)
    assert got.spikes == expected.spikes
    assert got.final_v == expected.final_v
    assert got.final_refractory == expected.final_refractory
    assert got.events_delivered == expected.events_delivered


def test_keyed_generator_deterministic_and_keyed():
    a = keyed_generator(7, 42, 0, 1)
    b = keyed_generator(7, 42, 0, 1)
    assert torch.rand(5, generator=a).tolist() == torch.rand(5, generator=b).tolist()
    c = keyed_generator(7, 43, 0, 1)
    assert torch.rand(5, generator=c).tolist() != torch.rand(5, generator=a).tolist()


def test_b002_timing_reported():
    spec = StaticNetSpec(n_neurons=256, ticks=300, out_degree=8, seed=9)
    start = time.perf_counter()
    result = run_torch(spec)
    elapsed = time.perf_counter() - start
    print(f"\nB002 torch-cpu={elapsed:.2f}s spikes={result.total_spikes}")
    assert result.total_spikes == 151


def test_vram_status_none_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("machine has CUDA; no-CUDA branch untestable here")
    assert vram_status() is None


def test_torch_rejects_bad_size():
    with pytest.raises(ValueError, match="positive int"):
        TorchLIF(0)
