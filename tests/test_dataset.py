"""PH4-WI01 tests: dataset protocol conformance + determinism + structure."""

import math

import pytest

from vnr.connectome.dataset import RandomDataset, SyntheticDataset, ToyDataset


def _assert_protocol(ds, n: int) -> None:
    assert isinstance(ds.name, str) and ds.name
    assert isinstance(ds.provenance, dict) and "source" in ds.provenance
    assert ds.neurons() == list(range(n))
    total = 0
    for nid in ds.neurons():
        for tgt, w in ds.successors(nid):
            assert 0 <= tgt < n
            assert not math.isnan(w) and math.isfinite(w)
            total += 1
    assert ds.edge_count() == total


def test_toy_exact_structure():
    ds = ToyDataset()
    _assert_protocol(ds, 12)
    assert ds.successors(0) == [(1, 0.6)]
    assert ds.successors(7) == [(2, 0.8)]
    assert len(ds.successors(8)) == 8
    assert ds.successors(9) == [] and ds.successors(11) == []
    assert ds.edge_count() == 7 + 1 + 8
    with pytest.raises(KeyError):
        ds.successors(12)


def test_synthetic_determinism_and_modularity():
    a = SyntheticDataset(seed=5)
    b = SyntheticDataset(seed=5)
    assert a.edge_count() == b.edge_count()
    assert all(a.successors(i) == b.successors(i) for i in range(300))
    c = SyntheticDataset(seed=6)
    assert c.edge_count() != a.edge_count() or any(
        c.successors(i) != a.successors(i) for i in range(300)
    )
    _assert_protocol(a, 300)
    same_block = sum(
        1 for s in range(300) for t, _ in a.successors(s) if (s % 3) == (t % 3)
    )
    assert same_block > a.edge_count() // 2, "modules must dominate bridges"


def test_random_determinism_and_density():
    a = RandomDataset(seed=5)
    b = RandomDataset(seed=5)
    assert a.edge_count() == b.edge_count()
    assert all(a.successors(i) == b.successors(i) for i in range(300))
    _assert_protocol(a, 300)
    # G(300, 0.02): expectation 300*299*0.02 ≈ 1794, generous 5σ band.
    assert 1200 <= a.edge_count() <= 2400


def test_invalid_params_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        SyntheticDataset(n_neurons=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        SyntheticDataset(n_neurons=2, n_modules=3)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        RandomDataset(p_edge=1.5)
    with pytest.raises(ValueError, match="weight_hi"):
        RandomDataset(weight_lo=0.9, weight_hi=0.1)
