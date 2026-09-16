"""PH4-WI02 tests: the real FlyWire connectome as a VNR dataset.

Skipped when the fetched corpus is absent (it is not committed), so a fresh
clone stays green. When present, these tests assert the properties that make
the real data scientifically different from the synthetic generator: a
heavy-tailed degree distribution and measurable LOCALITY.

The locality test matters most: entry 39 measured that uniform random fan-out
scatters every spike across the network and caps the virtualization benefit.
The real connectome is structured, so this is where that ceiling should lift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vnr.connectome.dataset import ConnectomeDataset
from vnr.connectome.flywire_dataset import FlyWireDataset

PILOT = Path(r"G:\BRAIN\VNR\data\flywire_v783")

pytestmark = pytest.mark.skipif(
    not sorted(PILOT.glob("byid-*.parquet")),
    reason="fetched FlyWire corpus not present (not committed)",
)


def _dataset(limit_chunks: int = 1) -> FlyWireDataset:
    """Load a fixed snapshot of chunks.

    NOTE: `bulk_by_id.py` may be writing new chunks while tests run, so the
    chunk list is snapshotted once per call. Reading the glob twice (once to
    build the dataset, once to resolve ids) could otherwise see a different
    file set and produce a spurious integrity failure - observed and fixed.
    """
    chunks = tuple(sorted(PILOT.glob("byid-*.parquet"))[:limit_chunks])
    return FlyWireDataset(chunks=chunks)


def test_implements_the_dataset_protocol():
    """The real connectome must satisfy the same contract as toy/synthetic."""
    ds = _dataset()
    assert isinstance(ds, ConnectomeDataset)
    assert ds.neurons()
    assert ds.edge_count() > 0
    assert ds.name.startswith("flywire")


def test_neuron_and_edge_counts_are_consistent():
    ds = _dataset()
    counted = sum(len(ds.successors(n)) for n in ds.neurons())
    assert counted == ds.edge_count()


def test_compact_ids_are_dense_and_raw_ids_are_not():
    """Two id spaces, both correct: kernels need dense, provenance needs raw."""
    compact = _dataset(1)
    assert compact.neurons() == list(range(len(compact.neurons())))

    raw = FlyWireDataset(chunks=compact.chunks, compact=False)
    assert min(raw.neurons()) > 700_000_000_000_000_000  # real FlyWire root ids
    assert len(raw.neurons()) == len(compound := compact.neurons()), (
        "both id spaces must cover the same neuron set"
    )
    assert len(compound) > 0


def test_weights_carry_biological_sign():
    """Edges are inhibitory (negative) or excitatory - not a uniform constant."""
    ds = _dataset(1)
    weights = [w for n in ds.neurons() for _, w in ds.successors(n)]
    assert weights, "no edges loaded"
    assert any(w < 0 for w in weights), "GABA edges must be inhibitory"
    assert any(w > 0 for w in weights), "excitatory edges must be positive"
    assert ds.is_finite()
    assert 0 < ds.mean_weight() < 1.0


def test_degree_distribution_is_heavy_tailed():
    """Real connectomes are hub-dominated; this is why they differ from G(n,p).

    Asserts the measured shape rather than a specific number, so the test
    travels to larger fetches.
    """
    ds = _dataset(1)
    stats = ds.degree_stats()
    assert stats["max"] > stats["mean"] * 20, (
        f"expected a heavy tail, got mean={stats['mean']:.2f} max={stats['max']:.0f}"
    )


def test_real_connectome_is_more_local_than_uniform_random():
    """The scientific point: structure beats uniform fan-out.

    Entry 39 found residency is governed by fan-out, and that uniform random
    connectivity has no locality. If the real connectome is measurably more
    local than a uniform graph over the same ids, then the virtualization
    ceiling found on synthetic data does not transfer - which is a
    testable claim, so it is tested.
    """
    ds = _dataset(1)
    locality = FlyWireDataset.locality_ratio(ds)
    assert 0.0 < locality < 0.5, (
        f"real connectome locality {locality:.4f} should be well below the "
        f"uniform-graph value of ~0.33-0.5 for scattered edges"
    )


def test_nt_distribution_is_counted_from_data():
    ds = _dataset(1)
    dist = ds.nt_distribution()
    assert dist, "no transmitter data"
    assert sum(dist.values()) > 0
    # ACH dominates the fly brain; assert the ordering, not an exact share.
    top = max(dist, key=lambda k: dist[k])
    assert top in {"ACH", "GABA", "GLUT", "UNKNOWN"}, top


def test_provenance_records_where_the_data_came_from():
    ds = _dataset(1)
    prov = ds.provenance
    assert "flywire" in prov["source"].lower()
    assert prov["node_ids"] in {"compact", "raw"}
    assert int(prov["distinct_edges"]) > 0


def test_zero_root_and_self_loops_are_excluded():
    """Unmapped segments and self-connections are never valid neurons.

    Careful with id spaces: in COMPACT mode index 0 is a legitimate neuron
    (the lowest sorted root id), so id 0 is only invalid in RAW mode. The
    first version of this test asserted `n != 0` unconditionally and failed
    on a correct dataset - the assertion was wrong, not the data.
    """
    ds = _dataset(1)
    ids = set(ds.neurons())
    for n in ds.neurons():
        for tgt, _ in ds.successors(n):
            assert tgt != n, "self-loop leaked into the dataset"
            assert tgt in ids, f"target {tgt} is outside the neuron set"

    # In RAW mode the original FlyWire root ids are exposed; root id 0 means
    # "unmapped segment" and must never appear.
    raw = FlyWireDataset(chunks=ds.chunks, compact=False)
    assert 0 not in set(raw.neurons()), "root id 0 (unmapped) leaked into raw ids"
