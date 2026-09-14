"""PH4-WI03 tests: canonical census, properties, instantiate, report."""

import pytest

from vnr.connectome.dataset import ToyDataset
from vnr.connectome.motifs import (
    MotifCatalog,
    canonical_code,
    census,
    code_properties,
    instantiate,
)
from vnr.connectome.report import connectome_report, write_report


def _toy_adj() -> dict[int, set[int]]:
    ds = ToyDataset()
    return {nid: {t for t, _ in ds.successors(nid)} for nid in ds.neurons()}


def test_canonical_order_independent_and_complete():
    adj = _toy_adj()
    assert canonical_code(0, 1, 8, adj) == canonical_code(8, 0, 1, adj)
    assert canonical_code(1, 0, 8, adj) == canonical_code(0, 1, 8, adj)
    # all 64 masks canonicalize; every code has consistent properties
    seen = {canonical_code(0, 1, 2, {0: set(), 1: set(), 2: set()})}
    assert seen == {0}
    props = code_properties(0)
    assert props["n_edges"] == 0 and props["cyclic"] is False
    with pytest.raises(ValueError, match="not a canonical"):
        code_properties(9)  # 9 is not a canonical representative


def test_toy_census_exact_sums_and_hub_signatures():
    cat = census(_toy_adj())
    assert cat.exact is True
    # Completeness invariant: single-edge (code 1), mutual-dyad (code 3),
    # and null (code 0) triples are counted combinatorially, so EVERY triple
    # lands somewhere: C(12,3) = 220 exactly.
    assert cat.triples_counted == 220
    assert sum(m.count for m in cat.motifs.values()) == 220
    # hub 8 fans to 0..7: {8,a,b} triples carry exactly the 2 hub edges
    two_edge = [m for m in cat.motifs.values() if m.n_edges == 2]
    assert sum(m.count for m in two_edge) >= 20
    assert cat.top(1)[0].count >= cat.top(5)[-1].count


def test_query_and_instantiate_roundtrip():
    cat = census(_toy_adj())
    cyclic = cat.query(cyclic=True)
    assert all(m.cyclic for m in cyclic)
    assert isinstance(cat, MotifCatalog)
    nontrivial = [m for m in cat.top(64) if m.n_edges > 0]
    assert nontrivial, "toy must contain non-trivial motifs"
    target = nontrivial[0]
    with pytest.raises(ValueError, match="edgeless"):
        instantiate(0, 5, 0)
    edges = instantiate(target.code, 100_000, 1_000_000)
    assert len(edges) == 100_000 * target.n_edges
    assert len(set(edges)) == len(edges)  # no duplicate pairs anywhere
    assert min(s for s, _ in edges) >= 1_000_000
    assert max(t for _, t in edges) < 1_000_000 + 300_000
    # structure preserved: re-census one copy recovers the code
    one = {s - 1_000_000: set() for s in range(1_000_000, 1_000_003)}
    for s, t in edges[: target.n_edges]:
        one[s - 1_000_000].add(t - 1_000_000)
    assert canonical_code(0, 1, 2, one) == target.code
    with pytest.raises(ValueError, match="not a canonical"):
        instantiate(9, 3, 0)
    with pytest.raises(ValueError, match="positive int"):
        census(_toy_adj(), sample_every=0)


def test_report_on_toy_exact():
    report = connectome_report(ToyDataset())
    assert report["neurons"] == 12
    assert report["edges"] == 16
    assert report["isolated_neurons"] == 3
    assert report["reciprocity"] == 0.0
    assert report["max_out_degree"] == 8


def test_report_writers(tmp_path):
    report = connectome_report(ToyDataset())
    jp = write_report(report, tmp_path / "r.json")
    import json

    assert json.loads(jp.read_text(encoding="utf-8"))["edges"] == 16
    hp = write_report(report, tmp_path / "r.html")
    text = hp.read_text(encoding="utf-8")
    assert "Connectome report: toy-v1" in text and "16" in text
