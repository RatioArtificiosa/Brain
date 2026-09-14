"""PH4-WI04 tests: all 8 generators + roles (determinism, scale, metadata)."""

import pytest

from vnr.connectome.dataset import ToyDataset
from vnr.connectome.motifs import census
from vnr.generate import GenerationMetadata
from vnr.generate.generators import (
    assign_roles,
    duplication_divergence,
    hybrid,
    modular,
    motif_fill,
    population,
    random_control,
    replication,
    spatial,
)


def _toy():
    ds = ToyDataset()
    return {nid: list(ds.successors(nid)) for nid in ds.neurons()}, 12


def test_replication_exact_copies():
    src, n = _toy()
    edges, meta = replication(src, n, copies=3, p_cross=0.0, seed=1)
    assert meta.n_out == 36 and meta.generator == "replication"
    assert isinstance(meta, GenerationMetadata) and meta.parent == "source"
    assert (
        edges[0] == [(1, 0.6)] and edges[12] == [(13, 0.6)] and edges[24] == [(25, 0.6)]
    )
    again, _ = replication(src, n, copies=3, p_cross=0.0, seed=1)
    assert again == edges
    with pytest.raises(ValueError, match="copies"):
        replication(src, n, copies=0)


def test_duplication_divergence_bounds():
    src, n = _toy()
    edges, meta = duplication_divergence(src, n, p_drop=0.1, p_rewire=0.05, seed=2)
    assert meta.n_out == 24
    assert meta.e_out <= 32  # 16 edges duplicated, then only drops/rewires
    again, _ = duplication_divergence(src, n, p_drop=0.1, p_rewire=0.05, seed=2)
    assert again == edges
    pure, _ = duplication_divergence(src, n, p_drop=0.0, p_rewire=0.0, seed=2)
    assert sum(len(v) for v in pure.values()) == 32


def test_modular_preserves_counts():
    src, n = _toy()
    edges, meta = modular(src, n, n_modules=3, copies_per_module=2, p_inter=0.0, seed=3)
    assert meta.n_out == 24  # 4 members/module x 3 modules x 2 copies
    # Hash partition shatters the chain: only hub 8's edges to {2, 5} stay
    # intra-module (2 edges x 2 copies = 4). Honest structural consequence,
    # not a bug — biological modules would partition by connectivity, not id.
    assert meta.e_out == 4
    again, _ = modular(src, n, n_modules=3, copies_per_module=2, p_inter=0.0, seed=3)
    assert again == edges


def test_spatial_calibrates_and_varies():
    src, n = _toy()
    edges, meta = spatial(src, n, length_scale=0.25, seed=4)
    assert meta.n_out == 12
    assert abs(meta.e_out - 16) <= 8  # calibrated near source count
    again, _ = spatial(src, n, length_scale=0.25, seed=4)
    assert again == edges
    with pytest.raises(ValueError, match="length_scale"):
        spatial(src, n, length_scale=0.0)


def test_motif_fill_from_real_catalog():
    ds = ToyDataset()
    adj = {nid: {t for t, _ in ds.successors(nid)} for nid in ds.neurons()}
    cat = census(adj)
    code = next(m.code for m in cat.top(64) if m.n_edges >= 2)
    codes = {m.code: m.count for m in cat.motifs.values()}
    edges, meta = motif_fill(codes, code, n_instances=10, p_inter=0.0, seed=5)
    assert meta.n_out == 30
    assert meta.e_out == 10 * next(
        m.n_edges for m in cat.motifs.values() if m.code == code
    )
    assert len(edges) == meta.n_out
    again, _ = motif_fill(codes, code, n_instances=10, p_inter=0.0, seed=5)
    assert again == edges
    with pytest.raises(ValueError, match="not in catalog"):
        motif_fill(codes, 999, 3)


def test_population_expands_and_inherits():
    src, n = _toy()
    edges, meta = population(src, n, pop_size=5, p_intra=1.0, p_inter=1.0, seed=6)
    assert meta.n_out == 60
    # full intra + full inheritance: 12 pops x 5x4 intra + 16 edges x 25 inherited
    assert meta.e_out == 12 * 20 + 16 * 25
    assert len(edges) == meta.n_out
    again, _ = population(src, n, pop_size=5, p_intra=1.0, p_inter=1.0, seed=6)
    assert again == edges
    small, _ = population(src, n, pop_size=2, p_intra=0.0, p_inter=0.0, seed=6)
    assert sum(len(v) for v in small.values()) == 0


def test_hybrid_composes():
    src, n = _toy()
    edges, meta = hybrid(src, n, n_modules=2, copies_per_module=2, p_drop=0.0, seed=7)
    assert meta.generator == "hybrid" and meta.n_out == 48
    again, _ = hybrid(src, n, n_modules=2, copies_per_module=2, p_drop=0.0, seed=7)
    assert again == edges


def test_random_control_preserves_degrees():
    src, n = _toy()
    edges, meta = random_control(src, n, seed=8)
    for s in range(n):
        assert len(edges[s]) == len(src.get(s, []))
    indeg_src = sorted(
        sum(1 for outs in src.values() for t, _ in outs if t == x) for x in range(n)
    )
    indeg_new = sorted(
        sum(1 for outs in edges.values() for t, _ in outs if t == x) for x in range(n)
    )
    assert indeg_src == indeg_new
    assert meta.e_out == 16


def test_roles_structural_only():
    src, n = _toy()
    roles = assign_roles(src, n)
    assert set(roles) == set(range(n))
    assert roles[8] == "sensory"  # net source: nothing points to the hub
    assert roles[2] == "association"  # highest-degree non-source node
    assert all(
        v
        in {
            "sensory",
            "association",
            "memory",
            "valuation",
            "goal",
            "motor",
            "unassigned",
        }
        for v in roles.values()
    )
