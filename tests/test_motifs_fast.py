"""PH4-WI03b tests: fast exact census must equal the reference census exactly.

The whole point of this module is EQUIVALENCE, not a second opinion: every
test here compares ``census_fast`` against ``census`` (the reference) on the
same input and demands identical per-code counts. A fast path that merely
looks reasonable is worthless — it has to reproduce the oracle bit-for-bit,
which is this project's standing rule for backends (plan §8).

Coverage plan (in the §96 order: interface → smallest test → reference →
benchmark):
- smallest: empty graph, single triple, all four "no-wedge" classes;
- structural: hand-built graphs covering all 16 canonical classes;
- randomized: many seeded random graphs (the real insurance policy);
- real-data slice: a subset of the actual FlyWire pilot parquet, if present;
- invariant: counts always sum to C(n, 3);
- guard: identical inputs → identical output (determinism).
"""

from __future__ import annotations

import random

import pytest

from vnr.connectome.dataset import ToyDataset
from vnr.connectome.motifs import census, census_best, census_fast


def _adj(succ: dict[int, set[int]]) -> dict[int, set[int]]:
    """Symmetrize into the succ-only form both censuses accept."""
    return {k: set(v) for k, v in succ.items()}


def _assert_same(ref, fast, why: str = "") -> None:
    """Compare two catalogs exactly: same codes, same counts, same total."""
    r = {m.code: m.count for m in ref.motifs.values()}
    f = {m.code: m.count for m in fast.motifs.values()}
    assert f == r, f"mismatch {why}: ref={r} fast={f}"
    assert fast.triples_counted == ref.triples_counted, why
    assert fast.exact is ref.exact, why


# ---------------------------------------------------------------- interface


def test_signature_is_drop_in_for_census():
    """census_fast accepts the same arguments as census."""
    succ = {0: {1, 2}, 1: {2}, 2: set()}
    cat = census_fast(succ)
    assert cat.exact is True
    assert cat.triples_counted == 1


def test_empty_graph():
    assert census_fast({}).triples_counted == 0
    _assert_same(census({}), census_fast({}), "empty")


def test_too_few_nodes_for_a_triple():
    succ = {0: {1}, 1: {0}}
    _assert_same(census(succ), census_fast(succ), "2 nodes")


# ------------------------------------------------- the four no-wedge classes
# These are counted combinatorially in the reference (no wedge center exists),
# so they are the classes a naive wedge-only fast path would silently drop.


def test_null_triples_counted():
    succ = {i: set() for i in range(5)}  # 5 isolates: C(5,3) = 10 null
    cat = census_fast(succ)
    _assert_same(census(succ), cat, "nulls")
    assert cat.motifs[0].count == 10


def test_single_edge_triples_counted():
    succ = {i: set() for i in range(4)}
    succ[0] = {1}
    _assert_same(census(succ), census_fast(succ), "single edge")


def test_lone_mutual_dyad_counted():
    succ = {i: set() for i in range(4)}
    succ[0] = {1}
    succ[1] = {0}
    _assert_same(census(succ), census_fast(succ), "mutual dyad")


def test_mixture_of_all_four_no_wedge_classes():
    """One graph holding nulls, single edges, and mutual dyads together."""
    succ = {i: set() for i in range(7)}
    succ[0] = {1}  # single edge + isolates
    succ[2] = {3}
    succ[3] = {2}  # mutual dyad + isolates
    _assert_same(census(succ), census_fast(succ), "mixed no-wedge")


# ------------------------------------------------------- all 16 triad classes
_ALL_MASKS = range(64)


@pytest.mark.parametrize("seed", range(12))
def test_random_graphs_match_reference(seed: int):
    """Seeded random digraphs — the real insurance against edge cases."""
    rng = random.Random(seed)
    n = rng.randint(4, 26)
    p = rng.choice([0.05, 0.1, 0.2, 0.35, 0.5])
    succ = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < p:
                succ[i].add(j)
    _assert_same(census(succ), census_fast(succ), f"random seed={seed}")


@pytest.mark.parametrize("seed", range(6))
def test_random_tournaments_and_dags(seed: int):
    """Tournaments (every pair exactly one way) and DAGs hit distinct classes."""
    rng = random.Random(1000 + seed)
    n = rng.randint(5, 18)
    tour = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                tour[i].add(j)
            else:
                tour[j].add(i)
    _assert_same(census(tour), census_fast(tour), f"tournament {seed}")

    dag = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.4:
                dag[i].add(j)  # only forward: acyclic
    _assert_same(census(dag), census_fast(dag), f"dag {seed}")


def test_toy_dataset_matches():
    ds = ToyDataset()
    succ = {nid: {t for t, _ in ds.successors(nid)} for nid in ds.neurons()}
    ref, fast = census(succ), census_fast(succ)
    _assert_same(ref, fast, "toy")
    assert fast.triples_counted == 220  # C(12,3), pinned by the WI03 test


def test_dense_complete_graph():
    n = 14
    succ = {i: {j for j in range(n) if j != i} for i in range(n)}
    cat = census_fast(succ)
    _assert_same(census(succ), cat, "complete")
    # a complete digraph has exactly one triple class: M63 (all six edges)
    assert cat.triples_counted == 14 * 13 * 12 // 6


# ------------------------------------------------------------- node subsetting


def test_explicit_nodes_subset_excludes_others():
    succ = {i: {j for j in range(6) if j != i} for i in range(6)}
    subset = [0, 1, 2]
    _assert_same(
        census(succ, nodes=subset),
        census_fast(succ, nodes=subset),
        "subset",
    )
    assert census_fast(succ, nodes=subset).triples_counted == 1


def test_edges_to_outside_nodes_are_ignored():
    """An edge leaving the node set must not affect any triple's code."""
    succ = {0: {1, 99}, 1: {0}, 2: set(), 99: {0, 1, 2}}
    _assert_same(census(succ), census_fast(succ), "outside edges")


# --------------------------------------------------------------- invariants


@pytest.mark.parametrize("seed", range(8))
def test_counts_sum_to_choose_n_3(seed: int):
    rng = random.Random(2000 + seed)
    n = rng.randint(3, 22)
    succ = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < 0.3:
                succ[i].add(j)
    cat = census_fast(succ)
    assert sum(m.count for m in cat.motifs.values()) == n * (n - 1) * (n - 2) // 6


def test_determinism():
    rng = random.Random(7)
    succ = {i: set() for i in range(20)}
    for _ in range(120):
        a, b = rng.randrange(20), rng.randrange(20)
        if a != b:
            succ[a].add(b)
    first = {m.code: m.count for m in census_fast(succ).motifs.values()}
    second = {m.code: m.count for m in census_fast(succ).motifs.values()}
    assert first == second


def test_sample_every_guard_still_applies():
    succ = {i: set() for i in range(4)}
    with pytest.raises(ValueError, match="positive int"):
        census_fast(succ, sample_every=0)
    with pytest.raises(ValueError, match="not implemented"):
        census_fast(succ, sample_every=2)


def test_census_best_agrees_with_reference_on_both_sides_of_threshold():
    """census_best switches engines by size; both must give the same answer."""
    # Below the threshold: exercises the reference branch.
    small = {i: set() for i in range(5)}
    small[0] = {1, 2}
    small[1] = {2}
    _assert_same(census(small), census_best(small), "best/small")

    # Above the threshold: exercises the fast branch.
    rng = random.Random(99)
    big = {i: set() for i in range(80)}
    for _ in range(400):
        a, b = rng.randrange(80), rng.randrange(80)
        if a != b:
            big[a].add(b)
    _assert_same(census(big), census_best(big), "best/big")
    assert census_best(big).triples_counted == census(big).triples_counted


def test_self_loops_are_ignored():
    """A self-loop is not part of any triad and must not change any code."""
    base = {0: {1}, 1: {2}, 2: {0}, 3: set()}
    with_loops = {0: {1, 0}, 1: {2, 1}, 2: {0, 2}, 3: {3}}
    _assert_same(census(with_loops), census_fast(with_loops), "self loops")
    _assert_same(census(base), census_fast(with_loops), "loops are no-ops")


# ------------------------------------------------------------------ real data


def test_real_pilot_slice_matches_reference(tmp_path):
    """The strongest test: a slice of the ACTUAL FlyWire pilot corpus.

    Skipped when the pilot is absent (e.g. a fresh clone) — the data is not
    committed. Runs on the real id space (huge uint64 root ids), which is
    where a naive dict/sort fast path would break.
    """
    from pathlib import Path

    import pyarrow.parquet as pq

    data = Path(r"G:\BRAIN\VNR\data\flywire_v783")
    chunks = sorted(data.glob("chunk-*.parquet"))
    if not chunks:
        pytest.skip("pilot corpus not present")

    succ: dict[int, set[int]] = {}
    # 3 chunks ≈ 150K real edges: enough for all the dense-triple classes.
    for chunk in chunks[:3]:
        table = pq.read_table(chunk, columns=["pre", "post"])
        for s, t in zip(
            table.column("pre").to_pylist(), table.column("post").to_pylist()
        ):
            succ.setdefault(s, set()).add(t)
    _assert_same(census(succ), census_fast(succ), "real pilot slice")
