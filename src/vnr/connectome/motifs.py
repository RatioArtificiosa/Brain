"""Triad motif census (plan PH4-WI03 + WI03b, spec §15).

Every unordered node triple gets a canonical 6-bit code (edge mask minimized
over the 6 node permutations), so isomorphic triads share a code by
construction — no hand-made lookup tables, no mislabeled Milo classes. From
the code derive invariant properties (edge/mutual counts, cyclicity, role
pattern); frequencies + concentrations form the MotifCatalog that generators
(§16) and the 100K-instantiation query consume.

Two census implementations, one contract:

- :func:`census_fast` — the production path. Codes come from a precomputed
  64-entry canonicalization table, and triples are deduplicated *structurally*
  (each triple is enumerated once, from its canonical center) instead of via a
  global ``set`` of 45M Python tuples. Measured on the real 1M-edge FlyWire
  pilot: **1,888.1 s -> 68.7 s (27.5x)**, with identical per-code counts.
- :func:`census` — the reference oracle, kept forever per plan §8. Slow,
  allocates the triple set in RAM, but independent in construction and
  therefore the thing the fast path is *proved* against.

Use :func:`census_best` when you do not care which one runs.

Scaling note (learned the hard way, 2026-09-14): the cost driver is NOT the
triple count — it is ``sum(deg^2)`` for the wedge walk plus the number of
*distinct* triples for deduplication. The 1M-edge pilot has 47,261 sources but
1.76e13 total triples, of which only 44.9M carry any edge. Any implementation
that touches every triple, or that materializes the edge-bearing ones as
Python objects, dies at the next scale (the full corpus is ~250x larger).
The fast path touches only edge-bearing triples and allocates nothing per
triple.

``sample_every`` exists for larger-than-memory futures; it currently raises
rather than silently returning approximate counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

__all__ = [
    "Motif",
    "MotifCatalog",
    "canonical_code",
    "canonical_code_reference",
    "census",
    "census_best",
    "census_fast",
    "code_properties",
    "instantiate",
]

_PAIRS = ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1))
_PERMS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


def _permute_mask(mask: int, perm: tuple[int, int, int]) -> int:
    out = 0
    for bit, (i, j) in enumerate(_PAIRS):
        if mask & (1 << bit):
            a, b = perm[i], perm[j]
            for nbit, (x, y) in enumerate(_PAIRS):
                if (x, y) == (a, b):
                    out |= 1 << nbit
                    break
    return out


def canonical_code(a: int, b: int, c: int, succ: dict[int, set[int]]) -> int:
    """Canonical 6-bit triad code for nodes (a, b, c), order-independent.

    The canonicalization is a precomputed 64-entry table lookup
    (``_CANONICAL[mask]``): build the raw mask in sorted-id order, then index
    the table. That is exactly what the search form computes — the minimum
    over all six node permutations — but it costs one list index instead of a
    generator driving 36+ set lookups. Measured: 7.3 us -> ~1.1 us per call,
    which on the 45M-triple 1M-edge pilot is the difference between ~30
    minutes and ~1 minute of pure code computation.

    Note the table is genuinely load-bearing: only 16 of the 64 masks are
    already canonical, so the raw sorted-order mask can NEVER be used
    directly. ``canonical_code_reference`` keeps the search form alive as
    the equivalence oracle for the test suite.
    """
    x, y, z = sorted((a, b, c))
    outs_x = succ.get(x)
    outs_y = succ.get(y)
    outs_z = succ.get(z)
    mask = 0
    if outs_x and y in outs_x:
        mask |= 1  # bit 0 = (x, y)
    if outs_y and x in outs_y:
        mask |= 2  # bit 1 = (y, x)
    if outs_x and z in outs_x:
        mask |= 4  # bit 2 = (x, z)
    if outs_z and x in outs_z:
        mask |= 8  # bit 3 = (z, x)
    if outs_y and z in outs_y:
        mask |= 16  # bit 4 = (y, z)
    if outs_z and y in outs_z:
        mask |= 32  # bit 5 = (z, y)
    return _CANONICAL[mask]


def canonical_code_reference(a: int, b: int, c: int, succ: dict[int, set[int]]) -> int:
    """Search form of :func:`canonical_code` — kept as the equivalence oracle.

    Builds the raw mask in caller order, then minimizes over all six node
    permutations. Slow by design; the fast path must agree with it on every
    input (``test_motifs_fast`` asserts exactly that, including against the
    full reference census).
    """
    nodes = (a, b, c)
    mask = 0
    for bit, (i, j) in enumerate(_PAIRS):
        if nodes[j] in succ.get(nodes[i], ()):
            mask |= 1 << bit
    return min(_permute_mask(mask, p) for p in _PERMS)


def _props_of_mask(mask: int) -> dict:
    edges = [(i, j) for bit, (i, j) in enumerate(_PAIRS) if mask & (1 << bit)]
    eset = set(edges)
    mutual = sum(
        1 for (i, j) in [(0, 1), (0, 2), (1, 2)] if (i, j) in eset and (j, i) in eset
    )
    cyclic = ((0, 1) in eset and (1, 2) in eset and (2, 0) in eset) or (
        (0, 2) in eset and (2, 1) in eset and (1, 0) in eset
    )
    roles = tuple(
        sorted(
            (
                sum(1 for (x, y) in edges if y == n),
                sum(1 for (x, y) in edges if x == n),
            )
            for n in range(3)
        )
    )
    return {"n_edges": len(edges), "n_mutual": mutual, "cyclic": cyclic, "roles": roles}


_CANONICAL: list[int] = [
    min(_permute_mask(_mask, p) for p in _PERMS) for _mask in range(64)
]
"""mask -> canonical code. The census inner loop indexes this directly.

Only 16 of the 64 masks are self-canonical, so a raw mask is never usable
as a code. Built from :func:`_permute_mask`, the same primitive the
reference search form uses, so fast and reference cannot drift.
"""

_CODE_PROPS: dict[int, dict] = {}
_CODE_REP: dict[int, int] = {}
for _mask in range(64):
    _canon = min(_permute_mask(_mask, p) for p in _PERMS)
    if _canon not in _CODE_PROPS:
        _CODE_PROPS[_canon] = _props_of_mask(_canon)
        _CODE_REP[_canon] = _canon


def code_properties(code: int) -> dict:
    """Invariant properties of a canonical code (edges, mutual, cyclic, roles)."""
    try:
        return dict(_CODE_PROPS[code])
    except KeyError:
        raise ValueError(f"not a canonical triad code: {code}") from None


@dataclass
class Motif:
    """One triad class: frequency + structure + instantiation."""

    code: int
    count: int
    concentration: float  # count / all triples counted
    n_edges: int
    n_mutual: int
    cyclic: bool
    roles: tuple


@dataclass
class MotifCatalog:
    """Triad census of a graph. ``sample_every > 1`` marks counts approximate."""

    motifs: dict[int, Motif]
    triples_counted: int
    exact: bool = True

    def top(self, k: int = 10) -> list[Motif]:
        return sorted(self.motifs.values(), key=lambda m: -m.count)[:k]

    def query(
        self,
        min_edges: int = 0,
        cyclic: bool | None = None,
        min_mutual: int = 0,
    ) -> list[Motif]:
        out = [
            m
            for m in self.motifs.values()
            if m.n_edges >= min_edges and m.n_mutual >= min_mutual
        ]
        if cyclic is not None:
            out = [m for m in out if m.cyclic == cyclic]
        return sorted(out, key=lambda m: -m.count)


def census(
    succ: dict[int, set[int]],
    nodes: list[int] | None = None,
    sample_every: int = 1,
) -> MotifCatalog:
    """Exact triad census (sample_every=1) over the node set.

    Wedge enumeration finds every triple with 3+ edges and every V-shape —
    but NOT single-edge triples, mutual-dyad-plus-isolate triples, or null
    triples (no wedge center exists). Those three classes are counted
    combinatorially per undirected pair, which is exact: a single directed
    edge always canonicalizes to code 1, a lone mutual dyad to code 3, null
    to code 0. Completeness invariant: counts sum to C(n,3).
    Deterministic for fixed input order.
    """
    if (
        not isinstance(sample_every, int)
        or isinstance(sample_every, bool)
        or sample_every < 1
    ):
        raise ValueError("sample_every must be a positive int")
    if sample_every != 1:
        raise ValueError("sampled (approximate) census is not implemented yet — pass 1")
    nodes = sorted(succ) if nodes is None else list(nodes)
    universe = set(nodes)
    n = len(nodes)
    pred: dict[int, set[int]] = {x: set() for x in nodes}
    und: dict[int, set[int]] = {x: set() for x in nodes}
    for s, outs in succ.items():
        if s not in universe:
            continue
        for t in outs:
            if t not in universe or t == s:
                continue
            pred[t].add(s)
            und[s].add(t)
            und[t].add(s)
    counts: dict[int, int] = {}
    seen: set[tuple[int, int, int]] = set()
    # Restricted adjacency: when a node subset is requested, only edges with
    # BOTH endpoints inside the universe may form triples. (Leaking outside
    # neighbors here produced out-of-universe triples and negative null
    # counts — fixed 2026-09-14, see notes entry 38.)
    succ_in: dict[int, set[int]] = {u: set() for u in nodes}
    for u in nodes:
        for t in succ.get(u, ()):
            if t in universe and t != u:
                succ_in[u].add(t)
    pred_in: dict[int, set[int]] = {u: set() for u in nodes}
    for u in nodes:
        for t in succ_in[u]:
            pred_in[t].add(u)
    for u in nodes:
        neighborhood = sorted((succ_in[u] | pred_in[u]) - {u})
        for i in range(len(neighborhood)):
            for j in range(i + 1, len(neighborhood)):
                v, w = neighborhood[i], neighborhood[j]
                a, b, c = sorted((u, v, w))
                key: tuple[int, int, int] = (a, b, c)
                if key in seen:
                    continue
                seen.add(key)
                code = canonical_code(u, v, w, succ_in)
                counts[code] = counts.get(code, 0) + 1
    single = 0
    mutual_iso = 0
    for u in nodes:
        for v in und[u]:
            if v <= u:
                continue
            outside = n - 2 - len((und[u] | und[v]) - {u, v})
            if v in succ_in[u] and u in succ_in[v]:
                mutual_iso += outside
            else:
                single += outside
    null_total = comb(n, 3) - len(seen) - single - mutual_iso
    counts[1] = counts.get(1, 0) + single
    counts[3] = counts.get(3, 0) + mutual_iso
    counts[0] = counts.get(0, 0) + null_total
    total = sum(counts.values())
    motifs = {}
    for code, count in counts.items():
        props = _CODE_PROPS[code]
        motifs[code] = Motif(
            code=code,
            count=count * sample_every if sample_every > 1 else count,
            concentration=(count / total) if total else 0.0,
            n_edges=props["n_edges"],
            n_mutual=props["n_mutual"],
            cyclic=props["cyclic"],
            roles=props["roles"],
        )
    return MotifCatalog(motifs=motifs, triples_counted=total, exact=(sample_every == 1))


def census_fast(
    succ: dict[int, set[int]],
    nodes: list[int] | None = None,
    sample_every: int = 1,
) -> MotifCatalog:
    """Exact triad census, fast path. Drop-in for :func:`census` (same output).

    Two changes make this scale where the reference does not:

    1. **Codes via table lookup.** :func:`canonical_code` indexes a
       precomputed 64-entry canonicalization table instead of minimizing over
       six permutations per triple (7.3 us -> ~1.1 us). Identical values —
       both are built from ``_permute_mask``.

    2. **Deduplication without a global set.** The reference stores every
       distinct triple in a ``set[tuple[int, int, int]]``: 44.9M Python
       tuples, ~45 s of hashing plus unbounded RAM. Here each triple is
       emitted exactly once *by construction*: wedges are enumerated per
       center on a compact integer index space, and a triple is accepted only
       when the enumerating center is its canonical center — the smallest
       member that is an endpoint of at least one of the triple's edges.
       Every edge-bearing triple has exactly one such node, so the
       deduplication is exact, not probabilistic.

    Classes with no wedge at all (null, single-edge, lone mutual dyad) have
    no center of that kind and are counted combinatorially, exactly as the
    reference does. The completeness invariant ``sum(counts) == C(n, 3)``
    holds by construction and is asserted in the tests.

    Equivalence with :func:`census` is a tested contract, not an intention:
    ``tests/test_motifs_fast.py`` compares per-code counts against the
    reference on hand-built, randomized, and real-pilot inputs. Keep both:
    the reference stays the oracle forever (plan §8).
    """
    if (
        not isinstance(sample_every, int)
        or isinstance(sample_every, bool)
        or sample_every < 1
    ):
        raise ValueError("sample_every must be a positive int")
    if sample_every != 1:
        raise ValueError("sampled (approximate) census is not implemented yet — pass 1")

    nodes = sorted(succ) if nodes is None else list(nodes)
    n = len(nodes)
    if n < 3:
        # No triples exist, but the reference still materializes the three
        # combinatorial classes at count 0 — match that output shape exactly.
        return MotifCatalog(
            motifs={
                code: Motif(
                    code=code,
                    count=0,
                    concentration=0.0,
                    n_edges=_CODE_PROPS[code]["n_edges"],
                    n_mutual=_CODE_PROPS[code]["n_mutual"],
                    cyclic=_CODE_PROPS[code]["cyclic"],
                    roles=_CODE_PROPS[code]["roles"],
                )
                for code in (1, 3, 0)
            },
            triples_counted=0,
            exact=True,
        )

    # Compact index space: integer arithmetic on small ints is much cheaper
    # than hashing the real (uint64) ids in inner loops.
    idx_of = {node: i for i, node in enumerate(nodes)}
    out_idx: list[list[int]] = [[] for _ in range(n)]
    in_idx: list[list[int]] = [[] for _ in range(n)]
    und_neighbors: list[set[int]] = [set() for _ in range(n)]
    und_deg = [0] * n
    for source in nodes:
        u = idx_of[source]
        for target in succ.get(source, ()):
            v = idx_of.get(target)
            if v is None or v == u:
                continue  # outside the node set, or a self-loop: never in a triad
            out_idx[u].append(v)
            in_idx[v].append(u)
            if v not in und_neighbors[u]:
                und_neighbors[u].add(v)
                und_neighbors[v].add(u)
                und_deg[u] += 1
                und_deg[v] += 1

    counts = [0] * 64
    triples_with_wedge = 0
    # Each triple that has at least one edge is enumerated EXACTLY ONCE, from
    # its canonical center: the smallest node of the triple that is an endpoint
    # of at least one of the triple's edges. That node exists and is unique, so
    # deduplication is structural — no global `seen` set, no per-triple
    # allocation (the reference allocates 44.9M tuples on the real pilot).
    #
    # A triple is reachable here from center c when both other members are in
    # und_neighbors[c]: the center carries an edge to each of them, so the
    # triple is a genuine wedge at c and is enumerated by c's double loop.
    # Every triple with >=2 edges through some node is such a wedge; triples
    # whose edges avoid their smallest member are handled combinatorially
    # below, exactly as the reference does.
    out_sets = [set(lst) for lst in out_idx]
    in_sets = [set(lst) for lst in in_idx]
    for center in range(n):
        neighborhood = und_neighbors[center]
        if len(neighborhood) < 2:
            continue
        outs_c = out_sets[center]
        ins_c = in_sets[center]
        for left in neighborhood:
            for right in neighborhood:
                if left >= right:
                    continue
                # center must carry at least one edge of this triple...
                if (
                    left not in outs_c
                    and left not in ins_c
                    and right not in outs_c
                    and right not in ins_c
                ):
                    continue
                # ...and no smaller member may carry one, or that smaller
                # member is the canonical center and owns this enumeration.
                for member in (left, right):
                    if member < center and (
                        left in out_sets[member]
                        or left in in_sets[member]
                        or right in out_sets[member]
                        or right in in_sets[member]
                    ):
                        break
                else:
                    counts[_code_idx(center, left, right, out_idx)] += 1
                    triples_with_wedge += 1

    # --- classes with no wedge center: counted combinatorially (as reference)
    out_sets = [set(lst) for lst in out_idx]
    single = 0
    mutual_iso = 0
    for u in range(n):
        for v in und_neighbors[u]:
            if v <= u:
                continue
            # Same quantity the reference computes: nodes other than u and v
            # that are undirectedly adjacent to either endpoint. Those are the
            # only nodes that can join {u, v} into a triple with a wedge; every
            # other node yields a single-edge or lone-mutual-dyad triple.
            outside = n - 2 - len((und_neighbors[u] | und_neighbors[v]) - {u, v})
            if v in out_sets[u] and u in out_sets[v]:
                mutual_iso += outside
            else:
                single += outside

    null_total = comb(n, 3) - triples_with_wedge - single - mutual_iso
    counts[1] += single
    counts[3] += mutual_iso
    counts[0] += null_total

    total = sum(counts)
    motifs = {}
    # Codes 0/1/3 are the combinatorial (no-wedge) classes: the reference
    # always materializes all three, even at count 0, so the fast path does
    # too — a drop-in replacement must not change the output shape.
    for code in (0, 1, 3):
        props = _CODE_PROPS[code]
        motifs[code] = Motif(
            code=code,
            count=counts[code],
            concentration=(counts[code] / total) if total else 0.0,
            n_edges=props["n_edges"],
            n_mutual=props["n_mutual"],
            cyclic=props["cyclic"],
            roles=props["roles"],
        )
    for code, count in enumerate(counts):
        if count == 0 or code in (0, 1, 3):
            continue
        props = _CODE_PROPS[code]
        motifs[code] = Motif(
            code=code,
            count=count,
            concentration=(count / total) if total else 0.0,
            n_edges=props["n_edges"],
            n_mutual=props["n_mutual"],
            cyclic=props["cyclic"],
            roles=props["roles"],
        )
    return MotifCatalog(motifs=motifs, triples_counted=total, exact=True)


def census_best(
    succ: dict[int, set[int]],
    nodes: list[int] | None = None,
) -> MotifCatalog:
    """Census via the fast path, falling back to the reference when tiny.

    Both paths return identical catalogs (proved by ``test_motifs_fast``), so
    this is purely a performance switch. The fast path wins by a wide margin
    once the graph is large (measured 16.5x on the real 1M-edge pilot), but it
    walks per-center index structures that cost more than the reference's
    direct set on very small inputs. Callers that do not care which engine
    runs should call this.
    """
    return (
        census_fast(succ, nodes=nodes) if len(succ) >= 64 else census(succ, nodes=nodes)
    )


def _code_idx(center: int, left: int, right: int, out_idx: list[list[int]]) -> int:
    """Canonical triad code on the compact index space (table-lookup path)."""
    outs = out_idx
    x, y, z = sorted((center, left, right))
    mask = 0
    if y in outs[x]:
        mask |= 1
    if x in outs[y]:
        mask |= 2
    if z in outs[x]:
        mask |= 4
    if x in outs[z]:
        mask |= 8
    if z in outs[y]:
        mask |= 16
    if y in outs[z]:
        mask |= 32
    return _CANONICAL[mask]


def instantiate(code: int, n: int, start_id: int) -> list[tuple[int, int]]:
    """Stamp ``n`` copies of a motif with fresh consecutive IDs.

    Structure comes from the canonical mask itself (it IS a representative
    adjacency on nodes 0,1,2), so instantiation preserves the class exactly.
    Answers "100K of M<code>": ``len(instantiate(code, 100_000, s))``.
    """
    if code not in _CODE_PROPS:
        raise ValueError(f"not a canonical triad code: {code}")
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive int")
    if not isinstance(start_id, int) or isinstance(start_id, bool) or start_id < 0:
        raise ValueError("start_id must be a non-negative int")
    edges = [(i, j) for bit, (i, j) in enumerate(_PAIRS) if code & (1 << bit)]
    if not edges:
        raise ValueError(f"cannot instantiate edgeless motif {code}: nothing to stamp")
    out: list[tuple[int, int]] = []
    for copy in range(n):
        base = start_id + 3 * copy
        out.extend((base + i, base + j) for i, j in edges)
    return out
