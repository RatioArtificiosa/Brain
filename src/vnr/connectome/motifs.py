"""Triad motif census (plan PH4-WI03, spec §15).

Every unordered node triple gets a canonical 6-bit code (edge mask minimized
over the 6 node permutations), so isomorphic triads share a code by
construction — no hand-made lookup tables, no mislabeled Milo classes. From
the code derive invariant properties (edge/mutual counts, cyclicity, role
pattern); frequencies + concentrations form the MotifCatalog that generators
(§16) and the 100K-instantiation query consume.

Scaling: exact census via wedge enumeration with dedupe. Cost is
sum-of-squared-degrees wedge visits; the 1M-edge pilot runs exact in ~1 min
on this machine. ``sample_every`` exists for larger-than-memory futures and
is clearly marked approximate when used.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

__all__ = [
    "Motif",
    "MotifCatalog",
    "canonical_code",
    "census",
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
    """Canonical 6-bit triad code for nodes (a, b, c), order-independent."""
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
    for u in nodes:
        neighborhood = sorted((succ.get(u, set()) | pred.get(u, set())) - {u})
        for i in range(len(neighborhood)):
            for j in range(i + 1, len(neighborhood)):
                v, w = neighborhood[i], neighborhood[j]
                a, b, c = sorted((u, v, w))
                key: tuple[int, int, int] = (a, b, c)
                if key in seen:
                    continue
                seen.add(key)
                code = canonical_code(u, v, w, succ)
                counts[code] = counts.get(code, 0) + 1
    single = 0
    mutual_iso = 0
    for u in nodes:
        for v in und[u]:
            if v <= u:
                continue
            outside = n - 2 - len((und[u] | und[v]) - {u, v})
            if v in succ.get(u, ()) and u in succ.get(v, ()):
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
