"""The eight scaling generators (plan PH4-WI04, spec §16–18).

All operate on edge dicts ``{src: [(tgt, weight)]}`` with ``0..n-1`` ids and
return renumbered ``0..m-1`` dicts plus lineage metadata. Deterministic per
(seed, params). Weights of synthesized edges default to the source mean
unless the generator has a better rule (documented per function).
"""

from __future__ import annotations

from vnr.generate import (
    GENERATOR_VERSION,
    Edges,
    GenerationMetadata,
    check_edges,
    mean_weight,
    rng_for,
)

__all__ = [
    "duplication_divergence",
    "hybrid",
    "modular",
    "motif_fill",
    "population",
    "random_control",
    "replication",
    "spatial",
]


def _meta(
    name: str, seed: int, params: dict, parent: str, n_in: int, e_in: int, edges: Edges
) -> tuple[Edges, GenerationMetadata]:
    n_out = (
        max([s for s in edges] + [t for outs in edges.values() for t, _ in outs] + [-1])
        + 1
    )
    e_out = sum(len(v) for v in edges.values())
    return edges, GenerationMetadata(
        generator=name,
        version=GENERATOR_VERSION,
        seed=seed,
        params=params,
        parent=parent,
        n_in=n_in,
        e_in=e_in,
        n_out=n_out,
        e_out=e_out,
    )


def replication(
    source: Edges,
    n_source: int,
    copies: int,
    p_cross: float = 0.0,
    seed: int = 0,
    parent: str = "source",
) -> tuple[Edges, GenerationMetadata]:
    """Tile the graph ``copies`` times; sparse cross-tile links with prob ``p_cross``."""
    if copies < 1:
        raise ValueError("copies must be >= 1")
    if not 0.0 <= p_cross <= 1.0:
        raise ValueError("p_cross must be in [0, 1]")
    rng = rng_for(seed, "replication")
    w = mean_weight(source)
    out: Edges = {}
    for c in range(copies):
        base = c * n_source
        for s in range(n_source):
            out[base + s] = [(base + t, wt) for t, wt in source.get(s, [])]
    if p_cross > 0 and copies > 1:
        all_ids = list(out)
        for i in range(len(all_ids)):
            for j in range(i + 1, len(all_ids)):
                a, b = all_ids[i], all_ids[j]
                if a // n_source != b // n_source and rng.random() < p_cross:
                    out.setdefault(a, []).append((b, w))
                    out.setdefault(b, []).append((a, w))
    n_out = copies * n_source
    for nid in range(n_out):
        out.setdefault(nid, [])
    check_edges(out, n_out, "replication")
    return _meta(
        "replication",
        seed,
        {"copies": copies, "p_cross": p_cross},
        parent,
        n_source,
        sum(len(v) for v in source.values()),
        out,
    )


def duplication_divergence(
    source: Edges,
    n_source: int,
    p_drop: float = 0.1,
    p_rewire: float = 0.05,
    seed: int = 0,
    parent: str = "source",
) -> tuple[Edges, GenerationMetadata]:
    """Duplicate once, then drop edges (p_drop) and rewire targets (p_rewire) — §17."""
    if not 0.0 <= p_drop <= 1.0 or not 0.0 <= p_rewire <= 1.0:
        raise ValueError("probabilities must be in [0, 1]")
    rng = rng_for(seed, "duplication")
    out: Edges = {s: list(source.get(s, [])) for s in range(n_source)}
    out.update(
        {
            n_source + s: [(n_source + t, w) for t, w in source.get(s, [])]
            for s in range(n_source)
        }
    )
    for s in list(out):
        kept = []
        for t, w in out[s]:
            r = rng.random()
            if r < p_drop:
                continue
            if r < p_drop + p_rewire:
                t = rng.randrange(2 * n_source)
            kept.append((t, w))
        out[s] = kept
    check_edges(out, 2 * n_source, "duplication")
    return _meta(
        "duplication_divergence",
        seed,
        {"p_drop": p_drop, "p_rewire": p_rewire},
        parent,
        n_source,
        sum(len(v) for v in source.values()),
        out,
    )


def modular(
    source: Edges,
    n_source: int,
    n_modules: int,
    copies_per_module: int,
    p_inter: float = 0.01,
    seed: int = 0,
    parent: str = "source",
) -> tuple[Edges, GenerationMetadata]:
    """Partition by id hash (§18), replicate each module, sparse inter-module links."""
    if n_modules < 1 or copies_per_module < 1:
        raise ValueError("n_modules and copies_per_module must be >= 1")
    if not 0.0 <= p_inter <= 1.0:
        raise ValueError("p_inter must be in [0, 1]")
    rng = rng_for(seed, "modular")
    w = mean_weight(source)
    mod_of = {s: s % n_modules for s in range(n_source)}
    base_of: dict[tuple[int, int], int] = {}
    nxt = 0
    for m in range(n_modules):
        for c in range(copies_per_module):
            base_of[(m, c)] = nxt
            nxt += len([s for s in range(n_source) if mod_of[s] == m])
    members = {
        m: [s for s in range(n_source) if mod_of[s] == m] for m in range(n_modules)
    }
    out: Edges = {}
    for m in range(n_modules):
        for c in range(copies_per_module):
            base = base_of[(m, c)]
            idx = {s: base + k for k, s in enumerate(members[m])}
            for s in members[m]:
                out[idx[s]] = [
                    (idx[t], wt)
                    for t, wt in source.get(s, [])
                    if mod_of.get(t, -1) == m
                ]
    ids = list(out)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if rng.random() < p_inter:
                a, b = ids[i], ids[j]
                out.setdefault(a, []).append((b, w))
    n_out = nxt
    for nid in range(n_out):
        out.setdefault(nid, [])
    check_edges(out, n_out, "modular")
    return _meta(
        "modular",
        seed,
        {
            "n_modules": n_modules,
            "copies_per_module": copies_per_module,
            "p_inter": p_inter,
        },
        parent,
        n_source,
        sum(len(v) for v in source.values()),
        out,
    )


def spatial(
    source: Edges,
    n_source: int,
    length_scale: float = 0.25,
    target_edges: int | None = None,
    seed: int = 0,
    parent: str = "source",
) -> tuple[Edges, GenerationMetadata]:
    """Seeded unit-square positions; p(i→j) ∝ exp(-d/length_scale), calibrated
    to ``target_edges`` (default: source edge count) by deterministic search."""
    if length_scale <= 0:
        raise ValueError("length_scale must be positive")
    rng = rng_for(seed, "spatial")
    pos = [(rng.random(), rng.random()) for _ in range(n_source)]
    import math

    dist = [[0.0] * n_source for _ in range(n_source)]
    for i in range(n_source):
        for j in range(n_source):
            if i != j:
                dist[i][j] = math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])
    want = (
        target_edges
        if target_edges is not None
        else sum(len(v) for v in source.values())
    )
    w = mean_weight(source)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        exp_edges = sum(
            math.exp(-dist[i][j] / length_scale) * mid
            for i in range(n_source)
            for j in range(n_source)
            if i != j
        )
        if exp_edges < want:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    base = (lo + hi) / 2
    out: Edges = {s: [] for s in range(n_source)}
    for i in range(n_source):
        for j in range(n_source):
            if i != j and rng.random() < base * math.exp(-dist[i][j] / length_scale):
                out[i].append((j, w))
    check_edges(out, n_source, "spatial")
    return _meta(
        "spatial",
        seed,
        {"length_scale": length_scale, "target_edges": want},
        parent,
        n_source,
        sum(len(v) for v in source.values()),
        out,
    )


def motif_fill(
    catalog_motifs: dict[int, int],
    motif_code: int,
    n_instances: int,
    p_inter: float = 0.01,
    weight: float = 0.5,
    seed: int = 0,
    parent: str = "catalog",
) -> tuple[Edges, GenerationMetadata]:
    """Tile ``n_instances`` of a catalog motif + sparse inter-instance links."""
    from vnr.connectome.motifs import instantiate as _instantiate

    if motif_code not in catalog_motifs:
        raise ValueError(f"motif {motif_code} not in catalog")
    rng = rng_for(seed, "motif")
    pairs = _instantiate(motif_code, n_instances, 0)
    out: Edges = {}
    for s, t in pairs:
        out.setdefault(s, []).append((t, weight))
    ids = list(out)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if rng.random() < p_inter:
                a, b = ids[i], ids[j]
                out.setdefault(a, []).append((b, weight))
    n_out = 3 * n_instances
    for nid in range(n_out):
        out.setdefault(nid, [])
    check_edges(out, n_out, "motif_fill")
    return _meta(
        "motif_fill",
        seed,
        {"motif_code": motif_code, "n_instances": n_instances, "p_inter": p_inter},
        parent,
        0,
        0,
        out,
    )


def population(
    source: Edges,
    n_source: int,
    pop_size: int = 5,
    p_intra: float = 0.5,
    p_inter: float = 0.3,
    seed: int = 0,
    parent: str = "source",
) -> tuple[Edges, GenerationMetadata]:
    """Expand each neuron into a micro-population (§76 seed): dense intra-pop
    links, source edges inherited probabilistically between populations."""
    if pop_size < 1:
        raise ValueError("pop_size must be >= 1")
    for p in (p_intra, p_inter):
        if not 0.0 <= p <= 1.0:
            raise ValueError("probabilities must be in [0, 1]")
    rng = rng_for(seed, "population")
    w = mean_weight(source)
    out: Edges = {}
    for s in range(n_source):
        members = list(range(s * pop_size, (s + 1) * pop_size))
        for a in members:
            for b in members:
                if a != b and rng.random() < p_intra:
                    out.setdefault(a, []).append((b, w))
    for s in range(n_source):
        for t, wt in source.get(s, []):
            for a in range(s * pop_size, (s + 1) * pop_size):
                for b in range(t * pop_size, (t + 1) * pop_size):
                    if rng.random() < p_inter:
                        out.setdefault(a, []).append((b, wt))
    n_out = n_source * pop_size
    for nid in range(n_out):
        out.setdefault(nid, [])
    check_edges(out, n_out, "population")
    return _meta(
        "population",
        seed,
        {"pop_size": pop_size, "p_intra": p_intra, "p_inter": p_inter},
        parent,
        n_source,
        sum(len(v) for v in source.values()),
        out,
    )


def hybrid(
    source: Edges,
    n_source: int,
    n_modules: int = 2,
    copies_per_module: int = 2,
    p_drop: float = 0.05,
    p_inter: float = 0.01,
    seed: int = 0,
    parent: str = "source",
) -> tuple[Edges, GenerationMetadata]:
    """Compose: modular expansion, then a duplication-divergence pass (§16)."""
    mod_edges, mod_meta = modular(
        source,
        n_source,
        n_modules,
        copies_per_module,
        p_inter=p_inter,
        seed=seed,
        parent=parent,
    )
    n_mod = mod_meta.n_out
    dup_edges, _ = duplication_divergence(
        mod_edges,
        n_mod,
        p_drop=p_drop,
        p_rewire=p_drop,
        seed=seed + 1,
        parent="modular",
    )
    check_edges(dup_edges, 2 * n_mod, "hybrid")
    return _meta(
        "hybrid",
        seed,
        {
            "n_modules": n_modules,
            "copies_per_module": copies_per_module,
            "p_drop": p_drop,
            "p_inter": p_inter,
        },
        parent,
        n_source,
        sum(len(v) for v in source.values()),
        dup_edges,
    )


def random_control(
    source: Edges, n_source: int, seed: int = 0, parent: str = "source"
) -> tuple[Edges, GenerationMetadata]:
    """Configuration model: EXACT out-degree (and in-degree multiset) preserved,
    wiring randomized. The null control structural claims must beat."""
    rng = rng_for(seed, "random")
    out_stubs: list[int] = []
    indeg: dict[int, int] = {s: 0 for s in range(n_source)}
    for s in range(n_source):
        for t, _ in source.get(s, []):
            out_stubs.append(s)
            indeg[t] = indeg.get(t, 0) + 1
    in_stubs: list[int] = []
    for t, d in indeg.items():
        in_stubs.extend([t] * d)
    rng.shuffle(in_stubs)
    w = mean_weight(source)
    out: Edges = {s: [] for s in range(n_source)}
    for s, t in zip(out_stubs, in_stubs):
        out[s].append((t, w))
    check_edges(out, n_source, "random_control")
    return _meta(
        "random_control",
        seed,
        {},
        parent,
        n_source,
        sum(len(v) for v in source.values()),
        out,
    )


class CognitiveRole:
    """Computational role labels — structural heuristics, NEVER human regions (§19)."""

    SENSORY = "sensory"  # net sources (no incoming): entry points
    ASSOCIATION = "association"  # hubs: high total degree
    MEMORY = "memory"  # recurrently embedded: high mutual fraction
    VALUATION = "valuation"  # sinks with convergent fan-in
    GOAL = "goal"  # net sinks (no outgoing): readouts
    MOTOR = "motor"  # alias-safe output role (same as GOAL class here)
    UNASSIGNED = "unassigned"


def assign_roles(
    source: Edges, n_source: int, hub_quantile: float = 0.9
) -> dict[int, str]:
    """Topology-only role assignment. Documented heuristic, not a claim."""
    indeg = {s: 0 for s in range(n_source)}
    outdeg = {s: len(source.get(s, [])) for s in range(n_source)}
    for s in range(n_source):
        for t, _ in source.get(s, []):
            indeg[t] = indeg.get(t, 0) + 1
    degs = sorted(outdeg[s] + indeg.get(s, 0) for s in range(n_source))
    hub_at = degs[min(len(degs) - 1, int(hub_quantile * len(degs)))]
    roles: dict[int, str] = {}
    for s in range(n_source):
        if indeg.get(s, 0) == 0 and outdeg[s] > 0:
            roles[s] = CognitiveRole.SENSORY
        elif outdeg[s] == 0 and indeg.get(s, 0) > 0:
            roles[s] = CognitiveRole.GOAL
        elif outdeg[s] + indeg.get(s, 0) >= hub_at > 0:
            roles[s] = CognitiveRole.ASSOCIATION
        else:
            roles[s] = CognitiveRole.UNASSIGNED
    return roles
