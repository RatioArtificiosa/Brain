"""E004 scaling sweep with controls - the knee graph (plan PH5-WI02, spec 37-39).

THE project's central experiment. It asks one question at increasing scale:
as we compress the network, at what point does behaviour break, and what does
the curve look like just before it does?

Design (per plan 38-39):
  * Scale sweep: 1x-100x on a real connectome base.
  * COMPRESSION axis: how aggressively the network is virtualized. Each step
    keeps a smaller resident budget and evicts harder, so the run holds less
    of the network physically resident.
  * FIDELITY axis: scored against the SAME network run fully explicit - the
    reference - using per-metric distances (spec 72), never a collapsed
    scalar. The gate is an AND of thresholds.
  * CONTROLS (six, per spec 38-39): the structure must earn its keep, so the
    same sweep runs against null models. If the real connectome does not beat
    them, that is the result and it ships.

Every number below is measured in the run that prints it.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vnr.backend.fidelity import compare_spike_trains
from vnr.connectome.flywire_dataset import FlyWireDataset
from vnr.core.events import EventQueue, NeuralEvent
from vnr.core.frontier import (
    ActiveFrontier,
    FrontierParams,
    FrontierState,
)
from vnr.core.materialize import Materializer
from vnr.core.neuron import LIFNeuron
from vnr.core.procedural import keyed_uniform

DATA = Path(r"G:\BRAIN\VNR\data\flywire_v783")
DRIVE_POP = 0xD1
WEIGHT = 0.4


@dataclass
class SweepPoint:
    """One measured point on the compression-vs-fidelity curve."""

    condition: str
    scale: int
    budget_fraction: float
    ticks: int
    spikes_reference: int
    spikes_candidate: int
    peak_resident: int
    resident_fraction: float
    evictions: int
    rate_ratio: float
    count_correlation: float
    binned_cosine: float
    max_count_diff: int
    gate_pass: bool
    wall_seconds: float


@dataclass
class SweepResult:
    """The full curve plus the controls it must beat."""

    base_neurons: int
    base_edges: int
    ticks: int
    seed: int
    points: list[SweepPoint] = field(default_factory=list)
    controls: dict[str, list[SweepPoint]] = field(default_factory=dict)


def _select_neurons(ds: FlyWireDataset, n: int) -> list[int]:
    """The n highest-degree neurons: a connected core, not degree-0 noise."""
    return sorted(ds.neurons(), key=lambda x: -len(ds.successors(x)))[:n]


def _control_edges(ds: FlyWireDataset, keep: list[int], kind: str, seed: int):
    """Build the six control adjacency variants (spec 38-39)."""
    import random

    rng = random.Random(seed)
    keep_set = set(keep)
    real = {n: [t for t, _ in ds.successors(n) if t in keep_set] for n in keep}
    out_degrees = {n: len(real[n]) for n in keep}

    if kind == "real":
        return real
    if kind == "random":
        # Degree-matched random: same out-degree, targets drawn uniformly.
        return {n: [rng.choice(keep) for _ in range(out_degrees[n])] for n in keep}
    if kind == "degree_matched_shuffle":
        # Preserve the exact target multiset, shuffle which source sends it.
        stubs = [t for n in keep for t in real[n]]
        rng.shuffle(stubs)
        out: dict[int, list[int]] = {n: [] for n in keep}
        i = 0
        for n in keep:
            out[n] = stubs[i : i + out_degrees[n]]
            i += out_degrees[n]
        return {n: [t for t in out[n] if t is not None] for n in keep}
    if kind == "motif_destroyed":
        # Break TRIADIC structure while preserving BOTH out-degree and the
        # edge-length distribution, so the control isolates motif content
        # rather than re-testing locality (which `random` already covers).
        #
        # Method: collect every edge's SPAN (target - source), shuffle the
        # span multiset, and reassign spans to sources. Out-degrees are
        # untouched and the overall edge length distribution is identical,
        # but which triangles exist is randomised. If the real graph still
        # wins here, the advantage is triadic structure rather than merely
        # having shorter edges.
        spans = [t - n for n in keep for t in real[n]]
        rng.shuffle(spans)
        out_m: dict[int, list[int]] = {}
        cursor = 0
        for n in keep:
            degree = out_degrees[n]
            assigned = []
            for span in spans[cursor : cursor + degree]:
                cand = n + span
                if cand not in keep_set:
                    # Fall back to the nearest in-set node at that distance,
                    # preserving the span as closely as the id space allows.
                    cand = min(keep, key=lambda x: abs((x - n) - span))
                assigned.append(cand)
            out_m[n] = assigned
            cursor += degree
        if out_m == real:
            raise ValueError(
                "motif_destroyed produced no change; it is not a distinct control"
            )
        return out_m
    if kind == "hub_removed":
        ranked = sorted(keep, key=lambda x: -out_degrees[x])
        drop = set(ranked[: max(1, len(ranked) // 100)])
        return {
            n: [t for t in real[n] if t not in drop] if n not in drop else []
            for n in keep
        }
    if kind == "locality_preserved_only":
        # Keep only edges whose span is below the median: purely local wiring.
        spans = sorted(abs(t - n) for n in keep for t in real[n])
        cutoff = spans[len(spans) // 2] if spans else 0
        return {n: [t for t in real[n] if abs(t - n) <= cutoff] for n in keep}
    raise ValueError(f"unknown control {kind!r}")


def run_explicit(nodes: list[int], adj, ticks: int, hot: int, amp: float, seed: int):
    neurons = {n: LIFNeuron() for n in nodes}
    queue = EventQueue()
    spikes: dict[int, list[int]] = {n: [] for n in nodes}
    for t in range(ticks):
        inp: dict[int, float] = {}
        for ev in queue.drain_tick(t):
            inp[ev.target_id] = inp.get(ev.target_id, 0.0) + ev.weight
        for s in range(hot):
            n = nodes[int(keyed_uniform(seed, s, DRIVE_POP, 1, t // 20) * len(nodes))]
            inp[n] = inp.get(n, 0.0) + amp
        for n in nodes:
            if neurons[n].step(t, inp.get(n, 0.0)):
                spikes[n].append(t)
                for tgt in adj[n]:
                    queue.push(
                        NeuralEvent(
                            tick=t + 1, source_id=n, target_id=tgt, weight=WEIGHT
                        )
                    )
    return spikes


def run_virtualized(
    nodes: list[int],
    adj,
    ticks: int,
    hot: int,
    amp: float,
    seed: int,
    budget_fraction: float,
    evict_ticks: int,
):
    """Virtualized run under a hard resident budget (the compression axis)."""
    budget = max(2, int(len(nodes) * budget_fraction))
    mat = Materializer()
    frontier = ActiveFrontier(
        params=FrontierParams(
            activate_count=2,
            quiet_ticks=max(2, evict_ticks // 4),
            evict_ticks=evict_ticks,
            candidate_timeout_ticks=max(2, evict_ticks // 4),
        )
    )
    for n in nodes:
        mat.register(n)
    queue = EventQueue()
    spikes: dict[int, list[int]] = {n: [] for n in nodes}
    last_live: dict[int, int] = {}
    peak = 0

    def ensure(n: int, tick: int) -> None:
        if not mat.is_resident(n):
            mat.materialize_neuron(n)
            for missed in range(last_live.get(n, -1) + 1, tick):
                mat.step_neuron(n, missed, 0.0)
        frontier.observe(n, tick)

    for t in range(ticks):
        inp: dict[int, float] = {}
        for ev in queue.drain_tick(t):
            ensure(ev.target_id, t)
            inp[ev.target_id] = inp.get(ev.target_id, 0.0) + ev.weight
        for s in range(hot):
            n = nodes[int(keyed_uniform(seed, s, DRIVE_POP, 1, t // 20) * len(nodes))]
            ensure(n, t)
            inp[n] = inp.get(n, 0.0) + amp
        for n in mat.resident_ids():
            if mat.step_neuron(n, t, inp.get(n, 0.0)):
                spikes[n].append(t)
                for tgt in adj[n]:
                    queue.push(
                        NeuralEvent(
                            tick=t + 1, source_id=n, target_id=tgt, weight=WEIGHT
                        )
                    )
            last_live[n] = t
        peak = max(peak, len(mat.resident_ids()))
        # Hard budget: evict stalest first, in addition to frontier eviction.
        over = len(mat.resident_ids()) - budget
        if over > 0:
            stalest = sorted(mat.resident_ids(), key=lambda n: last_live.get(n, -1))
            for n in stalest[:over]:
                mat.evict_neuron(n)
        frontier.update(t)
        for n in mat.resident_ids():
            if frontier.state_of(n) is FrontierState.EVICTING:
                mat.evict_neuron(n)
    return spikes, peak, mat.stats()["evictions"]


def sweep(
    ds: FlyWireDataset,
    n_base: int,
    ticks: int,
    seed: int,
    budget_fraction: float = 0.25,
    evict_ticks: int = 20,
    controls: tuple[str, ...] = ("real",),
) -> SweepResult:
    """Run the compression-vs-fidelity sweep at each scale, plus controls."""
    nodes = _select_neurons(ds, n_base)
    hot = max(1, n_base // 50)
    result = SweepResult(
        base_neurons=len(nodes), base_edges=ds.edge_count(), ticks=ticks, seed=seed
    )

    for kind in controls:
        adj = _control_edges(ds, nodes, kind, seed)
        points: list[SweepPoint] = []
        # ONE reference for the whole curve, computed over the FULL node set.
        # The compression axis is the resident budget, not the network size:
        # shrinking the network while scoring against a full-size reference
        # changes two things at once and produces a meaningless rate_ratio
        # (measured: 1.00 -> 5.65 on an early buggy revision of this script).
        ref = run_explicit(nodes, adj, ticks, hot, 12.0, seed)
        ref_total = sum(len(v) for v in ref.values())
        for budget in (1.0, 0.5, 0.25, 0.125, 0.0625):
            t0 = time.perf_counter()
            cand, peak, evict = run_virtualized(
                nodes, adj, ticks, hot, 12.0, seed, budget, evict_ticks
            )
            wall = time.perf_counter() - t0
            score = compare_spike_trains(ref, cand, ticks)
            point = SweepPoint(
                condition=kind,
                scale=len(nodes),
                budget_fraction=budget,
                ticks=ticks,
                spikes_reference=ref_total,
                spikes_candidate=sum(len(v) for v in cand.values()),
                peak_resident=peak,
                resident_fraction=peak / len(nodes) if nodes else 0.0,
                evictions=evict,
                rate_ratio=score.rate_ratio,
                count_correlation=score.count_correlation,
                binned_cosine=score.binned_cosine,
                max_count_diff=score.max_count_diff,
                gate_pass=score.overall_pass,
                wall_seconds=wall,
            )
            points.append(point)
        if kind == "real":
            result.points = points
        else:
            result.controls[kind] = points
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons", type=int, default=600)
    parser.add_argument("--ticks", type=int, default=400)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--evict-ticks", type=int, default=20)
    parser.add_argument("--controls", default="real")
    parser.add_argument("--out", default="artifacts/e004_sweep.json")
    args = parser.parse_args()

    chunks = tuple(sorted(DATA.glob("byid-*.parquet")))
    if not chunks:
        print("no corpus present")
        return 1
    print(f"loading real FlyWire corpus ({len(chunks)} chunks)...")
    ds = FlyWireDataset(chunks=chunks)
    print(f"corpus: {len(ds.neurons()):,} neurons, {ds.edge_count():,} edges")

    kinds = tuple(k.strip() for k in args.controls.split(",") if k.strip())
    print(f"controls: {kinds}")
    result = sweep(
        ds,
        args.neurons,
        args.ticks,
        args.seed,
        budget_fraction=args.budget,
        evict_ticks=args.evict_ticks,
        controls=kinds,
    )

    header = (
        f"{'condition':<26}{'budget':>8}{'spikes':>9}{'peak':>7}"
        f"{'%res':>8}{'evict':>8}{'rate':>9}{'cos':>10}{'gate':>6}"
    )
    print()
    print(header)
    print("-" * len(header))

    def _show(point: SweepPoint, label: str) -> None:
        print(
            f"{label:<26}{point.budget_fraction * 100:>7.2f}%"
            f"{point.spikes_candidate:>9,}{point.peak_resident:>7,}"
            f"{point.resident_fraction * 100:>7.1f}%{point.evictions:>8,}"
            f"{point.rate_ratio:>9.4f}{point.binned_cosine:>10.6f}"
            f"{'PASS' if point.gate_pass else 'FAIL':>6}"
        )

    for point in result.points:
        _show(point, point.condition)
    for kind, points in result.controls.items():
        for point in points:
            _show(point, kind if point is points[0] else "")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_neurons": result.base_neurons,
        "base_edges": result.base_edges,
        "ticks": result.ticks,
        "seed": result.seed,
        "real": [asdict(p) for p in result.points],
        "controls": {k: [asdict(p) for p in v] for k, v in result.controls.items()},
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
