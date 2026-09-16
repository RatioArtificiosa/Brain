"""Does the REAL connectome lift the virtualization ceiling? (entry 39 test)

Entry 39 measured that residency is governed by fan-out, and that uniform
random connectivity has no locality, so the touched set grows toward the whole
network. That capped synthetic compression at ~24-40% resident.

Hypothesis: on real FlyWire connectivity, activity stays local, so a sparse
drive keeps a much smaller fraction resident at the same fidelity.

This script tests it directly, and reports the honest answer either way.
"""

from __future__ import annotations

import sys
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
from vnr.core.procedural import keyed_uniform

DATA = Path(r"G:\BRAIN\VNR\data\flywire_v783")


def run_real_network(
    ds: FlyWireDataset,
    n_neurons: int,
    ticks: int,
    hot_fraction: float,
    drive_amplitude: float,
    seed: int,
    virtualized: bool,
    quiet_ticks: int = 10,
    evict_ticks: int = 40,
) -> tuple[dict[int, list[int]], int, int]:
    """Run the REAL adjacency (optionally with materialize/evict)."""
    drive_pop = 0xF1  # arbitrary domain tag separating drive draws from edges
    hot = max(1, int(n_neurons * hot_fraction))

    def drive_at(tick: int) -> set[int]:
        block = tick // 20
        return {
            int(keyed_uniform(seed, s, drive_pop, 1, block) * n_neurons)
            for s in range(hot)
        }

    # Decide which neurons are "logical": take the n_neurons with most edges so
    # the network is connected rather than dominated by degree-0 nodes.
    ranked = sorted(ds.neurons(), key=lambda n: -len(ds.successors(n)))[:n_neurons]
    keep = set(ranked)
    adjacency = {n: [t for t, _ in ds.successors(n) if t in keep] for n in ranked}

    if not virtualized:
        from vnr.core.neuron import LIFNeuron

        neurons = {n: LIFNeuron() for n in ranked}
        queue = EventQueue()
        spikes: dict[int, list[int]] = {n: [] for n in ranked}
        for t in range(ticks):
            inp: dict[int, float] = {}
            for ev in queue.drain_tick(t):
                inp[ev.target_id] = inp.get(ev.target_id, 0.0) + ev.weight
            for n in drive_at(t):
                if n in keep:
                    inp[n] = inp.get(n, 0.0) + drive_amplitude
            for n in ranked:
                if neurons[n].step(t, inp.get(n, 0.0)):
                    spikes[n].append(t)
                    for tgt in adjacency[n]:
                        queue.push(
                            NeuralEvent(
                                tick=t + 1, source_id=n, target_id=tgt, weight=0.4
                            )
                        )
        return spikes, len(ranked), 0

    mat = Materializer()
    frontier = ActiveFrontier(
        params=FrontierParams(
            activate_count=2,
            quiet_ticks=quiet_ticks,
            evict_ticks=evict_ticks,
            candidate_timeout_ticks=quiet_ticks,
        )
    )
    for n in ranked:
        mat.register(n)
    queue = EventQueue()
    spikes = {n: [] for n in ranked}
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
            if ev.target_id in keep:
                ensure(ev.target_id, t)
                inp[ev.target_id] = inp.get(ev.target_id, 0.0) + ev.weight
        for n in drive_at(t):
            if n in keep:
                ensure(n, t)
                inp[n] = inp.get(n, 0.0) + drive_amplitude
        for n in mat.resident_ids():
            if mat.step_neuron(n, t, inp.get(n, 0.0)):
                spikes[n].append(t)
                for tgt in adjacency[n]:
                    queue.push(
                        NeuralEvent(tick=t + 1, source_id=n, target_id=tgt, weight=0.4)
                    )
            last_live[n] = t
        peak = max(peak, len(mat.resident_ids()))
        frontier.update(t)
        for n in mat.resident_ids():
            if frontier.state_of(n) is FrontierState.EVICTING:
                mat.evict_neuron(n)
    return spikes, peak, mat.stats()["evictions"]


def main() -> int:
    chunks = tuple(sorted(DATA.glob("byid-*.parquet")))
    if not chunks:
        print("no corpus")
        return 1
    print(f"loading {len(chunks)} chunks of REAL FlyWire data...")
    ds = FlyWireDataset(chunks=chunks)
    print(f"connectome: {len(ds.neurons()):,} neurons, {ds.edge_count():,} edges")
    print(f"degree: {ds.degree_stats()}")
    print(f"locality: {FlyWireDataset.locality_ratio(ds):.4f} (uniform ~0.33-0.5)")
    print()

    n = 2000
    ticks = 600
    print(
        f"{'hot%':>6}{'amp':>6}{'spikes':>9}{'peak':>8}{'%res':>9}{'evict':>8}{'cos':>11}"
    )
    print("-" * 60)

    for hot_fraction, amp in [(0.02, 12.0), (0.05, 12.0), (0.10, 12.0)]:
        ref_spikes, _, _ = run_real_network(
            ds, n, ticks, hot_fraction, amp, 7, virtualized=False
        )
        virt_spikes, peak, evict = run_real_network(
            ds, n, ticks, hot_fraction, amp, 7, virtualized=True
        )
        total_ref = sum(len(v) for v in ref_spikes.values())
        total_virt = sum(len(v) for v in virt_spikes.values())
        score = compare_spike_trains(ref_spikes, virt_spikes, ticks)
        pct = peak / n * 100
        print(
            f"{hot_fraction * 100:>5.0f}%{amp:>6.0f}{total_ref:>9,}{peak:>8,}"
            f"{pct:>8.1f}%{evict:>8,}{score.binned_cosine:>11.6f}"
        )
        if total_ref != total_virt:
            print(
                f"        ! spike mismatch: explicit {total_ref} vs virtualized {total_virt}"
            )

    print()
    print("Compare with the SYNTHETIC result (entry 39): degree 8 -> 76.6%,")
    print("degree 2 -> 36.7% resident. If the real connectome holds a much")
    print("smaller fraction resident at the same fidelity, structure is doing")
    print("the work that uniform fan-out could not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
