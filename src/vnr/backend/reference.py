"""Reference backend (plan §8 oracle, PH2-WI01, spec §69).

Two event-driven simulators over the same static network, sharing one driver
discipline: every neuron steps every tick in sorted-ID order, deliveries
accumulate before drive, pushes follow spikes in order. Given identical
inputs the float operations run in identical order, so agreement is bit-exact
— any divergence is a virtualization bug, never rounding.

- ``run_explicit``: all neurons resident, stored adjacency list.
- ``run_virtualized``: edges regenerated procedurally per spike, neurons
  materialized on first input and evicted through the WI04 frontier; missed
  ticks replay as exact zero-input steps on rematerialize (identical op
  sequence to the explicit path, so lazy reactivation stays bit-exact).

Target hashes are mapped ``% n_neurons`` to close the test network; real
population addressing arrives at PH6. The drive alternates halves across
thirds (A-B-A) so evictions AND rematerializations both occur mid-run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vnr.core.events import EventQueue, NeuralEvent
from vnr.core.frontier import ActiveFrontier, FrontierParams, FrontierState
from vnr.core.materialize import Materializer
from vnr.core.neuron import LIFNeuron
from vnr.core.procedural import (
    ConnectivityParams,
    ProceduralConnectivity,
    keyed_uniform,
)

__all__ = [
    "RunResult",
    "StaticNetSpec",
    "VirtualStats",
    "build_drive",
    "run_explicit",
    "run_virtualized",
]

_DRIVE_POPULATION = 0xD21E
_DRIVE_BLOCK = 100


@dataclass(frozen=True)
class StaticNetSpec:
    """Closed static network + phased drive. All ticks are ints (D2.4)."""

    n_neurons: int = 512
    seed: int = 1
    out_degree: int = 8
    weight: float = 0.4
    delay_ticks: int = 2
    ticks: int = 1500
    drive_density: float = 0.4
    drive_amplitude: float = 3.0
    quiet_ticks: int = 15
    evict_ticks: int = 60

    def __post_init__(self) -> None:
        for name in ("n_neurons", "ticks", "out_degree", "quiet_ticks", "evict_ticks"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
        if self.n_neurons < 2:
            raise ValueError("n_neurons must be >= 2 (two drive halves)")
        if self.ticks < 3:
            raise ValueError("ticks must be >= 3 (three drive phases)")
        if self.out_degree < 1:
            raise ValueError("out_degree must be positive")
        if not 0.0 < self.drive_density < 1.0:
            raise ValueError("drive_density must be in (0, 1)")
        if not isinstance(self.drive_amplitude, (int, float)) or isinstance(
            self.drive_amplitude, bool
        ):
            raise TypeError("drive_amplitude must be a number")
        if not math.isfinite(self.drive_amplitude) or self.drive_amplitude <= 0:
            raise ValueError("drive_amplitude must be positive and finite")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an int")
        if self.quiet_ticks <= 0 or self.evict_ticks <= self.quiet_ticks:
            raise ValueError("need 0 < quiet_ticks < evict_ticks (hysteresis band)")
        if not isinstance(self.weight, (int, float)) or isinstance(self.weight, bool):
            raise TypeError("weight must be a number")
        if not math.isfinite(self.weight):
            raise ValueError("weight must be finite")
        if not isinstance(self.delay_ticks, int) or isinstance(self.delay_ticks, bool):
            raise TypeError("delay_ticks must be an int")
        if self.delay_ticks < 0:
            raise ValueError("delay_ticks must be non-negative")


@dataclass(frozen=True)
class RunResult:
    """Bit-comparable outcome of one static-net run."""

    spikes: dict[int, list[int]]
    final_v: dict[int, float]
    final_refractory: dict[int, int]
    events_delivered: int

    @property
    def total_spikes(self) -> int:
        """Total spikes across all neurons."""
        return sum(len(s) for s in self.spikes.values())


@dataclass(frozen=True)
class VirtualStats:
    """Residency proof: the run stayed bounded while matching exactly."""

    max_resident: int
    materializations: int
    evictions: int


def _rule(spec: StaticNetSpec) -> ProceduralConnectivity:
    return ProceduralConnectivity(
        params=ConnectivityParams(
            global_seed=spec.seed,
            target_population_id=0,
            connectivity_version=1,
            out_degree=spec.out_degree,
            weight=spec.weight,
            delay_ticks=spec.delay_ticks,
        )
    )


def build_drive(spec: StaticNetSpec) -> list[list[int]]:
    """Per-tick driven-neuron lists, shared verbatim by both paths.

    Halves alternate A-B-A across thirds: the idle half cools, evicts, and
    rematerializes — the comparison is vacuous unless that cycle fires.
    Drive comes in sustained ``_DRIVE_BLOCK``-tick blocks (salt-and-pepper
    ticks never accumulate past threshold on this integrator — steady
    states, not pulse amplitudes, cf. WI02 notes).
    """
    drive: list[list[int]] = []
    half_n = spec.n_neurons // 2
    active_block: dict[int, bool] = {}
    for t in range(spec.ticks):
        third = min(2, t * 3 // spec.ticks)
        want_half = 0 if third in (0, 2) else 1
        tick_ids = []
        for nid in range(spec.n_neurons):
            if (0 if nid < half_n else 1) != want_half:
                continue
            block = t // _DRIVE_BLOCK
            key = nid * 1_000_003 + block
            if key not in active_block:
                active_block[key] = (
                    keyed_uniform(spec.seed, nid, _DRIVE_POPULATION, 1, block)
                    < spec.drive_density
                )
            if active_block[key]:
                tick_ids.append(nid)
        drive.append(tick_ids)
    return drive


def _push(
    queue: EventQueue, targets: list[int], source: int, tick: int, spec: StaticNetSpec
) -> None:
    for target in targets:
        queue.push(
            NeuralEvent(
                tick=tick + spec.delay_ticks,
                source_id=source,
                target_id=target,
                weight=spec.weight,
            )
        )


def run_explicit(
    spec: StaticNetSpec, drive: list[list[int]] | None = None
) -> RunResult:
    """All-resident reference: stored adjacency, resident LIF per neuron."""
    rule = _rule(spec)
    adjacency = {
        nid: [tgt % spec.n_neurons for tgt in rule.sample_targets(nid)]
        for nid in range(spec.n_neurons)
    }
    neurons = {nid: LIFNeuron() for nid in range(spec.n_neurons)}
    queue: EventQueue = EventQueue()
    spikes: dict[int, list[int]] = {nid: [] for nid in range(spec.n_neurons)}
    drive = build_drive(spec) if drive is None else drive
    delivered = 0
    for t in range(spec.ticks):
        inp: dict[int, float] = {}
        for event in queue.drain_tick(t):
            inp[event.target_id] = inp.get(event.target_id, 0.0) + event.weight
            delivered += 1
        for nid in drive[t]:
            inp[nid] = inp.get(nid, 0.0) + spec.drive_amplitude
        for nid in range(spec.n_neurons):
            if neurons[nid].step(t, inp.get(nid, 0.0)):
                spikes[nid].append(t)
                _push(queue, adjacency[nid], nid, t, spec)
    return RunResult(
        spikes=spikes,
        final_v={nid: n.state.v for nid, n in neurons.items()},
        final_refractory={
            nid: n.state.refractory_until_tick for nid, n in neurons.items()
        },
        events_delivered=delivered,
    )


def run_virtualized(
    spec: StaticNetSpec, drive: list[list[int]] | None = None
) -> tuple[RunResult, VirtualStats]:
    """Virtualized twin: procedural fan-out, materialize-on-input, eviction
    through the frontier. Missed ticks replay as zero-input steps."""
    rule = _rule(spec)
    mat = Materializer()
    for nid in range(spec.n_neurons):
        mat.register(nid)
    frontier = ActiveFrontier(
        params=FrontierParams(
            activate_count=2,
            quiet_ticks=spec.quiet_ticks,
            evict_ticks=spec.evict_ticks,
            candidate_timeout_ticks=spec.quiet_ticks,
        )
    )
    queue: EventQueue = EventQueue()
    spikes: dict[int, list[int]] = {nid: [] for nid in range(spec.n_neurons)}
    last_live: dict[int, int] = {}
    drive = build_drive(spec) if drive is None else drive
    delivered = 0
    max_resident = 0

    def ensure(nid: int, tick: int) -> None:
        if not mat.is_resident(nid):
            mat.materialize_neuron(nid)
            for missed in range(last_live.get(nid, -1) + 1, tick):
                mat.step_neuron(nid, missed, 0.0)
        frontier.observe(nid, tick)

    for t in range(spec.ticks):
        inp: dict[int, float] = {}
        for event in queue.drain_tick(t):
            ensure(event.target_id, t)
            inp[event.target_id] = inp.get(event.target_id, 0.0) + event.weight
            delivered += 1
        for nid in drive[t]:
            ensure(nid, t)
            inp[nid] = inp.get(nid, 0.0) + spec.drive_amplitude
        for nid in mat.resident_ids():
            if mat.step_neuron(nid, t, inp.get(nid, 0.0)):
                spikes[nid].append(t)
                _push(
                    queue,
                    [tgt % spec.n_neurons for tgt in rule.sample_targets(nid)],
                    nid,
                    t,
                    spec,
                )
            last_live[nid] = t
        max_resident = max(max_resident, len(mat.resident_ids()))
        frontier.update(t)
        for nid in mat.resident_ids():
            if frontier.state_of(nid) is FrontierState.EVICTING:
                mat.evict_neuron(nid)
    for nid in range(spec.n_neurons):  # logical final state: catch up the evicted
        if not mat.is_resident(nid):
            mat.materialize_neuron(nid)
            for missed in range(last_live.get(nid, -1) + 1, spec.ticks):
                mat.step_neuron(nid, missed, 0.0)
        last_live[nid] = spec.ticks - 1
    final = {nid: mat.materialize_neuron(nid) for nid in range(spec.n_neurons)}
    stats = mat.stats()
    return (
        RunResult(
            spikes=spikes,
            final_v={nid: s.v for nid, s in final.items()},
            final_refractory={nid: s.refractory_until_tick for nid, s in final.items()},
            events_delivered=delivered,
        ),
        VirtualStats(
            max_resident=max_resident,
            materializations=stats["materializations"],
            evictions=stats["evictions"],
        ),
    )
