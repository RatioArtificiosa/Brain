"""Budget-bounded scaling driver (plan PH2-WI02, spec §70).

Runs a large VIRTUAL population (default 1M logical neurons) under a hard
resident budget (default 50K) and proves the run completes without ever
materializing the whole network. Mechanism:

- logical IDs live in ``range(virtual_n)``; procedural targets fold into range
  with ``% virtual_n`` (real population addressing arrives at PH6);
- structural registration is LAZY (register-on-first-touch), so even metadata
  stays proportional to touched neurons, not virtual size;
- residency is capped every tick by stalest-first eviction: ``peak_resident``
  can never exceed ``resident_budget``;
- all drive randomness flows through ``keyed_uniform`` (D2.3) — no ``random``
  module anywhere, so two runs are bit-identical.

This is the reference-oracle scaling path: exact, single-threaded, slow. The
PH3 backends must reproduce its spike trains; performance lives there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vnr.core.events import EventQueue, NeuralEvent
from vnr.core.materialize import Materializer
from vnr.core.procedural import (
    ConnectivityParams,
    ProceduralConnectivity,
    keyed_uniform,
)

__all__ = ["ScalingResult", "ScalingSpec", "run_scaled"]

_DRIVE_POPULATION = 0x5CA1E


@dataclass(frozen=True)
class ScalingSpec:
    """1M-virtual / 50K-budget run. All ticks are ints (D2.4)."""

    virtual_n: int = 1_000_000
    resident_budget: int = 50_000
    ticks: int = 150
    seed: int = 7
    out_degree: int = 16
    weight: float = 0.5
    delay_ticks: int = 1
    hot_neurons: int = 1500
    drive_block_ticks: int = 30
    drive_amplitude: float = 12.0

    def __post_init__(self) -> None:
        for name in (
            "virtual_n",
            "resident_budget",
            "ticks",
            "out_degree",
            "delay_ticks",
            "hot_neurons",
            "drive_block_ticks",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.resident_budget >= self.virtual_n:
            raise ValueError(
                "resident_budget must be < virtual_n (else the test is vacuous)"
            )
        if not isinstance(self.weight, (int, float)) or isinstance(self.weight, bool):
            raise TypeError("weight must be a number")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("weight must be positive and finite")
        if not isinstance(self.drive_amplitude, (int, float)) or isinstance(
            self.drive_amplitude, bool
        ):
            raise TypeError("drive_amplitude must be a number")
        if not math.isfinite(self.drive_amplitude) or self.drive_amplitude <= 0:
            raise ValueError("drive_amplitude must be positive and finite")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an int")
        if self.delay_ticks < 0:
            raise ValueError("delay_ticks must be non-negative")


@dataclass(frozen=True)
class ScalingResult:
    """Outcome + residency proof of one bounded scaling run."""

    total_spikes: int
    events_delivered: int
    peak_resident: int
    materializations: int
    evictions: int
    ticks: int


def _rule(spec: ScalingSpec) -> ProceduralConnectivity:
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


def _drive_ids(spec: ScalingSpec, tick: int) -> list[int]:
    """Assembly-style drive: a HOT set sustained for a whole block, carrying
    half its members into the next block.

    Rationale (WI02 lesson, cf. neuron notes): on this integrator a single
    tick rises at most ``amplitude * (1 - decay)`` — salt-and-pepper pulses
    never accumulate (measured: 0 spikes from 1500 random hits/tick). Only
    SUSTAINED drive reaches steady state (``Vinf = amplitude``) and ignites.
    The half-carry compounds recurrent fan-out into fresh members, so activity
    cascades instead of dying at every block boundary. All keyed (D2.3).
    """
    block = tick // spec.drive_block_ticks
    fresh = {
        int(keyed_uniform(spec.seed, s, _DRIVE_POPULATION, 1, block) * spec.virtual_n)
        for s in range(spec.hot_neurons)
    }
    if block == 0:
        return sorted(fresh)
    carry = {
        int(
            keyed_uniform(spec.seed, s, _DRIVE_POPULATION, 1, block - 1)
            * spec.virtual_n
        )
        for s in range(spec.hot_neurons // 2)
    }
    return sorted(fresh | carry)


def run_scaled(spec: ScalingSpec) -> ScalingResult:
    """Virtual-N run capped at ``resident_budget`` residents every tick."""
    rule = _rule(spec)
    mat = Materializer()
    registered: set[int] = set()
    queue: EventQueue = EventQueue()
    last_live: dict[int, int] = {}
    total_spikes = 0
    delivered = 0
    peak_resident = 0

    def ensure(nid: int, tick: int) -> None:
        if nid not in registered:
            mat.register(nid)
            registered.add(nid)
        if not mat.is_resident(nid):
            mat.materialize_neuron(nid)
            for missed in range(last_live.get(nid, -1) + 1, tick):
                mat.step_neuron(nid, missed, 0.0)

    def enforce_budget() -> None:
        over = len(mat.resident_ids()) - spec.resident_budget
        if over <= 0:
            return
        stalest = sorted(mat.resident_ids(), key=lambda nid: last_live.get(nid, -1))
        for nid in stalest[:over]:
            mat.evict_neuron(nid)

    for t in range(spec.ticks):
        inp: dict[int, float] = {}
        for event in queue.drain_tick(t):
            ensure(event.target_id, t)
            inp[event.target_id] = inp.get(event.target_id, 0.0) + event.weight
            delivered += 1
        for nid in _drive_ids(spec, t):
            ensure(nid, t)
            inp[nid] = inp.get(nid, 0.0) + spec.drive_amplitude
        for nid in mat.resident_ids():
            if mat.step_neuron(nid, t, inp.get(nid, 0.0)):
                total_spikes += 1
                for tgt in rule.sample_targets(nid):
                    queue.push(
                        NeuralEvent(
                            tick=t + spec.delay_ticks,
                            source_id=nid,
                            target_id=tgt % spec.virtual_n,
                            weight=spec.weight,
                        )
                    )
            last_live[nid] = t
        enforce_budget()
        peak_resident = max(peak_resident, len(mat.resident_ids()))

    stats = mat.stats()
    return ScalingResult(
        total_spikes=total_spikes,
        events_delivered=delivered,
        peak_resident=peak_resident,
        materializations=stats["materializations"],
        evictions=stats["evictions"],
        ticks=spec.ticks,
    )
