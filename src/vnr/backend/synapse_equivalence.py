"""Explicit-vs-procedural synapse equivalence (plan PH2-WI03, spec §71).

Question: when synapses are regenerated procedurally (optionally quantized)
instead of stored, how far do the dynamics diverge? The harness runs one
small static network twice — once with stored per-edge weights, once with
procedurally regenerated weights — and reports divergence metrics. With
``weight_bits=None`` the weights are identical by construction, so the gate
is bit-exactness; with quantization it turns into the project's first
compression-vs-fidelity data point (rate ratio + count-vector correlation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vnr.core.events import EventQueue, NeuralEvent
from vnr.core.neuron import LIFNeuron
from vnr.core.procedural import (
    ConnectivityParams,
    ProceduralConnectivity,
    keyed_uniform,
)

__all__ = ["EquivalenceResult", "EquivalenceSpec", "compare"]

_DRIVE_POP = 0xE017


@dataclass(frozen=True)
class EquivalenceSpec:
    """Small static net with per-edge weight variation. Ticks are ints (D2.4)."""

    n_neurons: int = 256
    seed: int = 21
    out_degree: int = 16
    weight: float = 0.5
    weight_std: float = 0.2
    delay_ticks: int = 1
    ticks: int = 400
    hot_neurons: int = 40
    drive_block_ticks: int = 50
    drive_amplitude: float = 12.0

    def __post_init__(self) -> None:
        for name in (
            "n_neurons",
            "out_degree",
            "ticks",
            "delay_ticks",
            "hot_neurons",
            "drive_block_ticks",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_neurons < 2:
            raise ValueError("n_neurons must be >= 2")
        for name in ("weight", "drive_amplitude"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.weight_std, (int, float)) or isinstance(
            self.weight_std, bool
        ):
            raise TypeError("weight_std must be a number")
        if not math.isfinite(self.weight_std) or self.weight_std < 0:
            raise ValueError("weight_std must be non-negative and finite")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an int")


@dataclass(frozen=True)
class EquivalenceResult:
    """Divergence between stored-weight and procedural-weight runs."""

    spikes_explicit: int
    spikes_procedural: int
    rate_ratio: float  # procedural / explicit total spikes
    count_correlation: float  # Pearson r of per-neuron spike counts
    max_count_diff: int
    weight_bits: int | None


def _rule(spec: EquivalenceSpec, weight_bits: int | None) -> ProceduralConnectivity:
    return ProceduralConnectivity(
        params=ConnectivityParams(
            global_seed=spec.seed,
            target_population_id=0,
            connectivity_version=1,
            out_degree=spec.out_degree,
            weight=spec.weight,
            weight_std=spec.weight_std,
            weight_bits=weight_bits,
            delay_ticks=spec.delay_ticks,
        )
    )


def _drive_ids(spec: EquivalenceSpec, tick: int) -> list[int]:
    block = tick // spec.drive_block_ticks
    fresh = {
        int(keyed_uniform(spec.seed, s, _DRIVE_POP, 1, block) * spec.n_neurons)
        for s in range(spec.hot_neurons)
    }
    if block == 0:
        return sorted(fresh)
    carry = {
        int(keyed_uniform(spec.seed, s, _DRIVE_POP, 1, block - 1) * spec.n_neurons)
        for s in range(spec.hot_neurons // 2)
    }
    return sorted(fresh | carry)


def _run(
    weights_of: dict[int, list[float]],
    targets_of: dict[int, list[int]],
    spec: EquivalenceSpec,
) -> dict[int, int]:
    """All-resident event-driven run. Returns per-neuron spike counts."""
    neurons = {nid: LIFNeuron() for nid in range(spec.n_neurons)}
    queue: EventQueue = EventQueue()
    counts = {nid: 0 for nid in range(spec.n_neurons)}
    for t in range(spec.ticks):
        inp: dict[int, float] = {}
        for event in queue.drain_tick(t):
            inp[event.target_id] = inp.get(event.target_id, 0.0) + event.weight
        for nid in _drive_ids(spec, t):
            inp[nid] = inp.get(nid, 0.0) + spec.drive_amplitude
        for nid in range(spec.n_neurons):
            if neurons[nid].step(t, inp.get(nid, 0.0)):
                counts[nid] += 1
                for tgt, w in zip(targets_of[nid], weights_of[nid]):
                    queue.push(
                        NeuralEvent(
                            tick=t + spec.delay_ticks,
                            source_id=nid,
                            target_id=tgt,
                            weight=w,
                        )
                    )
    return counts


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 1.0 if vx == vy else 0.0
    return cov / math.sqrt(vx * vy)


def compare(spec: EquivalenceSpec, weight_bits: int | None) -> EquivalenceResult:
    """Run stored-weight vs procedural-weight (optionally quantized) networks."""
    rule = _rule(spec, weight_bits)
    targets_of = {
        nid: [t % spec.n_neurons for t in rule.sample_targets(nid)]
        for nid in range(spec.n_neurons)
    }
    exact_rule = _rule(spec, None)
    stored = {nid: exact_rule.assign_weights(nid) for nid in range(spec.n_neurons)}
    procedural = {nid: rule.assign_weights(nid) for nid in range(spec.n_neurons)}
    counts_e = _run(stored, targets_of, spec)
    counts_p = _run(procedural, targets_of, spec)
    se = sum(counts_e.values())
    sp = sum(counts_p.values())
    xs = [counts_e[n] for n in range(spec.n_neurons)]
    ys = [counts_p[n] for n in range(spec.n_neurons)]
    return EquivalenceResult(
        spikes_explicit=se,
        spikes_procedural=sp,
        rate_ratio=(sp / se) if se else (1.0 if sp == 0 else float("inf")),
        count_correlation=_pearson([float(x) for x in xs], [float(y) for y in ys]),
        max_count_diff=max(abs(a - b) for a, b in zip(xs, ys)),
        weight_bits=weight_bits,
    )
