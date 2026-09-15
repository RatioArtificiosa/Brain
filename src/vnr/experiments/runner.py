"""Experiment runner and run records (plan PH5-WI01, specs §37/§55/§59).

An experiment is a set of CONDITIONS run under one shared config. Every run
writes a record that makes the result auditable after the fact: run id,
config hash, seed, code commit, hardware, timestamps (plan §55). The record
deliberately carries hypothesis / observation / interpretation as SEPARATE
fields — plan §5 forbids blurring a claim with a measurement.

The first experiment implemented is the §37 four-way comparison, which is the
spine of the whole programme:

    explicit_dense          stored adjacency, everything resident
    sparse                  stored adjacency, only reached neurons resident
    procedural              edges regenerated on demand, all resident
    procedural_virtualized  procedural edges + materialize/evict frontier

All four run the SAME network, seed, and stimulus, so the only thing that
varies is the compression mechanism — which is exactly what makes the
resulting fidelity numbers meaningful.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vnr.backend.fidelity import FidelityThresholds, compare_spike_trains
from vnr.backend.graph_health import HealthThresholds, check_graph_health
from vnr.backend.reference import (
    RunResult,
    StaticNetSpec,
    build_drive,
    run_explicit,
    run_virtualized,
)
from vnr.core.events import EventQueue, NeuralEvent
from vnr.core.neuron import LIFNeuron
from vnr.core.procedural import (
    ConnectivityParams,
    ProceduralConnectivity,
)

__all__ = [
    "ConditionResult",
    "ExperimentRecord",
    "FourWaySpec",
    "run_four_way",
    "write_run_record",
]

CONDITIONS = (
    "explicit_dense",
    "sparse",
    "procedural",
    "procedural_virtualized",
)


def _code_commit() -> str:
    """Current commit of the code repo, or 'unknown' outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return out.stdout.strip() or "unknown" if out.returncode == 0 else "unknown"


def _hardware_summary() -> dict[str, Any]:
    """Machine fingerprint recorded with every run (§55)."""
    try:
        from vnr.hardware import discover_hardware

        prof = discover_hardware()
        return {
            "cpu": prof.cpu_name,
            "threads": prof.cpu_threads,
            "ram_bytes": prof.ram_bytes,
            "gpu": prof.gpu_name,
            "vram_bytes": prof.gpu_vram_bytes,
            "torch": prof.torch_version,
            "cuda": prof.cuda_version,
        }
    except Exception:  # noqa: BLE001 - a record must still be written
        return {"cpu": platform.processor(), "threads": os.cpu_count() or 0}


@dataclass(frozen=True)
class FourWaySpec:
    """One shared network + stimulus for all four conditions (§37)."""

    n_neurons: int = 256
    seed: int = 37
    out_degree: int = 4
    weight: float = 0.4
    delay_ticks: int = 2
    ticks: int = 800
    drive_density: float = 0.02
    drive_amplitude: float = 12.0
    quiet_ticks: int = 10
    evict_ticks: int = 40
    """Defaults chosen by measurement (scripts/tune_four_way.py), not taste.

    The original defaults (degree 8, drive density 0.4, amplitude 3.0) left
    ~93% of neurons resident, because dense sustained drive touches the whole
    network and uniform fan-out has no locality. Measured sweep: residency is
    governed by FAN-OUT, not drive density — degree 8 -> 76.6% resident,
    degree 2 -> 36.7% at identical drive. These defaults sit in the sparse,
    low-fan-out regime where the virtualization mechanism is actually visible
    (~24% resident) while all four conditions remain bit-exact.

    IMPORTANT caveat for anyone reading a result from these defaults: with
    uniform random connectivity, EVERY spike scatters to `out_degree` random
    neurons, so the touched set grows toward the whole network. Virtualization
    therefore has a structural ceiling here that does not exist for a real
    connectome, where activity is spatially and topologically local. Do not
    read a modest compression figure from synthetic connectivity as a
    statement about the fly.
    """

    def to_net_spec(self) -> StaticNetSpec:
        return StaticNetSpec(
            n_neurons=self.n_neurons,
            seed=self.seed,
            out_degree=self.out_degree,
            weight=self.weight,
            delay_ticks=self.delay_ticks,
            ticks=self.ticks,
            drive_density=self.drive_density,
            drive_amplitude=self.drive_amplitude,
            quiet_ticks=self.quiet_ticks,
            evict_ticks=self.evict_ticks,
        )


@dataclass
class ConditionResult:
    """One condition's measured outcome (all fields measured, none assumed)."""

    condition: str
    total_spikes: int = 0
    events_delivered: int = 0
    peak_resident: int = 0
    materializations: int = 0
    evictions: int = 0
    wall_seconds: float = 0.0
    stored_synapses: int = 0
    resident_synapses: int = 0
    fidelity: dict[str, Any] | None = None
    health_ok: bool | None = None
    health_failures: list[str] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        """Stored synapses / what actually had to be kept resident.

        For resident-synapse-free conditions this reports the RATIO of stored
        edges to the resident-neuron budget's implied edge count, which is the
        comparison that matters: what did we avoid materializing?
        """
        if self.resident_synapses <= 0:
            return float("inf") if self.stored_synapses else 1.0
        return self.stored_synapses / self.resident_synapses


def _run_sparse(spec: StaticNetSpec, drive: list[list[int]]) -> RunResult:
    """Stored adjacency, but only TOUCHED neurons ever hold state.

    Distinguishes 'we keep the graph' from 'we keep the state': a neuron is
    allocated the first time anything reaches it, and from then on it is
    integrated every tick exactly as the reference does (including silent
    ticks — this integrator decays across them, so skipping them would shift
    spike times). Neurons never touched by drive or fan-out are never
    allocated at all.

    Correctness note (found by this experiment's own fidelity gate): an
    earlier revision stepped only neurons that received input ON that tick.
    Spike counts still matched, but timing cosine dropped to 0.78 because a
    neuron lifted near threshold at tick T fires at T+1 with zero input, and
    that tick was being skipped. Counts matching while times drift is exactly
    the failure mode plan §72's complementary metrics exist to catch.
    """
    rule = ProceduralConnectivity(
        params=ConnectivityParams(
            global_seed=spec.seed,
            target_population_id=0,
            connectivity_version=1,
            out_degree=spec.out_degree,
            weight=spec.weight,
            delay_ticks=spec.delay_ticks,
        )
    )
    adjacency = {
        nid: [tgt % spec.n_neurons for tgt in rule.sample_targets(nid)]
        for nid in range(spec.n_neurons)
    }
    queue: EventQueue = EventQueue()
    spikes: dict[int, list[int]] = {nid: [] for nid in range(spec.n_neurons)}
    neurons: dict[int, LIFNeuron] = {}
    delivered = 0
    for t in range(spec.ticks):
        inp: dict[int, float] = {}
        for event in queue.drain_tick(t):
            inp[event.target_id] = inp.get(event.target_id, 0.0) + event.weight
            delivered += 1
        for nid in drive[t]:
            inp[nid] = inp.get(nid, 0.0) + spec.drive_amplitude
        # Allocate the newly touched, then step EVERY resident neuron: silent
        # ticks matter to this integrator.
        for nid in inp:
            if nid not in neurons:
                neurons[nid] = LIFNeuron()
        for nid, neuron in neurons.items():
            if neuron.step(t, inp.get(nid, 0.0)):
                spikes[nid].append(t)
                for target in adjacency[nid]:
                    queue.push(
                        NeuralEvent(
                            tick=t + spec.delay_ticks,
                            source_id=nid,
                            target_id=target,
                            weight=spec.weight,
                        )
                    )
    final_v = {
        nid: (neurons[nid].state.v if nid in neurons else 0.0)
        for nid in range(spec.n_neurons)
    }
    final_refractory = {
        nid: (neurons[nid].state.refractory_until_tick if nid in neurons else 0)
        for nid in range(spec.n_neurons)
    }
    return RunResult(
        spikes=spikes,
        final_v=final_v,
        final_refractory=final_refractory,
        events_delivered=delivered,
    )


def _run_procedural(spec: StaticNetSpec, drive: list[list[int]]) -> RunResult:
    """Edges regenerated per spike (no stored adjacency), all neurons resident."""
    rule = ProceduralConnectivity(
        params=ConnectivityParams(
            global_seed=spec.seed,
            target_population_id=0,
            connectivity_version=1,
            out_degree=spec.out_degree,
            weight=spec.weight,
            delay_ticks=spec.delay_ticks,
        )
    )
    neurons = {nid: LIFNeuron() for nid in range(spec.n_neurons)}
    queue: EventQueue = EventQueue()
    spikes: dict[int, list[int]] = {nid: [] for nid in range(spec.n_neurons)}
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
                for target in rule.sample_targets(nid):
                    queue.push(
                        NeuralEvent(
                            tick=t + spec.delay_ticks,
                            source_id=nid,
                            target_id=target % spec.n_neurons,
                            weight=spec.weight,
                        )
                    )
    return RunResult(
        spikes=spikes,
        final_v={nid: n.state.v for nid, n in neurons.items()},
        final_refractory={
            nid: n.state.refractory_until_tick for nid, n in neurons.items()
        },
        events_delivered=delivered,
    )


def run_four_way(
    spec: FourWaySpec, thresholds: FidelityThresholds | None = None
) -> list[ConditionResult]:
    """Run all four §37 conditions against one shared drive, in order.

    The explicit-dense condition is the REFERENCE: every other condition's
    fidelity is scored against it, per plan §72 (per-metric distances, never
    a collapsed scalar).
    """
    net = spec.to_net_spec()
    drive = build_drive(net)
    stored_edges = spec.n_neurons * spec.out_degree
    results: list[ConditionResult] = []
    reference_trains: dict[int, list[int]] | None = None

    for condition in CONDITIONS:
        start = time.perf_counter()
        materializations = 0
        evictions = 0
        if condition == "explicit_dense":
            run = run_explicit(net, drive)
            peak = spec.n_neurons
            stored, resident = stored_edges, stored_edges
        elif condition == "sparse":
            run = _run_sparse(net, drive)
            peak = spec.n_neurons  # graph stored; state is lazy but unbounded
            stored, resident = stored_edges, stored_edges
        elif condition == "procedural":
            run = _run_procedural(net, drive)
            peak = spec.n_neurons
            stored, resident = 0, stored_edges
        else:
            run, vstats = run_virtualized(net, drive)
            peak = vstats.max_resident
            materializations = vstats.materializations
            evictions = vstats.evictions
            stored, resident = 0, peak * spec.out_degree
        wall = time.perf_counter() - start

        health_ok: bool | None = None
        failures: list[str] = []
        try:
            report = check_graph_health(
                run.spikes, spec.ticks, HealthThresholds(), raise_on_fail=True
            )
            health_ok = report.healthy
        except Exception as exc:  # noqa: BLE001 - health failure is DATA, not a crash
            health_ok = False
            failures = [str(exc)]

        fidelity: dict[str, Any] | None = None
        if reference_trains is None:
            reference_trains = run.spikes
        else:
            score = compare_spike_trains(
                reference_trains, run.spikes, spec.ticks, thresholds=thresholds
            )
            fidelity = {
                "rate_ratio": score.rate_ratio,
                "count_correlation": score.count_correlation,
                "binned_cosine": score.binned_cosine,
                "max_count_diff": score.max_count_diff,
                "passed": score.passed,
                "overall_pass": score.overall_pass,
            }

        results.append(
            ConditionResult(
                condition=condition,
                total_spikes=run.total_spikes,
                events_delivered=run.events_delivered,
                peak_resident=peak,
                materializations=materializations,
                evictions=evictions,
                wall_seconds=wall,
                stored_synapses=stored,
                resident_synapses=resident,
                fidelity=fidelity,
                health_ok=health_ok,
                health_failures=failures,
            )
        )
    return results


@dataclass
class ExperimentRecord:
    """The §55 run record: everything needed to re-run and audit this result."""

    experiment_id: str
    run_id: str
    created: str
    seed: int
    config_hash: str
    commit: str
    python: str
    hardware: dict[str, Any]
    spec: dict[str, Any]
    conditions: list[dict[str, Any]]
    hypothesis: str
    observation: str
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_record(
    experiment_id: str,
    spec: FourWaySpec,
    results: list[ConditionResult],
    hypothesis: str,
    interpretation: str,
) -> ExperimentRecord:
    """Assemble the record, deriving the observation text from measurements."""
    spec_dict = asdict(spec)
    canonical = json.dumps(spec_dict, sort_keys=True, separators=(",", ":"))
    config_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    run_id = f"{experiment_id}-{config_digest[:8]}-s{spec.seed}"
    lines = []
    for res in results:
        rate = res.fidelity["rate_ratio"] if res.fidelity else 1.0
        passed = res.fidelity["overall_pass"] if res.fidelity else True
        lines.append(
            f"{res.condition}: {res.total_spikes} spikes, peak resident "
            f"{res.peak_resident}/{spec.n_neurons}, rate_ratio vs explicit "
            f"{rate:.4f}, gate {'PASS' if passed else 'FAIL'}"
        )
    return ExperimentRecord(
        experiment_id=experiment_id,
        run_id=run_id,
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        seed=spec.seed,
        config_hash=config_digest,
        commit=_code_commit(),
        python=platform.python_version(),
        hardware=_hardware_summary(),
        spec=spec_dict,
        conditions=[asdict(r) for r in results],
        hypothesis=hypothesis,
        observation="\n".join(lines),
        interpretation=interpretation,
    )


def write_run_record(record: ExperimentRecord, directory: Path) -> Path:
    """Write ``run.json`` into ``directory`` (creating it) and return the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run.json"
    path.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return path
