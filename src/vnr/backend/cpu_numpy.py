"""NumPy vectorized CPU backend (plan PH3-WI01, benchmarks B001/B003).

Same mathematics as the oracle, reorganized for arrays: one float64 voltage
vector + one int refractory vector replace N ``LIFNeuron`` objects, and each
tick steps the whole population in a handful of vector ops. Per-neuron float
operations are identical to ``LIFNeuron.step`` in identical order, so spike
trains, final voltages, and refractory states match ``run_explicit``
bit-exactly — the gate this module must pass forever. Event routing stays in
``EventQueue`` (Python) at this stage; kernels take over routing at PH3-WI02.
"""

from __future__ import annotations

import numpy as np

from vnr.backend.reference import RunResult, StaticNetSpec, build_drive
from vnr.core.events import EventQueue, NeuralEvent
from vnr.core.neuron import LIFParams
from vnr.core.procedural import ConnectivityParams, ProceduralConnectivity

__all__ = ["NumpyLIF", "run_numpy"]


class NumpyLIF:
    """Vectorized LIF population with oracle-identical per-element semantics."""

    def __init__(self, n: int, params: LIFParams | None = None) -> None:
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError("n must be a positive int")
        self.params = params or LIFParams()
        self.v = np.full(n, self.params.v_init, dtype=np.float64)
        self.refractory_until = np.zeros(n, dtype=np.int64)

    def step(self, tick: int, currents: np.ndarray) -> np.ndarray:
        """Advance one tick. Returns boolean spike mask.

        Mirrors ``LIFNeuron.step`` element-wise: refractory ticks hold reset
        and ignore input; otherwise integrate exactly, threshold, reset, arm.
        """
        p = self.params
        active = tick >= self.refractory_until
        v = self.v
        v[active] = currents[active] + (v[active] - currents[active]) * p.decay
        v[~active] = p.v_reset
        fired = active & (v >= p.v_threshold)
        v[fired] = p.v_reset
        self.refractory_until[fired] = tick + 1 + p.refractory_ticks
        return fired


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


def run_numpy(spec: StaticNetSpec, drive: list[list[int]] | None = None) -> RunResult:
    """All-resident vectorized run. Contract: bit-identical to ``run_explicit``."""
    rule = _rule(spec)
    adjacency = [
        np.fromiter(
            (tgt % spec.n_neurons for tgt in rule.sample_targets(nid)), dtype=np.int64
        )
        for nid in range(spec.n_neurons)
    ]
    pop = NumpyLIF(spec.n_neurons)
    queue: EventQueue = EventQueue()
    spikes: dict[int, list[int]] = {nid: [] for nid in range(spec.n_neurons)}
    drive = build_drive(spec) if drive is None else drive
    delivered = 0
    currents = np.zeros(spec.n_neurons, dtype=np.float64)
    for t in range(spec.ticks):
        currents.fill(0.0)
        for event in queue.drain_tick(t):
            currents[event.target_id] += event.weight
            delivered += 1
        for nid in drive[t]:
            currents[nid] += spec.drive_amplitude
        fired = pop.step(t, currents)
        for nid in np.flatnonzero(fired):
            nid = int(nid)
            spikes[nid].append(t)
            for target in adjacency[nid]:
                queue.push(
                    NeuralEvent(
                        tick=t + spec.delay_ticks,
                        source_id=nid,
                        target_id=int(target),
                        weight=spec.weight,
                    )
                )
    return RunResult(
        spikes=spikes,
        final_v={nid: float(pop.v[nid]) for nid in range(spec.n_neurons)},
        final_refractory={
            nid: int(pop.refractory_until[nid]) for nid in range(spec.n_neurons)
        },
        events_delivered=delivered,
    )
