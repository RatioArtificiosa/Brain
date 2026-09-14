"""PyTorch backend (plan PH3-WI02, benchmarks B002/B004–B008).

Status on this machine (2026-09-14, recorded, not hidden): CPU-only torch
(2.14.0+cpu) — the CUDA line this driver (537.99) can load (cu121) is retired
from publication, and newer CUDA lines need a driver update. So this module
ships the COMPUTE path proven bit-exact on CPU (float64, same op order as the
oracle) plus the keyed-RNG design unit-tested; GPU execution (B004–B008,
VRAM manager hot paths) is implemented, guarded, and explicitly UNTESTED
until a CUDA-capable torch lands. Unblock steps: update NVIDIA driver past
R560 → ``pip install torch`` (CUDA build) → rerun B004–B008.

Keyed RNG (D2.3): ``torch.Generator`` seeded from the project's blake2b-8
lineage hash — stateless per (seed, source, population, version) key,
reproducible across processes. Philox is torch's CPU/CUDA generator
algorithm; the key layout is what makes it deterministic, and that layout
is what's pinned here.
"""

from __future__ import annotations

import torch

from vnr.backend.reference import RunResult, StaticNetSpec, build_drive
from vnr.core.events import EventQueue, NeuralEvent
from vnr.core.ids import virtual_id
from vnr.core.neuron import LIFParams
from vnr.core.procedural import ConnectivityParams, ProceduralConnectivity

__all__ = ["TorchLIF", "keyed_generator", "run_torch", "vram_status"]


def keyed_generator(
    global_seed: int, source_id: int, population_id: int, version: int
) -> torch.Generator:
    """Deterministic torch.Generator for one D2.3 key (any device)."""
    gen = torch.Generator()
    gen.manual_seed(virtual_id(global_seed, source_id, population_id, version, 0))
    return gen


def vram_status() -> dict[str, int | bool] | None:
    """Live VRAM numbers, or None when no CUDA device exists.

    The None branch is unit-tested (this machine). The CUDA branch is
    implemented but untestable here — it activates with a CUDA torch build.
    """
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info(0)
    return {
        "cuda": True,
        "total_bytes": total,
        "free_bytes": free,
        "allocated_bytes": torch.cuda.memory_allocated(0),
        "reserved_bytes": torch.cuda.memory_reserved(0),
    }


class TorchLIF:
    """Tensor LIF population with oracle-identical element semantics (float64)."""

    def __init__(
        self, n: int, params: LIFParams | None = None, device: str = "cpu"
    ) -> None:
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError("n must be a positive int")
        self.params = params or LIFParams()
        self.device = torch.device(device)
        self.v = torch.full(
            (n,), self.params.v_init, dtype=torch.float64, device=self.device
        )
        self.refractory_until = torch.zeros(n, dtype=torch.int64, device=self.device)

    def step(self, tick: int, currents: torch.Tensor) -> torch.Tensor:
        """Advance one tick. Returns boolean spike mask (same math as oracle)."""
        p = self.params
        active = tick >= self.refractory_until
        updated = currents + (self.v - currents) * p.decay
        self.v = torch.where(active, updated, torch.full_like(self.v, p.v_reset))
        fired = active & (self.v >= p.v_threshold)
        self.v = torch.where(fired, torch.full_like(self.v, p.v_reset), self.v)
        self.refractory_until = torch.where(
            fired,
            torch.full_like(self.refractory_until, tick + 1 + p.refractory_ticks),
            self.refractory_until,
        )
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


def run_torch(
    spec: StaticNetSpec, drive: list[list[int]] | None = None, device: str = "cpu"
) -> RunResult:
    """Tensor run. Contract: bit-identical to ``run_explicit`` on any device
    where float64 is IEEE-exact (CPU verified; CUDA pending a CUDA build)."""
    rule = _rule(spec)
    adjacency = [
        [tgt % spec.n_neurons for tgt in rule.sample_targets(nid)]
        for nid in range(spec.n_neurons)
    ]
    pop = TorchLIF(spec.n_neurons, device=device)
    queue: EventQueue = EventQueue()
    spikes: dict[int, list[int]] = {nid: [] for nid in range(spec.n_neurons)}
    drive = build_drive(spec) if drive is None else drive
    delivered = 0
    currents = torch.zeros(spec.n_neurons, dtype=torch.float64, device=pop.device)
    for t in range(spec.ticks):
        currents.zero_()
        for event in queue.drain_tick(t):
            currents[event.target_id] += event.weight
            delivered += 1
        for nid in drive[t]:
            currents[nid] += spec.drive_amplitude
        for nid in (
            torch.nonzero(pop.step(t, currents), as_tuple=False).flatten().tolist()
        ):
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
    return RunResult(
        spikes=spikes,
        final_v={nid: float(pop.v[nid].item()) for nid in range(spec.n_neurons)},
        final_refractory={
            nid: int(pop.refractory_until[nid].item()) for nid in range(spec.n_neurons)
        },
        events_delivered=delivered,
    )
