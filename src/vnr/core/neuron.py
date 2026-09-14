"""Leaky integrate-and-fire neuron (plan PH1-WI02, spec §33).

Exact discrete dynamics of ``dV/dt = (-V + I_syn) / tau`` on an integer-tick
schedule (plan D2.4, ``DT = 0.1 ms`` default). Per-step closed form for
constant input over one tick::

    V' = I_syn + (V - I_syn) * decay,  decay = exp(-dt / tau)

Spike when ``V >= threshold``: record the tick, reset ``V = v_reset``, hold
there until ``refractory_until_tick``. Ticks are ints; no float scheduling.
This module is the reference oracle: the PH3 NumPy/CUDA backends must
reproduce its spike trains exactly for static networks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["LIFNeuron", "LIFParams", "LIFState", "closed_form_voltage"]


@dataclass(frozen=True)
class LIFParams:
    """Deterministic LIF parameters. All voltages in mV, times in ms."""

    tau_ms: float = 20.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    v_init: float = 0.0
    dt_ms: float = 0.1
    refractory_ms: float = 2.0

    def __post_init__(self) -> None:
        if self.tau_ms <= 0:
            raise ValueError("tau_ms must be positive")
        if self.dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        if self.refractory_ms < 0:
            raise ValueError("refractory_ms must be non-negative")
        if self.v_threshold <= self.v_reset:
            raise ValueError("v_threshold must exceed v_reset")

    @property
    def decay(self) -> float:
        """Per-tick leak factor exp(-dt/tau), precomputed per step."""
        return math.exp(-self.dt_ms / self.tau_ms)

    @property
    def refractory_ticks(self) -> int:
        """Refractory period in integer ticks (D2.4: no float scheduling)."""
        return round(self.refractory_ms / self.dt_ms)


@dataclass
class LIFState:
    """Transient per-neuron state (evictable layer, plan §26)."""

    v: float = 0.0
    refractory_until_tick: int = 0


def closed_form_voltage(v0: float, i_syn: float, steps: int, decay: float) -> float:
    """Exact voltage after ``steps`` ticks of constant input (test oracle)."""
    return i_syn + (v0 - i_syn) * decay**steps


@dataclass
class LIFNeuron:
    """Single LIF unit stepped on integer ticks."""

    params: LIFParams = field(default_factory=LIFParams)
    state: LIFState = field(default_factory=LIFState)

    def __post_init__(self) -> None:
        self.state.v = self.params.v_init

    def reset(self) -> None:
        """Return to initial voltage with no refractory debt."""
        self.state.v = self.params.v_init
        self.state.refractory_until_tick = 0

    def step(self, tick: int, i_syn: float = 0.0) -> bool:
        """Advance one tick. Returns True on spike.

        Refractory ticks hold ``V = v_reset`` and ignore input; otherwise
        integrate exactly, then threshold, reset, and arm refractoriness.
        """
        p = self.params
        s = self.state
        if tick < s.refractory_until_tick:
            s.v = p.v_reset
            return False
        s.v = i_syn + (s.v - i_syn) * p.decay
        if s.v >= p.v_threshold:
            s.v = p.v_reset
            s.refractory_until_tick = tick + 1 + p.refractory_ticks
            return True
        return False

    def run(self, start_tick: int, currents: list[float]) -> list[int]:
        """Step over ``currents``; return the spike tick list."""
        spikes: list[int] = []
        for offset, i_syn in enumerate(currents):
            if self.step(start_tick + offset, i_syn):
                spikes.append(start_tick + offset)
        return spikes
