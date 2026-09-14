"""GeNN adapter, phase 1: spec translation (plan PH3-WI03, WSL2-only).

Environment verdict on this machine (2026-09-14, checked at runtime):
Ubuntu-20.04 has g++ 9.4 but only Python 3.8 (no PyGeNN wheels exist for
3.8); the newer Ubuntu has Python 3.10 but no C++ compiler and no
passwordless sudo. No live GeNN model can build here today. Unblock:
``sudo apt install g++ python3-pip`` in the newer Ubuntu, then
``pip install pygenn`` — at which point phase 2 (compile + run + exactness
test vs oracle) activates.

What ships now (complete and tested): the runtime ``available()`` probe
plus ``build_spec()``, which translates a ``StaticNetSpec`` into the plain
parameter dict a live GeNN backend consumes (neuron model, synapse
parameters, schedule). Pure function, fully unit-tested, zero GeNN
dependency — the compile/run layer plugs into exactly this contract later.
"""

from __future__ import annotations

from typing import Any

from vnr.backend.reference import StaticNetSpec

__all__ = ["GennNotAvailable", "available", "build_spec"]


class GennNotAvailable(RuntimeError):
    """Raised when live GeNN work is requested but PyGeNN cannot import."""


def available() -> bool:
    """True iff PyGeNN imports (WSL2 Ubuntu with pygenn installed)."""
    try:
        import pygenn  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False
    return True


def require_available() -> None:
    """Fail loud with the unblock recipe when PyGeNN is missing."""
    if not available():
        raise GennNotAvailable(
            "PyGeNN is not installed here (WSL2 newer-Ubuntu needs g++ via "
            "passworded sudo, then pip install pygenn — see module docstring)"
        )


def build_spec(spec: StaticNetSpec) -> dict[str, Any]:
    """Translate a static net into GeNN backend parameters.

    Neuron model is LIF with the oracle's exact constants (20 ms tau,
    0.1 ms tick, 1.0 mV threshold, 2.0 ms refractory); synapses are static
    pulse couplings with the spec weight and integer-tick delay; the drive
    schedule reuses the reference ``build_drive`` block discipline.
    """
    return {
        "backend": "genn",
        "device": "cpu",  # cuda only after a measured B006 justification
        "dt_ms": 0.1,
        "neuron_model": "LIF",
        "neuron_params": {
            "tau_ms": 20.0,
            "v_threshold": 1.0,
            "v_reset": 0.0,
            "refractory_ms": 2.0,
        },
        "synapse_model": "static_pulse",
        "weight": spec.weight,
        "delay_ticks": spec.delay_ticks,
        "network": {
            "n_neurons": spec.n_neurons,
            "seed": spec.seed,
            "out_degree": spec.out_degree,
        },
        "drive": {
            "density": spec.drive_density,
            "amplitude": spec.drive_amplitude,
            "block_ticks": 100,
        },
        "ticks": spec.ticks,
    }
