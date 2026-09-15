"""Measure which FourWaySpec parameters actually demonstrate virtualization.

This is the EVIDENCE behind the defaults in ``vnr/experiments/runner.py``.
The original defaults (degree 8, drive density 0.4, amplitude 3.0) left ~93%
of neurons resident: dense sustained drive touches the whole network and
uniform fan-out has no locality.

Run this instead of guessing if you ever revisit those defaults.

    python scripts/tune_four_way.py
"""

from vnr.experiments.runner import FourWaySpec, run_four_way

# (label, overrides) - each ran on 256 neurons / 800 ticks.
CANDS = [
    (
        "original default",
        {"drive_density": 0.4, "drive_amplitude": 3.0, "out_degree": 8},
    ),
    (
        "sparse drive 2%",
        {"drive_density": 0.02, "drive_amplitude": 12.0, "out_degree": 8},
    ),
    (
        "sparse drive 5%",
        {"drive_density": 0.05, "drive_amplitude": 12.0, "out_degree": 8},
    ),
    (
        "low fan-out 4",
        {"drive_density": 0.02, "drive_amplitude": 12.0, "out_degree": 4},
    ),
    (
        "low fan-out 2",
        {"drive_density": 0.02, "drive_amplitude": 12.0, "out_degree": 2},
    ),
]


def main() -> None:
    header = f"{'variant':<20}{'spikes':>8}{'peak':>7}{'resident%':>11}{'evict':>8}{'cos':>11}{'gate':>7}"
    print(header)
    print("-" * len(header))
    for label, overrides in CANDS:
        spec = FourWaySpec(
            n_neurons=256,
            ticks=800,
            quiet_ticks=10,
            evict_ticks=40,
            **overrides,
        )
        results = run_four_way(spec)
        reference, virtualized = results[0], results[-1]
        cosine = virtualized.fidelity["binned_cosine"] if virtualized.fidelity else 1.0
        all_pass = all(
            r.fidelity is None or r.fidelity["overall_pass"] for r in results
        )
        pct = virtualized.peak_resident / spec.n_neurons * 100
        print(
            f"{label:<20}{reference.total_spikes:>8}{virtualized.peak_resident:>7}"
            f"{pct:>10.1f}%{virtualized.evictions:>8}{cosine:>11.6f}"
            f"{'PASS' if all_pass else 'FAIL':>7}"
        )
    print()
    print("Finding: residency is governed by FAN-OUT, not drive density.")
    print("Uniform fan-out scatters every spike across the network, so the")
    print("touched set grows toward the whole population regardless of how")
    print("few neurons are driven. A real connectome is locally structured,")
    print("so this ceiling does not transfer to the fly.")


if __name__ == "__main__":
    main()
