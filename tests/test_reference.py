"""PH2-WI01 tests: §69 virtualization-vs-explicit, exact on static nets."""

import time

import pytest

from vnr.backend.reference import (
    StaticNetSpec,
    build_drive,
    run_explicit,
    run_virtualized,
)


def test_section69_virtualization_matches_explicit_exact():
    spec = StaticNetSpec()
    drive = build_drive(spec)
    start = time.perf_counter()
    ref = run_explicit(spec, drive)
    explicit_s = time.perf_counter() - start
    start = time.perf_counter()
    virt, stats = run_virtualized(spec, drive)
    virtual_s = time.perf_counter() - start
    assert virt.spikes == ref.spikes
    assert virt.final_v == ref.final_v
    assert virt.final_refractory == ref.final_refractory
    assert virt.events_delivered == ref.events_delivered
    assert ref.total_spikes > 1_000, "comparison vacuous without spikes"
    assert stats.evictions > 0, "no eviction cycle exercised"
    assert stats.materializations > spec.n_neurons, "no rematerialization seen"
    assert stats.max_resident < spec.n_neurons, "residency never bounded"
    print(
        f"\nN={spec.n_neurons} T={spec.ticks} spikes={ref.total_spikes} "
        f"delivered={ref.events_delivered} max_resident={stats.max_resident} "
        f"evictions={stats.evictions} explicit={explicit_s:.1f}s "
        f"virtual={virtual_s:.1f}s"
    )


def test_virtualized_deterministic_for_same_spec():
    spec = StaticNetSpec(n_neurons=128, ticks=600)
    drive = build_drive(spec)
    first, _ = run_virtualized(spec, drive)
    second, _ = run_virtualized(spec, drive)
    assert first.spikes == second.spikes
    assert first.final_v == second.final_v


def test_comparison_is_sensitive_to_seed():
    assert (
        run_explicit(StaticNetSpec(seed=1)).spikes
        != run_explicit(StaticNetSpec(seed=2)).spikes
    )


def test_invalid_specs_rejected():
    with pytest.raises(ValueError):
        StaticNetSpec(n_neurons=1)
    with pytest.raises(ValueError):
        StaticNetSpec(ticks=2)
    with pytest.raises(ValueError):
        StaticNetSpec(drive_density=0.0)
    with pytest.raises(ValueError):
        StaticNetSpec(drive_density=1.0)
    with pytest.raises(ValueError):
        StaticNetSpec(drive_amplitude=0.0)
    with pytest.raises(ValueError):
        StaticNetSpec(quiet_ticks=120, evict_ticks=120)
    with pytest.raises(ValueError):
        StaticNetSpec(quiet_ticks=200, evict_ticks=100)
    with pytest.raises(TypeError):
        StaticNetSpec(n_neurons=True)  # type: ignore[arg-type]
