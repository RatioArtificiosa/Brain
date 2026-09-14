"""PH2-WI02 tests: 1M-virtual run under a 50K resident budget (spec §70)."""

import pytest

from vnr.backend.scaling import ScalingResult, ScalingSpec, run_scaled


def test_1M_virtual_50K_budget_completes_bounded():
    spec = ScalingSpec()
    result = run_scaled(spec)
    assert isinstance(result, ScalingResult)
    assert result.ticks == spec.ticks
    assert result.peak_resident <= spec.resident_budget, (
        f"budget violated: peak {result.peak_resident} > {spec.resident_budget}"
    )
    assert result.peak_resident < spec.virtual_n // 10, (
        "must stay far below virtual size"
    )
    assert result.total_spikes > 0, (
        "a silent run proves nothing — drive must ignite activity"
    )
    assert result.events_delivered > 0
    assert result.materializations > 0 and result.evictions > 0, (
        "no evictions means the budget was never exercised"
    )
    print(
        f"\n1M-virtual/50K-budget: spikes={result.total_spikes:,} "
        f"delivered={result.events_delivered:,} peak={result.peak_resident:,} "
        f"materializations={result.materializations:,} evictions={result.evictions:,}"
    )


def test_scaled_run_is_deterministic():
    spec = ScalingSpec(virtual_n=20_000, resident_budget=2_000, ticks=60, seed=11)
    first = run_scaled(spec)
    second = run_scaled(spec)
    assert first == second


def test_budget_enforcement_under_heavy_drive():
    spec = ScalingSpec(
        virtual_n=1_000, resident_budget=50, ticks=30, hot_neurons=300, seed=3
    )
    result = run_scaled(spec)
    assert result.peak_resident <= 50


def test_invalid_specs_rejected():
    with pytest.raises(ValueError, match="resident_budget must be < virtual_n"):
        ScalingSpec(virtual_n=100, resident_budget=100)
    with pytest.raises(TypeError):
        ScalingSpec(ticks=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="drive_amplitude must be positive"):
        ScalingSpec(drive_amplitude=0.0)
