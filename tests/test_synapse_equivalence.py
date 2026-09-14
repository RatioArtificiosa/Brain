"""PH2-WI03 tests: stored vs procedural synapse equivalence (spec §71)."""

from vnr.backend.synapse_equivalence import EquivalenceSpec, compare


def test_exact_weights_reproduce_bit_for_bit():
    spec = EquivalenceSpec()
    result = compare(spec, None)
    assert result.rate_ratio == 1.0
    assert result.count_correlation == 1.0
    assert result.max_count_diff == 0
    assert result.spikes_explicit > 0


def test_8bit_quantization_preserves_dynamics():
    spec = EquivalenceSpec()
    result = compare(spec, 8)
    print(
        f"\n8-bit: explicit={result.spikes_explicit} procedural={result.spikes_procedural} "
        f"ratio={result.rate_ratio:.4f} r={result.count_correlation:.4f} "
        f"maxdiff={result.max_count_diff}"
    )
    assert 0.95 <= result.rate_ratio <= 1.05
    assert result.count_correlation >= 0.99


def test_4bit_quantization_characterized():
    spec = EquivalenceSpec()
    result = compare(spec, 4)
    print(
        f"\n4-bit: explicit={result.spikes_explicit} procedural={result.spikes_procedural} "
        f"ratio={result.rate_ratio:.4f} r={result.count_correlation:.4f} "
        f"maxdiff={result.max_count_diff}"
    )
    assert result.count_correlation >= 0.90


def test_comparison_is_deterministic():
    spec = EquivalenceSpec()
    assert compare(spec, 8) == compare(spec, 8)


def test_coarse_quantization_finds_the_edge():
    # Characterization, not a gate: walk down to 1-bit to find where the
    # dynamics first break. First fidelity-curve data point (§89-90).
    spec = EquivalenceSpec()
    for bits in (2, 1):
        result = compare(spec, bits)
        print(
            f"\n{bits}-bit: explicit={result.spikes_explicit} "
            f"procedural={result.spikes_procedural} ratio={result.rate_ratio:.4f} "
            f"r={result.count_correlation:.4f} maxdiff={result.max_count_diff}"
        )
    assert compare(spec, 2).count_correlation >= 0.90
