"""Graph-health gate (plan PH2-WI05, spec §73).

A run that "completes" can still be garbage: runaway excitation, total
silence, lockstep synchrony, one hub doing everything, or a desert of
isolated neurons. This gate fails LOUD on all five, with per-check values
and thresholds in the report — warnings explain, failures raise.

Checks (all pure functions of spike trains + time base):
- exploding: mean firing rate above ``max_mean_hz`` → FAIL;
- dead: zero spikes at all → FAIL;
- synchrony: share of spikes in the single busiest tick above
  ``sync_share_max`` → FAIL (lockstep is pathology, not assembly);
- hub-dominated: Gini of per-neuron counts above ``hub_gini_max`` → FAIL;
- isolation: fraction of silent neurons above ``isolation_max`` → FAIL.

``GraphHealthError`` carries every fired message: a failed gate never
passes silently and never reports a bare boolean.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "GraphHealthError",
    "HealthCheck",
    "HealthReport",
    "HealthThresholds",
    "check_graph_health",
]


@dataclass(frozen=True)
class HealthThresholds:
    """Per-check bands. Defaults suit small reference nets (PH2)."""

    max_mean_hz: float = 100.0
    min_active_fraction: float = 0.01
    sync_share_max: float = 0.5
    hub_gini_max: float = 0.9
    isolation_max: float = 0.9
    dt_ms: float = 0.1


@dataclass(frozen=True)
class HealthCheck:
    """One check outcome: value measured, band applied, verdict recorded."""

    name: str
    value: float
    threshold: float
    passed: bool
    message: str


@dataclass(frozen=True)
class HealthReport:
    """Full gate outcome. ``failures`` is empty iff the run is healthy."""

    checks: tuple[HealthCheck, ...]
    thresholds: HealthThresholds | None = None

    @property
    def failures(self) -> list[str]:
        return [c.message for c in self.checks if not c.passed]

    @property
    def healthy(self) -> bool:
        return not self.failures


class GraphHealthError(AssertionError):
    """Raised by ``check_graph_health(..., raise_on_fail=True)`` with every message."""


def _gini(counts: list[float]) -> float:
    n = len(counts)
    total = sum(counts)
    if total == 0:
        return 0.0
    ordered = sorted(counts)
    cum = 0.0
    for i, c in enumerate(ordered, start=1):
        cum += i * c
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


def check_graph_health(
    spikes: dict[int, list[int]],
    ticks: int,
    thresholds: HealthThresholds | None = None,
    raise_on_fail: bool = True,
) -> HealthReport:
    """Run all five checks. Fails loud via ``GraphHealthError`` by default."""
    if not isinstance(ticks, int) or isinstance(ticks, bool) or ticks <= 0:
        raise ValueError("ticks must be a positive int")
    if not spikes:
        raise ValueError("spikes must cover at least one neuron")
    t = thresholds or HealthThresholds()
    n = len(spikes)
    counts = {
        nid: len([s for s in trains if 0 <= s < ticks])
        for nid, trains in spikes.items()
    }
    total = sum(counts.values())
    duration_s = ticks * t.dt_ms / 1000.0
    mean_hz = total / n / duration_s if duration_s > 0 else 0.0
    active = sum(1 for c in counts.values() if c > 0)
    active_fraction = active / n
    tick_hist: dict[int, int] = {}
    for trains in spikes.values():
        for s in trains:
            if 0 <= s < ticks:
                tick_hist[s] = tick_hist.get(s, 0) + 1
    sync_share = (max(tick_hist.values()) / total) if total else 0.0
    gini = _gini([float(c) for c in counts.values()])
    isolation = 1.0 - active_fraction
    checks = (
        HealthCheck(
            "exploding",
            mean_hz,
            t.max_mean_hz,
            mean_hz <= t.max_mean_hz,
            f"mean rate {mean_hz:.1f} Hz exceeds {t.max_mean_hz:.1f} Hz",
        ),
        HealthCheck(
            "dead",
            float(total),
            1.0,
            total > 0,
            "network is completely silent (0 spikes)",
        ),
        HealthCheck(
            "synchrony",
            sync_share,
            t.sync_share_max,
            sync_share <= t.sync_share_max,
            f"busiest tick holds {sync_share:.2f} of all spikes (max {t.sync_share_max:.2f})",
        ),
        HealthCheck(
            "hub_dominated",
            gini,
            t.hub_gini_max,
            gini <= t.hub_gini_max,
            f"count Gini {gini:.3f} exceeds {t.hub_gini_max:.3f}",
        ),
        HealthCheck(
            "isolation",
            isolation,
            t.isolation_max,
            isolation <= t.isolation_max,
            f"{isolation:.2f} of neurons silent (max {t.isolation_max:.2f})",
        ),
    )
    if active_fraction < t.min_active_fraction and total > 0:
        checks = checks + (
            HealthCheck(
                "sparse_activity",
                active_fraction,
                t.min_active_fraction,
                True,
                f"only {active_fraction:.3f} of neurons active (below {t.min_active_fraction:.3f}); "
                "passes but suspicious — investigate drive",
            ),
        )
    report = HealthReport(checks=checks, thresholds=t)
    if raise_on_fail and not report.healthy:
        raise GraphHealthError("graph unhealthy:\n- " + "\n- ".join(report.failures))
    return report
