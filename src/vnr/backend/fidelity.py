"""Structural fidelity scoring (plan PH2-WI04, spec §72).

Compares two spike-train sets (reference vs candidate) across COMPLEMENTARY
per-metric distances — never collapsed into one scalar prematurely:

- ``rate_ratio``: total-spike ratio (fires if overall activity differs);
- ``count_r``: Pearson r of per-neuron spike counts (fires on rate redistribution);
- ``binned_cosine``: cosine of time-binned population vectors (fires on
  timing shifts that preserve counts — the case the other two miss);
- ``max_count_diff``: worst single-neuron count deviation (fires on outliers).

``overall_pass`` is a gate (AND of per-metric thresholds), not a score: it
answers "ship / don't ship", never "how good". Thresholds travel with the
result so a pass is always auditable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["FidelityThresholds", "StructuralFidelityScore", "compare_spike_trains"]


@dataclass(frozen=True)
class FidelityThresholds:
    """Per-metric pass bands. Defaults suit reference-backend gates (PH2)."""

    rate_ratio_lo: float = 0.95
    rate_ratio_hi: float = 1.05
    count_r_min: float = 0.99
    binned_cosine_min: float = 0.99
    max_count_diff_max: int = 0


@dataclass(frozen=True)
class StructuralFidelityScore:
    """Per-metric distances + gate outcome. No scalar collapse, ever."""

    rate_ratio: float
    count_correlation: float
    binned_cosine: float
    max_count_diff: int
    thresholds: FidelityThresholds = field(default_factory=FidelityThresholds)

    @property
    def passed(self) -> dict[str, bool]:
        t = self.thresholds
        return {
            "rate_ratio": t.rate_ratio_lo <= self.rate_ratio <= t.rate_ratio_hi,
            "count_r": self.count_correlation >= t.count_r_min,
            "binned_cosine": self.binned_cosine >= t.binned_cosine_min,
            "max_count_diff": self.max_count_diff <= t.max_count_diff_max,
        }

    @property
    def overall_pass(self) -> bool:
        return all(self.passed.values())


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 1.0 if vx == vy else 0.0
    return cov / math.sqrt(vx * vy)


def _cosine(xs: list[float], ys: list[float]) -> float:
    dot = sum(x * y for x, y in zip(xs, ys))
    nx = math.sqrt(sum(x * x for x in xs))
    ny = math.sqrt(sum(y * y for y in ys))
    if nx == 0 or ny == 0:
        return 1.0 if nx == ny else 0.0
    return dot / (nx * ny)


def compare_spike_trains(
    reference: dict[int, list[int]],
    candidate: dict[int, list[int]],
    ticks: int,
    bin_width: int = 10,
    thresholds: FidelityThresholds | None = None,
) -> StructuralFidelityScore:
    """Score candidate trains against reference trains.

    Both maps cover the same neuron IDs; ticks are ints (D2.4). Pure function:
    same inputs always give the same score (determinism is structural here).
    """
    if not isinstance(ticks, int) or isinstance(ticks, bool) or ticks <= 0:
        raise ValueError("ticks must be a positive int")
    if not isinstance(bin_width, int) or isinstance(bin_width, bool) or bin_width <= 0:
        raise ValueError("bin_width must be a positive int")
    if set(reference) != set(candidate):
        raise ValueError("reference and candidate must cover the same neuron IDs")
    n_bins = (ticks + bin_width - 1) // bin_width
    ref_counts: list[float] = []
    cand_counts: list[float] = []
    ref_binned: list[float] = []
    cand_binned: list[float] = []
    max_diff = 0
    for nid in sorted(reference):
        r = [t for t in reference[nid] if 0 <= t < ticks]
        c = [t for t in candidate[nid] if 0 <= t < ticks]
        ref_counts.append(float(len(r)))
        cand_counts.append(float(len(c)))
        max_diff = max(max_diff, abs(len(r) - len(c)))
        rb = [0.0] * n_bins
        cb = [0.0] * n_bins
        for t in r:
            rb[t // bin_width] += 1.0
        for t in c:
            cb[t // bin_width] += 1.0
        ref_binned.extend(rb)
        cand_binned.extend(cb)
    total_ref = sum(ref_counts)
    total_cand = sum(cand_counts)
    ratio = (
        (total_cand / total_ref)
        if total_ref
        else (1.0 if total_cand == 0 else float("inf"))
    )
    return StructuralFidelityScore(
        rate_ratio=ratio,
        count_correlation=_pearson(ref_counts, cand_counts),
        binned_cosine=_cosine(ref_binned, cand_binned),
        max_count_diff=max_diff,
        thresholds=thresholds or FidelityThresholds(),
    )
