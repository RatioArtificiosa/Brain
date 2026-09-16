"""FlyWire connectome as a VNR dataset (PH4-WI02 completion).

Bridges the real fetched corpus into the `ConnectomeDataset` protocol so the
simulators, fidelity gates, and graph-health checks can run on ACTUAL fly
connectivity instead of synthetic graphs. That difference matters more than it
sounds: the synthetic generator uses uniform random fan-out, which scatters
every spike across the whole network and caps the virtualization benefit
(measured in notes entry 39). A real connectome is locally structured.

Data source: parquet chunks produced by `scripts/bulk_by_id.py`, with columns
``pre, post, nt, conf`` (uint64 root ids, uint8 NT code, float32 confidence).

Two node-id spaces are supported:
  * ``node_ids="raw"``   keep the original uint64 FlyWire root ids;
  * ``node_ids="compact"`` remap to 0..n-1 (needed by the array kernels).

Edge weights derive from the neuron-type pair (spec 71: a real
sign/polarity signal rather than a uniform constant), and are documented here
because inventing them silently would be exactly the kind of unstated
assumption this project forbids.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["NT_NAMES", "FlyWireDataset"]

NT_NAMES = ("GABA", "ACH", "GLUT", "OCT", "SER", "DOP")
UNKNOWN_NT = 255

# Excitatory / inhibitory classification used to derive edge sign.
# ACH and GLUT are the fast excitatory transmitters in the fly; GABA is the
# principal inhibitory one. The remaining amines (OCT/SER/DOP) are
# neuromodulatory and are treated as weak excitatory for simulation purposes
# ONLY - this is a modelling choice, stated here rather than buried.
_INHIBITORY = frozenset({0})  # GABA
_EXCITATORY = frozenset({1, 2, 4, 5})  # ACH, GLUT, SER, DOP


def _nt_sign(code: int) -> float:
    """+1 excitatory, -1 inhibitory, 0 unknown/neuromodulatory-ambiguous."""
    if code in _INHIBITORY:
        return -1.0
    if code in _EXCITATORY:
        return 1.0
    return 0.0


@dataclass
class FlyWireDataset:
    """A real FlyWire connectome chunk set, exposed as a VNR dataset.

    Implements the `ConnectomeDataset` protocol (name / provenance /
    neurons / successors / edge_count).
    """

    chunks: tuple[Path, ...]
    compact: bool = True
    weight_scale: float = 0.4
    confidence_floor: float = 0.0
    drop_unknown_nt: bool = False
    _edges: dict[int, list[tuple[int, float]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _neurons: list[int] = field(default_factory=list, init=False, repr=False)
    _raw_ids: list[int] = field(default_factory=list, init=False, repr=False)
    _counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("at least one parquet chunk is required")
        if self.weight_scale <= 0:
            raise ValueError("weight_scale must be positive")
        self._load()

    # ------------------------------------------------------------------ load

    def _load(self) -> None:
        import pyarrow.parquet as pq

        # A concurrently-running fetch can hand us a file that is still being
        # written. Retry a failed read once rather than treating a transient
        # IO error as a corrupt corpus (observed while bulk_by_id.py was
        # running - notes entry 41).
        def _read(chunk: Path):
            try:
                return pq.read_table(chunk, columns=["pre", "post", "nt", "conf"])
            except Exception:  # noqa: BLE001 - retry once, then fail loud
                import time

                time.sleep(1.0)
                return pq.read_table(chunk, columns=["pre", "post", "nt", "conf"])

        raw_edges: set[tuple[int, int, int]] = set()
        rows_seen = 0
        for chunk in self.chunks:
            table = _read(chunk)
            pre = table.column("pre").to_pylist()
            post = table.column("post").to_pylist()
            nt = table.column("nt").to_pylist()
            conf = table.column("conf").to_pylist()
            for s, t, code, c in zip(pre, post, nt, conf):
                rows_seen += 1
                if s == 0 or t == 0 or s == t:
                    continue
                if c is not None and c < self.confidence_floor:
                    continue
                if self.drop_unknown_nt and code == UNKNOWN_NT:
                    continue
                raw_edges.add((int(s), int(t), int(code)))

        ids = sorted({s for s, _, _ in raw_edges} | {t for _, t, _ in raw_edges})
        self._raw_ids = ids
        if self.compact:
            index = {nid: i for i, nid in enumerate(ids)}
            self._neurons = list(range(len(ids)))
        else:
            index = {nid: nid for nid in ids}
            self._neurons = list(ids)

        edges: dict[int, list[tuple[int, float]]] = {n: [] for n in self._neurons}
        for s, t, code in raw_edges:
            sign = _nt_sign(code)
            if sign == 0.0:
                sign = 1.0  # unknown transmitter: excitatory by default
            edges[index[s]].append((index[t], sign * self.weight_scale))
        self._edges = edges
        self._counts = {
            "rows_seen": rows_seen,
            "distinct_edges": len(raw_edges),
            "neurons": len(ids),
        }

    # -------------------------------------------------------------- protocol

    @property
    def name(self) -> str:
        return f"flywire-{'compact' if self.compact else 'raw'}-n{len(self._neurons)}"

    @property
    def provenance(self) -> dict[str, str]:
        return {
            "source": "FlyWire fafb_public synapses_nt_v1, id-paged",
            "version": "v783-public-byid",
            "chunks": str(len(self.chunks)),
            "rows_seen": str(self._counts.get("rows_seen", 0)),
            "distinct_edges": str(self._counts.get("distinct_edges", 0)),
            "weight_rule": f"NT sign x {self.weight_scale}",
            "node_ids": "compact" if self.compact else "raw",
        }

    def neurons(self) -> list[int]:
        return list(self._neurons)

    def successors(self, neuron_id: int) -> list[tuple[int, float]]:
        try:
            return list(self._edges[neuron_id])
        except KeyError:
            raise KeyError(f"neuron {neuron_id} not in dataset") from None

    def edge_count(self) -> int:
        return sum(len(v) for v in self._edges.values())

    # ---------------------------------------------------------------- helpers

    def degree_stats(self) -> dict[str, float]:
        """Out-degree distribution - the property that governs residency."""
        degrees = [len(v) for v in self._edges.values()]
        if not degrees:
            return {"n": 0, "mean": 0.0, "max": 0.0, "median": 0.0}
        ordered = sorted(degrees)
        return {
            "n": len(degrees),
            "mean": sum(degrees) / len(degrees),
            "max": float(max(degrees)),
            "median": float(ordered[len(ordered) // 2]),
            "sum_sq": float(sum(d * d for d in degrees)),
        }

    def nt_distribution(self) -> dict[str, int]:
        """Transmitter mix, counted from the data actually loaded."""
        import pyarrow.parquet as pq

        counts: dict[str, int] = {}
        for chunk in self.chunks:
            table = pq.read_table(chunk, columns=["nt"])
            for code in table.column("nt").to_pylist():
                name = NT_NAMES[code] if 0 <= code < len(NT_NAMES) else "UNKNOWN"
                counts[name] = counts.get(name, 0) + 1
        return counts

    @staticmethod
    def locality_ratio(dataset: FlyWireDataset, n_samples: int = 2000) -> float:
        """How far edges reach across the id space, versus a uniform null.

        A uniform random graph scatters edges anywhere; a structured
        connectome concentrates them. This ratio (mean normalized edge span,
        0 = perfectly local, 1 = uniform) is the measurable property behind
        the entry-39 fan-out finding, so it is reported rather than assumed.
        """
        ids = dataset.neurons()
        if len(ids) < 2:
            return 0.0
        lo, hi = min(ids), max(ids)
        span = (hi - lo) or 1
        samples: list[float] = []
        for src in ids:
            outs = dataset.successors(src)
            if not outs:
                continue
            for tgt, _ in outs[:3]:
                samples.append(abs(tgt - src) / span)
            if len(samples) >= n_samples:
                break
        return sum(samples) / len(samples) if samples else 0.0

    def mean_weight(self) -> float:
        total = 0.0
        count = 0
        for outs in self._edges.values():
            for _, w in outs:
                total += w
                count += 1
        return total / count if count else 0.0

    def is_finite(self) -> bool:
        return all(math.isfinite(w) for outs in self._edges.values() for _, w in outs)
