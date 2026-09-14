"""Connectome reporting: stats JSON + readable HTML for any dataset (WI02 leftover).

Runs on any ``ConnectomeDataset`` — toy today, FlyWire the moment bytes land.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vnr.connectome.dataset import ConnectomeDataset

__all__ = ["connectome_report", "write_report"]


def connectome_report(dataset: ConnectomeDataset) -> dict[str, Any]:
    """Structural stats: counts, density, degrees, isolation, reciprocity."""
    neurons = dataset.neurons()
    n = len(neurons)
    out_deg: dict[int, int] = {}
    edges = 0
    mutual = 0
    adj: dict[int, set[int]] = {}
    for nid in neurons:
        outs = dataset.successors(nid)
        adj[nid] = {t for t, _ in outs}
        out_deg[nid] = len(outs)
        edges += len(outs)
    for nid in neurons:
        for tgt in adj[nid]:
            if nid in adj.get(tgt, ()):
                mutual += 1
    isolated = sum(
        1
        for nid in neurons
        if out_deg[nid] == 0 and not any(nid in adj[o] for o in neurons)
    )
    possible = n * (n - 1) if n > 1 else 0
    return {
        "dataset": dataset.name,
        "provenance": dataset.provenance,
        "neurons": n,
        "edges": edges,
        "density": (edges / possible) if possible else 0.0,
        "mean_out_degree": (edges / n) if n else 0.0,
        "max_out_degree": max(out_deg.values()) if out_deg else 0,
        "isolated_neurons": isolated,
        "reciprocal_edges": mutual,
        "reciprocity": (mutual / edges) if edges else 0.0,
    }


_HTML = """<html><head><title>Connectome report: {dataset}</title>
<style>body{{font-family:sans-serif;background:#0b1220;color:#e2e8f0;margin:40px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #334155;padding:6px 14px;text-align:left}}
th{{background:#1e293b}}</style></head><body>
<h1>Connectome report: {dataset}</h1>
<table>{rows}</table>
<p>Provenance: {provenance}</p>
</body></html>"""


def write_report(report: dict[str, Any], path: Path) -> Path:
    """Write JSON, or HTML when the path ends in .html."""
    path = Path(path)
    if path.suffix == ".html":
        rows = "".join(
            f"<tr><th>{k}</th><td>{v}</td></tr>"
            for k, v in report.items()
            if k != "provenance"
        )
        path.write_text(
            _HTML.format(
                dataset=report.get("dataset"),
                rows=rows,
                provenance=report.get("provenance"),
            ),
            encoding="utf-8",
        )
    else:
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path
