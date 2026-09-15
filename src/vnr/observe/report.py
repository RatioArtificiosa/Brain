"""Report rendering from a run record (plan §83, spec §5).

A report is where the project's central discipline becomes visible: the
HYPOTHESIS, the OBSERVATION, and the INTERPRETATION are three separate
sections and never merged into one confident paragraph. A reader must always
be able to tell which sentences are measurements and which are opinion.

Renders JSON (machine-readable, stable keys) or a self-contained dark HTML
page with no external assets — it opens from disk, offline, forever.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

__all__ = ["load_run", "render_html", "render_json", "report_from_directory"]

_CONDITION_LABEL = {
    "explicit_dense": "Explicit dense",
    "sparse": "Sparse (lazy state)",
    "procedural": "Procedural edges",
    "procedural_virtualized": "Procedural + virtualized",
}


def load_run(run_dir: Path) -> dict[str, Any]:
    """Read ``run.json`` from a run directory."""
    path = Path(run_dir) / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"no run.json in {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def report_from_directory(run_dir: Path) -> dict[str, Any]:
    """Build the report structure from a run record."""
    record = load_run(run_dir)
    conditions = record.get("conditions", [])
    spec = record.get("spec", {})
    n_neurons = spec.get("n_neurons", 0)

    rows = []
    for cond in conditions:
        fid = cond.get("fidelity") or {}
        stored = cond.get("stored_synapses", 0)
        resident = cond.get("resident_synapses", 0)
        peak = cond.get("peak_resident", 0)
        ratio = (stored / resident) if resident else float("inf")
        rows.append(
            {
                "condition": cond["condition"],
                "label": _CONDITION_LABEL.get(cond["condition"], cond["condition"]),
                "spikes": cond.get("total_spikes", 0),
                "events": cond.get("events_delivered", 0),
                "peak_resident": peak,
                "resident_fraction": (peak / n_neurons) if n_neurons else 0.0,
                "evictions": cond.get("evictions", 0),
                "materializations": cond.get("materializations", 0),
                "stored_synapses": stored,
                "resident_synapses": resident,
                "compression_ratio": ratio,
                "rate_ratio": fid.get("rate_ratio"),
                "count_correlation": fid.get("count_correlation"),
                "binned_cosine": fid.get("binned_cosine"),
                "gate_pass": fid.get("overall_pass"),
                "wall_seconds": cond.get("wall_seconds", 0.0),
                "health_ok": cond.get("health_ok"),
            }
        )

    return {
        "experiment_id": record.get("experiment_id"),
        "run_id": record.get("run_id"),
        "created": record.get("created"),
        "seed": record.get("seed"),
        "config_hash": record.get("config_hash"),
        "commit": record.get("commit"),
        "python": record.get("python"),
        "hardware": record.get("hardware", {}),
        "spec": spec,
        "conditions": rows,
        "hypothesis": record.get("hypothesis", ""),
        "observation": record.get("observation", ""),
        "interpretation": record.get("interpretation", ""),
    }


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, default=str)


_CSS = """
:root{--bg:#0a0e14;--panel:#121821;--line:#1e2836;--ink:#e6edf3;--dim:#8b98a8;
--accent:#4da3ff;--good:#3fb950;--warn:#d29922;--bad:#f85149}
*{box-sizing:border-box}
body{margin:0;padding:48px 32px;background:var(--bg);color:var(--ink);
font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1040px;margin:0 auto}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.02em;font-weight:650}
h2{font-size:15px;text-transform:uppercase;letter-spacing:.09em;color:var(--dim);
margin:38px 0 12px;font-weight:600}
.sub{color:var(--dim);font-size:13px;margin-bottom:4px}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 8px}
.chip{background:var(--panel);border:1px solid var(--line);border-radius:999px;
padding:4px 12px;font-size:12px;color:var(--dim)}
.chip b{color:var(--ink);font-weight:600}
table{width:100%;border-collapse:collapse;background:var(--panel);
border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:11px 14px;text-align:right;border-bottom:1px solid var(--line);
font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left;font-variant-numeric:normal}
th{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);
font-weight:600;background:#0e141c}
tr:last-child td{border-bottom:none}
.pass{color:var(--good);font-weight:600}
.fail{color:var(--bad);font-weight:600}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:18px 20px;margin:10px 0}
.card.hyp{border-left:3px solid var(--accent)}
.card.obs{border-left:3px solid var(--good)}
.card.int{border-left:3px solid var(--warn)}
.card pre{margin:0;white-space:pre-wrap;font:13px/1.7 ui-monospace,SFMono-Regular,
Consolas,monospace;color:var(--ink)}
.note{color:var(--dim);font-size:13px;margin-top:10px}
footer{margin-top:44px;color:var(--dim);font-size:12px;border-top:1px solid var(--line);
padding-top:16px}
"""


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return html.escape(str(value))


def render_html(report: dict[str, Any]) -> str:
    """Self-contained dark report page (no external assets, opens offline)."""
    esc = html.escape
    n_neurons = report.get("spec", {}).get("n_neurons", 0)
    ticks = report.get("spec", {}).get("ticks", 0)

    head = [
        f"<h1>{esc(str(report.get('experiment_id')))}</h1>",
        f'<div class="sub">{esc(str(report.get("run_id")))}</div>',
        '<div class="meta">',
        f'<span class="chip">seed <b>{esc(str(report.get("seed")))}</b></span>',
        f'<span class="chip">neurons <b>{_fmt(n_neurons, 0)}</b></span>',
        f'<span class="chip">ticks <b>{_fmt(ticks, 0)}</b></span>',
        f'<span class="chip">commit <b>{esc(str(report.get("commit")))}</b></span>',
        f'<span class="chip">python <b>{esc(str(report.get("python")))}</b></span>',
        f'<span class="chip">config <b>{esc(str(report.get("config_hash"))[:12])}</b></span>',
        f'<span class="chip">created <b>{esc(str(report.get("created")))}</b></span>',
        "</div>",
    ]

    cols = [
        "Condition",
        "Spikes",
        "Peak resident",
        "Resident %",
        "Evictions",
        "Rate ratio",
        "Count r",
        "Timing cos",
        "Gate",
        "Wall",
    ]
    body = ["<h2>Conditions - measured</h2>", "<table><thead><tr>"]
    body += [f"<th>{esc(c)}</th>" for c in cols]
    body += ["</tr></thead><tbody>"]
    for row in report.get("conditions", []):
        frac = row.get("resident_fraction", 0.0) * 100.0
        gate = row.get("gate_pass")
        gate_html = (
            '<span class="pass">PASS</span>'
            if gate
            else '<span class="fail">FAIL</span>'
        )
        cells = [
            esc(str(row.get("label"))),
            _fmt(row.get("spikes"), 0),
            _fmt(row.get("peak_resident"), 0),
            f"{frac:.1f}%",
            _fmt(row.get("evictions"), 0),
            _fmt(row.get("rate_ratio")),
            _fmt(row.get("count_correlation")),
            _fmt(row.get("binned_cosine")),
            gate_html,
            f"{row.get('wall_seconds', 0.0):.3f}s",
        ]
        body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    body += ["</tbody></table>"]
    body.append(
        '<p class="note">The explicit-dense condition is the reference: every '
        "ratio is scored against its spike trains. Colors report the gate, not "
        "a quality judgement.</p>"
    )

    sections = [
        ("hyp", "Hypothesis (stated before the run)", report.get("hypothesis", "")),
        (
            "obs",
            "Observation (what the instruments reported)",
            report.get("observation", ""),
        ),
        (
            "int",
            "Interpretation (what we think it means)",
            report.get("interpretation", ""),
        ),
    ]
    for cls, title, text in sections:
        body.append(f"<h2>{esc(title)}</h2>")
        body.append(f'<div class="card {cls}"><pre>{esc(str(text))}</pre></div>')

    body.append("<footer>")
    body.append(
        "Generated by <code>vnr report</code>. Hypothesis, observation, and "
        "interpretation are recorded separately and never merged (plan §5). "
        "Null results ship like any other."
    )
    body.append("</footer>")

    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(str(report.get('experiment_id')))} - VNR report</title>"
        f'<style>{_CSS}</style></head><body><div class="wrap">'
        + "".join(head)
        + "".join(body)
        + "</div></body></html>\n"
    )
