"""Analyze an E004 sweep result: where is the knee, and does structure win?

Reads the JSON written by scripts/e004_sweep.py and reports:
  1. the compression-vs-fidelity curve for each condition;
  2. the CONTROL comparison - does real connectome beat the null models;
  3. whether the knee was found at all (a fidelity failure), and if not,
     what compression headroom remains unexplored.
"""

import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/e004_sweep.json")
data = json.loads(path.read_text(encoding="utf-8"))

print(
    f"base: {data['base_neurons']:,} neurons, {data['base_edges']:,} edges, "
    f"{data['ticks']} ticks, seed {data['seed']}"
)
print()

order = ["real", *data["controls"].keys()]
print(
    f"{'condition':<26}{'budget':>8}{'%res':>8}{'evict':>8}"
    f"{'rate':>9}{'cos':>11}{'gate':>6}"
)
print("-" * 76)

knee_found = False
best = None
for kind in order:
    points = data["real"] if kind == "real" else data["controls"][kind]
    for i, p in enumerate(points):
        label = kind if i == 0 else ""
        print(
            f"{label:<26}{p['budget_fraction'] * 100:>7.2f}%"
            f"{p['resident_fraction'] * 100:>7.1f}%{p['evictions']:>8,}"
            f"{p['rate_ratio']:>9.4f}{p['binned_cosine']:>11.6f}"
            f"{'PASS' if p['gate_pass'] else 'FAIL':>6}"
        )
        if not p["gate_pass"]:
            knee_found = True
        elif kind == "real" and (
            best is None or p["resident_fraction"] < best["resident_fraction"]
        ):
            best = p

print()
print("=== MINIMUM ACHIEVED RESIDENCY AT PASSING FIDELITY ===")
for kind in order:
    points = data["real"] if kind == "real" else data["controls"][kind]
    passing = [p for p in points if p["gate_pass"]]
    if passing:
        low = min(passing, key=lambda p: p["resident_fraction"])
        print(
            f"  {kind:<26} {low['resident_fraction'] * 100:>5.1f}% resident "
            f"at {low['budget_fraction'] * 100:.2f}% budget, cos={low['binned_cosine']:.6f}"
        )
    else:
        print(f"  {kind:<26} no passing point")

print()
print("=== CONTROL COMPARISON (the scientific claim) ===")
if best:
    real_low = best["resident_fraction"]
    print(f"real connectome minimum passing residency: {real_low * 100:.1f}%")
    for ctrl, points in data["controls"].items():
        passing = [p for p in points if p["gate_pass"]]
        if not passing:
            continue
        c_low = min(passing, key=lambda p: p["resident_fraction"])["resident_fraction"]
        ratio = c_low / real_low if real_low else float("inf")
        verdict = "real WINS" if c_low > real_low else "control ties/wins"
        print(f"  vs {ctrl:<26} {c_low * 100:>5.1f}%  ({ratio:.2f}x)  {verdict}")

print()
if knee_found:
    print("KNEE LOCATED: at least one condition failed fidelity at some budget.")
else:
    print("NO KNEE FOUND: fidelity held at every tested budget.")
    print("That means compression headroom remains UNEXPLORED - the curve must")
    print("be pushed further (smaller budget, weaker drive, or longer runs)")
    print("before any compression claim can be made.")
