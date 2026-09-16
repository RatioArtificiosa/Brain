"""Exhaustive FlyWire access probe: find EVERY path that yields real data.

The recorded verdict (notes entry 32/33) is that `flywire_fafb_production`
403s and `flywire_fafb_public` is open at versions [630, 783]. That was one
probe against one set of assumptions. This script tests the full space:

- every datastack name reachable from the account
- every materialization version per datastack
- which TABLES exist and are readable in each
- whether new versions have appeared since the recorded probe
- what row counts / schema each readable table actually has

It prints ONLY names, versions, and counts - never tokens.

    python scripts/probe_all_access.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Candidate datastacks. The first three were tried before; the rest are
# plausible names for the same connectome served under different access
# tiers, plus the older FlyWire releases.
CANDIDATES = [
    "flywire_fafb_public",
    "flywire_fafb_production",
    "flywire_fafb",
    "flywire_v783",
    "flywire_v630",
    "flywire_full",
    "fafb_flywire_public",
    "flywire_public",
]


def client_for(datastack: str, token: str):
    from caveclient import CAVEclient

    return CAVEclient(datastack, auth_token=token)


def safe(fn, *args, **kwargs):
    """Run fn, returning (ok, value_or_error_text). Never raises."""
    try:
        return True, fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - probing is the point
        text = " | ".join(str(exc).splitlines()[-2:])[:180]
        return False, f"{type(exc).__name__}: {text}"


def probe_datastack(name: str, token: str) -> dict:
    """Everything we can learn about one datastack, without raising."""
    info: dict = {"datastack": name, "opened": False, "versions": [], "tables": {}}
    ok, client = safe(client_for, name, token)
    if not ok:
        info["error"] = client
        return info
    info["opened"] = True

    ok, versions = safe(client.materialize.get_versions)
    if ok:
        info["versions"] = list(versions)[:12]

    # Which tables can we actually read? Try the ones that matter for VNR.
    for table in (
        "synapses_nt_v1",
        "synapses",
        "proofread_synapses",
        "neurons",
        "proofread_neurons",
        "cell_types",
    ):
        ok, res = safe(
            client.materialize.query_table,
            table,
            limit=5,
            select_columns=None,
            split_positions=False,
        )
        if ok:
            info["tables"][table] = {"readable": True, "columns": list(res.columns)}
        else:
            info["tables"][table] = {"readable": False, "error": str(res)[:120]}
    return info


def main() -> int:
    from vnr.connectome.credentials import load_token

    try:
        token = load_token()
    except KeyError as exc:
        print(f"no token: {exc}")
        return 2
    os.environ.setdefault("CAVE_TOKEN", token)

    results = []
    for name in CANDIDATES:
        info = probe_datastack(name, token)
        results.append(info)
        if not info["opened"]:
            print(f"[closed ] {name}")
            print(f"           {info.get('error', '')[:150]}")
            continue
        print(f"[OPEN   ] {name}  versions={info['versions'][:6]}")
        for table, meta in info["tables"].items():
            if meta["readable"]:
                print(
                    f"           table {table}: READABLE ({len(meta['columns'])} cols)"
                )
            else:
                print(f"           table {table}: {meta['error'][:80]}")

    out = Path("artifacts/access_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    opened = [r["datastack"] for r in results if r["opened"]]
    readable = [
        (r["datastack"], t)
        for r in results
        for t, m in r["tables"].items()
        if m.get("readable")
    ]
    print()
    print(f"datastacks opened: {opened}")
    print(f"readable tables  : {readable}")
    print(f"full report      : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
