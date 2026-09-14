"""Find synapse tables + counts on public FlyWire (ops script, read-only)."""

import os

from vnr.connectome.credentials import load_token

os.environ["CAVE_TOKEN"] = load_token()

from caveclient import CAVEclient

client = CAVEclient("flywire_fafb_public", auth_token=os.environ["CAVE_TOKEN"])
for version in (783, 630):
    try:
        tables = client.materialize.get_tables(version=version)
    except Exception as exc:  # noqa: BLE001 — reconnaissance
        print(version, "get_tables failed:", type(exc).__name__, str(exc)[:150])
        continue
    syn = [t for t in tables if "synapse" in t.lower()]
    print(version, "tables:", len(tables), "synapse-like:", syn[:6])
    for table in syn[:2]:
        try:
            count = client.materialize.get_annotation_count(table, version=version)
            print("  ", table, "rows:", count)
        except Exception as exc:  # noqa: BLE001 — reconnaissance
            print("  ", table, "count failed:", type(exc).__name__, str(exc)[:120])
